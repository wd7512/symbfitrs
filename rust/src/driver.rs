//! The installable fitter's entry points: model evaluation, one fit of a compiled model, and a
//! threaded batch of fits.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::expr::Compiled;
use crate::fit::fit_with;
use crate::problem::Problem;
use crate::solvers::core::Opts;
use crate::solvers::Status;
use crate::tape::Bound;
use crate::tracker::Tracker;

/// Data for one fit: one column per variable, targets, optional weights and start.
#[derive(Clone, Debug, Default)]
pub struct Data {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
    pub w: Option<Vec<f64>>,
    pub p0: Option<Vec<f64>>,
    /// Record θ after every accepted update (entry 0 is the start).
    pub trace: bool,
    /// Stop after this many accepted updates.
    pub max_steps: Option<usize>,
    /// Further starts after `p0`: each is fitted in turn and the lowest loss is kept.
    pub more_starts: Vec<Vec<f64>>,
    /// Box constraints per parameter (±∞ = free): an active-set LM keeps every iterate inside,
    /// and each start is first moved into the box.
    pub lower: Option<Vec<f64>>,
    pub upper: Option<Vec<f64>>,
}

#[derive(Clone, Debug)]
pub struct Outcome {
    /// Parameter names, in the order of `params`.
    pub names: Vec<String>,
    /// Best parameters found (the start if no evaluation was finite).
    pub params: Vec<f64>,
    /// Mean of the squared (weighted) residuals at `params`.
    pub loss: f64,
    /// converged | stalled | deadline | failure
    pub status: &'static str,
    pub message: String,
    pub nfev: usize,
    /// Evaluations that gave a non-finite loss (skipped by the solver).
    pub n_nonfinite: usize,
    pub time_ms: f64,
    /// The best loss each time it improved, and when (ms since the call).
    pub history_ms: Vec<f64>,
    pub history_loss: Vec<f64>,
    /// Accepted updates completed, and (with `trace`) θ at step 0..=steps.
    pub steps: usize,
    pub trace: Option<Vec<Vec<f64>>>,
    /// Every start's final loss and parameters, in start order (one entry for a single start).
    pub start_losses: Vec<f64>,
    pub start_params: Vec<Vec<f64>>,
}

/// f(x; θ) at every data point.
pub fn predict(c: &Compiled, x: &[Vec<f64>], theta: &[f64]) -> Result<Vec<f64>, String> {
    let p = c.params.len();
    if theta.len() != p {
        return Err(format!("params has {} values, the model has {p} parameters", theta.len()));
    }
    if x.len() != c.variables.len() {
        return Err(format!("expected {} variable columns, got {}", c.variables.len(), x.len()));
    }
    let m = x.first().map_or(1, Vec::len);
    if x.iter().any(|col| col.len() != m) {
        return Err("x columns have different lengths".into());
    }
    let mut b = Bound::new(&c.f, x, m, p)?;
    b.eval(theta);
    let mut out = vec![0.0; m];
    b.output_into(0, &mut out);
    Ok(out)
}

/// Fit `c` to `data` within `budget_ms`, timed from the call. With several starts, each start
/// gets an equal share of what is left of the budget when it begins; the lowest loss is kept, and
/// the result's status, steps, history and trace are that start's.
pub fn fit_one(c: &Compiled, data: Data, budget_ms: f64) -> Result<Outcome, String> {
    let t0 = Instant::now();
    let p = c.params.len();
    let mut starts = vec![match data.p0 {
        Some(ref p0) if p0.len() != p => return Err(format!("p0 has {} values, the model has {p} parameters", p0.len())),
        Some(ref p0) => p0.clone(),
        None => vec![1.0; p],
    }];
    for s in &data.more_starts {
        if s.len() != p {
            return Err(format!("a start has {} values, the model has {p} parameters", s.len()));
        }
        starts.push(s.clone());
    }
    for (name, b) in [("lower", &data.lower), ("upper", &data.upper)] {
        if b.as_ref().is_some_and(|b| b.len() != p) {
            return Err(format!("{name} bounds must have {p} values, one per parameter"));
        }
    }
    let lo = data.lower.clone().unwrap_or_else(|| vec![f64::NEG_INFINITY; p]);
    let hi = data.upper.clone().unwrap_or_else(|| vec![f64::INFINITY; p]);
    if (0..p).any(|j| !(lo[j] <= hi[j] && lo[j] < f64::INFINITY && hi[j] > f64::NEG_INFINITY)) {
        return Err("bounds must have lower <= upper, lower < +inf and upper > -inf".into());
    }
    if data.lower.is_some() || data.upper.is_some() {
        // every start is moved into the box, so even a fit that fails at once reports a point inside it
        for theta in &mut starts {
            for (j, t) in theta.iter_mut().enumerate() {
                *t = t.clamp(lo[j], hi[j]);
            }
        }
    }
    let opts = Opts { lower: data.lower.clone(), upper: data.upper.clone(), ..Opts::default() };
    // bound once: the problem holds only evaluation buffers, so every start reuses it
    let mut prob = Problem::new(c, &data.x, data.y, data.w)?;
    let k = starts.len();
    let (mut nfev, mut n_nonfinite) = (0, 0);
    let (mut start_losses, mut start_params) = (Vec::with_capacity(k), Vec::with_capacity(k));
    let mut best: Option<(f64, Outcome)> = None;
    for (s, theta0) in starts.into_iter().enumerate() {
        let begun_ms = t0.elapsed().as_secs_f64() * 1e3;
        let share = (budget_ms - begun_ms) / (k - s) as f64;
        let mut tr = Tracker::new(share, &[], prob.inv_scale);
        tr.enable_trace(data.trace, data.max_steps);
        let status = match catch_unwind(AssertUnwindSafe(|| fit_with(&mut prob, &theta0, &mut tr, &opts).0)) {
            Ok(s) => s,
            Err(e) => {
                let msg = e.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| e.downcast_ref::<String>().cloned());
                Status::Failure(format!("panicked: {}", msg.unwrap_or_default()))
            }
        };
        nfev += tr.nfev;
        n_nonfinite += tr.n_nonfinite;
        let loss = if tr.best_theta.is_some() { tr.best_loss } else { f64::NAN };
        let params = tr.best_theta.clone().unwrap_or(theta0);
        start_losses.push(loss);
        start_params.push(params.clone());
        let better = match &best {
            None => true,
            Some((b, _)) => loss < *b || (b.is_nan() && !loss.is_nan()),
        };
        if better {
            let message = match &status {
                Status::Failure(m) => m.clone(),
                _ => String::new(),
            };
            let o = Outcome {
                names: c.params.clone(),
                params,
                loss,
                status: status.kind(),
                message,
                nfev: 0,
                n_nonfinite: 0,
                time_ms: 0.0,
                history_ms: tr.hist_t_ms.iter().map(|t| t + begun_ms).collect(),
                history_loss: tr.hist_loss,
                steps: tr.steps,
                trace: tr.trace,
                start_losses: Vec::new(),
                start_params: Vec::new(),
            };
            best = Some((loss, o));
        }
    }
    let (_, mut o) = best.expect("at least one start");
    o.nfev = nfev;
    o.n_nonfinite = n_nonfinite;
    o.time_ms = t0.elapsed().as_secs_f64() * 1e3;
    o.start_losses = start_losses;
    o.start_params = start_params;
    Ok(o)
}

/// One batch job: a compiled model (or why it could not be compiled) and its data.
pub type Job = (Result<Arc<Compiled>, String>, Data);

/// Run `jobs` on `threads` threads, results in job order. Each job's budget starts when a thread
/// picks it up. `done` counts finished jobs; setting `cancel` stops picking up new ones (their
/// results are `Err("cancelled")`).
pub fn fit_many(
    jobs: Vec<Job>,
    budget_ms: f64,
    threads: usize,
    done: &AtomicUsize,
    cancel: &AtomicBool,
) -> Vec<Result<Outcome, String>> {
    let n = jobs.len();
    let slots: Vec<Mutex<Option<Job>>> = jobs.into_iter().map(|j| Mutex::new(Some(j))).collect();
    let results: Vec<Mutex<Result<Outcome, String>>> = (0..n).map(|_| Mutex::new(Err("cancelled".into()))).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads.clamp(1, n.max(1)) {
            s.spawn(|| loop {
                if cancel.load(Ordering::Relaxed) {
                    return;
                }
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    return;
                }
                let (model, data) = slots[i].lock().unwrap().take().expect("each job is taken once");
                let r = model.and_then(|c| fit_one(&c, data, budget_ms));
                *results[i].lock().unwrap() = r;
                done.fetch_add(1, Ordering::Relaxed);
            });
        }
    });
    results.into_iter().map(|r| r.into_inner().unwrap()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::compile;

    fn exp_data(a: f64, b: f64, c: f64) -> Data {
        let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let y = x.iter().map(|t| a * (-b * t).exp() + c).collect();
        Data { x: vec![x], y, ..Default::default() }
    }

    #[test]
    fn fits_an_exponential() {
        let c = compile("a*exp(-b*x) + c", &["x".into()], None).unwrap();
        let o = fit_one(&c, exp_data(2.0, 0.7, 0.3), 100.0).unwrap();
        assert_eq!(o.status, "converged", "{o:?}");
        for (got, want) in o.params.iter().zip([2.0, 0.7, 0.3]) {
            assert!((got - want).abs() < 1e-8, "{o:?}");
        }
        assert!(o.loss < 1e-20);
        assert_eq!(o.history_loss.last(), Some(&o.loss));
        assert!(o.history_ms.windows(2).all(|w| w[0] <= w[1]));
        assert!(o.history_loss.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn trace_and_step_cap() {
        let c = compile("a*exp(-b*x) + c", &["x".into()], None).unwrap();
        let mut d = exp_data(2.0, 0.7, 0.3);
        d.p0 = Some(vec![1.0, 1.0, 0.0]);
        d.trace = true;
        let full = fit_one(&c, d.clone(), 100.0).unwrap();
        let t = full.trace.as_ref().unwrap();
        assert_eq!(t.len(), full.steps + 1);
        assert_eq!(t[0].len(), 3);
        d.max_steps = Some(2);
        let capped = fit_one(&c, d, 100.0).unwrap();
        assert_eq!((capped.status, capped.steps), ("max_steps", 2));
        assert_eq!(capped.trace.as_ref().unwrap(), &t[..3]);
        // max_steps = 0 stops at the start
        let mut d2 = exp_data(2.0, 0.7, 0.3);
        d2.p0 = Some(vec![1.0, 1.0, 0.0]);
        d2.trace = true;
        d2.max_steps = Some(0);
        let z = fit_one(&c, d2, 100.0).unwrap();
        assert_eq!((z.status, z.steps, z.trace.as_ref().map(Vec::len)), ("max_steps", 0, Some(1)));
        // tracing off: nothing recorded
        assert!(fit_one(&c, exp_data(2.0, 0.7, 0.3), 100.0).unwrap().trace.is_none());
    }

    #[test]
    fn several_starts_keep_the_best_and_report_every_start() {
        // sin(c*x) has many local minima in c; a start near the truth finds it, a far one does not
        let c = compile("a*sin(c*x)", &["x".into()], None).unwrap();
        let x: Vec<f64> = (0..200).map(|i| i as f64 * 0.05).collect();
        let y: Vec<f64> = x.iter().map(|t| 1.5 * (3.0 * t).sin()).collect();
        let far = Data { x: vec![x.clone()], y: y.clone(), p0: Some(vec![1.0, 0.5]), ..Default::default() };
        let one = fit_one(&c, far.clone(), 200.0).unwrap();
        assert!(one.loss > 1e-3, "the far start alone should miss: {one:?}");
        assert_eq!((one.start_losses.len(), one.start_params.len()), (1, 1));
        let d = Data { more_starts: vec![vec![1.0, 2.9], vec![1.0, 9.0]], trace: true, ..far };
        let o = fit_one(&c, d, 300.0).unwrap();
        assert!(o.loss < 1e-20 && (o.params[1] - 3.0).abs() < 1e-8, "{o:?}");
        assert_eq!(o.start_losses.len(), 3);
        assert_eq!(o.start_losses[1], o.loss);
        assert_eq!(o.start_params[1], o.params);
        assert_eq!(o.start_losses[0], one.loss, "the first start is fitted exactly as on its own");
        assert_eq!(o.trace.unwrap()[0][1], 2.9, "the trace is the winning start's (a, linear, is solved at step 0)");
        assert!(o.nfev > one.nfev);
        let bad = Data { more_starts: vec![vec![1.0]], ..exp_data(1.0, 1.0, 1.0) };
        assert!(fit_one(&compile("a*exp(-b*x) + c", &["x".into()], None).unwrap(), bad, 100.0).is_err());
    }

    #[test]
    fn bounds_hold_the_optimum_on_the_bound_and_inactive_bounds_change_nothing() {
        let c = compile("a*exp(-b*x) + c", &["x".into()], None).unwrap();
        let free = fit_one(&c, exp_data(2.0, 0.7, 0.3), 100.0).unwrap();
        // b is the only nonlinear parameter: the true value 0.7 lies above the cap of 0.5
        let mut d = exp_data(2.0, 0.7, 0.3);
        d.p0 = Some(vec![1.0, 0.2, 0.0]);
        d.trace = true;
        d.lower = Some(vec![f64::NEG_INFINITY; 3]);
        d.upper = Some(vec![f64::INFINITY, 0.5, f64::INFINITY]);
        let o = fit_one(&c, d, 100.0).unwrap();
        assert!((o.params[1] - 0.5).abs() < 1e-12, "{o:?}");
        assert!(o.trace.unwrap().iter().all(|t| t[1] <= 0.5 + 1e-12));
        // bounds that never bind give the unbounded trajectory exactly
        let mut d = exp_data(2.0, 0.7, 0.3);
        d.lower = Some(vec![f64::NEG_INFINITY, -1e6, f64::NEG_INFINITY]); // only the nonlinear b: bounding a linear one turns projection off
        d.upper = Some(vec![f64::INFINITY, 1e6, f64::INFINITY]);
        let wide = fit_one(&c, d, 100.0).unwrap();
        assert_eq!((wide.status, wide.steps), (free.status, free.steps));
        assert_eq!(wide.params, free.params);
    }

    #[test]
    fn a_bounded_linear_parameter_is_held_at_its_bound() {
        let c = compile("a*exp(-b*x) + c", &["x".into()], None).unwrap();
        let mut d = exp_data(-2.0, 0.7, 0.3); // the true amplitude is negative
        d.lower = Some(vec![0.0, f64::NEG_INFINITY, f64::NEG_INFINITY]);
        d.upper = Some(vec![f64::INFINITY; 3]);
        let o = fit_one(&c, d, 200.0).unwrap();
        assert!(o.params[0].abs() < 1e-12 && o.params[0] >= 0.0, "{o:?}");
    }

    #[test]
    fn bad_bounds_are_rejected() {
        let c = compile("a*exp(-b*x) + c", &["x".into()], None).unwrap();
        let mut d = exp_data(2.0, 0.7, 0.3);
        d.lower = Some(vec![0.0; 2]);
        assert!(fit_one(&c, d, 100.0).is_err());
        let mut d = exp_data(2.0, 0.7, 0.3);
        d.lower = Some(vec![1.0; 3]);
        d.upper = Some(vec![0.0; 3]);
        assert!(fit_one(&c, d, 100.0).is_err());
        let mut d = exp_data(2.0, 0.7, 0.3);
        d.lower = Some(vec![f64::INFINITY; 3]);
        assert!(fit_one(&c, d, 100.0).is_err(), "a lower bound of +inf leaves no point in the box");
    }

    #[test]
    fn every_start_is_bounded() {
        let c = compile("a*exp(-b*x) + c", &["x".into()], None).unwrap();
        let mut d = exp_data(2.0, 0.7, 0.3);
        d.p0 = Some(vec![1.0, 3.0, 0.0]);
        d.more_starts = vec![vec![1.0, 0.1, 0.0], vec![1.0, -4.0, 0.0]];
        d.lower = Some(vec![f64::NEG_INFINITY, 0.0, f64::NEG_INFINITY]);
        d.upper = Some(vec![f64::INFINITY, 0.5, f64::INFINITY]);
        let o = fit_one(&c, d, 300.0).unwrap();
        assert!(o.start_params.iter().all(|t| (0.0..=0.5).contains(&t[1])), "{o:?}");
        assert!((o.params[1] - 0.5).abs() < 1e-12, "{o:?}");
    }

    #[test]
    fn predict_matches_the_expression() {
        let c = compile("a*exp(-b*x) + c", &["x".into()], None).unwrap();
        let f = predict(&c, &[vec![0.0, 1.0]], &[2.0, 0.5, 1.0]).unwrap();
        assert!((f[0] - 3.0).abs() < 1e-15 && (f[1] - (2.0 * (-0.5f64).exp() + 1.0)).abs() < 1e-15);
        assert!(predict(&c, &[vec![0.0]], &[1.0]).is_err());
    }

    #[test]
    fn batch_keeps_order_and_reports_bad_jobs() {
        let vars = vec!["x".to_string()];
        let model = |e: &str| compile(e, &vars, None).map(Arc::new);
        let jobs: Vec<Job> = vec![
            (model("a*exp(-b*x) + c"), exp_data(2.0, 0.7, 0.3)),
            (model("a*exp(-b*x) +"), exp_data(1.0, 1.0, 1.0)),
            (model("a*exp(-b*x) + c"), exp_data(-1.0, 0.2, 5.0)),
            (model("a*x + c"), Data { x: vec![vec![1.0, 2.0]], y: vec![1.0], ..Default::default() }),
        ];
        let (done, cancel) = (AtomicUsize::new(0), AtomicBool::new(false));
        let r = fit_many(jobs, 100.0, 3, &done, &cancel);
        assert_eq!(done.load(Ordering::Relaxed), 4);
        assert!((r[0].as_ref().unwrap().params[0] - 2.0).abs() < 1e-8);
        assert!(r[1].is_err() && r[3].is_err());
        assert!((r[2].as_ref().unwrap().params[2] - 5.0).abs() < 1e-8);
    }
}
