//! Levenberg–Marquardt.
//!
//! Every iteration factorises the column-scaled Jacobian once (QR→SVD, see [`crate::linalg`]);
//! retries with a larger damping reuse that factorisation. A trial is accepted iff its residual
//! sum of squares is finite and strictly smaller.

use nalgebra::{DMatrix, DVector};

use super::{Lsq, Status};
use crate::linalg::factor;
use crate::tracker::Stop;

#[derive(Debug, Clone)]
pub struct Opts {
    pub ftol: f64,
    pub xtol: f64,
    pub gtol: f64,
    pub tau: f64,
    pub loss_floor: f64,
    /// Box constraints on the variables (one entry each, ±∞ = free). Active-set: a variable on a
    /// bound whose descent direction points outward is held fixed for that iteration, and every
    /// trial point is clipped into the box. `None` on both = unbounded (the default behaviour).
    pub lower: Option<Vec<f64>>,
    pub upper: Option<Vec<f64>>,
}

impl Default for Opts {
    fn default() -> Self {
        Opts { ftol: 1e-15, xtol: 1e-15, gtol: 1e-15, tau: 1e-3, loss_floor: 1e-32, lower: None, upper: None }
    }
}

pub fn solve<F: Lsq>(f: &mut F, x0: &DVector<f64>, o: &Opts) -> Status {
    match run(f, x0, o) {
        Ok(s) => s,
        Err(Stop::Deadline) => Status::Deadline,
    }
}

fn run<F: Lsq>(f: &mut F, x0: &DVector<f64>, o: &Opts) -> Result<Status, Stop> {
    let (m, n) = (f.m(), f.n());
    let bounded = o.lower.is_some() || o.upper.is_some();
    let lo = DVector::from_vec(o.lower.clone().unwrap_or_else(|| vec![f64::NEG_INFINITY; n]));
    let hi = DVector::from_vec(o.upper.clone().unwrap_or_else(|| vec![f64::INFINITY; n]));
    if lo.len() != n || hi.len() != n || (0..n).any(|j| lo[j].is_nan() || hi[j].is_nan() || lo[j] > hi[j]) {
        return Ok(Status::Failure("bounds must have one entry per parameter, with lower <= upper".into()));
    }
    let mut x = x0.clone();
    if bounded {
        for j in 0..n {
            x[j] = x[j].clamp(lo[j], hi[j]); // the start is projected into the box
        }
    }
    let mut r = DVector::zeros(m);
    let mut sse = f.residual(&x, &mut r)?;
    if !sse.is_finite() {
        return Ok(Status::Failure("non-finite initial residual".into()));
    }
    f.accept();
    if f.tracker().accept_step(true) {
        return Ok(Status::MaxSteps);
    }
    let floor = o.loss_floor / f.tracker().inv_scale();
    let mut jac = DMatrix::zeros(m, n);
    let mut js = DMatrix::zeros(m, n);
    let mut d: DVector<f64> = DVector::from_element(n, 0.0);
    let mut delta = DVector::zeros(n);
    let mut step = DVector::zeros(n);
    let mut xt = DVector::zeros(n);
    let mut rt = DVector::zeros(m);
    let mut lam = -1.0; // initialised from the first factorisation that needs it
    let mut nu = 2.0;
    loop {
        if sse <= floor {
            return Ok(Status::Converged);
        }
        if !f.jacobian(&x, &mut jac)? {
            return Ok(Status::Failure("non-finite Jacobian".into()));
        }
        // MINPACK scaling: d_j = max over iterations of ‖J_j‖ (1 for zero columns)
        for j in 0..n {
            let cn = jac.column(j).norm();
            d[j] = d[j].max(if cn > 0.0 { cn } else { 1.0 });
            let mut col = js.column_mut(j);
            col.copy_from(&jac.column(j));
            col.unscale_mut(d[j]);
        }
        // gradient (cosine) test on the scaled problem
        let rn = r.norm();
        let grad = js.tr_mul(&r);
        // variables held fixed this iteration: on a bound with the descent direction pointing outward
        let free: Vec<usize> = (0..n)
            .filter(|&j| !(bounded && ((x[j] <= lo[j] && grad[j] > 0.0) || (x[j] >= hi[j] && grad[j] < 0.0))))
            .collect();
        let all_free = free.len() == n;
        let gmax: f64 = free
            .iter()
            .map(|&j| {
                let cn = js.column(j).norm();
                if cn > 0.0 { grad[j].abs() / (cn * rn) } else { 0.0_f64 }
            })
            .fold(0.0, f64::max);
        if gmax <= o.gtol {
            return Ok(Status::Converged);
        }
        let jf;
        let jsr: &DMatrix<f64> = if all_free {
            &js
        } else {
            jf = js.select_columns(free.iter());
            &jf
        };
        let Some(fac) = factor(jsr, &r, false) else {
            return Ok(Status::Failure("factorisation failed".into()));
        };
        let mut stepf = DVector::zeros(free.len());
        let xs_norm = x.component_mul(&d).norm();
        if lam < 0.0 {
            lam = o.tau * fac.smax * fac.smax;
        }
        let (pred, sse_t) = loop {
            let mut pred = if all_free {
                fac.step(lam, &mut step)
            } else {
                let pf = fac.step(lam, &mut stepf);
                step.fill(0.0);
                for (k, &j) in free.iter().enumerate() {
                    step[j] = stepf[k];
                }
                pf
            };
            if step.norm() <= o.xtol * (o.xtol + xs_norm) {
                return Ok(Status::Converged);
            }
            delta.copy_from(&step.component_div(&d));
            let mut modified = false;
            xt.copy_from(&(&x + &delta));
            if bounded {
                for &j in &free {
                    let c = xt[j].clamp(lo[j], hi[j]);
                    if c != xt[j] {
                        xt[j] = c;
                        modified = true;
                    }
                }
            }
            if modified {
                // the step was clipped into the box: its predicted reduction is that of the step actually taken
                delta.copy_from(&(&xt - &x));
                step.copy_from(&delta.component_mul(&d));
                let jd = &jac * &delta;
                pred = -2.0 * r.dot(&jd) - jd.norm_squared();
            }
            let sse_t = f.residual(&xt, &mut rt)?;
            if sse_t.is_finite() && sse_t < sse {
                let rho = if modified && pred <= 0.0 { 0.0 } else { (sse - sse_t) / pred };
                lam *= f64::max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0).powi(3));
                nu = 2.0;
                break (pred, sse_t);
            }
            lam *= nu;
            nu *= 2.0;
            if !lam.is_finite() || lam > 1e32 * fac.smax * fac.smax {
                return Ok(Status::Stalled);
            }
        };
        // accepted
        let actual = sse - sse_t;
        let small_f = actual <= o.ftol * sse && pred <= o.ftol * sse;
        let small_x = step.norm() <= o.xtol * (o.xtol + xs_norm);
        std::mem::swap(&mut x, &mut xt);
        std::mem::swap(&mut r, &mut rt);
        sse = sse_t;
        f.accept();
        let capped = f.tracker().accept_step(false);
        if small_f || small_x {
            return Ok(Status::Converged);
        }
        if capped {
            return Ok(Status::MaxSteps);
        }
    }
}
