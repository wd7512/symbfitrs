//! The fit: variable projection + Levenberg–Marquardt.
//!
//! Structural degenerate cases: no eligible linear parameters → ordinary LM; all parameters
//! linear → one linear solve.

use nalgebra::DVector;

use crate::problem::Problem;
use crate::solvers::core::{solve, Opts};
use crate::solvers::full::FullLsq;
use crate::solvers::varpro::VarPro;
use crate::solvers::{Lsq, Status};
use crate::tracker::{Stop, Tracker};

/// Which route `fit` took.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Path {
    /// Variable projection + LM: the normal route.
    VarPro,
    /// No eligible linear parameters: LM on all parameters.
    Full,
    /// Every parameter linear: one linear solve.
    Linear,
}

impl Path {
    /// Suffix shown after the fitter's name in reports; `None` for the normal route.
    pub fn label(self) -> Option<&'static str> {
        match self {
            Path::VarPro => None,
            Path::Full => Some("LM fallback"),
            Path::Linear => Some("linear solve"),
        }
    }
}

/// Fit `prob` from `theta0` under `tr`'s deadline. Returns the status and the route taken.
pub fn fit(prob: &mut Problem, theta0: &[f64], tr: &mut Tracker) -> (Status, Path) {
    fit_with(prob, theta0, tr, &Opts::default())
}

/// `fit` with solver options. `base.lower`/`upper` are per parameter (length p). Bounds on a
/// linear parameter turn variable projection off (all parameters are then ordinary variables);
/// otherwise the bounds of the nonlinear parameters are passed to the projected problem.
pub fn fit_with(prob: &mut Problem, theta0: &[f64], tr: &mut Tracker, base: &Opts) -> (Status, Path) {
    let mut opts = base.clone();
    let finite = |v: &Option<Vec<f64>>, i: usize, inf: f64| v.as_ref().is_some_and(|b| b[i] != inf);
    let lin_bounded = prob.lin.iter().any(|&i| finite(&base.lower, i, f64::NEG_INFINITY) || finite(&base.upper, i, f64::INFINITY));
    if !prob.lin.is_empty() && prob.sep.is_some() && !lin_bounded {
        let pick = |v: &Option<Vec<f64>>| v.as_ref().map(|b| prob.nonlin.iter().map(|&i| b[i]).collect::<Vec<f64>>());
        opts.lower = pick(&base.lower);
        opts.upper = pick(&base.upper);
        let mut vp = VarPro::new(prob, tr);
        let beta0 = vp.beta_of(theta0);
        if beta0.is_empty() {
            // fully linear: the single projected solve is the exact answer
            let mut r = DVector::zeros(vp.m());
            return match vp.residual(&beta0, &mut r) {
                Ok(sse) if sse.is_finite() => {
                    vp.tracker().accept_step(true);
                    (Status::Converged, Path::Linear)
                }
                Ok(_) => (Status::Failure("non-finite linear solve".into()), Path::Linear),
                Err(Stop::Deadline) => (Status::Deadline, Path::Linear),
            };
        }
        (solve(&mut vp, &beta0, &opts), Path::VarPro)
    } else {
        let mut full = FullLsq { prob, tr };
        let x0 = DVector::from_column_slice(theta0);
        (solve(&mut full, &x0, &opts), Path::Full)
    }
}
