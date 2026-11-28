//! Python bindings (`symbfitrs._core`). Arrays cross as native-endian float64 bytes, so the
//! module needs only the stable ABI; `python/symbfitrs` wraps it in the public API.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::driver::{self, Data, Outcome};
use crate::expr::{self, Compiled};

fn f64s(b: &[u8]) -> PyResult<Vec<f64>> {
    match b.as_chunks::<8>() {
        (words, []) => Ok(words.iter().map(|w| f64::from_ne_bytes(*w)).collect()),
        _ => Err(PyValueError::new_err("array buffer is not float64")),
    }
}

fn columns(x: &[Vec<u8>]) -> PyResult<Vec<Vec<f64>>> {
    x.iter().map(|c| f64s(c)).collect()
}

/// x columns, y, weights (float64 bytes) and p0.
type RawData = (Vec<Vec<u8>>, Vec<u8>, Option<Vec<u8>>, Option<Vec<f64>>);

/// (model or expression, variables, x, y, w, p0, more_starts, (lower, upper)) as passed from
/// Python; the bounds have one value per parameter (±∞ = free).
type Job<'py> = (
    Bound<'py, PyAny>,
    Vec<String>,
    Vec<Vec<u8>>,
    Vec<u8>,
    Option<Vec<u8>>,
    Option<Vec<f64>>,
    Vec<Vec<f64>>,
    (Option<Vec<f64>>, Option<Vec<f64>>),
);

fn data((x, y, w, p0): RawData) -> PyResult<Data> {
    Ok(Data { x: columns(&x)?, y: f64s(&y)?, w: w.map(|w| f64s(&w)).transpose()?, p0, ..Default::default() })
}

/// (names, params, loss, status, message, nfev, n_nonfinite, time_ms, history_ms, history_loss,
/// steps, (trace, start_losses, start_params)); pyo3 converts tuples of at most 12
type PyOutcome = (
    Vec<String>,
    Vec<f64>,
    f64,
    &'static str,
    String,
    usize,
    usize,
    f64,
    Vec<f64>,
    Vec<f64>,
    usize,
    (Option<Vec<Vec<f64>>>, Vec<f64>, Vec<Vec<f64>>),
);

fn out(o: Outcome) -> PyOutcome {
    (
        o.names,
        o.params,
        o.loss,
        o.status,
        o.message,
        o.nfev,
        o.n_nonfinite,
        o.time_ms,
        o.history_ms,
        o.history_loss,
        o.steps,
        (o.trace, o.start_losses, o.start_params),
    )
}

#[pyclass(frozen, module = "symbfitrs._core")]
struct Model(Arc<Compiled>);

#[pymethods]
impl Model {
    #[getter]
    fn params(&self) -> Vec<String> {
        self.0.params.clone()
    }

    #[getter]
    fn variables(&self) -> Vec<String> {
        self.0.variables.clone()
    }

    #[getter]
    fn linear(&self) -> Vec<usize> {
        self.0.linear.clone()
    }

    fn predict(&self, py: Python<'_>, x: Vec<Vec<u8>>, params: Vec<f64>) -> PyResult<Vec<f64>> {
        let x = columns(&x)?;
        let c = self.0.clone();
        py.detach(move || driver::predict(&c, &x, &params)).map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (x, y, w, p0, budget_ms, trace=false, max_steps=None, more_starts=Vec::new(), lower=None, upper=None))]
    #[allow(clippy::too_many_arguments)] // the pyo3 signature mirrors `Model.fit`
    fn fit(
        &self,
        py: Python<'_>,
        x: Vec<Vec<u8>>,
        y: Vec<u8>,
        w: Option<Vec<u8>>,
        p0: Option<Vec<f64>>,
        budget_ms: f64,
        trace: bool,
        max_steps: Option<usize>,
        more_starts: Vec<Vec<f64>>,
        lower: Option<Vec<f64>>,
        upper: Option<Vec<f64>>,
    ) -> PyResult<PyOutcome> {
        let d = Data { trace, max_steps, more_starts, lower, upper, ..data((x, y, w, p0))? };
        let c = self.0.clone();
        py.detach(move || driver::fit_one(&c, d, budget_ms)).map(out).map_err(PyValueError::new_err)
    }
}

#[pyfunction]
#[pyo3(signature = (expr, variables, params=None))]
fn compile(expr: &str, variables: Vec<String>, params: Option<Vec<String>>) -> PyResult<Model> {
    expr::compile(expr, &variables, params.as_deref()).map(|c| Model(Arc::new(c))).map_err(PyValueError::new_err)
}

/// Jobs are (model or expression, variables, x, y, w, p0, more_starts, (lower, upper)); an
/// expression is compiled once per
/// batch over `variables`. Results come in job order, each (outcome tuple, "") or (None, error).
/// `progress(n)` is called with the number of newly finished jobs.
#[pyfunction]
#[pyo3(signature = (jobs, budget_ms, threads, progress=None, trace=false, max_steps=None))]
fn fit_many<'py>(
    py: Python<'py>,
    jobs: Vec<Job<'py>>,
    budget_ms: f64,
    threads: usize,
    progress: Option<Py<PyAny>>,
    trace: bool,
    max_steps: Option<usize>,
) -> PyResult<Vec<(Option<PyOutcome>, String)>> {
    let n = jobs.len();
    let mut compiled: HashMap<(String, Vec<String>), Result<Arc<Compiled>, String>> = HashMap::new();
    let mut batch = Vec::with_capacity(n);
    for (src, variables, x, y, w, p0, more_starts, (lower, upper)) in jobs {
        let model = match src.extract::<PyRef<Model>>() {
            Ok(m) => Ok(m.0.clone()),
            Err(_) => {
                let e: String = src.extract()?;
                compiled
                    .entry((e, variables))
                    .or_insert_with_key(|(e, v)| expr::compile(e, v, None).map(Arc::new))
                    .clone()
            }
        };
        batch.push((model, Data { trace, max_steps, more_starts, lower, upper, ..data((x, y, w, p0))? }));
    }
    let (done, cancel) = (AtomicUsize::new(0), AtomicBool::new(false));
    let mut interrupt: Option<PyErr> = None;
    let results = py.detach(|| {
        std::thread::scope(|s| {
            let worker = s.spawn(|| driver::fit_many(batch, budget_ms, threads, &done, &cancel));
            let mut reported = 0;
            loop {
                let finished = worker.is_finished();
                let d = done.load(Ordering::Relaxed);
                let err = Python::attach(|py| -> PyResult<()> {
                    py.check_signals()?;
                    if let (Some(cb), true) = (&progress, d > reported) {
                        cb.call1(py, (d - reported,))?;
                    }
                    Ok(())
                });
                reported = d;
                if let Err(e) = err {
                    cancel.store(true, Ordering::Relaxed);
                    interrupt.get_or_insert(e);
                }
                if finished || reported == n {
                    return worker.join().expect("fit_many worker panicked");
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        })
    });
    if let Some(e) = interrupt {
        return Err(e);
    }
    Ok(results
        .into_iter()
        .map(|r| match r {
            Ok(o) => (Some(out(o)), String::new()),
            Err(e) => (None, e),
        })
        .collect())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(fit_many, m)?)?;
    Ok(())
}
