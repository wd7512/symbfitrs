# Changelog

Changes to the installed `symbfitrs` package. The benchmark in `bench/` is not shipped and is
not listed here.

## Unreleased

- `Model.fit` and `Pool.fit_many` take `bounds={name: (lo, hi)}` (`None` = unbounded on that
  side). An active-set Levenberg–Marquardt step keeps every iterate inside the box: a parameter on
  a bound is held there while its gradient points out of the box and released when it turns
  inward, and every start is moved into the box first. In `Pool.fit_many` each job uses the
  entries that name its own model's parameters, so one mapping serves a batch of different
  expressions; a name that matches no job raises ValueError. `lo == hi` fixes a parameter.
  Bounds replace clipping inside the expression, whose clipped region has zero gradient.
  Bounding a linear parameter turns variable projection off. Without bounds the solver path is unchanged.
- `p0` may be 2-D (one start per row) in `Model.fit` and `Pool.fit_many` jobs. Each start is
  fitted in turn with an equal share of the budget left when it begins; the lowest loss is kept,
  and the status, steps, history and trace are that start's. `FitResult.start_losses` and
  `FitResult.start_params` report every start, which shows whether the fit has more than one
  optimum. A 1-D `p0` behaves as before. The data are bound to the model once for all starts.
- `Model.fit` and `Pool.fit_many` take `trace=True` and `max_steps=`. With `trace`, `FitResult.trace`
  holds the parameter vector (in `names` order) at step 0 (the start, with eliminated linear
  parameters already solved at it) and after every accepted update, shape `(steps + 1, n_params)`; rejected trials and residual or Jacobian evaluations do
  not count as steps. `max_steps=k` stops after `k` accepted updates with status `"max_steps"`;
  a capped run is a prefix of the uncapped trace. A converged, stalled or deadline result keeps
  the trace so far. `FitResult.steps` is the number of accepted updates. Both default to off,
  and then nothing is recorded.

## 0.1.1 (2026-11-27)

- `FitResult.history_ms` and `history_loss`: the best loss at the first evaluation and at each
  improvement, timed from the call like `time_ms`.
- `FitResult.n_nonfinite`: evaluations that gave a non-finite loss.
- `Model.predict(x, params)`.
- `Pool.fit_many` jobs can be `(model, x, y, p0, weights)`, and `model` can be a compiled
  `Model` with its own variables.
- Expressions: `atan2`, `wrap`, `step` and `expwerfc(a, b)`. A declared parameter named `pi`
  wins over the constant.
- Faster VarPro+LM steps: three redundant copies are skipped, with the same solver trajectories.

## 0.1.0 (2026-11-26)

First release: `compile`, `Model.fit` and `Pool.fit_many`, with wheels for Linux and macOS
(x86_64, aarch64).
