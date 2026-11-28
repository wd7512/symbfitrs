"""The installed package: compile, fit and Pool.fit_many."""

from pathlib import Path

import numpy as np
import pytest
from scipy.special import erfc

import symbfitrs as sf
from bench.suites import nist

NIST = Path(__file__).resolve().parents[1] / "data" / "external" / "nist"
X = np.linspace(0, 5, 60)
Y = 2.0 * np.exp(-0.7 * X) + 0.3


def test_compile_names_params_and_linear_set():
    m = sf.compile("a*exp(-b*x) + c")
    assert (m.params, m.variables, m.linear) == (["a", "b", "c"], ["x"], ["a", "c"])
    assert sf.compile("a*exp(-b*x) + c", params=["c", "b", "a"]).params == ["c", "b", "a"]
    for bad in ("a*exp(-b*x", "a*foo(x)", "2*x", "a $ x"):
        with pytest.raises(ValueError):
            sf.compile(bad)


def test_fit_recovers_parameters():
    r = sf.compile("a*exp(-b*x) + c").fit(X, Y, budget_ms=50)
    assert r.success and r.status == "converged"
    np.testing.assert_allclose(r.params, [2.0, 0.7, 0.3], rtol=1e-10)
    assert r["b"] == pytest.approx(0.7) and set(r.as_dict()) == {"a", "b", "c"}
    assert r.loss < 1e-25 and r.nfev > 0 and 0 < r.time_ms < 50


def test_fit_input_forms():
    m = sf.compile("a*x + b*sin(k*t)", variables=["x", "t"])
    x, t = np.linspace(0, 1, 40), np.linspace(0, 3, 40)
    y = 1.5 * x - 0.5 * np.sin(2.0 * t)
    for cols in ({"x": x, "t": t}, np.vstack([x, t]), [x, t]):
        r = m.fit(cols, y, p0=[1, 1, 1.8], budget_ms=50)
        np.testing.assert_allclose(r.params, [1.5, -0.5, 2.0], rtol=1e-8)
    w = np.r_[np.ones(39), 0.0]  # a zero weight hides the outlier
    r = m.fit([x, t], np.r_[y[:-1], 99.0], p0=[1, 1, 1.8], weights=w, budget_ms=50)
    np.testing.assert_allclose(r.params, [1.5, -0.5, 2.0], rtol=1e-8)
    with pytest.raises(ValueError):
        m.fit(x, y)  # one column for two variables
    with pytest.raises(ValueError):
        m.fit([x, t], y, p0=[1, 2])
    with pytest.raises(ValueError):
        m.fit([x, t[:-1]], y)


def test_deadline_returns_best_so_far():
    m = sf.compile(" + ".join(f"a{k}*exp(-b{k}*x)" for k in range(12)))
    r = m.fit(X, Y, p0=np.linspace(0.1, 3, 24), budget_ms=0.01)
    assert r.status == "deadline" and r.params.shape == (24,)


def test_fit_many_keeps_order_and_isolates_errors():
    jobs = [
        ("a*exp(-b*x) + c", X, Y),
        ("a*exp(-b*x", X, Y),
        ("a*x + c", X, 3 * X - 1),
        ("a*exp(-b*x) + c", X, Y[:-1]),
        ("a*exp(-b*x) + c", X, -Y, [-1, 1, 0]),
    ] * 50
    out = sf.Pool(threads=4).fit_many(jobs, budget_ms=20, progress=False)
    assert len(out) == len(jobs)
    for i in range(0, len(jobs), 5):
        np.testing.assert_allclose(out[i].params, [2.0, 0.7, 0.3], rtol=1e-8)
        assert out[i + 1].status == "error" and "expected" in out[i + 1].message
        assert out[i + 2].as_dict() == pytest.approx({"a": 3.0, "c": -1.0})
        assert out[i + 3].status == "error" and not out[i + 3].success
        np.testing.assert_allclose(out[i + 4].params, [-2.0, 0.7, -0.3], rtol=1e-8)


@pytest.mark.skipif(not NIST.exists(), reason="run python -m bench.fetch (nist)")
def test_nist_certified_rss_from_both_starts():
    for nb in nist.load(NIST):
        m = sf.compile(nb.expr, nb.variables, nb.params)
        x = nb.x if nb.x.ndim == 1 else nb.x.T
        for start in ("start1", "start2"):
            r = m.fit(x, nb.y, p0=nb.starts[start], budget_ms=500)
            rss = r.loss * nb.y.size
            assert rss == pytest.approx(nb.certified_rss, rel=1e-6, abs=1e-20), (nb.name, start)


def test_predict_and_history():
    m = sf.compile("a*exp(-b*x) + c")
    np.testing.assert_allclose(m.predict(X, [2.0, 0.7, 0.3]), Y, rtol=1e-15)
    with pytest.raises(ValueError):
        m.predict(X, [1.0, 2.0])
    r = m.fit(X, Y, budget_ms=50)
    assert r.history_loss.size > 0 and r.history_loss[-1] == r.loss
    assert np.all(np.diff(r.history_loss) < 0) and np.all(np.diff(r.history_ms) >= 0)
    assert r.history_ms[-1] <= r.time_ms and r.n_nonfinite == 0


def test_wrap_step_expwerfc_and_declared_pi():
    x = np.linspace(-6, 6, 80)
    m = sf.compile("a*wrap(x - b) + c*step(x - 1) + expwerfc(d*x, 2 - x)")
    np.testing.assert_allclose(
        m.predict(x, [1.5, 0.5, 2.0, 0.3]),
        1.5 * np.angle(np.exp(1j * (x - 0.5))) + 2.0 * (x >= 1) + np.exp(0.3 * x) * erfc(2 - x),
        rtol=1e-13,
    )
    assert sf.compile("pi*x", params=["pi"]).predict(X, [2.0])[1] == 2.0 * X[1]


def test_fit_many_takes_models_with_their_own_variables_and_weights():
    m = sf.compile("a*u + b", variables=["u"])
    w = np.r_[np.ones(59), 0.0]
    jobs = [(m, X, np.r_[3 * X[:-1] + 1, 99.0], None, w), ("a*exp(-b*x) + c", X, Y)]
    out = sf.Pool(threads=2).fit_many(jobs, budget_ms=50, progress=False)
    np.testing.assert_allclose(out[0].params, [3.0, 1.0], rtol=1e-10)
    np.testing.assert_allclose(out[1].params, [2.0, 0.7, 0.3], rtol=1e-8)


def test_atan2_matches_numpy():
    x = np.linspace(-3, 3, 41)
    m = sf.compile("atan2(a*x - 1, b)")
    np.testing.assert_allclose(m.predict(x, [1.2, -0.5]), np.arctan2(1.2 * x - 1, -0.5), rtol=1e-15)


def _exp_problem():
    x = np.linspace(0, 5, 60)
    return x, 2.0 * np.exp(-0.7 * x) + 0.3


def test_trace_is_opt_in_and_starts_at_the_start():
    x, y = _exp_problem()
    m = sf.compile("a*exp(-b*x) + c")
    assert m.fit(x, y, p0=[1, 1, 0]).trace is None
    r = m.fit(x, y, p0=[1, 1, 0], trace=True)
    assert r.trace.shape == (r.steps + 1, 3)
    # step 0 is the start in names order; the linear parameters (a, c) are already solved at it
    assert r.trace[0][1] == 1.0
    full_p0 = sf.compile("exp(-b*x)*sin(c*x)").fit(x, y, p0=[0.5, 2.0], trace=True, max_steps=0)
    np.testing.assert_array_equal(full_p0.trace, [[0.5, 2.0]])  # no linear parameters: exactly p0
    np.testing.assert_allclose(r.trace[-1], r.params)  # accepted updates only improve the loss


def test_max_steps_caps_accepted_updates_and_is_a_trace_prefix():
    x, y = _exp_problem()
    m = sf.compile("a*exp(-b*x) + c")
    full = m.fit(x, y, p0=[1, 1, 0], trace=True)
    assert full.steps > 3
    for k in (0, 1, 3):
        r = m.fit(x, y, p0=[1, 1, 0], trace=True, max_steps=k)
        assert (r.status, r.steps, len(r.trace)) == ("max_steps", k, k + 1)
        np.testing.assert_array_equal(r.trace, full.trace[: k + 1])
        np.testing.assert_array_equal(r.params, r.trace[-1])
    # a cap beyond convergence changes nothing
    r = m.fit(x, y, p0=[1, 1, 0], trace=True, max_steps=10_000)
    assert r.status == full.status and r.steps == full.steps


def test_trace_follows_names_order_on_the_fallback_and_linear_paths():
    x, y = _exp_problem()
    lin = sf.compile("a*x + c").fit(x, 3 * x + 1, trace=True)  # fully linear: one solve
    assert lin.status == "converged" and lin.trace.shape == (1, 2)
    full = sf.compile("exp(-b*x)*sin(c*x)").fit(x, y, p0=[0.5, 2.0], trace=True, max_steps=2)
    assert full.trace.shape[1] == 2 and full.steps <= 2


def test_deadline_keeps_the_trace_so_far():
    x, y = _exp_problem()
    r = sf.compile("a*exp(-b*x) + c").fit(x, y, p0=[1, 1, 0], budget_ms=0.0, trace=True)
    assert r.status == "deadline" and r.trace.shape[1] == 3
    assert len(r.trace) in (0, r.steps + 1)  # nothing evaluated, or the steps taken so far


def test_fit_many_trace_matches_fit():
    x, y = _exp_problem()
    m = sf.compile("a*exp(-b*x) + c")
    jobs = [(m, x, y, [1, 1, 0]), (m, x, y * 2, [1, 1, 0])]
    pooled = sf.Pool(threads=2).fit_many(jobs, progress=False, trace=True, max_steps=4)
    for (_, xx, yy, p0), got in zip(jobs, pooled, strict=True):
        want = m.fit(xx, yy, p0=p0, trace=True, max_steps=4)
        assert got.status == want.status == "max_steps" and got.steps == 4
        np.testing.assert_array_equal(got.trace, want.trace)
    assert sf.Pool(threads=1).fit_many(jobs, progress=False)[0].trace is None


def test_bounds_keep_the_fit_inside_the_box_and_validate_names():
    x, y = _exp_problem()
    m = sf.compile("a*exp(-b*x) + c")
    r = m.fit(x, y, p0=[1, 0.2, 0], bounds={"b": (None, 0.5)}, trace=True)  # the true b is 0.7
    assert r["b"] == pytest.approx(0.5) and np.all(r.trace[:, m.params.index("b")] <= 0.5 + 1e-12)
    with pytest.raises(ValueError, match="unknown"):
        m.fit(x, y, bounds={"nope": (0, 1)})
    # bounds that never bind change nothing
    assert (
        m.fit(x, y, p0=[1, 1, 0], bounds={"b": (-1e6, 1e6)}).steps
        == m.fit(x, y, p0=[1, 1, 0]).steps
    )


def test_a_bounded_linear_parameter_is_held_at_its_bound():
    x, y = _exp_problem()
    m = sf.compile("a*exp(-b*x) + c")
    r = m.fit(x, -y, bounds={"a": (0, None)})  # the true amplitude is negative: held at its bound
    assert abs(r["a"]) < 1e-12
    with pytest.raises(ValueError):
        m.fit(x, y, bounds={"b": (1.0, 0.0)})


def test_fit_many_takes_bounds_by_name():
    x, y = _exp_problem()
    expr = "a*exp(-b*x) + c"
    jobs = [
        (expr, x, -y),  # an expression: its parameters are found by name for the bounds
        (sf.compile(expr), x, -y),
        ("k*x", x, y),  # no parameter "a": fitted without bounds
    ]
    r = sf.Pool(threads=2).fit_many(jobs, bounds={"a": (0, None)}, progress=False)
    assert abs(r[0]["a"]) < 1e-12 and abs(r[1]["a"]) < 1e-12
    assert r[2].status == "converged"
    with pytest.raises(ValueError, match="no job"):  # a name that no job has is a typo
        sf.Pool(threads=1).fit_many(jobs, bounds={"aa": (0, None)}, progress=False)


def test_bounds_apply_to_every_start_and_reject_empty_boxes():
    x, y = _exp_problem()
    m = sf.compile("a*exp(-b*x) + c")
    r = m.fit(x, y, p0=[[1, 3, 0], [1, -4, 0], [1, 0.1, 0]], bounds={"b": (0, 0.5)})
    assert np.all((r.start_params[:, 1] >= 0) & (r.start_params[:, 1] <= 0.5))
    assert r["b"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        m.fit(x, y, bounds={"b": (np.inf, np.inf)})


def test_several_starts_keep_the_best():
    m = sf.compile("a*sin(c*x)")
    x = np.arange(200) * 0.05
    y = 1.5 * np.sin(3.0 * x)
    one = m.fit(x, y, p0=[1.0, 0.5], budget_ms=200)
    assert one.loss > 1e-3 and one.start_losses.shape == (1,) and one.start_params.shape == (1, 2)
    r = m.fit(x, y, p0=[[1.0, 0.5], [1.0, 2.9], [1.0, 9.0]], budget_ms=300)
    assert r.loss < 1e-20 and r["c"] == pytest.approx(3.0, abs=1e-8)
    assert r.start_losses.shape == (3,) and r.start_params.shape == (3, 2)
    assert r.start_losses[1] == r.loss and r.start_losses[0] == one.loss
    with pytest.raises(ValueError):
        m.fit(x, y, p0=np.ones((1, 2, 2)))


def test_fit_many_takes_several_starts():
    m = sf.compile("a*sin(c*x)")
    x = np.arange(200) * 0.05
    y = 1.5 * np.sin(3.0 * x)
    jobs = [
        (m, x, y, [[1.0, 0.5], [1.0, 2.9]]),
        (m, x, y, [1.0, 0.5]),
        (m, x, y, np.ones((1, 2, 2))),
    ]
    r = sf.Pool(threads=2).fit_many(jobs, budget_ms=300, progress=False)
    assert r[0].loss < 1e-20 and r[0].start_losses.shape == (2,)
    assert r[1].start_losses.shape == (1,) and r[1].loss > 1e-3
    assert r[2].status == "error"
