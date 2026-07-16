"""Acceptance gates for the three NSR accuracy layers.

The boosting tests drive a stub engine rather than ``NSREngine``: layer 1 is
engine-agnostic by contract, and a real policy fit per round would make the
suite unusably slow.  ``test_boosting_accepts_a_real_engine`` covers the
integration with the actual engine on a tiny configuration.
"""

import numpy as np
import pandas as pd
import pytest

sp = pytest.importorskip("sympy")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from nsr_engine import NSREngine, ParetoFront, ParetoPoint, ResidualBoostedNSR
from nsr_engine._expr import eval_sympy_on
from nsr_engine.refinement import joint_refit_prune, optimize_constants, optimize_front

X2 = sp.Symbol("x2")
X4 = sp.Symbol("x4")


def _make_data(n: int = 1500, seed: int = 0, noise: float = 0.05):
    """Two additive terms of comparable magnitude: `exp(x2) - 1.5*log(x4)`."""
    rng = np.random.default_rng(seed)
    x2 = rng.uniform(-1.0, 1.0, n)
    x4 = rng.uniform(0.5, 5.0, n)
    X = pd.DataFrame({"x2": x2, "x4": x4})
    y = pd.Series(np.exp(x2) - 1.5 * np.log(x4) + noise * rng.standard_normal(n))
    return X, y


def _r2(expr, X, y) -> float:
    pred = eval_sympy_on(expr, X)
    truth = y.to_numpy(dtype=np.float64)
    mask = np.isfinite(pred)
    sse = float(np.sum((truth[mask] - pred[mask]) ** 2))
    sst = float(np.sum((truth[mask] - truth[mask].mean()) ** 2))
    return 1.0 - sse / sst


class _StubEngine:
    """Offers a fixed candidate list, affine-fitted to whatever `y` it is given.

    Mimics the real engine's contract: expressions come back in raw feature
    terms already wrapped as ``b0 + b1*expr``.
    """

    def __init__(self, candidates):
        self._candidates = candidates

    def fit(self, X, y):
        y_arr = y.to_numpy(dtype=np.float64)
        points = []
        for i, expr in enumerate(self._candidates):
            pred = eval_sympy_on(expr, X)
            # The engine fits and scores on the finite subset only, so a
            # partially-defined candidate still reaches the front.
            mask = np.isfinite(pred) & np.isfinite(y_arr)
            if int(mask.sum()) < 2:
                continue
            pm, ym = pred[mask], y_arr[mask]
            b1 = float(np.cov(pm, ym)[0, 1] / np.var(pm))
            b0 = float(ym.mean() - b1 * pm.mean())
            fitted = sp.Float(b0) + sp.Float(b1) * expr
            resid = ym - (b0 + b1 * pm)
            points.append(
                ParetoPoint(
                    equation=str(fitted),
                    sympy_expr=fitted,
                    complexity=3 + i,
                    mse=float(np.mean(resid**2)),
                )
            )
        if not points:
            return ParetoFront([])
        return ParetoFront(points).dominance_filter()


def _two_term_factory(round_idx: int) -> _StubEngine:
    """Round 1 finds `exp`; only once its residual is formed does `log` win."""
    if round_idx == 1:
        return _StubEngine([sp.exp(X2), sp.log(X4)])
    return _StubEngine([sp.log(X4), sp.exp(X2)])


# ---------------------------------------------------------------------------
# Layer 1 — residual boosting (spec section 4.5)
# ---------------------------------------------------------------------------

def test_boosting_beats_single_fit_on_two_additive_terms():
    X, y = _make_data()

    plain = _two_term_factory(1).fit(X, y)
    boosted = ResidualBoostedNSR(_two_term_factory, max_rounds=3).fit(X, y)

    assert _r2(boosted.elbow().sympy_expr, X, y) > _r2(plain.elbow().sympy_expr, X, y)
    assert _r2(boosted.points[-1].sympy_expr, X, y) > 0.98


def test_boosting_stops_early_on_a_single_term_target():
    rng = np.random.default_rng(1)
    X, _ = _make_data()
    y = pd.Series(np.exp(X["x2"].to_numpy()) + 0.05 * rng.standard_normal(len(X)))

    booster = ResidualBoostedNSR(_two_term_factory, max_rounds=3, min_gain=0.02)
    front = booster.fit(X, y)

    assert len(front) == 1
    assert len(booster.terms_) == 1
    assert [r["added"] for r in booster.rounds_] == [True, False]


def test_boosting_never_underperforms_a_single_fit_on_pure_noise():
    rng = np.random.default_rng(2)
    X, _ = _make_data()
    y = pd.Series(rng.standard_normal(len(X)))

    booster = ResidualBoostedNSR(_two_term_factory, max_rounds=3, min_gain=0.02)
    front = booster.fit(X, y)

    assert len(booster.terms_) == 1  # round 1 only; nothing else earns its place
    assert front.points[0].mse <= float(np.var(y.to_numpy()))


def test_terms_summed_complexity_matches_last_front_point():
    X, y = _make_data()

    booster = ResidualBoostedNSR(_two_term_factory, max_rounds=3)
    front = booster.fit(X, y)

    expected = sum(c for _, c in booster.terms_) + (len(booster.terms_) - 1)
    assert front.points[-1].complexity == expected


def test_boosted_front_is_non_dominated_by_construction():
    X, y = _make_data()

    front = ResidualBoostedNSR(_two_term_factory, max_rounds=3).fit(X, y)

    assert len(front.dominance_filter()) == len(front)
    complexities = [p.complexity for p in front.points]
    mses = [p.mse for p in front.points]
    assert complexities == sorted(complexities) and len(set(complexities)) == len(complexities)
    assert mses == sorted(mses, reverse=True)


def test_term_refiner_hook_is_applied_before_subtraction():
    X, y = _make_data()
    seen = []

    def refiner(expr, X_, residual):
        seen.append(expr)
        return expr

    booster = ResidualBoostedNSR(_two_term_factory, max_rounds=2, term_refiner=refiner)
    booster.fit(X, y)

    assert len(seen) == 2


def test_min_mse_term_selection_picks_the_most_accurate_point():
    X, y = _make_data()

    booster = ResidualBoostedNSR(
        _two_term_factory, max_rounds=1, term_selection="min_mse"
    )
    booster.fit(X, y)

    assert len(booster.terms_) == 1


def test_invalid_term_selection_is_rejected():
    with pytest.raises(ValueError, match="term_selection"):
        ResidualBoostedNSR(_two_term_factory, term_selection="best")


def test_boosting_keeps_a_term_that_is_undefined_on_some_rows():
    """The engine scores partially-defined terms on their finite subset.

    `log(a)` with negative rows in `a` is a legitimate front point — the engine
    masks the non-finite rows rather than discarding the candidate.  Boosting
    must mask the same way instead of rejecting the term and returning nothing.
    """
    rng = np.random.default_rng(6)
    X = pd.DataFrame({"a": rng.standard_normal(400)})
    y = pd.Series(np.log(np.abs(X["a"].to_numpy())) + 0.05 * rng.standard_normal(400))
    a = sp.Symbol("a")

    booster = ResidualBoostedNSR(lambda k: _StubEngine([sp.log(a)]), max_rounds=2)
    front = booster.fit(X, y)

    assert len(front) >= 1
    assert len(booster.terms_) >= 1
    assert np.isfinite(front.points[0].mse)


def test_boosting_rejects_a_term_with_too_few_finite_rows():
    X = pd.DataFrame({"a": np.full(200, -1.0)})
    y = pd.Series(np.arange(200, dtype=float))
    a = sp.Symbol("a")

    booster = ResidualBoostedNSR(lambda k: _StubEngine([sp.log(a)]), max_rounds=2)
    front = booster.fit(X, y)

    assert len(front) == 0
    assert booster.rounds_[0]["added"] is False


def test_boosting_handles_an_empty_front():
    X, y = _make_data(n=100)

    booster = ResidualBoostedNSR(lambda k: _StubEngine([]), max_rounds=2)
    front = booster.fit(X, y)

    assert len(front) == 0
    assert booster.terms_ == []


@pytest.mark.slow
def test_boosting_accepts_a_real_engine():
    X, y = _make_data(n=400)

    def factory(round_idx: int) -> NSREngine:
        return NSREngine(
            n_lambda=2,
            n_iters=5,
            batch_size=16,
            max_len=7,
            unary_ops=("square", "abs", "log", "exp"),
            random_state=42 + round_idx,
            device="cpu",
        )

    booster = ResidualBoostedNSR(factory, max_rounds=2)
    front = booster.fit(X, y)

    assert isinstance(front, ParetoFront)
    assert len(booster.rounds_) >= 1


# ---------------------------------------------------------------------------
# Layer 2 — constant optimization (spec section 5.4)
# ---------------------------------------------------------------------------

def test_constant_opt_recovers_an_interior_weight():
    rng = np.random.default_rng(3)
    X, _ = _make_data()
    y = pd.Series(
        1.5 * np.log(X["x4"].to_numpy()) + 0.05 * rng.standard_normal(len(X))
    )
    start = sp.Float(1.0) * sp.log(X4)

    refined = optimize_constants(start, X, y)

    coeff = float(refined.coeff(sp.log(X4)))
    assert coeff == pytest.approx(1.5, abs=0.02)
    assert _r2(refined, X, y) >= _r2(start, X, y)


def test_constant_opt_preserves_complexity_and_returns_floats_only():
    X, y = _make_data()
    expr = sp.Float(0.5) * sp.exp(X2) + sp.Float(0.5) * sp.log(X4)

    refined = optimize_constants(expr, X, y)

    assert sp.count_ops(refined) == sp.count_ops(expr)


def test_equal_literals_become_independent_parameters():
    """Two `1.0`s must separate — replacement is per occurrence, not per value."""
    rng = np.random.default_rng(4)
    X, _ = _make_data()
    y = pd.Series(
        2.0 * np.exp(X["x2"].to_numpy())
        - 0.5 * np.log(X["x4"].to_numpy())
        + 0.01 * rng.standard_normal(len(X))
    )
    expr = sp.Float(1.0) * sp.exp(X2) + sp.Float(1.0) * sp.log(X4)

    refined = optimize_constants(expr, X, y)

    assert float(refined.coeff(sp.exp(X2))) == pytest.approx(2.0, abs=0.05)
    assert float(refined.coeff(sp.log(X4))) == pytest.approx(-0.5, abs=0.05)


def test_constant_opt_leaves_integer_exponents_alone():
    """A `square`'s `**2` must stay fixed — a free fractional exponent is unstable."""
    rng = np.random.default_rng(5)
    X, _ = _make_data()
    y = pd.Series(3.0 * X["x2"].to_numpy() ** 2 + 0.01 * rng.standard_normal(len(X)))
    expr = sp.Float(1.0) * X2**2

    refined = optimize_constants(expr, X, y)

    assert refined.has(X2**2)
    assert float(refined.coeff(X2**2)) == pytest.approx(3.0, abs=0.05)


def test_constant_opt_without_floats_returns_the_expression_unchanged():
    X, y = _make_data()
    expr = sp.log(X4)

    assert optimize_constants(expr, X, y) is expr


def _nested_const_expr(depth: int):
    """`depth` Float literals, nested so sympy cannot fold them together."""
    expr = X2
    for i in range(depth):
        expr = sp.log(sp.Float(0.1 * (i + 1)) + expr**2)
    return expr


def test_constant_opt_refuses_too_many_free_constants():
    X, y = _make_data(n=200)

    assert optimize_constants(_nested_const_expr(13), X, y) is _nested_const_expr(13)


def test_max_free_consts_is_configurable():
    X, y = _make_data(n=200)
    expr = _nested_const_expr(4)

    assert optimize_constants(expr, X, y, max_free_consts=3) is expr
    assert optimize_constants(expr, X, y, max_free_consts=9) is not expr


def test_fit_subsample_is_configurable():
    """`fit_subsample=0` means every row; a small value still recovers the weight."""
    rng = np.random.default_rng(11)
    X, _ = _make_data(n=1200)
    y = pd.Series(
        1.5 * np.log(X["x4"].to_numpy()) + 0.02 * rng.standard_normal(len(X))
    )
    start = sp.Float(1.0) * sp.log(X4)

    for subsample in (50, 0):
        refined = optimize_constants(start, X, y, fit_subsample=subsample)
        assert float(refined.coeff(sp.log(X4))) == pytest.approx(1.5, abs=0.1)


def test_sentinel_scales_with_the_target_magnitude():
    """A fixed 1e6 sentinel would sit on top of a target near 1e6.

    An overflowing row must always read as a bad fit, never a perfect one, so
    the refit may never make an expression less finite than it started.
    """
    rng = np.random.default_rng(12)
    n = 300
    X = pd.DataFrame({"x2": rng.uniform(0.1, 30.0, n)})
    y = pd.Series(1e6 + 1e4 * rng.standard_normal(n))  # centred on the old sentinel
    expr = sp.Float(1.0) * sp.exp(sp.Float(1.0) * X2)

    refined = optimize_constants(expr, X, y)

    pred = eval_sympy_on(refined, X)
    assert pred is not None
    assert int(np.isfinite(pred).sum()) >= int(np.isfinite(eval_sympy_on(expr, X)).sum())


def test_constant_opt_survives_exp_overflow():
    X, y = _make_data(n=200)
    expr = sp.Float(3.0) * sp.exp(sp.Float(500.0) * X4)

    refined = optimize_constants(expr, X, y)  # must not raise

    assert refined is not None


def test_constant_opt_is_deterministic():
    X, y = _make_data(n=200)
    expr = sp.Float(1.0) * sp.log(X4) + sp.Float(0.2) * sp.exp(X2)

    assert optimize_constants(expr, X, y, seed=7) == optimize_constants(
        expr, X, y, seed=7
    )


def test_optimize_front_never_worsens_a_point():
    X, y = _make_data()
    front = ParetoFront(
        [
            ParetoPoint(
                equation="1.0*log(x4)",
                sympy_expr=sp.Float(1.0) * sp.log(X4),
                complexity=3,
                mse=9.9,
            )
        ]
    )

    refined = optimize_front(front, X, y)

    assert refined.points[0].mse <= 9.9
    assert refined.points[0].complexity == 3


def test_optimize_front_preserves_a_non_mse_metric():
    X, y = _make_data()
    front = ParetoFront(
        [
            ParetoPoint(
                equation="1.0*log(x4)",
                sympy_expr=sp.Float(1.0) * sp.log(X4),
                complexity=3,
                mse=-5.0,
                score_metric="r2",
            )
        ]
    )

    refined = optimize_front(front, X, y)

    assert refined.points[0].score_metric == "r2"


# ---------------------------------------------------------------------------
# Layer 3 — joint refit + prune (spec section 6.4)
# ---------------------------------------------------------------------------

def test_joint_refit_beats_boosting_alone():
    X, y = _make_data()

    booster = ResidualBoostedNSR(_two_term_factory, max_rounds=3)
    boosted = booster.fit(X, y)
    refined = joint_refit_prune(booster.terms_, X, y)

    assert refined is not None
    expr, complexity, mse = refined
    assert mse <= boosted.points[-1].mse
    assert _r2(expr, X, y) >= _r2(boosted.points[-1].sympy_expr, X, y)


def test_joint_refit_recovers_the_true_coefficients():
    X, y = _make_data(noise=0.01)
    terms = [(sp.Float(1.0) * sp.exp(X2), 3), (sp.Float(1.0) * sp.log(X4), 3)]

    expr, _, _ = joint_refit_prune(terms, X, y)

    assert float(expr.coeff(sp.exp(X2))) == pytest.approx(1.0, abs=0.05)
    assert float(expr.coeff(sp.log(X4))) == pytest.approx(-1.5, abs=0.05)


def test_joint_refit_prunes_a_redundant_term():
    X, y = _make_data()
    terms = [
        (sp.Float(1.0) * sp.exp(X2), 3),
        (sp.Float(1.0) * sp.log(X4), 3),
        (sp.Float(2.0) * sp.exp(X2), 4),  # collinear with the first
    ]
    summed = sum(c for _, c in terms) + (len(terms) - 1)

    _, complexity, _ = joint_refit_prune(terms, X, y)

    assert complexity < summed


def test_joint_refit_returns_none_when_every_term_is_non_finite():
    X, y = _make_data(n=200)
    terms = [(sp.log(sp.Float(-1.0) * X4 * X4 - sp.Float(5.0)), 4)]

    assert joint_refit_prune(terms, X, y) is None


def test_joint_refit_returns_none_for_no_terms():
    X, y = _make_data(n=200)

    assert joint_refit_prune([], X, y) is None


def test_joint_refit_handles_a_single_term():
    X, y = _make_data(n=400)
    terms = [(sp.Float(1.0) * sp.log(X4), 3)]

    refined = joint_refit_prune(terms, X, y)

    assert refined is not None
    assert refined[1] == 3  # one term, no join nodes


def test_joint_refit_fit_subsample_caps_the_estimator_but_not_the_score():
    """The row cap applies to the LassoCV fit; the reported MSE uses every row."""
    X, y = _make_data(n=3000, noise=0.05)
    terms = [(sp.Float(1.0) * sp.exp(X2), 3), (sp.Float(1.0) * sp.log(X4), 3)]

    expr, _, mse = joint_refit_prune(terms, X, y, fit_subsample=200, polish=None)

    # Weights are well determined long before every row is used.
    assert float(expr.coeff(sp.exp(X2))) == pytest.approx(1.0, abs=0.05)
    assert float(expr.coeff(sp.log(X4))) == pytest.approx(-1.5, abs=0.05)

    # The returned MSE is the full-data MSE of the returned expression, not the
    # MSE over the 200 sampled rows — otherwise it would not be comparable to
    # the boosted front's points.
    pred = eval_sympy_on(expr, X)
    full = float(np.mean((y.to_numpy(dtype=np.float64) - pred) ** 2))
    assert mse == pytest.approx(full, rel=1e-9)


def test_joint_refit_fit_subsample_zero_uses_every_row():
    X, y = _make_data(n=400)
    terms = [(sp.Float(1.0) * sp.exp(X2), 3), (sp.Float(1.0) * sp.log(X4), 3)]

    capped = joint_refit_prune(terms, X, y, fit_subsample=0, polish=None)
    uncapped = joint_refit_prune(terms, X, y, fit_subsample=10_000, polish=None)

    # n < both caps, so the two agree exactly.
    assert capped[2] == pytest.approx(uncapped[2], rel=1e-12)


def test_joint_refit_passes_fit_subsample_to_the_polish():
    X, y = _make_data(n=400)
    terms = [(sp.Float(1.0) * sp.exp(X2), 3)]
    seen = {}

    def polish(expr, X_, y_, *, seed=0, fit_subsample=None):
        seen["fit_subsample"] = fit_subsample
        return expr

    joint_refit_prune(terms, X, y, fit_subsample=123, polish=polish)

    assert seen["fit_subsample"] == 123


def test_joint_refit_rejects_an_unknown_estimator():
    X, y = _make_data(n=100)

    with pytest.raises(ValueError, match="estimator"):
        joint_refit_prune([(sp.log(X4), 3)], X, y, estimator="ridge")


def test_joint_refit_ols_estimator_runs():
    X, y = _make_data(n=400)
    terms = [(sp.Float(1.0) * sp.exp(X2), 3), (sp.Float(1.0) * sp.log(X4), 3)]

    refined = joint_refit_prune(terms, X, y, estimator="ols")

    assert refined is not None
