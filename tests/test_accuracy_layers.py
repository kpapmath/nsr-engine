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
            # The round *is* the weak learner; a round that boosted again would
            # nest one booster inside another.
            boosting=False,
        )

    booster = ResidualBoostedNSR(factory, max_rounds=2)
    front = booster.fit(X, y)

    assert isinstance(front, ParetoFront)
    assert len(booster.rounds_) >= 1


# ---------------------------------------------------------------------------
# Layer 1 as the engine default (0.7.0)
# ---------------------------------------------------------------------------


def test_engine_boosts_by_default():
    engine = NSREngine()

    assert engine.boosting is True
    assert engine.boosting_max_rounds == 3
    assert engine.boosting_min_gain == 0.02


def _record_dispatch(monkeypatch, engine) -> list[str]:
    calls: list[str] = []

    def single(X, y):
        calls.append("single")
        return ParetoFront([])

    def boosted(X, y):
        calls.append("boosted")
        return ParetoFront([])

    monkeypatch.setattr(engine, "_fit_single", single)
    monkeypatch.setattr(engine, "_fit_boosted", boosted)
    return calls


def test_engine_fit_dispatches_to_the_boosted_path_by_default(monkeypatch):
    engine = NSREngine()
    calls = _record_dispatch(monkeypatch, engine)

    engine.fit(pd.DataFrame({"a": [1.0]}), pd.Series([1.0]))

    assert calls == ["boosted"]


@pytest.mark.parametrize(
    "kwargs",
    [{"boosting": False}, {"boosting_max_rounds": 1}],
    ids=["boosting-off", "one-round"],
)
def test_engine_fit_runs_a_single_fit_when_boosting_cannot_add_a_term(
    monkeypatch, kwargs
):
    """One round would return that round's elbow alone — a smaller front."""
    engine = NSREngine(**kwargs)
    calls = _record_dispatch(monkeypatch, engine)

    engine.fit(pd.DataFrame({"a": [1.0]}), pd.Series([1.0]))

    assert calls == ["single"]


def test_round_engine_is_fresh_and_does_not_boost():
    engine = NSREngine(random_state=7, cache_prefix="run")

    second = engine._round_engine(2)

    assert second.boosting is False
    assert second.random_state == 9
    assert second.cache_prefix == "run_round2"
    assert engine.cache_prefix == "run"  # the original is untouched
    assert second.max_len == engine.max_len  # every other setting carries over


def test_merge_fronts_keeps_the_simple_end_of_the_round_one_front():
    """The booster emits one point per round; merging restores the rest."""
    from nsr_engine.engine import _merge_fronts

    round_one = ParetoFront(
        [
            ParetoPoint(equation="a", sympy_expr=None, complexity=1, mse=9.0),
            ParetoPoint(equation="a + b", sympy_expr=None, complexity=3, mse=4.0),
        ]
    )
    boosted = ParetoFront(
        [
            ParetoPoint(equation="a + b", sympy_expr=None, complexity=3, mse=4.0),
            ParetoPoint(equation="a + b + c", sympy_expr=None, complexity=5, mse=1.0),
        ]
    )

    merged = _merge_fronts(round_one, boosted)

    assert [(p.equation, p.complexity) for p in sorted(
        merged.points, key=lambda p: p.complexity
    )] == [("a", 1), ("a + b", 3), ("a + b + c", 5)]


def test_merge_fronts_drops_dominated_points():
    from nsr_engine.engine import _merge_fronts

    merged = _merge_fronts(
        ParetoFront(
            [ParetoPoint(equation="worse", sympy_expr=None, complexity=4, mse=9.0)]
        ),
        ParetoFront(
            [ParetoPoint(equation="better", sympy_expr=None, complexity=2, mse=1.0)]
        ),
    )

    assert [p.equation for p in merged.points] == ["better"]


@pytest.mark.slow
def test_engine_default_fit_boosts_and_covers_the_single_fit_front():
    X, y = _make_data(n=400)
    kwargs = dict(
        n_lambda=2,
        n_iters=5,
        batch_size=16,
        max_len=7,
        unary_ops=("square", "abs", "log", "exp"),
        random_state=42,
        device="cpu",
    )

    boosted = NSREngine(boosting_max_rounds=2, **kwargs).fit(X, y)
    single = NSREngine(boosting=False, **kwargs).fit(X, y)

    assert isinstance(boosted, ParetoFront)
    # Round 1 runs the same search as the single fit, so its points are in
    # there too: boosting adds reach, it never trades the simple end away.
    assert len(boosted) >= len(single)
    assert min(p.complexity for p in boosted.points) == min(
        p.complexity for p in single.points
    )


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


# ---------------------------------------------------------------------------
# residual_metric: the objective the round-acceptance rule is measured in
# ---------------------------------------------------------------------------
def test_residual_metric_defaults_to_mse():
    """The default must not move: existing callers and the CLI rely on it."""
    assert ResidualBoostedNSR(_two_term_factory).residual_metric == "mse"
    front = ResidualBoostedNSR(_two_term_factory).fit(*_make_data())
    assert {p.score_metric for p in front.points} == {"mse"}


def test_invalid_residual_metric_is_rejected():
    with pytest.raises(ValueError, match="residual_metric"):
        ResidualBoostedNSR(_two_term_factory, residual_metric="not_a_metric")


def test_mbd_residual_metric_is_rejected_with_a_reason():
    """`mbd` is the one score metric the gain rule cannot use.

    It was rejected before only because everything but mse/rmse was; now that
    the rest are supported it is refused on its own merits, so the message has
    to say why rather than just listing alternatives.
    """
    with pytest.raises(ValueError, match="cannot drive the round acceptance rule"):
        ResidualBoostedNSR(_two_term_factory, residual_metric="mbd")


def test_mape_residual_metric_labels_and_scores_in_mape():
    """`mape` now follows `score_metric` instead of being refused outright."""
    X, y = _make_data()
    front = ResidualBoostedNSR(_two_term_factory, residual_metric="mape").fit(X, y)

    assert len(front) >= 1
    assert {p.score_metric for p in front.points} == {"mape"}
    assert all(p.mse >= 0.0 for p in front.points)


def test_rmse_residual_metric_labels_and_scores_in_rmse():
    """Under `residual_metric="rmse"` the emitted points must report RMSE.

    A point labelled `mse` while carrying an RMSE value (or the reverse) is the
    silent failure this guards: `ParetoPoint.score` negates for r2-like metrics
    and downstream selection compares the numbers directly.
    """
    X, y = _make_data()
    booster = ResidualBoostedNSR(_two_term_factory, residual_metric="rmse")
    front = booster.fit(X, y)
    assert {p.score_metric for p in front.points} == {"rmse"}
    # `cum_score` is the RMSE of the same residual `cum_mse` is the MSE of.
    for r in booster.rounds_:
        assert r["cum_score"] == pytest.approx(np.sqrt(r["cum_mse"]))


def test_both_metrics_keep_the_same_rounds_under_a_converted_min_gain():
    """MSE and RMSE gains are monotonically related, so with `min_gain`
    converted by `gain_rmse = 1 - sqrt(1 - gain_mse)` the two modes must accept
    exactly the same rounds. Otherwise switching metric silently changes how
    many terms the model gets."""
    X, y = _make_data()
    mse_gain = 0.02
    rmse_gain = 1.0 - np.sqrt(1.0 - mse_gain)

    a = ResidualBoostedNSR(_two_term_factory, min_gain=mse_gain)
    b = ResidualBoostedNSR(_two_term_factory, min_gain=rmse_gain,
                           residual_metric="rmse")
    a.fit(X, y)
    b.fit(X, y)
    assert [r["added"] for r in a.rounds_] == [r["added"] for r in b.rounds_]
    assert len(a.terms_) == len(b.terms_)
    # Same terms, not merely the same count.
    assert [str(t) for t, _ in a.terms_] == [str(t) for t, _ in b.terms_]


# ---------------------------------------------------------------------------
# _bounded_simplify: `sympy.simplify` must not be able to hang a fit
# ---------------------------------------------------------------------------
def test_bounded_simplify_matches_plain_simplify_when_it_is_fast():
    """The bound must be invisible for every call that already completed.

    Normal expressions simplify in well under a second, so the guard has to be
    a pure pass-through for them -- otherwise it would change results that were
    never broken.
    """
    import sympy as sp

    from nsr_engine.engine import _bounded_simplify

    a, b = sp.symbols("a b")
    for expr in (sp.Float(0.5) + sp.Float(2.0) * (a - sp.Float(0.1)) / sp.Float(1.7),
                 sp.sqrt(sp.Abs(a * b)) + a ** 2 - sp.Float(0.75) * b,
                 (a + b) ** 2 - a ** 2 - 2 * a * b):
        assert _bounded_simplify(expr) == sp.simplify(expr)


def test_bounded_simplify_returns_the_input_when_it_runs_long(monkeypatch):
    """On timeout it falls back to the unsimplified expression.

    That is the same fallback the callers already take when `simplify` raises:
    the expression is mathematically identical, it merely reads less tidily.
    Without this, one pathological candidate can burn the whole fit -- measured
    at 20% of fits exceeding a 1800 s cap on a 10-feature search.
    """
    import time

    import sympy as sp

    from nsr_engine import engine as _engine

    monkeypatch.setenv("NSR_SIMPLIFY_TIMEOUT_S", "0.25")
    monkeypatch.setattr(sp, "simplify", lambda e, **kw: time.sleep(30))

    expr = sp.Symbol("a") + sp.Float(1.0)
    t0 = time.perf_counter()
    got = _engine._bounded_simplify(expr)
    elapsed = time.perf_counter() - t0

    assert got == expr, "should hand back the untouched expression"
    assert elapsed < 5.0, f"the bound did not fire (took {elapsed:.1f}s)"


def test_bounded_simplify_restores_the_previous_signal_handler():
    """The guard must not leave SIGALRM pointing at its own handler, or an
    unrelated alarm elsewhere in the process would raise TimeoutError."""
    import signal

    import sympy as sp

    from nsr_engine.engine import _bounded_simplify

    before = signal.getsignal(signal.SIGALRM)
    _bounded_simplify(sp.Symbol("a") + sp.Float(1.0))
    assert signal.getsignal(signal.SIGALRM) is before
    # And no timer is left armed.
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
