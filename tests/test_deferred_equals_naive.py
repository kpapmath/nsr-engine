"""R3: the deferred assembly path must reproduce the naive full-dataset loop.

The engine defers exact scoring until after the lambda-sweep, then truncates to
top-K per complexity before sympy conversion.  Ranking that truncation by the
*exact* full-dataset score makes it front-preserving: a front member is
exact-optimal at its complexity, so it ranks first in its group and survives any
K >= 1.

Ranking it by the *approx* subsampled score -- the pre-0.4 behaviour -- does not
have that property, and `test_approx_prefilter_can_drop_exact_best` exhibits a
pool where it demonstrably drops the exact-best candidate.  That test is what
makes the guarantee falsifiable rather than decorative.
"""

import numpy as np
import pandas as pd
import pytest

from nsr_engine.engine import NSREngine, _OOCExpr


def _engine(**kw) -> NSREngine:
    # `refine_constants=False`: these tests pin the *scoring and assembly* path
    # -- that the exact prefilter never evicts a front member. Constant
    # refinement is a separate post-assembly stage that deliberately rewrites
    # coefficients, so leaving it on would compare a refined front against an
    # unrefined naive one and fail for reasons unrelated to the property named.
    defaults = dict(random_state=0, standardize=False, prefilter_per_complexity=2,
                    refine_constants=False)
    return NSREngine(**{**defaults, **kw})


def _data(n: int = 600, seed: int = 3) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """A target whose front genuinely spans several complexity levels.

    The interaction term matters: with a purely linear target the front collapses
    to a single complexity-1 point and truncation at higher complexities becomes
    unobservable, which would make these tests pass vacuously.  Here the exact
    front is c=1 `a` -> c=3 `a*b` -> c=5 `a*b + a`, with strictly decreasing MSE.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    y = (0.6 * a + 0.9 * (a * b) + 0.03 * rng.standard_normal(n)).astype(np.float64)
    return {"a": a, "b": b}, y


# Candidate tokens spanning complexities 1, 3 and 5.
_TOKENS: list[tuple[str, ...]] = [
    ("a",),
    ("b",),
    ("+", "a", "b"),
    ("-", "a", "b"),
    ("*", "a", "b"),
    ("/", "a", "b"),
    ("+", "a", "a"),
    ("+", "*", "a", "b", "a"),
    ("-", "*", "a", "b", "b"),
    ("+", "*", "a", "a", "b"),
    ("*", "+", "a", "b", "a"),
]

# The exact-best candidate at each complexity, verified against a full-dataset
# scan of `_TOKENS` over `_data()`.
_EXACT_BEST = {1: ("a",), 3: ("*", "a", "b"), 5: ("+", "*", "a", "b", "a")}


def _adversarial_pool() -> list[_OOCExpr]:
    """Pool whose approx ranking is worst exactly where the exact ranking is best.

    This is the failure mode the referee identified: `approx_mse` is measured on a
    random row subsample, so the exact-best candidate at a complexity can carry a
    poor approx score purely from sampling noise.  Encoding it explicitly keeps the
    test deterministic instead of hoping an unlucky seed reproduces it.
    """
    return [
        _OOCExpr(
            tokens=t,
            complexity=len(t),
            approx_mse=9.0 + i if t in _EXACT_BEST.values() else 0.1 * i,
        )
        for i, t in enumerate(_TOKENS)
    ]


def _naive_front(engine: NSREngine, pool: list[_OOCExpr], arrays, y):
    """Score every candidate on the full dataset, no prefilter at all."""
    return engine._assemble_front(engine._exact_eval_arrays(pool, arrays, y))


def _points(front) -> list[tuple[str, int, float]]:
    return sorted(
        (p.equation, p.complexity, round(p.score, 12)) for p in front.points
    )


# --------------------------------------------------------------------------
# The guarantee
# --------------------------------------------------------------------------


def test_deferred_exact_front_equals_naive_front():
    """Exact-ranked truncation yields element-wise the same front as no truncation."""
    arrays, y = _data()
    engine = _engine(prefilter_metric="exact")
    pool = _adversarial_pool()

    exact = engine._exact_eval_arrays(pool, arrays, y)
    truncated = engine._truncate_exact_per_complexity(
        exact, engine.prefilter_per_complexity
    )
    deferred = engine._assemble_front(truncated)
    naive = _naive_front(engine, pool, arrays, y)

    assert _points(deferred) == _points(naive)
    assert len(truncated) < len(exact), "truncation must actually bite"
    # Guard against the vacuous version of this test: if the front were a single
    # point, truncation at the other complexities would be unobservable.
    assert len({c for _, c, _ in _points(naive)}) >= 3, "front must span complexities"


def test_old_approx_first_assembly_differs_from_naive():
    """The pre-0.4 order (truncate by approx, then score) does NOT reproduce naive.

    This is the control for `test_deferred_exact_front_equals_naive_front`: it
    shows that test can fail, so its passing is evidence about the fix rather than
    about the pool being too easy.
    """
    arrays, y = _data()
    engine = _engine(prefilter_metric="approx")
    pool = _adversarial_pool()

    old_candidates = engine._prefilter_candidates(
        pool, per_complexity=engine.prefilter_per_complexity
    )
    old_front = engine._assemble_front(
        engine._exact_eval_arrays(old_candidates, arrays, y)
    )
    naive = _naive_front(engine, pool, arrays, y)

    assert _points(old_front) != _points(naive)


@pytest.mark.parametrize("metric", ["mse", "rmse", "mae", "r2"])
def test_front_preserved_for_maximize_and_minimize_metrics(metric):
    """Direction handling: r2 is higher-is-better, the rest are lower-is-better."""
    arrays, y = _data()
    engine = _engine(prefilter_metric="exact", score_metric=metric)
    pool = _adversarial_pool()

    exact = engine._exact_eval_arrays(pool, arrays, y)
    deferred = engine._assemble_front(
        engine._truncate_exact_per_complexity(exact, engine.prefilter_per_complexity)
    )
    naive = _naive_front(engine, pool, arrays, y)

    assert _points(deferred) == _points(naive)


def test_truncation_keeps_the_exact_best_at_each_complexity():
    """The rank-1 candidate per complexity survives even at K=1."""
    arrays, y = _data()
    engine = _engine(prefilter_metric="exact")
    pool = _adversarial_pool()

    exact = engine._exact_eval_arrays(pool, arrays, y)
    kept = {row[0].tokens for row in engine._truncate_exact_per_complexity(exact, 1)}

    by_complexity: dict[int, list] = {}
    for cand, score, _, _ in exact:
        if np.isfinite(score):
            by_complexity.setdefault(cand.complexity, []).append((score, cand.tokens))
    for complexity, rows in by_complexity.items():
        best = min(rows)[1]  # score_metric is "mse" here: lower is better
        assert best == _EXACT_BEST[complexity], "test fixture drifted from the data"
        assert best in kept, f"exact-best at complexity {complexity} was dropped"


# --------------------------------------------------------------------------
# The wired-up path, not just the helpers
# --------------------------------------------------------------------------


def _fit_with_fixed_pool(engine: NSREngine, X, y, pool):
    """Run `fit` end to end with the lambda-sweep stubbed out.

    Training is far too slow for a unit test, but stubbing only `_sweep_lambdas`
    leaves the entire assembly path -- selection, exact scoring, truncation,
    sympy conversion, dominance filtering -- under test.  Without this the
    truncation call in `fit` could be deleted and the helper tests would still
    pass.
    """
    engine._sweep_lambdas = lambda *a, **kw: {c.tokens: c for c in pool}  # type: ignore[method-assign]
    return engine.fit(X, y)


def test_fit_assembly_path_is_front_preserving():
    """End-to-end through `fit`: the default (exact) path reproduces the naive front."""
    arrays, y = _data()
    X = pd.DataFrame(arrays)
    y_series = pd.Series(y)
    pool = _adversarial_pool()

    front = _fit_with_fixed_pool(_engine(prefilter_metric="exact"), X, y_series, pool)
    naive = _naive_front(_engine(), pool, arrays, y)

    assert _points(front) == _points(naive)


def test_fit_assembly_path_under_approx_differs():
    """The same wiring under `prefilter_metric="approx"` loses a front member."""
    arrays, y = _data()
    X = pd.DataFrame(arrays)
    y_series = pd.Series(y)
    pool = _adversarial_pool()

    front = _fit_with_fixed_pool(_engine(prefilter_metric="approx"), X, y_series, pool)
    naive = _naive_front(_engine(), pool, arrays, y)

    assert _points(front) != _points(naive)


def test_exact_is_the_default():
    """The front-preserving path is what users get without opting in."""
    assert NSREngine().prefilter_metric == "exact"


# --------------------------------------------------------------------------
# Why the old claim failed
# --------------------------------------------------------------------------


def test_approx_prefilter_can_drop_exact_best():
    """The legacy approx ranking evicts the exact-best candidate at a complexity.

    `approx_mse` is a *subsampled* loss, so two candidates are compared on
    different random rows.  Here `("*", "a", "b")` is the exact-best complexity-3
    candidate but carries the worst approx score -- exactly the noise pattern the
    referee flagged.  With K=1 the approx path drops it and the exact path keeps
    it, so the two fronts differ.
    """
    n = 400
    rng = np.random.default_rng(7)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    # Target is exactly a*b, so ("*", "a", "b") is the unambiguous best at c=3.
    y = (a * b).astype(np.float64)
    arrays = {"a": a, "b": b}

    pool = [
        _OOCExpr(tokens=("*", "a", "b"), complexity=3, approx_mse=9.9),  # best, ranked last
        _OOCExpr(tokens=("+", "a", "b"), complexity=3, approx_mse=0.1),
        _OOCExpr(tokens=("-", "a", "b"), complexity=3, approx_mse=0.2),
    ]

    exact_engine = _engine(prefilter_metric="exact", prefilter_per_complexity=1)
    approx_engine = _engine(prefilter_metric="approx", prefilter_per_complexity=1)

    exact_kept = exact_engine._truncate_exact_per_complexity(
        exact_engine._exact_eval_arrays(pool, arrays, y), 1
    )
    approx_kept = approx_engine._prefilter_candidates(pool, per_complexity=1)

    assert exact_kept[0][0].tokens == ("*", "a", "b")
    assert approx_kept[0].tokens == ("+", "a", "b")

    # And the resulting fronts genuinely differ.
    exact_front = exact_engine._assemble_front(exact_kept)
    approx_front = approx_engine._assemble_front(
        approx_engine._exact_eval_arrays(approx_kept, arrays, y)
    )
    assert _points(exact_front) != _points(approx_front)


def test_duplicate_equations_resolve_to_lowest_complexity():
    """Redundant token sequences must not pin an equation to an inflated complexity.

    `a`, `+ a 0.0` and `* a 1.0` are the same function, so they score identically
    and convert to the same equation string.  `_assemble_front` dedups by that
    string; if it broke ties by insertion order the reported complexity would
    depend on candidate ordering, and truncating the pool could change the front
    even when truncation preserved the best candidate at every complexity.
    """
    arrays, y = _data()
    engine = _engine()

    # `a`, `a*1`, `(a*1)*1` are the same function; all use tokens in the default
    # constant pool so they actually evaluate.
    duplicates = [
        _OOCExpr(tokens=("*", "*", "a", "1.0", "1.0"), complexity=5, approx_mse=0.0),
        _OOCExpr(tokens=("*", "a", "1.0"), complexity=3, approx_mse=0.0),
        _OOCExpr(tokens=("a",), complexity=1, approx_mse=0.0),
    ]
    scored = engine._exact_eval_arrays(duplicates, arrays, y)

    # All three really are the same function to the scorer.
    finite = [s for _, s, _, _ in scored if np.isfinite(s)]
    assert len(finite) == 3
    assert max(finite) - min(finite) == 0.0

    # Worst case for insertion-order dedup: the simplest form is visited last.
    front = engine._assemble_front(scored)
    assert [p.complexity for p in front.points] == [1]

    # And the answer must not depend on the order candidates arrive in.
    reversed_front = engine._assemble_front(list(reversed(scored)))
    assert _points(front) == _points(reversed_front)


def test_selection_respects_the_cost_cap():
    """`exact_prefilter_multiple` bounds the exactly-scored set; None disables it."""
    pool = [
        _OOCExpr(tokens=("a",) * (i % 3 + 1), complexity=i % 3 + 1, approx_mse=float(i))
        for i in range(300)
    ]

    uncapped = _engine(prefilter_metric="exact", exact_prefilter_multiple=None)
    assert len(uncapped._select_for_exact_scoring(pool, per_complexity=2)) == len(pool)

    capped = _engine(prefilter_metric="exact", exact_prefilter_multiple=4)
    kept = capped._select_for_exact_scoring(pool, per_complexity=2)
    assert len(kept) == 3 * (4 * 2)  # 3 complexity levels x (multiple * K)

    legacy = _engine(prefilter_metric="approx")
    assert len(legacy._select_for_exact_scoring(pool, per_complexity=2)) == 3 * 2


def test_invalid_config_rejected():
    with pytest.raises(ValueError, match="prefilter_metric"):
        NSREngine(prefilter_metric="subsampled")
    with pytest.raises(ValueError, match="exact_prefilter_multiple"):
        NSREngine(exact_prefilter_multiple=0)
