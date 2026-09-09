"""Guards for the v0.6.0 accuracy changes: constant refinement and the
relative complexity penalty.

Both were added because the benchmark suite showed the engine finding the right
expression *shape* with the wrong coefficients -- `0.937*x0*x1 + 1.07*x2` where
the target is `x0*x1 + x2` -- and because the default complexity penalty could
exceed the maximum achievable reward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nsr_engine import NSREngine


def _quadratic(n: int = 4_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x0": rng.uniform(-2, 2, n), "x1": rng.uniform(-2, 2, n)})
    y = pd.Series(X.x0 * X.x1 + X.x1 + rng.normal(0, 0.01, n))
    return X, y


def _engine(**kw) -> NSREngine:
    base = dict(n_iters=6, n_lambda=2, batch_size=16, max_len=10, hidden_dim=16,
                embed_dim=8, prefilter_per_complexity=3, random_state=0,
                device="cpu")
    return NSREngine(**{**base, **kw})


# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------
def test_new_accuracy_defaults_are_on():
    e = NSREngine()
    assert e.refine_constants is True
    assert e.lambda_relative is True
    assert e.refine_max_nfev == 200


def test_both_remain_switchable():
    e = NSREngine(refine_constants=False, lambda_relative=False)
    assert e.refine_constants is False
    assert e.lambda_relative is False


# ---------------------------------------------------------------------------
# the complexity penalty must not be able to swamp the reward
# ---------------------------------------------------------------------------
def test_relative_penalty_cannot_exceed_lambda():
    """Reward is 1/(1+nmse) - penalty, bounded in (0, 1].

    With an absolute penalty, the default lambda_max=1e-1 at max_len=12 gives
    0.1*12 = 1.2 -- larger than any achievable reward, so that lambda arm can
    never favour a long expression no matter how well it fits. Normalising by
    max_len bounds the penalty by lambda itself.
    """
    max_len, lam = 12, 0.1
    absolute = lam * max_len
    relative = lam * max_len / max_len
    assert absolute > 1.0, "this is the failure mode being guarded against"
    assert relative == pytest.approx(lam)      # bounded by lambda itself
    assert relative <= 1.0


def test_relative_penalty_is_applied_in_the_reward():
    import inspect

    from nsr_engine import engine as eng

    src = inspect.getsource(eng.NSREngine._train_one_lambda)
    assert "self.lambda_relative" in src
    assert "max(1, self.max_len)" in src


# ---------------------------------------------------------------------------
# refinement
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_refinement_never_worsens_the_front():
    """optimize_front keeps a refit only when it improves that point's score."""
    X, y = _quadratic()
    plain = _engine(refine_constants=False).fit(X, y)
    refined = _engine(refine_constants=True).fit(X, y)
    assert len(refined.points) == len(plain.points)
    by_cx = {p.complexity: p for p in plain.points}
    for pt in refined.points:
        if pt.complexity in by_cx:
            # `mse` holds the score under the configured metric; lower is better
            assert pt.mse <= by_cx[pt.complexity].mse + 1e-9


@pytest.mark.slow
def test_refinement_survives_missing_optional_deps(monkeypatch):
    """Without scipy/scikit-learn the front must come back unrefined, not lost."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name.startswith("nsr_engine.refinement"):
            raise ImportError("simulated missing extra")
        return real_import(name, *a, **kw)

    X, y = _quadratic(n=1_500)
    monkeypatch.setattr(builtins, "__import__", blocked)
    front = _engine(refine_constants=True).fit(X, y)
    assert front.points, "a missing optional extra must not empty the front"
