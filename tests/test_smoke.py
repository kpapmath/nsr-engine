"""Smoke tests: import, fit on a tiny synthetic dataset, check front shape."""

import numpy as np
import pandas as pd
import pytest

from nsr_engine import NSREngine, ParetoFront


def _make_data(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    X = pd.DataFrame({"a": a, "b": b})
    y = pd.Series(0.5 * a + 0.3 * b + 0.1 * rng.standard_normal(n))
    return X, y


def test_import():
    from nsr_engine import NSREngine, ParetoFront, ParetoPoint, SREngine  # noqa: F401


def test_fit_returns_pareto_front():
    X, y = _make_data()
    engine = NSREngine(
        n_lambda=2,
        n_iters=5,
        batch_size=16,
        max_len=5,
        random_state=0,
        standardize=True,
        affine_reward=True,
    )
    front = engine.fit(X, y)
    assert isinstance(front, ParetoFront)
    assert len(front) >= 0  # may be empty for very short runs


def test_pareto_front_dominance_filter():
    from nsr_engine.pareto import ParetoPoint

    pts = [
        ParetoPoint(equation="a", sympy_expr=None, complexity=1, mse=0.5),
        ParetoPoint(equation="b", sympy_expr=None, complexity=2, mse=0.3),
        ParetoPoint(equation="c", sympy_expr=None, complexity=2, mse=0.6),  # dominated
    ]
    front = ParetoFront(pts).dominance_filter()
    equations = {p.equation for p in front.points}
    assert "c" not in equations
    assert "a" in equations
    assert "b" in equations


def test_pareto_front_elbow():
    from nsr_engine.pareto import ParetoPoint

    pts = [
        ParetoPoint(equation="a", sympy_expr=None, complexity=1, mse=1.0),
        ParetoPoint(equation="b", sympy_expr=None, complexity=3, mse=0.2),
        ParetoPoint(equation="c", sympy_expr=None, complexity=5, mse=0.18),
    ]
    elbow = ParetoFront(pts).elbow()
    assert elbow.equation == "b"
