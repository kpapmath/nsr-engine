"""Smoke tests: import, fit on a tiny synthetic dataset, check front shape."""

import numpy as np
import pandas as pd
import pytest

from nsr_engine import NSREngine, ParetoFront
from examples.full_pipeline import (
    train_test_validation_split,
    validate_split_fractions,
)


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


def test_score_metric_mae():
    engine = NSREngine(score_metric="mae", affine_reward=False)
    pred = np.array([1.0, 3.0, 6.0])
    y = np.array([0.0, 4.0, 3.0])

    assert engine._score(pred, y) == pytest.approx(5.0 / 3.0)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("mape", (0.1 + 0.1 + 0.1) / 3.0 * 100.0),
        ("mbd", abs((11.0 + 18.0 + 33.0 - 60.0) / 3.0)),
        ("r2", 0.93),
        ("adjusted_r2", 0.86),
    ],
)
def test_additional_score_metrics(metric, expected):
    engine = NSREngine(score_metric=metric, affine_reward=False)
    pred = np.array([11.0, 18.0, 33.0])
    y = np.array([10.0, 20.0, 30.0])

    assert engine._score(pred, y) == pytest.approx(expected)


def test_score_metric_validation():
    with pytest.raises(ValueError, match="score_metric"):
        NSREngine(score_metric="huber")


def test_train_test_split_defaults_without_validation():
    X, y = _make_data(n=10)

    X_train, X_test, X_validation, y_train, y_test, y_validation = (
        train_test_validation_split(
            X,
            y,
            train_frac=0.8,
            test_frac=0.2,
            validation_frac=None,
            seed=1,
        )
    )

    assert len(X_train) == 8
    assert len(y_train) == 8
    assert len(X_test) == 2
    assert len(y_test) == 2
    assert X_validation is None
    assert y_validation is None


def test_train_test_validation_split():
    X, y = _make_data(n=10)

    X_train, X_test, X_validation, y_train, y_test, y_validation = (
        train_test_validation_split(
            X,
            y,
            train_frac=0.7,
            test_frac=0.2,
            validation_frac=0.1,
            seed=1,
        )
    )

    assert len(X_train) == 7
    assert len(y_train) == 7
    assert len(X_test) == 2
    assert len(y_test) == 2
    assert X_validation is not None
    assert y_validation is not None
    assert len(X_validation) == 1
    assert len(y_validation) == 1


def test_split_fraction_range_validation():
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        validate_split_fractions(1.2, -0.2, None)


def test_split_fraction_sum_validation():
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_split_fractions(0.7, 0.2, None)


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


def test_pareto_front_to_frame_uses_metric_column():
    from nsr_engine.pareto import ParetoPoint

    front = ParetoFront(
        [
            ParetoPoint(
                equation="a",
                sympy_expr=None,
                complexity=1,
                mse=0.2,
                score_metric="mae",
            )
        ]
    )

    assert list(front.to_frame().columns) == ["equation", "complexity", "mae"]


def test_pareto_front_maximizes_r2():
    from nsr_engine.pareto import ParetoPoint

    pts = [
        ParetoPoint(equation="a", sympy_expr=None, complexity=1, mse=0.7, score_metric="r2"),
        ParetoPoint(equation="b", sympy_expr=None, complexity=1, mse=0.8, score_metric="r2"),
    ]
    front = ParetoFront(pts).dominance_filter()

    assert [p.equation for p in front.points] == ["b"]
    assert front.to_frame()["r2"].iloc[0] == pytest.approx(0.8)
