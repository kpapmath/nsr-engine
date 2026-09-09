"""R4 guard: the pipeline selects on validation and never looks ahead.

Three properties, so the no-leakage guarantee that holds today by construction
cannot silently regress:

    1. The temporal 3-way split is ordered and disjoint (train < val < test).
    2. Train-time statistics (feature standardization) are computed on train
       rows only — validation/test rows do not inform them.
    3. The val-aware selector chooses an operating point from validation alone;
       it has no test argument and drops points worse than the mean-predictor
       baseline.
"""

import numpy as np
import pandas as pd
import pytest

sp = pytest.importorskip("sympy")

from nsr_engine import NSREngine
from nsr_engine.pareto import ParetoFront, ParetoPoint
from nsr_engine.pipeline import select_on_validation, train_test_validation_split


def _data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    X = pd.DataFrame({"a": a, "b": b})
    y = pd.Series(2.0 * a + 0.5 * b)
    return X, y


def test_temporal_split_is_ordered_and_disjoint():
    X, y = _data(100)
    Xtr, Xte, Xva, ytr, yte, yva = train_test_validation_split(
        X, y, train_frac=0.6, test_frac=0.2, validation_frac=0.2, seed=0, shuffle=False
    )
    # Temporal split: contiguous prefixes, no overlap, full coverage.
    assert len(Xtr) == 60 and len(Xva) == 20 and len(Xte) == 20
    # train is the first 60 rows, val the next 20, test the last 20 (shuffle=False).
    assert np.allclose(Xtr["a"].to_numpy(), X["a"].to_numpy()[:60])
    assert np.allclose(Xva["a"].to_numpy(), X["a"].to_numpy()[60:80])
    assert np.allclose(Xte["a"].to_numpy(), X["a"].to_numpy()[80:100])


def test_standardization_uses_train_rows_only():
    X, y = _data(400)
    Xtr, Xte, Xva, ytr, yte, yva = train_test_validation_split(
        X, y, train_frac=0.6, test_frac=0.2, validation_frac=0.2, seed=0, shuffle=False
    )
    eng = NSREngine(n_lambda=2, n_iters=5, batch_size=16, max_len=5,
                    random_state=0, device="cpu")
    eng.fit(Xtr, ytr)
    # The engine's standardization statistics must match the TRAIN rows, and must
    # differ from statistics that would include validation/test rows. `_feat_mean`
    # is a {column: mean} mapping.
    feat_mean = dict(eng._feat_mean)
    for col in X.columns:
        train_mean = float(Xtr[col].to_numpy(dtype=np.float64).mean())
        full_mean = float(X[col].to_numpy(dtype=np.float64).mean())
        assert feat_mean[col] == pytest.approx(train_mean, abs=1e-4)
        assert abs(train_mean - full_mean) > 1e-9  # train-only stats genuinely differ


def _front_two_points():
    a, b = sp.Symbol("a"), sp.Symbol("b")
    good = ParetoPoint(equation="2*a + 0.5*b", sympy_expr=2 * a + 0.5 * b,
                       complexity=5, mse=0.0)
    bad = ParetoPoint(equation="100*a", sympy_expr=100 * a, complexity=3, mse=1e6)
    return ParetoFront([good, bad])


def test_val_selector_drops_below_baseline_and_ignores_test():
    X, y = _data(300)
    _, _, Xva, _, _, yva = train_test_validation_split(
        X, y, train_frac=0.6, test_frac=0.2, validation_frac=0.2, seed=0, shuffle=False
    )
    front = _front_two_points()
    chosen = select_on_validation(front, Xva, yva)
    assert chosen is not None
    # The good expression (matches y) beats the mean-predictor baseline; the
    # wildly-wrong one does not and must not be selected.
    assert chosen.equation == "2*a + 0.5*b"


def test_val_selector_returns_none_when_all_worse_than_baseline():
    X, y = _data(300)
    _, _, Xva, _, _, yva = train_test_validation_split(
        X, y, train_frac=0.6, test_frac=0.2, validation_frac=0.2, seed=0, shuffle=False
    )
    a = sp.Symbol("a")
    junk = ParetoFront([ParetoPoint(equation="1000*a", sympy_expr=1000 * a,
                                    complexity=3, mse=1e9)])
    assert select_on_validation(junk, Xva, yva) is None
