"""Reusable full-pipeline helpers for nsr-engine command-line runs."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def make_dataset(rows: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)

    a = rng.uniform(-2.0, 2.0, rows)
    b = rng.uniform(-1.5, 1.5, rows)
    c = rng.uniform(0.1, 3.0, rows)
    noise = 0.03 * rng.standard_normal(rows)

    X = pd.DataFrame({"a": a, "b": b, "c": c})
    y = pd.Series(0.7 * a + 0.25 * np.square(b) - 0.4 * np.log(c) + noise)
    return X, y


def load_csv_dataset(
    path: str,
    *,
    target_col: str,
    feature_cols: list[str] | None,
) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path)
    if target_col not in frame.columns:
        raise ValueError(f"target column {target_col!r} is not present in {path}")

    if feature_cols is None:
        feature_cols = [col for col in frame.columns if col != target_col]

    missing = [col for col in feature_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"feature columns not present in {path}: {missing}")

    X = frame.loc[:, feature_cols].copy()
    y = frame.loc[:, target_col].copy()
    return X, y


def validate_split_fractions(
    train_frac: float,
    test_frac: float,
    validation_frac: float | None,
) -> None:
    fractions = {
        "train": train_frac,
        "test": test_frac,
    }
    if validation_frac is not None:
        fractions["validation"] = validation_frac

    out_of_range = [
        f"{name}={value}"
        for name, value in fractions.items()
        if not 0.0 <= value <= 1.0
    ]
    if out_of_range:
        raise ValueError(
            "Split fractions must be in [0, 1]. Invalid values: "
            + ", ".join(out_of_range)
            + ". Please try again."
        )

    total = sum(fractions.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"Split fractions must sum to 1.0, got {total:.6f}. Please try again."
        )


def train_test_validation_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    train_frac: float,
    test_frac: float,
    validation_frac: float | None,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | None,
    pd.Series,
    pd.Series,
    pd.Series | None,
]:
    validate_split_fractions(train_frac, test_frac, validation_frac)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    train_end = int(train_frac * len(X))

    if validation_frac is None:
        validation_idx = None
        test_start = train_end
    else:
        validation_end = train_end + int(validation_frac * len(X))
        validation_idx = idx[train_end:validation_end]
        test_start = validation_end

    train_idx = idx[:train_end]
    test_idx = idx[test_start:]

    X_validation = (
        None
        if validation_idx is None
        else X.iloc[validation_idx].reset_index(drop=True)
    )
    y_validation = (
        None
        if validation_idx is None
        else y.iloc[validation_idx].reset_index(drop=True)
    )

    return (
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        X_validation,
        y.iloc[train_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
        y_validation,
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    resid = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(math.sqrt(float(np.mean(resid * resid))))


def evaluate_with_sympy(equation: object, X: pd.DataFrame, y: pd.Series) -> float | None:
    try:
        import sympy as sp
    except ImportError:
        print("Install sympy to evaluate the selected formula on held-out rows.")
        return None

    symbols = [sp.Symbol(col) for col in X.columns]
    fn = sp.lambdify(symbols, equation, modules="numpy")
    pred = np.asarray(fn(*(X[col].to_numpy() for col in X.columns)), dtype=np.float64)

    if pred.ndim == 0:
        pred = np.full(len(X), float(pred), dtype=np.float64)

    mask = np.isfinite(pred) & np.isfinite(y.to_numpy(dtype=np.float64))
    if int(mask.sum()) == 0:
        print("Selected formula produced no finite held-out predictions.")
        return None

    return rmse(y.to_numpy(dtype=np.float64)[mask], pred[mask])
