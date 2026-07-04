"""Run the nsr-engine pipeline end to end on synthetic data.

This example covers:

1. Creating a regression dataset.
2. Splitting train/test rows.
3. Training NSREngine across a lambda sweep.
4. Inspecting the Pareto front.
5. Selecting an elbow formula.
6. Evaluating the selected formula on held-out data when SymPy is installed.

Run from the repository root:

    python examples/full_pipeline.py

For a faster smoke run:

    python examples/full_pipeline.py --iters 20 --lambdas 3 --rows 1000
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from nsr_engine import NSREngine


def make_dataset(rows: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)

    a = rng.uniform(-2.0, 2.0, rows)
    b = rng.uniform(-1.5, 1.5, rows)
    c = rng.uniform(0.1, 3.0, rows)
    noise = 0.03 * rng.standard_normal(rows)

    X = pd.DataFrame({"a": a, "b": b, "c": c})
    y = pd.Series(0.7 * a + 0.25 * np.square(b) - 0.4 * np.log(c) + noise)
    return X, y


def train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    train_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    split = int(train_frac * len(X))
    train_idx = idx[:split]
    test_idx = idx[split:]
    return (
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lambdas", type=int, default=5)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=11)
    parser.add_argument("--metric", default="rmse")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/nsr-engine-full-pipeline-cache"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    X, y = make_dataset(args.rows, args.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_frac=0.8, seed=args.seed + 1
    )

    engine = NSREngine(
        n_lambda=args.lambdas,
        n_iters=args.iters,
        batch_size=args.batch_size,
        max_len=args.max_len,
        score_metric=args.metric,
        cache_dir=args.cache_dir,
        cache_prefix="full_pipeline",
        random_state=args.seed,
        standardize=True,
        affine_reward=True,
    )

    front = engine.fit(X_train, y_train)
    if len(front) == 0:
        raise RuntimeError(
            "No valid expressions were discovered. Increase --iters or --max-len."
        )

    frame = front.to_frame()
    print("\nPareto front")
    print(frame.to_string(index=False))

    selected = front.elbow()
    print("\nSelected elbow formula")
    print(selected.equation)
    print(f"complexity={selected.complexity} score_metric={selected.score_metric}")

    test_rmse = evaluate_with_sympy(selected.sympy_expr, X_test, y_test)
    if test_rmse is not None:
        print(f"held_out_rmse={test_rmse:.6f}")


if __name__ == "__main__":
    main()
