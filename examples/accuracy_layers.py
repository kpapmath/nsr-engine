"""The three accuracy layers stacked on a two-additive-term target.

Prints the test R2 of each stage, so the contribution of each layer is visible:

    plain NSR  ->  + boosting  ->  + constant opt  ->  + joint refit

The target is `exp(x2) - 1.5*log(x4)`: two additive terms of comparable
magnitude, which is exactly the case a single affine-rewarded fit cannot
express.  See docs/accuracy_layers.md.

Requires: pip install "nsr-engine[refine]"

Run:

    python examples/accuracy_layers.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from pipeline_common import add_common_args, print_front, small_engine

from nsr_engine import ParetoFront, ParetoPoint, ResidualBoostedNSR
from nsr_engine._expr import eval_sympy_on
from nsr_engine.refinement import joint_refit_prune, optimize_constants


def make_two_term_dataset(rows: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    x2 = rng.uniform(-1.0, 1.0, rows)
    x4 = rng.uniform(0.5, 5.0, rows)
    X = pd.DataFrame({"x2": x2, "x4": x4})
    y = pd.Series(np.exp(x2) - 1.5 * np.log(x4) + 0.15 * rng.standard_normal(rows))
    return X, y


def test_r2(expr, X: pd.DataFrame, y: pd.Series) -> float:
    pred = eval_sympy_on(expr, X)
    if pred is None:
        return float("nan")
    truth = y.to_numpy(dtype=np.float64)
    mask = np.isfinite(pred)
    sse = float(np.sum((truth[mask] - pred[mask]) ** 2))
    sst = float(np.sum((truth[mask] - truth[mask].mean()) ** 2))
    return 1.0 - sse / sst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    X, y = make_two_term_dataset(args.rows, args.seed)
    split = int(0.8 * len(X))
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    def engine_factory(round_idx: int):
        # A fresh engine per round, with a seed that moves, so rounds do not
        # repeat the same search on the residual.
        return small_engine(
            seed=args.seed + round_idx,
            n_iters=args.iters,
            n_lambda=args.lambdas,
            max_len=13,
            unary_ops=["square", "abs", "log", "exp", "sqrt"],
        )

    # Stage 1: plain NSR — one affine-fitted expression.
    plain = engine_factory(0).fit(X_train, y_train)
    print_front("plain NSR", plain)
    if len(plain) == 0:
        print("No expressions found; increase --iters or --rows.")
        return
    scores = {"plain NSR": test_r2(plain.elbow().sympy_expr, X_test, y_test)}

    # Stage 2: + residual boosting (layer 1).
    booster = ResidualBoostedNSR(engine_factory, max_rounds=3, min_gain=0.02)
    boosted = booster.fit(X_train, y_train)
    print_front("+ boosting", boosted)
    for record in booster.rounds_:
        print(
            f"  round {record['round']}: added={record['added']} "
            f"gain={record['gain']:.4f} cum_mse={record['cum_mse']:.6f} "
            f"({record['reason']})"
        )
    if len(boosted) == 0:
        return
    scores["+ boosting"] = test_r2(boosted.elbow().sympy_expr, X_test, y_test)

    # Stage 3: + constant optimization as the per-term refiner (layer 2).
    booster_c = ResidualBoostedNSR(
        engine_factory, max_rounds=3, min_gain=0.02, term_refiner=optimize_constants
    )
    boosted_c = booster_c.fit(X_train, y_train)
    print_front("+ constant opt", boosted_c)
    if len(boosted_c) == 0:
        return
    scores["+ constant opt"] = test_r2(boosted_c.elbow().sympy_expr, X_test, y_test)

    # Stage 4: + joint refit and prune over the discovered terms (layer 3).
    refined = joint_refit_prune(booster_c.terms_, X_train, y_train)
    if refined is None:
        print("\njoint refit produced no usable model")
    else:
        expr, complexity, mse = refined
        points = list(boosted_c.points)
        points.append(
            ParetoPoint(
                equation=str(expr),
                sympy_expr=expr,
                complexity=complexity,
                mse=mse,
            )
        )
        final = ParetoFront(points).dominance_filter()
        print_front("+ joint refit", final)
        scores["+ joint refit"] = test_r2(final.elbow().sympy_expr, X_test, y_test)

    print("\nElbow test R2 by stage")
    for label, value in scores.items():
        print(f"  {label:<16} {value: .4f}")


if __name__ == "__main__":
    main()
