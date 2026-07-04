"""Generated-data pipeline with the default train/test split.

Run:

    python examples/generated_train_test.py
"""

from __future__ import annotations

import argparse

from pipeline_common import add_common_args, print_front, small_engine

from nsr_engine.pipeline import evaluate_with_sympy, make_dataset, train_test_validation_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    X, y = make_dataset(args.rows, args.seed)
    X_train, X_test, _, y_train, y_test, _ = train_test_validation_split(
        X,
        y,
        train_frac=0.8,
        test_frac=0.2,
        validation_frac=None,
        seed=args.seed + 1,
    )

    front = small_engine(
        seed=args.seed,
        n_iters=args.iters,
        n_lambda=args.lambdas,
    ).fit(X_train, y_train)
    print_front("generated train/test", front)

    if len(front):
        test_rmse = evaluate_with_sympy(front.elbow().sympy_expr, X_test, y_test)
        if test_rmse is not None:
            print(f"test_rmse={test_rmse:.6f}")


if __name__ == "__main__":
    main()
