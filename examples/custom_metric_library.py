"""Pipeline using a custom score metric and reduced token library.

Run:

    python examples/custom_metric_library.py
"""

from __future__ import annotations

import argparse

from pipeline_common import add_common_args, print_front, small_engine

from nsr_engine.pipeline import make_dataset, train_test_validation_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    X, y = make_dataset(args.rows, args.seed)
    X_train, _, _, y_train, _, _ = train_test_validation_split(
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
        score_metric="mae",
        binary_ops=["+", "-", "*"],
        unary_ops=["square", "log"],
        const_tokens=["-1.0", "0.5", "1.0"],
    ).fit(X_train, y_train)
    print_front("custom metric/library", front)


if __name__ == "__main__":
    main()
