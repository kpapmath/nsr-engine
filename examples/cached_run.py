"""Pipeline using the lambda-sweep cache.

Run:

    python examples/cached_run.py
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

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

    with tempfile.TemporaryDirectory(prefix="nsr-pipeline-cache-") as tmp:
        cache_dir = Path(tmp)
        first = small_engine(
            seed=args.seed,
            n_iters=args.iters,
            n_lambda=args.lambdas,
            cache_dir=cache_dir,
        ).fit(X_train, y_train)
        print_front("cached first run", first)

        second = small_engine(
            seed=args.seed,
            n_iters=args.iters,
            n_lambda=args.lambdas,
            cache_dir=cache_dir,
        ).fit(X_train, y_train)
        print_front("cached second run", second)


if __name__ == "__main__":
    main()
