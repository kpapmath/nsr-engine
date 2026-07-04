"""CSV-input pipeline with explicit target and selected feature columns.

Run:

    python examples/csv_input.py
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import add_common_args, print_front, small_engine

from nsr_engine.pipeline import (
    evaluate_with_sympy,
    load_csv_dataset,
    train_test_validation_split,
)


def write_example_csv(path: Path, *, rows: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    temp = rng.uniform(-1.0, 1.0, rows)
    pressure = rng.uniform(0.2, 2.0, rows)
    drift = rng.normal(0.0, 0.03, rows)
    target = 1.2 * temp - 0.35 * np.log(pressure) + drift
    pd.DataFrame(
        {
            "temp": temp,
            "pressure": pressure,
            "unused_note": np.arange(rows),
            "target": target,
        }
    ).to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="nsr-pipeline-csv-") as tmp:
        csv_path = write_example_csv(
            Path(tmp) / "measurements.csv",
            rows=args.rows,
            seed=args.seed,
        )
        X, y = load_csv_dataset(
            str(csv_path),
            target_col="target",
            feature_cols=["temp", "pressure"],
        )
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
        print_front("csv input", front)

        if len(front):
            test_rmse = evaluate_with_sympy(front.elbow().sympy_expr, X_test, y_test)
            if test_rmse is not None:
                print(f"test_rmse={test_rmse:.6f}")


if __name__ == "__main__":
    main()
