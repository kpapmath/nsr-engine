"""Out-of-core pipeline using Parquet input, a memmap store, and fit_memmap.

Run:

    python examples/memmap_out_of_core.py
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from pipeline_common import add_common_args, print_front, small_engine

from nsr_engine.pipeline import make_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print(
            "Install pyarrow to run this example: "
            "pip install 'nsr-engine[memmap]'"
        )
        return

    from nsr_engine.memmap_store import build_memmap_dataset

    X, y = make_dataset(args.rows, args.seed)
    frame = X.assign(target=y)

    with tempfile.TemporaryDirectory(prefix="nsr-pipeline-memmap-") as tmp:
        tmp_path = Path(tmp)
        parquet_path = tmp_path / "train.parquet"
        pq.write_table(pa.Table.from_pandas(frame), parquet_path)

        store = build_memmap_dataset(
            files=[parquet_path],
            feature_cols=list(X.columns),
            target_col="target",
            memmap_path=tmp_path / "train.mmap",
            verbose=True,
        )

        front = small_engine(
            seed=args.seed,
            n_iters=args.iters,
            n_lambda=args.lambdas,
            step_subsample_size=min(128, args.rows),
        ).fit_memmap(store, train_lo=0, train_hi=store.n_rows, chunk_rows=200)
        print_front("memmap out-of-core", front)


if __name__ == "__main__":
    main()
