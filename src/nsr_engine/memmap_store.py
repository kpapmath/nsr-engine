from __future__ import annotations

"""On-disk float32 memmap store for out-of-core full-set NSR training.

A month of XBTUSD feature data is ~800M rows; as float64 that is ~100GB, far
beyond a typical workstation's RAM. To train on the *full* set without holding
it in memory, this module streams the projected feature + target columns once
into a single on-disk ``numpy.memmap`` of shape ``(n_rows, n_cols)`` (float32).

Downstream code (NSR REINFORCE) reads only random row subsamples per iteration
and evaluates final formulas by streaming contiguous chunks, so peak RAM stays
small while every row participates in training.

The build is cached: a JSON sidecar records ``n_rows``, the column order, and a
signature of the source files (name + size + row count). A subsequent run with
the same inputs re-opens the existing memmap read-only instead of rebuilding.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_SIDECAR_SUFFIX = ".meta.json"


@dataclass
class MemmapDataset:
    """A row-major float32 memmap of feature columns plus the target column.

    Attributes
    ----------
    data:
        ``np.memmap`` of shape ``(n_rows, n_cols)``.  Column ``j`` corresponds
        to ``columns[j]``.  Missing source values are stored as NaN.
    columns:
        Ordered list of all stored columns (features followed by the target).
    feature_cols:
        The subset of ``columns`` to expose as SR inputs (target excluded).
    target_col:
        Name of the target column.
    n_rows:
        Total number of rows.
    path:
        Path to the backing ``.npy``-style raw memmap file.
    """

    data: np.memmap
    columns: list[str]
    feature_cols: list[str]
    target_col: str
    n_rows: int
    path: Path

    @property
    def col_to_idx(self) -> dict[str, int]:
        return {c: j for j, c in enumerate(self.columns)}

    def gather(
        self, rows: np.ndarray | slice
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Return ``(feature_arrays, y)`` for the given rows, materialized in RAM.

        ``rows`` may be an integer index array (random subsample) or a slice
        (contiguous chunk).  Only the selected rows are pulled off disk.
        Features stay float32 (halving chunk RAM vs float64); the target is
        cast to float64 so downstream metric accumulation stays accurate.
        """
        idx = self.col_to_idx
        block = np.array(self.data[rows], dtype=np.float32)
        arrays = {c: block[:, idx[c]] for c in self.feature_cols}
        y = block[:, idx[self.target_col]].astype(np.float64)
        return arrays, y


def chunk_ranges(lo: int, hi: int, chunk_rows: int) -> list[tuple[int, int]]:
    """Split ``[lo, hi)`` into contiguous ``(start, stop)`` chunks."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be >= 1")
    return [(s, min(s + chunk_rows, hi)) for s in range(lo, hi, chunk_rows)]


# ---------------------------------------------------------------------------
# Build / cache
# ---------------------------------------------------------------------------


def _source_signature(files: list[Path]) -> list[dict[str, object]]:
    sig: list[dict[str, object]] = []
    for f in files:
        st = f.stat()
        sig.append(
            {
                "name": f.name,
                "size": st.st_size,
                "rows": pq.ParquetFile(f).metadata.num_rows,
            }
        )
    return sig


def _existing_columns(first_file: Path, requested: list[str]) -> list[str]:
    available = set(pq.ParquetFile(first_file).schema.names)
    return [c for c in requested if c in available]


def build_memmap_dataset(
    files: list[Path],
    feature_cols: list[str],
    target_col: str,
    memmap_path: Path,
    *,
    batch_size: int = 500_000,
    rebuild: bool = False,
    verbose: bool = True,
) -> MemmapDataset:
    """Stream *files* into a float32 memmap at *memmap_path*.

    Columns absent from the data are dropped (with a note); the target column
    must be present.  Returns a :class:`MemmapDataset` opened read-only.

    Caching: if *memmap_path* and its sidecar already describe the same source
    files and column set, the existing memmap is reopened without rebuilding.
    Pass ``rebuild=True`` to force a fresh build.
    """
    if not files:
        raise FileNotFoundError("build_memmap_dataset: no input files")
    memmap_path = Path(memmap_path)
    memmap_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = memmap_path.with_suffix(memmap_path.suffix + _SIDECAR_SUFFIX)

    requested = list(feature_cols)
    if target_col not in requested:
        requested = requested + [target_col]
    stored_cols = _existing_columns(files[0], requested)
    if target_col not in stored_cols:
        raise KeyError(
            f"target column '{target_col}' not present in {files[0].name}"
        )
    dropped = [c for c in requested if c not in stored_cols]
    if dropped and verbose:
        print(f"[memmap] dropping {len(dropped)} columns absent from data: {dropped}")
    stored_feature_cols = [c for c in stored_cols if c != target_col]

    signature = _source_signature(files)
    n_rows = int(sum(int(s["rows"]) for s in signature))
    n_cols = len(stored_cols)

    # --- cache check -------------------------------------------------------
    if not rebuild and memmap_path.exists() and sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            meta = None
        if (
            meta is not None
            and meta.get("n_rows") == n_rows
            and meta.get("columns") == stored_cols
            and meta.get("signature") == signature
            and meta.get("dtype") == "float32"
        ):
            if verbose:
                print(
                    f"[memmap] reusing cached store {memmap_path.name} "
                    f"({n_rows:,} rows x {n_cols} cols)"
                )
            data = np.memmap(
                memmap_path, dtype=np.float32, mode="r", shape=(n_rows, n_cols)
            )
            return MemmapDataset(
                data=data,
                columns=stored_cols,
                feature_cols=stored_feature_cols,
                target_col=target_col,
                n_rows=n_rows,
                path=memmap_path,
            )

    # --- build -------------------------------------------------------------
    if verbose:
        gib = n_rows * n_cols * 4 / 1024**3
        print(
            f"[memmap] building store: {n_rows:,} rows x {n_cols} cols "
            f"(~{gib:.1f} GiB float32) -> {memmap_path}"
        )
        print(f"[memmap] columns: {stored_cols}")

    from nsr_engine._logging import Heartbeat

    hb = Heartbeat("memmap-build", interval_s=15.0)
    data = np.memmap(
        memmap_path, dtype=np.float32, mode="w+", shape=(n_rows, n_cols)
    )
    offset = 0
    for fi, f in enumerate(files):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(columns=stored_cols, batch_size=batch_size):
            n = batch.num_rows
            block = np.empty((n, n_cols), dtype=np.float32)
            for j, c in enumerate(stored_cols):
                block[:, j] = batch.column(c).to_numpy(zero_copy_only=False)
            data[offset : offset + n] = block
            offset += n
            if verbose:
                pct = 100.0 * offset / n_rows
                hb.beat(f"{offset:,}/{n_rows:,} rows ({pct:.1f}%) file {fi + 1}/{len(files)}")
        if verbose:
            print(
                f"[memmap]   [{fi + 1}/{len(files)}] {f.name}: "
                f"{offset:,}/{n_rows:,} rows written",
                flush=True,
            )
    if offset != n_rows:
        raise RuntimeError(
            f"[memmap] row-count mismatch: wrote {offset:,}, expected {n_rows:,}"
        )
    data.flush()

    sidecar.write_text(
        json.dumps(
            {
                "n_rows": n_rows,
                "columns": stored_cols,
                "dtype": "float32",
                "target_col": target_col,
                "signature": signature,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if verbose:
        print(f"[memmap] build complete; sidecar -> {sidecar.name}")

    data_ro = np.memmap(
        memmap_path, dtype=np.float32, mode="r", shape=(n_rows, n_cols)
    )
    return MemmapDataset(
        data=data_ro,
        columns=stored_cols,
        feature_cols=stored_feature_cols,
        target_col=target_col,
        n_rows=n_rows,
        path=memmap_path,
    )
