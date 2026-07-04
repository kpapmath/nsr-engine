# nsr-engine Quickstart

`nsr-engine` is a Python library for neural symbolic regression. It does not
currently install a command-line executable or define `console_scripts` in
`pyproject.toml`; the public interface is the Python API. This guide documents
the install commands and every public argument exposed by the library.

## Install

Core install:

```bash
pip install nsr-engine
```

Install with SymPy support for readable formulas:

```bash
pip install "nsr-engine[sympy]"
```

Install with out-of-core Parquet to memmap support:

```bash
pip install "nsr-engine[memmap,sympy]"
```

Install from source for development:

```bash
git clone https://github.com/kpapmath/nsr-engine
cd nsr-engine
pip install -e ".[sympy,memmap,dev]"
```

## Minimal In-Memory Run

```python
import numpy as np
import pandas as pd

from nsr_engine import NSREngine

rng = np.random.default_rng(0)
n = 5000
a = rng.standard_normal(n)
b = rng.standard_normal(n)

X = pd.DataFrame({"a": a, "b": b})
y = pd.Series(0.5 * a + 0.3 * b + 0.05 * rng.standard_normal(n))

engine = NSREngine(
    n_lambda=8,
    n_iters=300,
    batch_size=64,
    max_len=10,
    score_metric="mse",
    random_state=42,
)

front = engine.fit(X, y)

print(front.to_frame())
print("elbow formula:", front.elbow().equation)
```

## Out-of-Core Run

Use the memmap path when the training table is too large to fit comfortably in
RAM. This requires the `memmap` extra, which installs `pyarrow`.

```python
from pathlib import Path

from nsr_engine import NSREngine
from nsr_engine.memmap_store import build_memmap_dataset

store = build_memmap_dataset(
    files=list(Path("data").glob("*.parquet")),
    feature_cols=["a", "b", "c"],
    target_col="y",
    memmap_path=Path("cache/train.mmap"),
)

engine = NSREngine(
    n_lambda=8,
    n_iters=300,
    batch_size=64,
    max_len=10,
    step_subsample_size=50_000,
    score_metric="mae",
    random_state=42,
)

front = engine.fit_memmap(
    store,
    train_lo=0,
    train_hi=store.n_rows,
    chunk_rows=5_000_000,
)

print(front.to_frame())
```

## `NSREngine` Arguments

These are the constructor arguments accepted by `NSREngine(...)`.

| Argument | Default | Explanation |
| --- | --- | --- |
| `lambda_grid` | `None` | Explicit list or tuple of complexity penalty values. When provided, it overrides `n_lambda`, `lambda_min`, and `lambda_max`. |
| `n_lambda` | `10` | Number of lambda values to generate when `lambda_grid` is not supplied. |
| `lambda_min` | `1e-4` | Lower bound for the auto-generated log-spaced lambda grid. |
| `lambda_max` | `1e-1` | Upper bound for the auto-generated log-spaced lambda grid. |
| `n_iters` | `200` | Number of REINFORCE optimization iterations to run for each lambda value. |
| `batch_size` | `64` | Number of expression trees sampled from the GRU policy per iteration. |
| `max_len` | `15` | Maximum prefix token sequence length. This is also the maximum expression tree node count. |
| `elite_frac` | `0.05` | Risk-seeking quantile epsilon. Updates use samples at or above the `1 - elite_frac` reward quantile. |
| `entropy_weight` | `0.005` | Weight for the entropy bonus in the policy loss. Higher values encourage more exploration. |
| `hidden_dim` | `128` | GRU hidden state dimension. Larger values increase model capacity and memory use. |
| `embed_dim` | `32` | Token embedding dimension used before the GRU cell. |
| `lr` | `1e-3` | Adam optimizer learning rate. |
| `random_state` | `42` | Base random seed. Lambda run `i` uses `random_state + i`. |
| `cache_dir` | `None` | Directory for JSON candidate caches. When set, discovered candidates are saved per lambda and reused on later runs. |
| `cache_prefix` | `None` | Optional prefix added to cache file names. Useful when sharing one cache directory across experiments. |
| `binary_ops` | `("+", "-", "*", "/")` | Binary operators available to the expression grammar. |
| `unary_ops` | `("square", "abs", "log")` | Unary operators available to the expression grammar. |
| `const_tokens` | `("-1.0", "-0.5", "0.5", "1.0", "2.0")` | Constant terminal tokens available to sampled expressions. Values are parsed with `float(...)`. |
| `device` | `"auto"` | Torch device. `"auto"` selects CUDA, then Apple MPS, then CPU. You may also pass values such as `"cpu"` or `"cuda"`. |
| `step_subsample_size` | `None` | Number of rows used for each training iteration reward calculation. For in-memory `fit`, `None` means use all rows. For `fit_memmap`, `None` is treated as `50_000`. |
| `standardize` | `True` | Whether feature columns are z-scored before training. Returned SymPy formulas are converted back to raw feature terms when possible. |
| `affine_reward` | `True` | Whether rewards and final scoring use a least-squares affine fit `b0 + b1 * expression` before applying `score_metric`. This makes scoring less sensitive to expression scale and offset. |
| `score_metric` | `"mse"` | Accuracy metric to minimize. Supported values are `"mse"`, `"rmse"`, and `"mae"`. |
| `prefilter_per_complexity` | `16` | Number of best approximate-score candidates to keep per complexity before exact full-set evaluation. |

## `fit` Arguments

```python
front = engine.fit(X, y)
```

| Argument | Type | Explanation |
| --- | --- | --- |
| `X` | `pandas.DataFrame` | Feature table. Column names become variable tokens in the expression grammar. |
| `y` | `pandas.Series` | Target values aligned row-for-row with `X`. |

Notes:

- `fit` warns if any input columns contain negative values because operators
  such as `log` and division can produce invalid numeric regions.
- If `step_subsample_size` is smaller than the row count, each training
  iteration scores rewards on a random row subset.
- If `X` contains a `regime_id` column and subsampling is enabled, the sampler
  tries to draw a balanced subset across regime values.

## `fit_memmap` Arguments

```python
front = engine.fit_memmap(
    store,
    train_lo=0,
    train_hi=store.n_rows,
    chunk_rows=5_000_000,
    prefilter_per_complexity=None,
)
```

| Argument | Default | Explanation |
| --- | --- | --- |
| `store` | required | `MemmapDataset` returned by `build_memmap_dataset(...)`. |
| `train_lo` | required | Inclusive lower row bound for training and exact scoring. |
| `train_hi` | required | Exclusive upper row bound for training and exact scoring. Must leave at least two rows. |
| `chunk_rows` | `5_000_000` | Number of rows streamed at a time during standardization stats and exact score evaluation. |
| `prefilter_per_complexity` | `None` | Per-call override for the engine-level `prefilter_per_complexity`. |

## `build_memmap_dataset` Arguments

```python
store = build_memmap_dataset(
    files,
    feature_cols,
    target_col,
    memmap_path,
    batch_size=500_000,
    rebuild=False,
    verbose=True,
)
```

| Argument | Default | Explanation |
| --- | --- | --- |
| `files` | required | List of Parquet file paths to stream into one row-major float32 memmap. |
| `feature_cols` | required | Requested feature columns. Missing feature columns are dropped with a message when `verbose=True`. |
| `target_col` | required | Target column name. It must exist in the input files. |
| `memmap_path` | required | Path where the raw float32 memmap file is created or reused. |
| `batch_size` | `500_000` | Parquet batch size used while streaming source files into the memmap. |
| `rebuild` | `False` | Force rebuilding even when the memmap sidecar indicates the cache is valid. |
| `verbose` | `True` | Print build, cache, progress, and dropped-column messages. |

The builder also writes a sidecar metadata file named
`<memmap_path>.meta.json`. The cache is reused when row counts, columns, dtype,
and source file signatures match.

## Pareto Front Helpers

`fit` and `fit_memmap` return `ParetoFront`.

| Method | Arguments | Explanation |
| --- | --- | --- |
| `front.to_frame()` | none | Returns a `pandas.DataFrame` with `equation`, `complexity`, and the selected metric column sorted by complexity. |
| `front.elbow()` | none | Returns the point with the largest score drop per unit complexity increase. |
| `front.dominance_filter()` | none | Returns a new front containing only non-dominated points. |
| `len(front)` | none | Returns the number of points in the front. |

Each point is a `ParetoPoint` with:

| Attribute | Explanation |
| --- | --- |
| `equation` | Human-readable equation string. Requires SymPy conversion to succeed. |
| `sympy_expr` | SymPy expression object for the equation. |
| `complexity` | Token count of the sampled prefix expression before affine wrapping. |
| `mse` | Backward-compatible score value field. It contains exact MSE when `score_metric="mse"` and the selected metric value otherwise. |
| `score` | Alias for the score value used for Pareto dominance. |
| `score_metric` | Metric used to compute the score: `"mse"`, `"rmse"`, or `"mae"`. |
