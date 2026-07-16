# nsr-engine Quickstart

`nsr-engine` is a Python library for neural symbolic regression. It does not
require a command-line workflow, but it includes one for running the full
pipeline end to end. This guide documents the install commands, CLI usage, and
every public argument exposed by the library.

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

Install with the accuracy layers (adds `scipy` and `scikit-learn`):

```bash
pip install "nsr-engine[refine]"
```

Install from source for development:

```bash
git clone https://github.com/kpapmath/nsr-engine
cd nsr-engine
pip install -e ".[sympy,memmap,refine,dev]"
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

## Full Pipeline Example

A complete runnable pipeline is available from the repository root as
`main.py`, from the package as `python -m nsr_engine.main`, and after
installation as the `nsr-engine` command. It can generate synthetic data or
load a CSV, optionally run holdout, K-fold, or time-series validation, train the
lambda sweep, print the Pareto front, and select the elbow formula.

```bash
python main.py
```

The example wrapper still works:

```bash
python examples/full_pipeline.py
```

After installation, use:

```bash
nsr-engine
```

By default the pipeline uses `--validation-mode none`, which fits the final
Pareto front on the entire dataset. Add `--validation-mode sequential` for a
single chronological train/test split:

```bash
python main.py --validation-mode sequential --train-frac 0.8 --test-frac 0.2
```

Add `--validation-frac` with `sequential` or `holdout` to use
train/validation/test splits:

```bash
python main.py --validation-mode sequential --train-frac 0.7 --validation-frac 0.1 --test-frac 0.2
```

K-fold, expanding-window, walk-forward, and blocked time-series validation are
also available:

```bash
python main.py --validation-mode k-fold --folds 5
python main.py --validation-mode expanding-window --folds 5
python main.py --validation-mode walk-forward --folds 5
python main.py --validation-mode blocked-time-series --folds 5
```

Use smaller settings for a fast smoke run:

```bash
python main.py --iters 20 --lambdas 3 --rows 1000
```

Run on CSV data by naming the target column. If `--feature-cols` is omitted,
all non-target columns are used as features.

```bash
python main.py --input-csv data.csv --target-col y --feature-cols a,b,c
```

Show all CLI inputs:

```bash
python main.py --help
```

## Full Pipeline CLI Arguments

The CLI exposes data, split, engine, and accuracy-layer options.

| Argument | Default | Explanation |
| --- | --- | --- |
| `--input-csv` | `None` | Optional CSV file to load instead of generating synthetic data. |
| `--target-col` | `None` | Target column for `--input-csv`. Required when `--input-csv` is used. |
| `--feature-cols` | `None` | Comma-separated feature columns. Defaults to all CSV columns except the target. |
| `--rows` | `3000` | Synthetic row count when `--input-csv` is not used. |
| `--seed` | `7` | Synthetic data seed and base split seed. |
| `--validation-mode` | `"none"` | Validation strategy: `none`, `sequential`, `holdout`, `k-fold`, `expanding-window`, `walk-forward`, or `blocked-time-series`. |
| `--train-frac` | `0.8` | Fraction of rows used for `fit` in `sequential` or `holdout` mode. |
| `--test-frac` | `0.2` | Fraction of rows used for held-out test evaluation in `sequential` or `holdout` mode. |
| `--validation-frac` | `None` | Optional validation fraction for `sequential` or `holdout` mode. When set, train/test/validation fractions must sum to `1.0`. |
| `--folds` | `5` | Number of folds for `k-fold`, `expanding-window`, `walk-forward`, or `blocked-time-series` validation. |
| `--shuffle` / `--no-shuffle` | `False` | Shuffle rows for `holdout` and `k-fold` validation. `sequential` and time-series modes preserve row order. |
| `--lambda-grid` | `None` | Comma-separated lambda values. Overrides `--lambdas`, `--lambda-min`, and `--lambda-max`. |
| `--lambdas`, `--n-lambda` | `10` | Number of lambda values to generate. |
| `--lambda-min` | `1e-4` | Lower bound for generated lambda grid. |
| `--lambda-max` | `1e-1` | Upper bound for generated lambda grid. |
| `--iters`, `--n-iters` | `200` | REINFORCE iterations per lambda. |
| `--batch-size` | `64` | Expressions sampled per iteration. |
| `--max-len` | `15` | Maximum prefix token sequence length. |
| `--elite-frac` | `0.05` | Risk-seeking elite quantile fraction. |
| `--entropy-weight` | `0.005` | Entropy bonus weight. |
| `--hidden-dim` | `128` | GRU hidden state size. |
| `--embed-dim` | `32` | Token embedding size. |
| `--lr` | `1e-3` | Adam learning rate. |
| `--random-state` | `None` | Engine random seed. Defaults to `--seed` when omitted. |
| `--cache-dir` | `None` | Directory for JSON candidate caches. |
| `--cache-prefix` | `"full_pipeline"` | Cache filename prefix. |
| `--binary-ops` | engine default | Comma-separated binary operators. |
| `--unary-ops` | engine default | Comma-separated unary operators. |
| `--const-tokens` | engine default | Comma-separated constant tokens. |
| `--device` | `"auto"` | Torch device string such as `"auto"`, `"cpu"`, or `"cuda"`. |
| `--step-subsample-size` | `None` | Rows used for each training reward calculation. Accepts an integer or `none`. |
| `--standardize` / `--no-standardize` | `True` | Enable or disable feature z-scoring. |
| `--affine-reward` / `--no-affine-reward` | `True` | Enable or disable least-squares affine scoring. |
| `--metric`, `--score-metric` | `"mse"` | Score metric passed to `NSREngine`. |
| `--prefilter-per-complexity` | `16` | Approximate-score candidates kept per complexity before exact evaluation. |
| `--boosting` / `--no-boosting` | `False` | Accuracy layer 1. Fit additive terms by re-running the engine on each round's residual. |
| `--boosting-max-rounds` | `3` | Hard cap on the number of additive terms. |
| `--boosting-min-gain` | `0.02` | Minimum relative training-MSE improvement for a round to be kept. |
| `--term-selection` | `"elbow"` | `elbow` or `min_mse`. Which point of each round's front becomes that round's term. |
| `--constant-opt` / `--no-constant-opt` | `False` | Accuracy layer 2. Refit float constants by least squares. |
| `--max-free-consts` | `12` | Skip constant optimization for expressions with more free constants than this. |
| `--max-nfev` | `200` | Maximum residual evaluations per least-squares solve. |
| `--joint-refit` / `--no-joint-refit` | `False` | Accuracy layer 3. Jointly re-weight and prune the boosted terms. Requires `--boosting`. |
| `--joint-refit-estimator` | `"lasso_cv"` | `lasso_cv` or `ols`. Estimator for the joint refit. |
| `--coef-rel-tol` | `1e-3` | Relative weight threshold below which a term is pruned. |
| `--fit-subsample` | `8000` | Shared by layers 2 and 3: maximum rows the constant fit and the joint refit see. `0` uses every row. |

The accuracy layers are off by default and need the `refine` extra. See the
[accuracy layers guide](accuracy_layers.md) and the
[CLI reference](cli_reference.md#accuracy-layer-arguments).

Validation modes:

| Mode | Behavior |
| --- | --- |
| `none` | Fit the final Pareto front on all rows. This is the default. |
| `sequential` | Chronological single train/test split; past rows train, immediately subsequent rows test. |
| `holdout` | Single train/test split like `sequential`, but may be randomized with `--shuffle`. |
| `k-fold` | K-fold validation; folds are ordered by default and randomized only with `--shuffle`. |
| `expanding-window` | Expanding-window time-series validation; each fold adds more historical rows to training. |
| `walk-forward` | Walk-forward mode using the same expanding training window behavior. |
| `blocked-time-series` | Contiguous non-overlapping train/validation blocks; each validation block immediately follows its training block. |

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
| `binary_ops` | `("+", "-", "*", "/")` | Binary operators available to the expression grammar. The four arithmetic operators are the full supported set. |
| `unary_ops` | `("square", "abs", "log")` | Unary operators available to the expression grammar. Extra operators (`sqrt`, `cbrt`, `exp`, `log10`, `log2`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `arcsin`, `arccos`, `arctan`, `arcsinh`, `arctanh`, `sigmoid`, `neg`, `sign`, `cube`, `reciprocal`) can be opted in. Unknown names raise a `ValueError`. See the [CLI reference](cli_reference.md#unary-operators). |
| `const_tokens` | `("-1.0", "-0.5", "0.5", "1.0", "2.0")` | Constant terminal tokens available to sampled expressions. Values are parsed with `float(...)`. |
| `device` | `"auto"` | Torch device. `"auto"` selects CUDA, then Apple MPS, then CPU. You may also pass values such as `"cpu"` or `"cuda"`. |
| `step_subsample_size` | `None` | Number of rows used for each training iteration reward calculation. For in-memory `fit`, `None` means use all rows. For `fit_memmap`, `None` is treated as `50_000`. |
| `standardize` | `True` | Whether feature columns are z-scored before training. Returned SymPy formulas are converted back to raw feature terms when possible. |
| `affine_reward` | `True` | Whether rewards and final scoring use a least-squares affine fit `b0 + b1 * expression` before applying `score_metric`. This makes scoring less sensitive to expression scale and offset. |
| `score_metric` | `"mse"` | Accuracy metric. Supported values are `"mse"`, `"rmse"`, `"mae"`, `"mape"`, `"mbd"`, `"r2"`, and `"adjusted_r2"`. |
| `prefilter_per_complexity` | `16` | Number of best approximate-score candidates to keep per complexity before exact full-set evaluation. |

### Score Metric Values

All metrics are computed on residuals after the optional affine wrapper
`b0 + b1 * expression`.

| Value | Meaning | Direction |
| --- | --- | --- |
| `"mse"` | Mean squared error. | Lower is better. |
| `"rmse"` | Root mean squared error. | Lower is better. |
| `"mae"` | Mean absolute error. | Lower is better. |
| `"mape"` | Mean absolute percentage error, reported as a percentage. Zero target denominators use a small epsilon. | Lower is better. |
| `"mbd"` | Absolute mean bias deviation. | Lower is better. |
| `"r2"` | Coefficient of determination. | Higher is better. |
| `"adjusted_r2"` | Adjusted R squared using one effective predictor, the generated expression. | Higher is better. |

For `"r2"` and `"adjusted_r2"`, `front.to_frame()` reports the actual metric
value. Pareto dominance still works correctly by maximizing those metrics
internally.

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
| `score` | Internal value used for Pareto dominance. It matches `mse` for lower-is-better metrics and is negated for `"r2"` and `"adjusted_r2"`. |

## Accuracy Layers

Three optional post-hoc passes over a front. They require the `refine` extra and
are documented in full, with measured results, in
[accuracy_layers.md](accuracy_layers.md).

```python
from nsr_engine import (
    NSREngine, ParetoFront, ParetoPoint, ResidualBoostedNSR,
    joint_refit_prune, optimize_constants,
)

def engine_factory(round_idx: int) -> NSREngine:
    return NSREngine(n_lambda=4, n_iters=150, max_len=17,
                     unary_ops=("square", "abs", "log", "exp", "sqrt"),
                     random_state=42 + round_idx)

booster = ResidualBoostedNSR(engine_factory, max_rounds=3, min_gain=0.02,
                             term_refiner=optimize_constants)
front = booster.fit(X, y)

refined = joint_refit_prune(booster.terms_, X, y)
if refined is not None:
    expr, complexity, mse = refined
    points = list(front.points)
    points.append(ParetoPoint(equation=str(expr), sympy_expr=expr,
                              complexity=complexity, mse=mse))
    front = ParetoFront(points).dominance_filter()
```

### `ResidualBoostedNSR` Arguments

Conforms to the same `fit(X, y) -> ParetoFront` contract as `NSREngine`.

| Argument | Default | Explanation |
| --- | --- | --- |
| `engine_factory` | required | `factory(round_idx) -> engine` with `engine.fit(X, y) -> ParetoFront`, called once per round with the 1-based round index. Must return a fresh engine; vary its seed with `round_idx`. |
| `max_rounds` | `3` | Hard cap on the number of additive terms. |
| `min_gain` | `0.02` | After round 1, a round is kept only if it cuts training MSE by at least this relative amount. |
| `term_refiner` | `None` | Optional `f(expr, X, residual) -> expr` hook applied to each picked term before it is subtracted. Pass `optimize_constants` to run layer 2 inside layer 1. |
| `term_selection` | `"elbow"` | `"elbow"` or `"min_mse"`. Which point of each round's front becomes that round's term. |

After `fit`:

| Attribute | Explanation |
| --- | --- |
| `rounds_` | Per-round diagnostics: `round`, `added`, `reason`, `gain`, `term`, `cum_mse`. |
| `terms_` | List of `(sympy_expr, complexity)` for each kept term. This is the input to `joint_refit_prune`. |

Front points are scored in MSE, and are non-dominated by construction.

### `optimize_constants` Arguments

```python
expr = optimize_constants(expr, X, y, max_nfev=200, seed=0)
```

| Argument | Default | Explanation |
| --- | --- | --- |
| `expr` | required | SymPy expression in raw feature terms. |
| `X` | required | Feature table whose columns are the expression's free symbols. |
| `y` | required | Target values aligned row-for-row with `X`. |
| `max_nfev` | `200` | Maximum residual evaluations for `scipy.optimize.least_squares`. |
| `seed` | `0` | Seed for the sub-sample RNG. Makes the call deterministic. |
| `max_free_consts` | `12` | Return `expr` unchanged when it has more free constants than this. Keyword-only. |
| `fit_subsample` | `8000` | Maximum rows the fit sees. `0` uses every row. Keyword-only. |

Returns a new expression with optimised constants, or `expr` unchanged if there
is nothing to optimise, the fit fails, or the result does not lower the
sub-sampled training MSE. Complexity is always preserved. Expressions with no
`Float` nodes, or with more than 12 of them, are returned unchanged.

`optimize_front(front, X, y, seed=0)` applies the same pass to every point of a
front, preserving each point's `score_metric`.

### `joint_refit_prune` Arguments

```python
refined = joint_refit_prune(terms, X, y, coef_rel_tol=1e-3, seed=0)
```

| Argument | Default | Explanation |
| --- | --- | --- |
| `terms` | required | `list[(sympy_expr, complexity)]`, normally `booster.terms_`. |
| `X`, `y` | required | Data to refit against. |
| `coef_rel_tol` | `1e-3` | Prune a term when `abs(w) <= coef_rel_tol * max(abs(w))`. |
| `seed` | `0` | Seed for the sub-sample RNG and `LassoCV`. |
| `estimator` | `"lasso_cv"` | `"lasso_cv"` for sparse weights, or `"ols"` to skip the sparsity penalty. Keyword-only. |
| `fit_subsample` | `8000` | Maximum rows the estimator and the polish see. `0` uses every row. Keyword-only. |
| `polish` | `optimize_constants` | Final polish applied to the reassembled sum, called as `polish(expr, X, y, seed=..., fit_subsample=...)`. Pass `None` to skip it. Keyword-only. |

Returns `(expr, complexity, train_mse)`, or `None` if no term survives — for
example when every term evaluates non-finite. `train_mse` is always computed on
every row, whatever `fit_subsample` is set to, so the point stays comparable to
the boosted front's points.
| `score_metric` | Metric used to compute the score: `"mse"`, `"rmse"`, `"mae"`, `"mape"`, `"mbd"`, `"r2"`, or `"adjusted_r2"`. |
