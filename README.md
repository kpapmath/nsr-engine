# nsr-engine

Neural Symbolic Regression engine: a GRU policy trained with risk-seeking REINFORCE (Petersen et al. 2021) that discovers closed-form mathematical expressions from data.

Single-objective training is turned into a **Pareto front** by sweeping a complexity penalty λ over a log-spaced grid and pooling all discovered expressions.

## Install

```bash
pip install nsr-engine                  # core (numpy, pandas, torch)
pip install "nsr-engine[sympy]"         # + sympy for human-readable formulas
pip install "nsr-engine[memmap,sympy]"  # + pyarrow for out-of-core fit_memmap
pip install "nsr-engine[refine]"        # + scipy/scikit-learn for the accuracy layers
```

From source:

```bash
git clone https://github.com/kpapmath/nsr-engine
cd nsr-engine
pip install -e ".[sympy,memmap,refine,dev]"
```

## Quick start

```python
import pandas as pd
import numpy as np
from nsr_engine import NSREngine

rng = np.random.default_rng(0)
n = 5000
a = rng.standard_normal(n)
b = rng.standard_normal(n)
X = pd.DataFrame({"a": a, "b": b})
y = pd.Series(0.5 * a + 0.3 * b + 0.05 * rng.standard_normal(n))

engine = NSREngine(
    n_lambda=8,       # number of lambda values to sweep
    n_iters=300,      # REINFORCE iterations per lambda
    batch_size=64,    # expressions sampled per iteration
    max_len=10,       # maximum token sequence length
    score_metric="mse",  # "mse", "rmse", "mae", "mape", "mbd", "r2", or "adjusted_r2"
    random_state=42,
)
front = engine.fit(X, y)

print(front.to_frame())
print("elbow formula:", front.elbow().equation)
```

## Accuracy layers

Three optional post-hoc passes sit on top of the engine's output. They operate on
the sympy expressions the engine emits — no change to the policy or the reward —
and each is **accept-if-better**, so enabling one can never make the training fit
worse.

| Layer | What it fixes | API |
|---|---|---|
| 1 — Residual boosting | The affine reward fits **one** expression, not a sum. Two additive terms of comparable magnitude collapse into a linear surrogate. Each round runs a fresh engine on the previous residual, so the rounds sum to `intercept + Σₖ bₖ·exprₖ`. | `ResidualBoostedNSR` |
| 2 — Constant optimization | Constants are quantized to the token set plus the affine `b0, b1`, so interior weights (the `1.5` in `1.5*log(x4)`) are unreachable. Refits every float by least squares. | `optimize_constants`, `optimize_front` |
| 3 — Joint refit + prune | Boosting scales each term once and never revisits it. Re-weights the discovered terms jointly with LASSO and drops the redundant ones. | `joint_refit_prune` |

```python
from nsr_engine import NSREngine, ResidualBoostedNSR, joint_refit_prune, optimize_constants

def engine_factory(round_idx: int) -> NSREngine:
    return NSREngine(n_lambda=4, n_iters=150, max_len=17, random_state=42 + round_idx,
                     unary_ops=("square", "abs", "log", "exp", "sqrt"))

booster = ResidualBoostedNSR(engine_factory, max_rounds=3, min_gain=0.02,
                             term_refiner=optimize_constants)   # layers 1 + 2
front = booster.fit(X, y)

refined = joint_refit_prune(booster.terms_, X, y)               # layer 3
```

On a two-additive-term target (`exp(x2) - 1.5*log(x4)`, medium noise) this moves
elbow test R² from **0.66** (plain NSR) to **0.835** (boosting) to **0.879**
(joint refit). From the CLI:

```bash
python main.py --boosting --constant-opt --joint-refit
```

Full detail, measured results across the benchmark grid, and known limitations
are in the [accuracy layers guide](docs/accuracy_layers.md). A runnable
progression is in `examples/accuracy_layers.py`.

## Full pipeline example

The repository includes a runnable end-to-end pipeline that generates data or
loads a CSV, optionally runs holdout, K-fold, or time-series validation, trains
the engine, prints the Pareto front, and selects the elbow formula.

```bash
python main.py
```

The example wrapper and package module entry point are equivalent:

```bash
python examples/full_pipeline.py
python -m nsr_engine.main
```

By default the example uses the entire dataset for the final fit
(`--validation-mode none`). Add `--validation-mode sequential` for a single
chronological train/test split:

```bash
python main.py --validation-mode sequential --train-frac 0.8 --test-frac 0.2
```

K-fold, expanding-window, walk-forward, and blocked time-series validation are
also available:

```bash
python main.py --validation-mode k-fold --folds 5
python main.py --validation-mode expanding-window --folds 5
python main.py --validation-mode walk-forward --folds 5
python main.py --validation-mode blocked-time-series --folds 5
```

Validation modes:

| Mode | Behavior |
| --- | --- |
| `none` | Fit the final Pareto front on all rows. This is the default. |
| `sequential` | Chronological single train/test split; past rows train, immediately subsequent rows test. |
| `holdout` | Single train/test split like `sequential`, but may be randomized with `--shuffle`. |
| `k-fold` | K-fold validation; folds are ordered by default and randomized only with `--shuffle`. |
| `expanding-window` | Expanding-window time-series validation; each fold adds more historical rows to training. |
| `walk-forward` | Alias-style walk-forward mode using the same expanding training window behavior. |
| `blocked-time-series` | Contiguous non-overlapping train/validation blocks; each validation block immediately follows its training block. |

For a quicker smoke run:

```bash
python main.py --iters 20 --lambdas 3 --rows 1000
```

To run on your own CSV data, provide the target column and optionally a
comma-separated feature list:

```bash
python main.py --input-csv data.csv --target-col y --feature-cols a,b,c
```

After installation, the same CLI is available as:

```bash
nsr-engine
nsr-engine --help
```

Every command-line input, with its default and available options, is documented
in the [CLI reference](docs/cli_reference.md).

## Pipeline case examples

Functional examples for each distinct pipeline path are split into standalone
scripts. See `docs/pipeline_examples.md` for the full summary.

```bash
python examples/generated_train_test.py
python examples/generated_validation.py
python examples/csv_input.py
python examples/custom_metric_library.py
python examples/cached_run.py
python examples/memmap_out_of_core.py
```

The defaults are small CPU-sized smoke examples; increase `--iters`,
`--lambdas`, and `--rows` for stronger searches.

## Token grammar

| Type       | Tokens                     |
|------------|----------------------------|
| Binary ops | `+ - * /`                  |
| Unary ops  | `square abs log` (default) |
| Constants  | `-1.0 -0.5 0.5 1.0 2.0`   |
| Variables  | column names of input `X`  |

Sequences are in prefix (Polish) notation; the arity-tracking constraint guarantees every sampled sequence is a valid, complete expression tree.

The default unary set is `square`, `abs`, `log`. Many more are available on
opt-in via `unary_ops=[...]` (or `--unary-ops`): `sqrt`, `cbrt`, `exp`,
`log10`, `log2`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `arcsin`,
`arccos`, `arctan`, `arcsinh`, `arctanh`, `sigmoid`, `neg`, `sign`, `cube`, and
`reciprocal`. See the [CLI reference](docs/cli_reference.md#unary-operators)
for the full list and each operator's numeric behavior.

## Out-of-core training

For datasets too large to fit in RAM, build a `MemmapDataset` from Parquet files and use `fit_memmap`:

```python
from nsr_engine.memmap_store import build_memmap_dataset

store = build_memmap_dataset(
    files=list(Path("data/").glob("*.parquet")),
    feature_cols=["a", "b", "c"],
    target_col="y",
    memmap_path=Path("cache/train.mmap"),
)

front = engine.fit_memmap(store, train_lo=0, train_hi=store.n_rows)
```

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_lambda` | 10 | Lambda grid size |
| `lambda_min / lambda_max` | 1e-4 / 1e-1 | Lambda sweep range |
| `n_iters` | 200 | REINFORCE iters per lambda |
| `batch_size` | 64 | Expressions per iteration |
| `max_len` | 15 | Max token sequence length |
| `elite_frac` | 0.05 | Risk-seeking quantile ε |
| `entropy_weight` | 0.005 | Entropy bonus coefficient |
| `standardize` | True | Z-score features before training |
| `affine_reward` | True | Score residuals after a least-squares affine fit |
| `score_metric` | `"mse"` | Accuracy metric: `"mse"`, `"rmse"`, `"mae"`, `"mape"`, `"mbd"`, `"r2"`, or `"adjusted_r2"` |
| `cache_dir` | None | Cache lambda runs to disk (JSON) |

## Scoring metrics

`score_metric` controls how expressions are ranked after the optional affine
fit `b0 + b1 * expression`.

| Value | Meaning | Direction |
|-------|---------|-----------|
| `"mse"` | Mean squared error | Lower is better |
| `"rmse"` | Root mean squared error | Lower is better |
| `"mae"` | Mean absolute error | Lower is better |
| `"mape"` | Mean absolute percentage error, reported as percent | Lower is better |
| `"mbd"` | Absolute mean bias deviation | Lower is better |
| `"r2"` | Coefficient of determination | Higher is better |
| `"adjusted_r2"` | Adjusted R squared using one effective predictor | Higher is better |

For `r2` and `adjusted_r2`, `front.to_frame()` reports the actual R squared
value while Pareto dominance maximizes it internally.

## Reference

Petersen et al. (2021). *Deep Symbolic Regression: Recovering Mathematical Expressions from Data via Risk-Seeking Policy Gradients*. ICLR 2021.
