# Full Pipeline CLI Reference

`nsr-engine` is a Python library, but it ships a runnable end-to-end pipeline
that can be driven entirely from the command line. This document lists every
CLI input, its default, and the available options it accepts.

For installation and a first run, see the [Quickstart](quickstart.md). For the
`NSREngine(...)` Python constructor arguments, see the
[library reference in the Quickstart](quickstart.md#nsrengine-arguments).

## Entry points

The same CLI is available three ways:

```bash
python main.py            # from the repository root
python -m nsr_engine.main # from the installed package
nsr-engine                # console script after `pip install`
```

Show all inputs with their argparse help:

```bash
python main.py --help
```

## Common usage

Run with synthetic data and the default train/test split (`0.8 / 0.2`):

```bash
python main.py
```

Add a validation split (train/test/validation fractions must sum to `1.0`):

```bash
python main.py --train-frac 0.7 --test-frac 0.2 --validation-frac 0.1
```

Fast smoke run with small settings:

```bash
python main.py --iters 20 --lambdas 3 --rows 1000
```

Run on your own CSV by naming the target column. When `--feature-cols` is
omitted, every non-target column is used as a feature:

```bash
python main.py --input-csv data.csv --target-col y --feature-cols a,b,c
```

Enable extra unary operators and pick a different accuracy metric:

```bash
python main.py --unary-ops square,abs,log,sqrt,sin,cos,tanh --metric r2
```

## Data arguments

| Argument | Default | Available options | Explanation |
| --- | --- | --- | --- |
| `--input-csv` | `None` | Path to a `.csv` file | Load a CSV instead of generating synthetic data. |
| `--target-col` | `None` | Any column name in the CSV | Target column. Required when `--input-csv` is used. |
| `--feature-cols` | `None` | Comma-separated column names | Feature columns. Defaults to all CSV columns except the target. |
| `--rows` | `3000` | Any positive integer | Synthetic row count when `--input-csv` is not used. |
| `--seed` | `7` | Any integer | Synthetic data seed and base split seed. |

## Split arguments

| Argument | Default | Available options | Explanation |
| --- | --- | --- | --- |
| `--train-frac` | `0.8` | Float in `(0, 1)` | Fraction of rows used for `fit`. |
| `--test-frac` | `0.2` | Float in `(0, 1)` | Fraction of rows used for final held-out evaluation. |
| `--validation-frac` | `None` | Float in `(0, 1)` or unset | Optional validation fraction. When set, train/test/validation fractions must sum to `1.0`. |

## Engine arguments

| Argument | Default | Available options | Explanation |
| --- | --- | --- | --- |
| `--lambda-grid` | `None` | Comma-separated floats | Explicit lambda values. Overrides `--lambdas`, `--lambda-min`, and `--lambda-max`. |
| `--lambdas`, `--n-lambda` | `10` | Any positive integer | Number of lambda values to generate. |
| `--lambda-min` | `1e-4` | Any positive float | Lower bound for the generated log-spaced lambda grid. |
| `--lambda-max` | `1e-1` | Any positive float | Upper bound for the generated log-spaced lambda grid. |
| `--iters`, `--n-iters` | `200` | Any positive integer | REINFORCE iterations per lambda. |
| `--batch-size` | `64` | Any positive integer | Expressions sampled per iteration. |
| `--max-len` | `15` | Any positive integer | Maximum prefix token sequence length (max expression node count). |
| `--elite-frac` | `0.05` | Float in `(0, 1]` | Risk-seeking elite quantile fraction. |
| `--entropy-weight` | `0.005` | Any non-negative float | Entropy bonus weight. Higher values encourage exploration. |
| `--hidden-dim` | `128` | Any positive integer | GRU hidden state size. |
| `--embed-dim` | `32` | Any positive integer | Token embedding size. |
| `--lr` | `1e-3` | Any positive float | Adam learning rate. |
| `--random-state` | `None` | Any integer or unset | Engine random seed. Defaults to `--seed` when omitted. |
| `--cache-dir` | `None` | Path to a directory | Directory for JSON candidate caches. |
| `--cache-prefix` | `"full_pipeline"` | Any string | Cache filename prefix. |
| `--binary-ops` | `+,-,*,/` | Subset of `+ - * /` | Comma-separated binary operators. See [Operator tokens](#operator-tokens). |
| `--unary-ops` | `square,abs,log` | Subset of the [available unary ops](#operator-tokens) | Comma-separated unary operators. |
| `--const-tokens` | `-1.0,-0.5,0.5,1.0,2.0` | Comma-separated `float(...)`-parseable values | Constant terminal tokens. |
| `--device` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"`, `"mps"`, or any Torch device string | `"auto"` selects CUDA, then Apple MPS, then CPU. |
| `--step-subsample-size` | `None` | Positive integer or `none` | Rows used per training reward calculation. `none` uses all rows. |
| `--standardize` / `--no-standardize` | `True` | Flag | Enable or disable feature z-scoring. |
| `--affine-reward` / `--no-affine-reward` | `True` | Flag | Enable or disable least-squares affine scoring. |
| `--metric`, `--score-metric` | `"mse"` | `"mse"`, `"rmse"`, `"mae"`, `"mape"`, `"mbd"`, `"r2"`, `"adjusted_r2"` | Accuracy metric. See [Score metric values](#score-metric-values). |
| `--prefilter-per-complexity` | `16` | Any positive integer | Approximate-score candidates kept per complexity before exact evaluation. |

## Operator tokens

The expression grammar is built from binary operators, unary operators, and
constant terminals. `--binary-ops`, `--unary-ops`, and `--const-tokens` select
which tokens are active; anything you pass must be a recognized token.

### Binary operators

| Default | Available options |
| --- | --- |
| `+ - * /` | `+`, `-`, `*`, `/` |

The four arithmetic operators are the complete set the evaluator understands.

### Unary operators

The three defaults are unchanged. The remaining operators are available on
opt-in via `--unary-ops`; passing an unrecognized name raises an error listing
the supported set. Every operator is domain-guarded so invalid inputs produce
`NaN` (which scoring masks out) rather than crashing.

| Token | Default? | Numeric behavior |
| --- | --- | --- |
| `square` | default | `x ** 2` |
| `abs` | default | `abs(x)` |
| `log` | default | `log(abs(x) + 1e-10)` |
| `cube` | opt-in | `x ** 3` (non-finite → NaN) |
| `neg` | opt-in | `-x` |
| `sign` | opt-in | `sign(x)` |
| `sqrt` | opt-in | `sqrt(abs(x))` |
| `cbrt` | opt-in | signed cube root of `x` |
| `reciprocal` | opt-in | `1 / x` with `abs(x) < 1e-9` → NaN |
| `log10` | opt-in | `log10(abs(x) + 1e-10)` |
| `log2` | opt-in | `log2(abs(x) + 1e-10)` |
| `exp` | opt-in | `exp(x)` (overflow → NaN) |
| `sin` | opt-in | `sin(x)` |
| `cos` | opt-in | `cos(x)` |
| `tan` | opt-in | `tan(x)` (non-finite → NaN) |
| `sinh` | opt-in | `sinh(x)` (overflow → NaN) |
| `cosh` | opt-in | `cosh(x)` (overflow → NaN) |
| `tanh` | opt-in | `tanh(x)` |
| `arcsin` | opt-in | `arcsin(clip(x, -1, 1))` |
| `arccos` | opt-in | `arccos(clip(x, -1, 1))` |
| `arctan` | opt-in | `arctan(x)` |
| `arcsinh` | opt-in | `arcsinh(x)` |
| `arctanh` | opt-in | `arctanh(clip(x, -1 + 1e-7, 1 - 1e-7))` |
| `sigmoid` | opt-in | `1 / (1 + exp(-clip(x, -50, 50)))` |

The same menu is available to the Python API through
`NSREngine(unary_ops=[...])`.

### Constant terminals

| Default | Available options |
| --- | --- |
| `-1.0 -0.5 0.5 1.0 2.0` | Any comma-separated values parseable by `float(...)`, e.g. `--const-tokens -2.0,-1.0,1.0,2.0,3.14` |

## Score metric values

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
