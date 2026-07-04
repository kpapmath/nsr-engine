# nsr-engine

Neural Symbolic Regression engine: a GRU policy trained with risk-seeking REINFORCE (Petersen et al. 2021) that discovers closed-form mathematical expressions from data.

Single-objective training is turned into a **Pareto front** by sweeping a complexity penalty λ over a log-spaced grid and pooling all discovered expressions.

## Install

```bash
pip install nsr-engine                  # core (numpy, pandas, torch)
pip install "nsr-engine[sympy]"         # + sympy for human-readable formulas
pip install "nsr-engine[memmap,sympy]"  # + pyarrow for out-of-core fit_memmap
```

From source:

```bash
git clone https://github.com/kpapmath/nsr-engine
cd nsr-engine
pip install -e ".[sympy,memmap,dev]"
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
    score_metric="mse",  # "mse", "rmse", or "mae"
    random_state=42,
)
front = engine.fit(X, y)

print(front.to_frame())
print("elbow formula:", front.elbow().equation)
```

## Token grammar

| Type       | Tokens                     |
|------------|----------------------------|
| Binary ops | `+ - * /`                  |
| Unary ops  | `square abs log`           |
| Constants  | `-1.0 -0.5 0.5 1.0 2.0`   |
| Variables  | column names of input `X`  |

Sequences are in prefix (Polish) notation; the arity-tracking constraint guarantees every sampled sequence is a valid, complete expression tree.

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
| `score_metric` | `"mse"` | Accuracy metric to minimize: `"mse"`, `"rmse"`, or `"mae"` |
| `cache_dir` | None | Cache lambda runs to disk (JSON) |

## Reference

Petersen et al. (2021). *Deep Symbolic Regression: Recovering Mathematical Expressions from Data via Risk-Seeking Policy Gradients*. ICLR 2021.
