# Pipeline Examples

The examples directory contains one runnable script for each distinct pipeline
case. Each script is self-contained and uses small CPU-sized defaults:

```bash
python examples/generated_train_test.py
python examples/generated_validation.py
python examples/csv_input.py
python examples/custom_metric_library.py
python examples/cached_run.py
python examples/memmap_out_of_core.py
python examples/accuracy_layers.py
```

All scripts accept the same basic sizing arguments:

```bash
--rows 600 --seed 7 --iters 12 --lambdas 2
```

Increase `--iters`, `--lambdas`, and `--rows` for stronger searches.
The full pipeline CLI uses the entire dataset by default
(`--validation-mode none`). Use `--validation-mode sequential`, `holdout`,
`k-fold`, `expanding-window`, `walk-forward`, or `blocked-time-series` when
held-out validation is desired.

```bash
python main.py --validation-mode sequential --train-frac 0.8 --test-frac 0.2
python main.py --validation-mode k-fold --folds 5
python main.py --validation-mode expanding-window --folds 5
python main.py --validation-mode blocked-time-series --folds 5
```

The standalone example scripts below demonstrate fixed pipeline paths. Use the
full CLI when you want to switch validation strategies from the command line.

## Cases

| Script | Pipeline case | What it demonstrates |
|--------|---------------|----------------------|
| `examples/generated_train_test.py` | Generated data, train/test | Builds synthetic data with `make_dataset`, splits rows 80/20 in order, fits `NSREngine`, prints the Pareto front, selects the elbow formula, and reports test RMSE when SymPy is available. |
| `examples/generated_validation.py` | Generated data, train/test/validation | Uses an ordered 70/10/20 train/validation/test split and reports validation RMSE plus test RMSE for the selected formula. |
| `examples/csv_input.py` | CSV input | Creates a temporary CSV, loads it through `load_csv_dataset`, selects explicit feature columns, and evaluates the selected formula on held-out rows. |
| `examples/custom_metric_library.py` | Custom engine configuration | Uses `score_metric="mae"` with a reduced binary, unary, and constant token library. |
| `examples/cached_run.py` | Cache reuse | Runs two fits against the same temporary cache directory so the second run loads cached lambda candidates. |
| `examples/memmap_out_of_core.py` | Out-of-core training | Writes generated data to Parquet, builds a `MemmapDataset`, and trains through `fit_memmap`. This case requires `pyarrow`. |
| `examples/accuracy_layers.py` | Accuracy layers | Stacks all three layers on a two-additive-term target (`exp(x2) - 1.5*log(x4)`) and prints the elbow test R2 after each stage, so each layer's contribution is visible. This case requires the `refine` extra. |

## Dependency Notes

The in-memory examples require the core package dependencies. Formula
evaluation on held-out rows requires SymPy:

```bash
pip install "nsr-engine[sympy]"
```

The memmap example additionally requires PyArrow:

```bash
pip install "nsr-engine[memmap]"
```

The accuracy layers example additionally requires SciPy and scikit-learn:

```bash
pip install "nsr-engine[refine]"
```
