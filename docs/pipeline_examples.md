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
```

All scripts accept the same basic sizing arguments:

```bash
--rows 600 --seed 7 --iters 12 --lambdas 2
```

Increase `--iters`, `--lambdas`, and `--rows` for stronger searches.

## Cases

| Script | Pipeline case | What it demonstrates |
|--------|---------------|----------------------|
| `examples/generated_train_test.py` | Generated data, train/test | Builds synthetic data with `make_dataset`, splits 80/20, fits `NSREngine`, prints the Pareto front, selects the elbow formula, and reports test RMSE when SymPy is available. |
| `examples/generated_validation.py` | Generated data, train/test/validation | Uses a 70/20/10 split and reports validation RMSE plus test RMSE for the selected formula. |
| `examples/csv_input.py` | CSV input | Creates a temporary CSV, loads it through `load_csv_dataset`, selects explicit feature columns, and evaluates the selected formula on held-out rows. |
| `examples/custom_metric_library.py` | Custom engine configuration | Uses `score_metric="mae"` with a reduced binary, unary, and constant token library. |
| `examples/cached_run.py` | Cache reuse | Runs two fits against the same temporary cache directory so the second run loads cached lambda candidates. |
| `examples/memmap_out_of_core.py` | Out-of-core training | Writes generated data to Parquet, builds a `MemmapDataset`, and trains through `fit_memmap`. This case requires `pyarrow`. |

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
