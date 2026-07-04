"""Compatibility wrapper for the nsr-engine full pipeline CLI.

The canonical repository-root command is:

    python main.py

This wrapper remains runnable from the repository root:

    python examples/full_pipeline.py

For all available arguments:

    python examples/full_pipeline.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from nsr_engine.main import main
from nsr_engine.pipeline import (
    evaluate_with_sympy,
    load_csv_dataset,
    make_dataset,
    rmse,
    train_test_validation_split,
    validate_split_fractions,
)

__all__ = [
    "evaluate_with_sympy",
    "load_csv_dataset",
    "make_dataset",
    "rmse",
    "train_test_validation_split",
    "validate_split_fractions",
]


if __name__ == "__main__":
    main()
