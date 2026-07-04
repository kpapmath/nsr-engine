"""Repository entry point for the nsr-engine full pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from nsr_engine.main import main


if __name__ == "__main__":
    main()
