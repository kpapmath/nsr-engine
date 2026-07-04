from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from nsr_engine.pareto import ParetoFront


@runtime_checkable
class SREngine(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> ParetoFront: ...
