from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ParetoPoint:
    equation: str
    sympy_expr: Any
    complexity: int
    mse: float


class ParetoFront:
    """Collection of (equation, complexity, mse) points on a Pareto front."""

    def __init__(self, points: list[ParetoPoint]) -> None:
        self.points = points

    def dominance_filter(self) -> ParetoFront:
        """Return a new front keeping only non-dominated points.

        Point A dominates B when A.complexity <= B.complexity AND A.mse <= B.mse
        with strict inequality in at least one dimension.
        """
        keep: list[ParetoPoint] = []
        for pt in self.points:
            dominated = False
            for other in self.points:
                if other is pt:
                    continue
                if (
                    other.complexity <= pt.complexity
                    and other.mse <= pt.mse
                    and (other.complexity < pt.complexity or other.mse < pt.mse)
                ):
                    dominated = True
                    break
            if not dominated:
                keep.append(pt)
        return ParetoFront(keep)

    def elbow(self) -> ParetoPoint:
        """Return the point with the highest MSE-drop per unit complexity increase."""
        sorted_pts = sorted(self.points, key=lambda p: p.complexity)
        if len(sorted_pts) == 1:
            return sorted_pts[0]

        best_ratio = -math.inf
        best_pt = sorted_pts[-1]
        for i in range(1, len(sorted_pts)):
            delta_mse = sorted_pts[i - 1].mse - sorted_pts[i].mse
            delta_complexity = sorted_pts[i].complexity - sorted_pts[i - 1].complexity
            if delta_complexity > 0 and delta_mse > 0:
                ratio = delta_mse / delta_complexity
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pt = sorted_pts[i]
        return best_pt

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"equation": p.equation, "complexity": p.complexity, "mse": p.mse}
                for p in sorted(self.points, key=lambda p: p.complexity)
            ]
        )

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        return f"ParetoFront({len(self.points)} points)"
