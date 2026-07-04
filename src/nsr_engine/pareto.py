from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd


_MAXIMIZE_METRICS = {"r2", "adjusted_r2"}


@dataclass
class ParetoPoint:
    equation: str
    sympy_expr: Any
    complexity: int
    mse: float
    score_metric: str = "mse"

    @property
    def score(self) -> float:
        """Accuracy score used for Pareto dominance.

        ``mse`` is retained as the backing field for compatibility with older
        callers. When ``score_metric`` is not ``"mse"``, it contains that
        metric's value.
        """
        if self.score_metric in _MAXIMIZE_METRICS:
            return -self.mse
        return self.mse


class ParetoFront:
    """Collection of (equation, complexity, score) points on a Pareto front."""

    def __init__(self, points: list[ParetoPoint]) -> None:
        self.points = points

    def dominance_filter(self) -> ParetoFront:
        """Return a new front keeping only non-dominated points.

        Point A dominates B when A.complexity <= B.complexity AND A.score <= B.score
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
                    and other.score <= pt.score
                    and (other.complexity < pt.complexity or other.score < pt.score)
                ):
                    dominated = True
                    break
            if not dominated:
                keep.append(pt)
        return ParetoFront(keep)

    def elbow(self) -> ParetoPoint:
        """Return the point with the highest score drop per complexity increase."""
        sorted_pts = sorted(self.points, key=lambda p: p.complexity)
        if len(sorted_pts) == 1:
            return sorted_pts[0]

        best_ratio = -math.inf
        best_pt = sorted_pts[-1]
        for i in range(1, len(sorted_pts)):
            delta_score = sorted_pts[i - 1].score - sorted_pts[i].score
            delta_complexity = sorted_pts[i].complexity - sorted_pts[i - 1].complexity
            if delta_complexity > 0 and delta_score > 0:
                ratio = delta_score / delta_complexity
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pt = sorted_pts[i]
        return best_pt

    def to_frame(self) -> pd.DataFrame:
        metric = self.points[0].score_metric if self.points else "mse"
        return pd.DataFrame(
            [
                {"equation": p.equation, "complexity": p.complexity, metric: p.mse}
                for p in sorted(self.points, key=lambda p: p.complexity)
            ]
        )

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        return f"ParetoFront({len(self.points)} points)"
