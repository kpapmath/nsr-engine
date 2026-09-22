from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


_MAXIMIZE_METRICS = {"r2", "adjusted_r2"}


def _fit_metrics(expr: Any, X: pd.DataFrame, y: Any) -> tuple[int, float, float]:
    """Rows scored, RMSE and R2 of ``expr`` on ``(X, y)``.

    Non-finite predictions are dropped rather than poisoning the mean, and the
    row count says how many survived -- an expression defined on a tenth of the
    sample should not read as a good fit.
    """
    import numpy as np

    from nsr_engine._expr import eval_sympy_on

    nan = float("nan")
    try:
        pred = eval_sympy_on(expr, X)
    except Exception:
        return 0, nan, nan
    if pred is None:
        return 0, nan, nan

    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(y)
    n = int(mask.sum())
    if n < 2:
        return n, nan, nan

    resid = y[mask] - pred[mask]
    rmse = float(math.sqrt(float(np.mean(resid * resid))))
    centered = y[mask] - float(np.mean(y[mask]))
    ss_tot = float(np.dot(centered, centered))
    ss_res = float(np.dot(resid, resid))
    if ss_tot == 0.0:
        r2 = 1.0 if ss_res == 0.0 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    return n, rmse, r2


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
    """Collection of (equation, complexity, score) points on a Pareto front.

    Parameters
    ----------
    points:
        The front's points.
    empty_reason:
        New in 0.9.0.  Optional diagnostic explaining *why* a front came back
        empty, quoted by :meth:`elbow` so the caller reads the cause rather
        than a bare "no candidate was accepted".  Producers that know the
        reason -- :class:`~nsr_engine.boosting.ResidualBoostedNSR` records the
        rejection of every round in ``rounds_`` -- pass it here.  Ignored when
        ``points`` is non-empty.
    """

    def __init__(
        self,
        points: list[ParetoPoint],
        *,
        empty_reason: str | None = None,
    ) -> None:
        self.points = points
        self.empty_reason = empty_reason

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
        return ParetoFront(keep, empty_reason=self.empty_reason)

    def elbow(self) -> ParetoPoint:
        """Return the point with the highest score drop per complexity increase.

        Raises
        ------
        ValueError
            If the front is empty.  An empty front is not a degenerate elbow but
            a failed search: no candidate was ever accepted, and there is no
            point to return.  Raising here keeps the failure at its cause
            instead of handing back ``None`` for a caller to trip over one
            attribute access later.  Callers that can legitimately see an empty
            front guard with ``len(front)`` first.
        """
        if not self.points:
            raise ValueError(
                "elbow() on an empty Pareto front; no candidate was accepted"
                + (f" ({self.empty_reason})" if self.empty_reason else "")
            )
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

    SAVE_COLUMNS = [
        "point",
        "complexity",
        "score_metric",
        "score",
        "is_elbow",
        "fit_rows",
        "fit_rmse",
        "fit_r2",
        "equation",
    ]

    def save(
        self,
        path: str | Path,
        *,
        X: pd.DataFrame | None = None,
        y: "pd.Series | Any" = None,
    ) -> Path:
        """Write the whole front to ``path`` as CSV and return that path.

        New in 0.9.0.  ``to_frame`` carries the three fields dominance is
        decided on; this carries what a reader needs *afterwards* -- which
        point :meth:`elbow` picks, and, when the data the front was fitted on
        is passed as ``X``/``y``, each point's RMSE and R2 on it.  Those two
        are the familiar scale to compare points on when ``score_metric`` is
        something else, and they are measured from the returned SymPy
        expression, so they describe the formula the caller actually gets.

        ``fit_rmse`` and ``fit_r2`` are in-sample and say nothing about
        generalization: a front is a menu of accuracy/complexity trade-offs,
        and choosing between its points on in-sample error alone picks the
        most complex one every time.  Score the candidates on held-out data
        for that.

        Rows are ordered by complexity, as ``to_frame`` orders them.  Points
        that cannot be evaluated get ``nan`` metrics and ``fit_rows`` of 0
        rather than being dropped -- the file always accounts for the whole
        front.  An empty front writes a header and no rows.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_save_frame(X=X, y=y).to_csv(path, index=False)
        return path

    def to_save_frame(
        self,
        *,
        X: pd.DataFrame | None = None,
        y: "pd.Series | Any" = None,
    ) -> pd.DataFrame:
        """The frame :meth:`save` writes.  New in 0.9.0."""
        import numpy as np

        elbow_eq: str | None = None
        if self.points:
            try:
                elbow_eq = self.elbow().equation
            except Exception:      # a front that cannot pick one is still savable
                elbow_eq = None

        y_arr = None
        if X is not None and y is not None:
            y_arr = np.asarray(y, dtype=np.float64)

        rows = []
        for i, p in enumerate(sorted(self.points, key=lambda pt: pt.complexity)):
            fit_rows, fit_rmse, fit_r2 = 0, float("nan"), float("nan")
            if y_arr is not None:
                fit_rows, fit_rmse, fit_r2 = _fit_metrics(p.sympy_expr, X, y_arr)
            rows.append({
                "point": i,
                "complexity": p.complexity,
                "score_metric": p.score_metric,
                "score": p.mse,
                "is_elbow": int(elbow_eq is not None and p.equation == elbow_eq),
                "fit_rows": fit_rows,
                "fit_rmse": fit_rmse,
                "fit_r2": fit_r2,
                "equation": p.equation,
            })
        return pd.DataFrame(rows, columns=self.SAVE_COLUMNS)

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        return f"ParetoFront({len(self.points)} points)"
