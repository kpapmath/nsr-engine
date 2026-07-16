"""Sympy evaluation helpers shared by the accuracy layers.

Deliberately self-contained: these depend on nothing but ``numpy``/``pandas``/
``sympy``, so :mod:`nsr_engine.boosting` and :mod:`nsr_engine.refinement` stay
drop-in for any engine that yields ``ParetoFront`` / ``ParetoPoint`` objects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def require_sympy() -> Any:
    """Import sympy or raise with an actionable install hint."""
    try:
        import sympy as sp
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "sympy is required for the NSR accuracy layers. "
            "Install with: pip install 'nsr-engine[refine]'"
        ) from exc
    return sp


def eval_sympy_on(expr: Any, X: pd.DataFrame) -> np.ndarray | None:
    """Evaluate ``expr`` over the columns of ``X``.

    Returns a float64 array of length ``len(X)``, or ``None`` if the expression
    cannot be lambdified or evaluated at all.  Non-finite entries are preserved
    for the caller to mask — a partially-defined expression (``log`` of a
    negative row) is still useful.
    """
    sp = require_sympy()
    n = len(X)
    symbols = [sp.Symbol(col) for col in X.columns]
    try:
        fn = sp.lambdify(symbols, expr, modules="numpy")
    except Exception:
        return None

    cols = [X[col].to_numpy(dtype=np.float64) for col in X.columns]
    with np.errstate(all="ignore"):
        try:
            raw = fn(*cols)
            pred = np.asarray(raw, dtype=np.float64)
        except Exception:
            return None

    if pred.ndim == 0:
        return np.full(n, float(pred), dtype=np.float64)
    if pred.shape != (n,):
        try:
            pred = np.broadcast_to(pred, (n,)).astype(np.float64)
        except ValueError:
            return None
    return pred


def mse_of(residual: np.ndarray) -> float:
    """Mean squared error over the finite entries of ``residual``."""
    resid = np.asarray(residual, dtype=np.float64)
    finite = resid[np.isfinite(resid)]
    if finite.size == 0:
        return float("inf")
    return float(np.mean(finite * finite))
