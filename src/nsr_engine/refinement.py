"""Accuracy layers 2 and 3: constant optimization and joint refit + prune.

Layer 2 (:func:`optimize_constants`) refits the floating-point constants of an
expression by least squares.  The engine's reward can only reach constants from
the fixed token set plus the affine ``b0``/``b1``, so interior weights (the
``1.5`` in ``1.5*log(x4)``) are not directly reachable and the search
substitutes operator surrogates instead.  This pass recovers them.

Layer 3 (:func:`joint_refit_prune`) takes the terms discovered by
:class:`~nsr_engine.boosting.ResidualBoostedNSR`, re-weights them jointly as a
fixed basis, drops redundant ones, and polishes the result with Layer 2.
Boosting picks each term greedily and never revisits it; this undoes that.

Both passes are *accept-if-better*: they never return something with a worse
training fit than what they were given.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from nsr_engine._expr import eval_sympy_on, require_sympy
from nsr_engine.pareto import ParetoFront, ParetoPoint

__all__ = ["optimize_constants", "optimize_front", "joint_refit_prune"]

# Defaults for the tunables below; every one is overridable per call.
# More free parameters than this is slow and overfits — leave the expression be.
_MAX_FREE_CONSTS = 12
# Optimised constants generalise; fitting on every row buys nothing but time.
_FIT_SUBSAMPLE = 8000
# Stand-in for a non-finite prediction, so an `exp` overflow cannot abort the
# solve.  Scaled by the target's magnitude at use: a fixed 1e6 would sit *on*
# top of a target that happens to be near 1e6, making an overflowing row score
# as a perfect fit rather than a rejected one.
_NONFINITE_SENTINEL = 1e6

_MAXIMIZE_METRICS = {"r2", "adjusted_r2"}


def _require_scipy() -> Any:
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "scipy is required for constant optimization (accuracy layer 2). "
            "Install with: pip install 'nsr-engine[refine]'"
        ) from exc
    return least_squares


# ---------------------------------------------------------------------------
# Layer 2 — constant optimization
# ---------------------------------------------------------------------------

def _parametrize(expr: Any, sp: Any) -> tuple[Any, list[tuple[Any, float]]]:
    """Replace every ``Float`` *occurrence* with a fresh symbol.

    ``Integer`` nodes are left alone: that keeps a ``square``'s ``**2`` exponent
    fixed, since a free fractional exponent is numerically unstable.  Replacing
    per occurrence rather than per value lets two equal literals separate.
    """
    consts: list[tuple[Any, float]] = []

    def rec(node: Any) -> Any:
        if isinstance(node, sp.Float):
            sym = sp.Symbol(f"__c{len(consts)}")
            consts.append((sym, float(node)))
            return sym
        if not node.args:
            return node
        return node.func(*[rec(arg) for arg in node.args])

    return rec(expr), consts


def optimize_constants(
    expr: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    max_nfev: int = 200,
    seed: int = 0,
    *,
    max_free_consts: int = _MAX_FREE_CONSTS,
    fit_subsample: int = _FIT_SUBSAMPLE,
) -> Any:
    """Refit the float constants of ``expr`` by least squares.

    Returns a new expression with optimised constants — or ``expr`` unchanged if
    there is nothing to optimise, the fit fails, or it does not reduce the
    (sub-sampled) training MSE.  Complexity is preserved: every constant is
    substituted back in place, so no node is added or removed.

    ``max_free_consts`` caps how many free parameters are worth fitting, and
    ``fit_subsample`` caps how many rows the fit sees.

    Deterministic: the same ``(expr, X, y, seed)`` yields the same output.
    """
    sp = require_sympy()
    if expr is None:
        return expr
    try:
        sexpr = sp.sympify(expr)
    except Exception:
        return expr

    param_expr, consts = _parametrize(sexpr, sp)
    # Guard rails: nothing to fit, or too many free parameters.
    if not consts or len(consts) > max_free_consts:
        return expr

    least_squares = _require_scipy()

    y_arr = np.asarray(
        y.to_numpy() if isinstance(y, pd.Series) else y, dtype=np.float64
    )
    if len(X) != y_arr.size or y_arr.size == 0:
        return expr

    # Sub-sample from a seeded RNG so the fit stays fast and reproducible.
    n = y_arr.size
    if fit_subsample > 0 and n > fit_subsample:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=fit_subsample, replace=False))
        X_fit = X.iloc[idx]
        y_fit = y_arr[idx]
    else:
        X_fit = X
        y_fit = y_arr

    feat_syms = [sp.Symbol(col) for col in X.columns]
    const_syms = [sym for sym, _ in consts]
    theta0 = np.array([val for _, val in consts], dtype=np.float64)

    try:
        fn = sp.lambdify(feat_syms + const_syms, param_expr, modules="numpy")
    except Exception:
        return expr

    cols = [X_fit[col].to_numpy(dtype=np.float64) for col in X_fit.columns]
    n_fit = y_fit.size
    finite_y = np.isfinite(y_fit)

    # Keep the sentinel far from the target on the target's own scale, so a
    # non-finite row always reads as a bad fit.  For an O(1) target this is the
    # plain 1e6; for a target near 1e6 a fixed 1e6 would read as a *perfect* fit.
    y_magnitude = (
        float(np.max(np.abs(y_fit[finite_y]))) if bool(finite_y.any()) else 1.0
    )
    sentinel = _NONFINITE_SENTINEL * max(1.0, y_magnitude)

    def predict(theta: np.ndarray) -> np.ndarray | None:
        with np.errstate(all="ignore"):
            try:
                raw = fn(*cols, *theta)
                pred = np.asarray(raw, dtype=np.float64)
            except Exception:
                return None
        if pred.ndim == 0:
            pred = np.full(n_fit, float(pred), dtype=np.float64)
        if pred.shape != (n_fit,):
            try:
                pred = np.broadcast_to(pred, (n_fit,)).astype(np.float64)
            except ValueError:
                return None
        return np.where(np.isfinite(pred), pred, sentinel)

    def residuals(theta: np.ndarray) -> np.ndarray:
        pred = predict(theta)
        if pred is None:
            return np.full(n_fit, sentinel, dtype=np.float64)
        resid = np.where(finite_y, pred - y_fit, 0.0)
        return np.where(np.isfinite(resid), resid, sentinel)

    def sub_mse(resid: np.ndarray) -> float:
        # A surviving-but-huge prediction can overflow when squared; that is a
        # rejection, not a failure, so let it become inf quietly.
        with np.errstate(over="ignore", invalid="ignore"):
            return float(np.mean(resid * resid))

    base_mse = sub_mse(residuals(theta0))

    try:
        with np.errstate(all="ignore"):
            result = least_squares(
                residuals,
                theta0,
                method="trf",
                loss="soft_l1",
                max_nfev=max_nfev,
            )
    except Exception:
        return expr

    new_mse = sub_mse(residuals(result.x))
    # Accept-if-better, on the same sub-sample the fit saw.
    if not np.isfinite(new_mse) or new_mse >= base_mse:
        return expr

    try:
        polished = param_expr.subs(
            {sym: sp.Float(float(val)) for sym, val in zip(const_syms, result.x)},
            simultaneous=True,
        )
    except Exception:
        return expr
    return polished


def _score_from_residual(
    resid: np.ndarray, y: np.ndarray, metric: str
) -> float | None:
    """Score ``resid = y - pred`` in the front's own metric."""
    from nsr_engine.engine import _metric_from_residuals

    try:
        value = _metric_from_residuals(resid, metric, y=y, n_params=1)
    except Exception:
        return None
    return value if np.isfinite(value) else None


def optimize_front(
    front: ParetoFront,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    max_nfev: int = 200,
    seed: int = 0,
    max_free_consts: int = _MAX_FREE_CONSTS,
    fit_subsample: int = _FIT_SUBSAMPLE,
) -> ParetoFront:
    """Apply :func:`optimize_constants` to every point of ``front``.

    Each point keeps its complexity and its ``score_metric``; the refit is kept
    only when it improves that point's score, so the returned front is never
    worse than the input.
    """
    y_arr = np.asarray(
        y.to_numpy() if isinstance(y, pd.Series) else y, dtype=np.float64
    )
    points: list[ParetoPoint] = []
    for point in front.points:
        refined = optimize_constants(
            point.sympy_expr,
            X,
            y_arr,
            max_nfev=max_nfev,
            seed=seed,
            max_free_consts=max_free_consts,
            fit_subsample=fit_subsample,
        )
        if refined is None or refined is point.sympy_expr:
            points.append(point)
            continue

        pred = eval_sympy_on(refined, X)
        if pred is None:
            points.append(point)
            continue
        mask = np.isfinite(pred) & np.isfinite(y_arr)
        if int(mask.sum()) < 2:
            points.append(point)
            continue

        score = _score_from_residual(
            y_arr[mask] - pred[mask], y_arr[mask], point.score_metric
        )
        if score is None:
            points.append(point)
            continue

        better = (
            -score < point.score
            if point.score_metric in _MAXIMIZE_METRICS
            else score < point.score
        )
        if not better:
            points.append(point)
            continue

        points.append(
            ParetoPoint(
                equation=str(refined),
                sympy_expr=refined,
                complexity=point.complexity,
                mse=score,
                score_metric=point.score_metric,
            )
        )
    return ParetoFront(points)


# ---------------------------------------------------------------------------
# Layer 3 — joint refit + LASSO prune
# ---------------------------------------------------------------------------

def joint_refit_prune(
    terms: list[tuple[Any, int]],
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    coef_rel_tol: float = 1e-3,
    seed: int = 0,
    *,
    estimator: str = "lasso_cv",
    fit_subsample: int = _FIT_SUBSAMPLE,
    polish: Callable[..., Any] | None = optimize_constants,
) -> tuple[Any, int, float] | None:
    """Re-weight boosted terms jointly, prune the redundant ones, polish.

    ``terms`` is the ``(sympy_expr, complexity)`` list a booster exposes as
    ``terms_``.  Returns ``(expr, complexity, train_mse)`` for the re-weighted,
    pruned and polished model, or ``None`` if nothing usable remains — every
    term evaluating non-finite is a ``None``, not an exception.

    Boosting scales each term once, when it is discovered, against the residual
    of that round only.  Fitting all terms at once finds the combination the
    greedy pass could not, and LASSO's sparse weights identify terms that earn
    nothing once the others are present.

    ``fit_subsample`` caps the rows the estimator and the polish see (``0`` uses
    every row).  The returned ``train_mse`` is always computed on every row, so
    it stays comparable to the boosted front's points.  ``polish`` is called as
    ``polish(expr, X, y, seed=..., fit_subsample=...)``.
    """
    sp = require_sympy()
    if estimator not in {"lasso_cv", "ols"}:
        raise ValueError("estimator must be 'lasso_cv' or 'ols'")
    if not terms:
        return None

    y_arr = np.asarray(
        y.to_numpy() if isinstance(y, pd.Series) else y, dtype=np.float64
    )

    # 1. Build the basis, dropping any term that cannot be evaluated.
    columns: list[np.ndarray] = []
    kept: list[tuple[Any, int]] = []
    for expr, complexity in terms:
        values = eval_sympy_on(expr, X)
        if values is None or not np.all(np.isfinite(values)):
            continue
        columns.append(values)
        kept.append((expr, int(complexity)))
    if not columns:
        return None

    phi = np.column_stack(columns)
    mask = np.isfinite(y_arr)
    if int(mask.sum()) < 2:
        return None
    phi_fit = phi[mask]
    y_fit = y_arr[mask]

    # 2. Sparse joint refit, on at most `fit_subsample` rows.  The weights of a
    #    handful of basis columns are well determined long before every row is
    #    used, and LassoCV over millions of rows is the expensive step here.
    #    The MSE reported below still uses every row.
    if fit_subsample > 0 and y_fit.size > fit_subsample:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(y_fit.size, size=fit_subsample, replace=False))
        phi_est = phi_fit[idx]
        y_est = y_fit[idx]
    else:
        phi_est = phi_fit
        y_est = y_fit

    try:
        from sklearn.linear_model import LassoCV, LinearRegression
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "scikit-learn is required for joint refit + prune (accuracy layer 3). "
            "Install with: pip install 'nsr-engine[refine]'"
        ) from exc

    use_lasso = estimator == "lasso_cv" and phi_est.shape[1] >= 2 and y_est.size >= 3
    model: Any = (
        LassoCV(cv=3, random_state=seed) if use_lasso else LinearRegression()
    )
    try:
        model.fit(phi_est, y_est)
    except Exception:
        return None

    weights = np.asarray(model.coef_, dtype=np.float64).ravel()
    intercept = float(model.intercept_)
    if not np.all(np.isfinite(weights)) or not np.isfinite(intercept):
        return None

    # 3. Prune terms that earn nothing next to the others.
    largest = float(np.max(np.abs(weights)))
    if largest <= 0.0:
        return None
    survivors = [i for i, w in enumerate(weights) if abs(w) > coef_rel_tol * largest]
    if not survivors:
        return None

    # 4. Reassemble, recomputing complexity on the survivors with the boosting
    #    convention (one `+` node per join) so it stays on the same axis.
    expr = sp.Float(intercept)
    for i in survivors:
        expr = expr + sp.Float(float(weights[i])) * kept[i][0]
    complexity = sum(kept[i][1] for i in survivors) + (len(survivors) - 1)

    # 5. Polish the reassembled sum (layer 2), under the same row cap.
    if polish is not None:
        polished = polish(expr, X, y_arr, seed=seed, fit_subsample=fit_subsample)
        if polished is not None:
            expr = polished

    # The reported MSE always uses every row, so this point stays comparable to
    # the boosted front's points regardless of the row cap above.
    pred = eval_sympy_on(expr, X)
    if pred is None:
        return None
    final_mask = np.isfinite(pred) & np.isfinite(y_arr)
    if int(final_mask.sum()) < 2:
        return None
    resid = y_arr[final_mask] - pred[final_mask]
    train_mse = float(np.mean(resid * resid))
    if not np.isfinite(train_mse):
        return None

    return expr, int(complexity), train_mse
