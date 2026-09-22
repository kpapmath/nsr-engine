"""Neural Symbolic Regression engine (PyTorch RNN + REINFORCE).

DSO (deep-symbolic-optimization) is not available for Python >= 3.11, so this
module implements the equivalent approach described in Petersen et al. 2021:

    An RNN policy autoregressively samples expression-tree tokens over a
    configurable operator/feature library, trained with risk-seeking policy
    gradient (REINFORCE against the epsilon-quantile reward baseline) plus an
    entropy bonus that sustains exploration.

Single-objective -> Pareto via lambda-sweep
-------------------------------------------
Reward: R = 1/(1+normalized_score) - lambda * complexity
Sweep lambda over a log-spaced grid; one policy trained per lambda.
Pool all discovered expressions across all lambda runs, evaluate on a common
split, then apply dominance_filter() to assemble the front.

Token grammar
-------------
Binary ops  : + - * /         (arity 2)
Unary ops   : square abs log  (arity 1, defaults)
              Extra unary ops are available on opt-in (via ``unary_ops=`` /
              ``--unary-ops``) without changing the defaults; see
              ``_SUPPORTED_UNARY_OPS`` for the full menu (sqrt, exp, sin, cos,
              tan, tanh, arctan, arctanh, log10, ...).
Variables   : feature columns  (arity 0 / terminal)
Constants   : -1 -0.5 0.5 1 2 (arity 0 / terminal)

Sequences are in prefix (Polish) notation; the arity-tracking constraint
ensures every sampled sequence yields a valid, complete expression tree.

Performance design
------------------
* The whole batch of expressions is sampled in parallel: one GRU step per
  token position for all ``batch_size`` sequences, with pre-computed
  arity-mask tensors, instead of a Python loop per sequence per step.
* During training only cheap numpy rewards are computed on the (standardized,
  float32) step data.  Candidates are tracked as token tuples with their best
  subsample score; the exact full-set score and the sympy conversion happen
  once after the lambda-sweep.
* The subsample score drives the policy only — it never decides front
  membership.  Candidates are scored exactly on the full dataset *before* the
  top-K-per-complexity truncation, which makes that truncation
  *front-preserving*: a front member is exact-optimal at its complexity, so it
  ranks first in its group and cannot be evicted (see ``prefilter_metric``).
  This is distinct from *numeric determinism* (float32 eval / float64 scoring /
  fixed seeds); the two properties are separate and separately testable.
* Expression evaluation runs in float32 (halving temporary-array RAM); the
  affine least-squares scoring casts the masked prediction vector to float64
  so accumulated statistics stay accurate.
"""

from __future__ import annotations

import copy
import heapq
import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

import numpy as np
import pandas as pd

from nsr_engine.pareto import ParetoFront, ParetoPoint

if TYPE_CHECKING:
    from nsr_engine.memmap_store import MemmapDataset

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Token vocabulary constants
# ---------------------------------------------------------------------------

_BINARY_OPS: tuple[str, ...] = ("+", "-", "*", "/")
_UNARY_OPS: tuple[str, ...] = ("square", "abs", "log")
_CONST_TOKENS: tuple[str, ...] = ("-1.0", "-0.5", "0.5", "1.0", "2.0")

# Rows gathered for constant refinement on the out-of-core path. Fitting a few
# coefficients by least squares does not need millions of rows, and holding the
# whole train range in memory would defeat `fit_memmap`.
_REFIT_SUBSAMPLE_ROWS = 200_000


# ---------------------------------------------------------------------------
# Extended unary-op registry
# ---------------------------------------------------------------------------
# ``_UNARY_OPS`` above is the *default* op set the engine ships with.  The
# registries below make a broader menu of unary functions *available* so callers
# can opt in via ``unary_ops=`` / ``--unary-ops`` without any code change --
# arity, vocab ordering and the sampling masks are all derived from the active
# op tuple, so a registered token "just works" once selected.  Defaults are left
# untouched; unregistered names are rejected at construction time.
#
# Every numeric branch is NaN/inf-safe: values are computed in the array dtype
# (float32 during training), restricted-domain ops guard their input, and
# non-finite results collapse to NaN so affine scoring can mask them out.

def _finite_np(x: "np.ndarray") -> "np.ndarray":
    """Replace +/-inf with NaN so downstream scoring can mask them out."""
    return np.where(np.isfinite(x), x, np.nan)


_NUMPY_UNARY: dict[str, Any] = {
    "square": lambda a: a ** 2,
    "cube": lambda a: _finite_np(a ** 3),
    "abs": lambda a: np.abs(a),
    "neg": lambda a: -a,
    "sign": lambda a: np.sign(a),
    "sqrt": lambda a: np.sqrt(np.abs(a)),
    "cbrt": lambda a: np.cbrt(a),
    "reciprocal": lambda a: 1.0 / np.where(np.abs(a) < 1e-9, np.nan, a),
    "log": lambda a: np.log(np.abs(a) + 1e-10),
    "log10": lambda a: np.log10(np.abs(a) + 1e-10),
    "log2": lambda a: np.log2(np.abs(a) + 1e-10),
    "exp": lambda a: _finite_np(np.exp(a)),
    "sin": lambda a: np.sin(a),
    "cos": lambda a: np.cos(a),
    "tan": lambda a: _finite_np(np.tan(a)),
    "sinh": lambda a: _finite_np(np.sinh(a)),
    "cosh": lambda a: _finite_np(np.cosh(a)),
    "tanh": lambda a: np.tanh(a),
    "arcsin": lambda a: np.arcsin(np.clip(a, -1.0, 1.0)),
    "arccos": lambda a: np.arccos(np.clip(a, -1.0, 1.0)),
    "arctan": lambda a: np.arctan(a),
    "arcsinh": lambda a: np.arcsinh(a),
    "arctanh": lambda a: np.arctanh(np.clip(a, -1.0 + 1e-7, 1.0 - 1e-7)),
    "sigmoid": lambda a: 1.0 / (1.0 + np.exp(-np.clip(a, -50.0, 50.0))),
}


def _sympy_unary_map(sp: Any) -> dict[str, Any]:
    """Sympy equivalents of :data:`_NUMPY_UNARY`, for equation rendering.

    Kept key-for-key in sync with :data:`_NUMPY_UNARY` so the printed formula
    matches what was evaluated numerically (same domain guards on log/sqrt/etc).
    """
    eps = sp.Float(1e-10)
    return {
        "square": lambda a: a ** 2,
        "cube": lambda a: a ** 3,
        "abs": lambda a: sp.Abs(a),
        "neg": lambda a: -a,
        "sign": lambda a: sp.sign(a),
        "sqrt": lambda a: sp.sqrt(sp.Abs(a)),
        "cbrt": lambda a: sp.cbrt(a),
        "reciprocal": lambda a: 1 / (a + sp.Float(1e-9)),
        "log": lambda a: sp.log(sp.Abs(a) + eps),
        "log10": lambda a: sp.log(sp.Abs(a) + eps) / sp.log(sp.Integer(10)),
        "log2": lambda a: sp.log(sp.Abs(a) + eps) / sp.log(sp.Integer(2)),
        "exp": lambda a: sp.exp(a),
        "sin": lambda a: sp.sin(a),
        "cos": lambda a: sp.cos(a),
        "tan": lambda a: sp.tan(a),
        "sinh": lambda a: sp.sinh(a),
        "cosh": lambda a: sp.cosh(a),
        "tanh": lambda a: sp.tanh(a),
        "arcsin": lambda a: sp.asin(a),
        "arccos": lambda a: sp.acos(a),
        "arctan": lambda a: sp.atan(a),
        "arcsinh": lambda a: sp.asinh(a),
        "arctanh": lambda a: sp.atanh(a),
        "sigmoid": lambda a: 1 / (1 + sp.exp(-a)),
    }


# Every unary token the evaluators understand (defaults + opt-in extras).
_SUPPORTED_UNARY_OPS: tuple[str, ...] = tuple(_NUMPY_UNARY)
_CONST_VALUES: dict[str, float] = {t: float(t) for t in _CONST_TOKENS}

# Semantic sampling constraints (Track A / A3).  Keys are unary parent tokens;
# values are the child tokens forbidden *immediately beneath* them in prefix
# notation.  Only compositions that collapse to a simpler expression are listed
# (e.g. exp(log x) == x, sqrt(x^2) == |x|, abs(abs x) == abs x), so masking them
# never removes a *needed* structure -- it only stops the sampler from spending
# its budget on provably redundant subtrees.  In prefix order a unary op is
# always followed by the root of its operand, so "forbidden child" maps exactly
# to "forbidden next token after this parent".
_UNARY_CHILD_FORBIDDEN: dict[str, tuple[str, ...]] = {
    "exp": ("log", "log10", "log2"),        # exp(log x) == x
    "log": ("exp", "abs"),                   # log(exp x) == x; log already |·|
    "log10": ("exp",),
    "log2": ("exp",),
    "sqrt": ("square", "abs"),               # sqrt(x^2) == |x|; sqrt already |·|
    "square": ("sqrt", "abs"),               # square(sqrt x) == |x|; sq(|x|)==x^2
    "cube": ("cbrt",),
    "cbrt": ("cube",),
    "abs": ("abs", "square", "sqrt"),        # idempotent / already-nonneg child
    "reciprocal": ("reciprocal",),           # 1/(1/x) == x
    "neg": ("neg",),                          # -(-x) == x
    "sign": ("sign", "abs"),
}

_START_TOKEN = "<s>"  # sentinel fed at step 0
_SCORE_METRICS: tuple[str, ...] = (
    "mse",
    "rmse",
    "mae",
    "mape",
    "mbd",
    "r2",
    "adjusted_r2",
)
_METRIC_EPS = 1e-10
_PREFILTER_METRICS: tuple[str, ...] = ("exact", "approx")

# Per-column feature scaling modes, all fitted on the training rows only.
# ``"zscore"`` is the historical behaviour.  The other three exist because it is
# the *centering* that leaves a strictly positive domain: ``x - mean`` is
# negative for roughly half the rows, so every log/sqrt/pow in the library
# spends those rows in its domain guard instead of on signal.  On a positive-only
# dataset the other modes condition the columns without leaving the positive
# orthant.
_SCALE_MODES: tuple[str, ...] = ("zscore", "scale", "minmax", "log", "geometric")
# The two modes fitted in log space; ``"geometric"`` maps back out of it.
_LOG_SCALE_MODES: frozenset[str] = frozenset({"log", "geometric"})


def _get_arity(
    token: str,
    binary_ops: tuple[str, ...] = _BINARY_OPS,
    unary_ops: tuple[str, ...] = _UNARY_OPS,
) -> int:
    if token in binary_ops:
        return 2
    if token in unary_ops:
        return 1
    return 0


def _build_vocab(
    features: list[str],
    binary_ops: tuple[str, ...] = _BINARY_OPS,
    unary_ops: tuple[str, ...] = _UNARY_OPS,
    const_tokens: tuple[str, ...] = _CONST_TOKENS,
) -> tuple[list[str], dict[str, int]]:
    """Build ordered vocab: [binary_ops, unary_ops, features, constants, start]."""
    tokens = (
        list(binary_ops)
        + list(unary_ops)
        + features
        + list(const_tokens)
        + [_START_TOKEN]
    )
    token_to_id = {t: i for i, t in enumerate(tokens)}
    return tokens, token_to_id


# ---------------------------------------------------------------------------
# Numeric prefix evaluator (runs during training — no sympy overhead)
# ---------------------------------------------------------------------------

def _eval_prefix_numpy(
    tokens: list[str],
    arrays: dict[str, np.ndarray],
    n_rows: int,
) -> np.ndarray | None:
    """Evaluate a prefix-notation token list numerically on *arrays*.

    Returns a float array of shape (n_rows,) in the dtype of *arrays*
    (float32 during training, halving temporary RAM), or None on parse
    failure.  Uses safe arithmetic (NaN propagation) to avoid crashes on
    invalid inputs.
    """
    pos_ref = [0]
    dtype = next(iter(arrays.values())).dtype if arrays else np.float64

    def _rec() -> np.ndarray | None:
        if pos_ref[0] >= len(tokens):
            return None
        tok = tokens[pos_ref[0]]
        pos_ref[0] += 1

        if tok == "+":
            l, r = _rec(), _rec()
            return None if l is None or r is None else l + r
        if tok == "-":
            l, r = _rec(), _rec()
            return None if l is None or r is None else l - r
        if tok == "*":
            l, r = _rec(), _rec()
            if l is None or r is None:
                return None
            with np.errstate(over="ignore", invalid="ignore"):
                prod = l * r
                return np.where(np.isfinite(prod), prod, np.nan)
        if tok == "/":
            l, r = _rec(), _rec()
            if l is None or r is None:
                return None
            safe_r = np.where(np.abs(r) < 1e-9, np.nan, r)
            return l / safe_r
        if tok in _NUMPY_UNARY:
            a = _rec()
            if a is None:
                return None
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                return _NUMPY_UNARY[tok](a)
        if tok in arrays:
            return arrays[tok]
        if tok in _CONST_VALUES:
            return np.full(n_rows, _CONST_VALUES[tok], dtype=dtype)
        return None

    result = _rec()
    if result is None:
        return None
    if len(tokens) == 1 and tokens[0] in arrays:
        result = result.copy()
    if not np.isfinite(result).any():
        return None
    return result


# ---------------------------------------------------------------------------
# Affine-fit reward: score the best  b0 + b1 * pred  against y.
# ---------------------------------------------------------------------------

def _affine_residual(
    pred: np.ndarray, y: np.ndarray
) -> tuple[float, float, float] | None:
    """Closed-form least-squares fit of ``b0 + b1 * pred`` to ``y``.

    Returns ``(residual_mse, b0, b1)`` over the finite-aligned rows, or ``None``
    if fewer than two valid rows.  Making the reward invariant to the scale and
    offset of ``pred`` lets a feature on any scale compete on correlation alone.

    This is **linear scaling** in the sense of Keijzer (2003) and is the same
    scaling Operon (Burlacu et al., 2020) applies inside its evolutionary
    fitness -- it is not original to this method.  The residual of the optimal
    affine fit is ``mse* = Var(y)*(1 - r^2)`` with ``r`` the Pearson correlation
    between ``pred`` and ``y``, so the inverse-normalised-MSE reward reduces to a
    strictly monotone function of ``r^2``::

        R = 1 / (1 + mse*/Var(y)) = 1 / (2 - r^2).

    The reward therefore ranks candidates purely by squared correlation; scale
    and offset are recovered for free by ``(b0, b1)`` and never have to be
    discovered by the search.

    References:
        Keijzer, M. (2003). Improving symbolic regression with interval
        arithmetic and linear scaling. EuroGP 2003.
        Burlacu, Kronberger, Kommenda (2020). Operon C++. GECCO 2020 Companion.
    """
    n = pred.size
    if n < 2:
        return None
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mp = float(pred.mean())
    my = float(y.mean())
    pc = pred - mp
    yc = y - my
    var_p = float(np.dot(pc, pc))
    syy = float(np.dot(yc, yc))
    if var_p < 1e-18:
        return syy / n, my, 0.0
    cov = float(np.dot(pc, yc))
    b1 = cov / var_p
    b0 = my - b1 * mp
    resid_sse = max(syy - cov * cov / var_p, 0.0)
    return resid_sse / n, b0, b1


def _metric_from_residuals(
    resid: np.ndarray,
    metric: str,
    *,
    y: np.ndarray | None = None,
    n_params: int = 1,
) -> float:
    resid = np.asarray(resid, dtype=np.float64)
    if metric == "mse":
        return float(np.mean(resid * resid))
    if metric == "rmse":
        return float(math.sqrt(float(np.mean(resid * resid))))
    if metric == "mae":
        return float(np.mean(np.abs(resid)))
    if metric == "mbd":
        return float(abs(np.mean(-resid)))

    if y is None:
        raise ValueError(f"{metric} requires target values")
    y = np.asarray(y, dtype=np.float64)
    if metric == "mape":
        denom = np.maximum(np.abs(y), _METRIC_EPS)
        return float(np.mean(np.abs(resid) / denom) * 100.0)
    if metric in {"r2", "adjusted_r2"}:
        n = int(y.size)
        if n == 0:
            return float("nan")
        centered = y - float(np.mean(y))
        ss_tot = float(np.dot(centered, centered))
        return _r2_from_sse(
            float(np.dot(resid, resid)), ss_tot, n, metric, n_params=n_params
        )
    raise ValueError(f"unsupported score_metric={metric!r}")


def _metric_to_loss(value: float, metric: str) -> float:
    if metric in {"r2", "adjusted_r2"}:
        return max(1.0 - value, 0.0)
    return value


def _r2_from_sse(
    sse: float,
    ss_tot: float,
    n: int,
    metric: str,
    *,
    n_params: int = 1,
) -> float:
    if ss_tot <= _METRIC_EPS:
        return 1.0 if sse <= _METRIC_EPS else 0.0
    r2 = 1.0 - sse / ss_tot
    if metric == "r2":
        return float(r2)
    denom = n - n_params - 1
    if denom <= 0:
        return float(r2)
    return float(1.0 - (1.0 - r2) * (n - 1) / denom)


def _target_metric_scale(y: np.ndarray, metric: str) -> float:
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return 1.0
    centered = y - float(np.mean(y))
    if metric == "mse":
        scale = float(np.mean(centered * centered))
    elif metric == "rmse":
        scale = float(math.sqrt(float(np.mean(centered * centered))))
    elif metric == "mae":
        scale = float(np.mean(np.abs(centered)))
    elif metric == "mape":
        scale = 100.0
    elif metric == "mbd":
        scale = float(np.mean(np.abs(centered)))
    elif metric in {"r2", "adjusted_r2"}:
        scale = 1.0
    else:
        raise ValueError(f"unsupported score_metric={metric!r}")
    return max(scale, _METRIC_EPS)


# ---------------------------------------------------------------------------
# Sympy conversion (runs only for surviving candidates, never in training)
# ---------------------------------------------------------------------------

# Wall-clock bound on `sympy.simplify`, in seconds. Overridable with
# NSR_SIMPLIFY_TIMEOUT_S.
#
# `simplify` is cosmetic here: it canonicalises the expression that gets printed
# and stored, and never changes what the model predicts. But on a small fraction
# of sampled expressions it rationalises the fitted floats into fractions with
# 50-digit numerators and then tries to factor them -- `Rational._eval_power` ->
# `perfect_power` -> `factorint` -> Miller-Rabin -- and a single candidate can
# then burn half an hour. Measured on a 10-feature search with `sqrt`/`sin`,
# 20% of fits exceeded a 1800 s cap this way while the median fit took 28 s.
#
# A typical simplify on these expressions takes 0.06-0.22 s, so this bound is
# ~25x headroom: it cannot change the result of any call that was already
# completing, only cap the ones that were not.
_SIMPLIFY_TIMEOUT_S = 5.0

# Wall-clock bound on *building* a candidate's sympy expression, in seconds.
# Overridable with NSR_CONVERT_TIMEOUT_S.
#
# Construction is not the cheap half of conversion it looks like.  SymPy
# evaluates as it builds, and for a nested unary chain that evaluation is
# exponential in the nesting depth: measured on `tanh` over three scaled
# features, one extra level costs ~3.5x, running 0.4 s at depth 3, 5 s at
# depth 5 and 19 s at depth 6 -- before `simplify` is even reached.  A
# 15-token expression can nest ten deep, and the policy does sample those.
#
# This is independent of `scale_mode`; centring merely adds a constant factor
# (~2.5x for "log", ~2.2x for the default "zscore") on top of the same curve.
#
# `_SIMPLIFY_TIMEOUT_S` does not cover any of it: that bound starts after the
# expression exists.  A candidate that blows this budget is dropped, which is
# the same treatment the engine already gives one that fails to convert.
_CONVERT_TIMEOUT_S = 10.0


def _budget_seconds(env_var: str, default: float) -> float:
    try:
        return float(os.environ.get(env_var, default))
    except ValueError:
        return default


@contextmanager
def _time_budget(limit: float) -> "Iterator[bool]":
    """Run the block under a wall-clock bound, yielding whether it is enforced.

    Yields ``True`` when the bound is active, so a caller can tell a genuine
    completion from one that was never bounded in the first place.

    The bound uses ``SIGALRM``, which is only available on POSIX and only from
    the main thread.  Where it is not, the block runs unbounded rather than
    silently doing something different; the benchmark harness runs each fit in
    its own process's main thread, so it is bounded there.

    Callers never nest these -- the phases they guard run in sequence -- so a
    single interval timer is enough and no deadline stack is needed.
    """
    import signal

    if limit <= 0 or not hasattr(signal, "SIGALRM"):
        yield False
        return

    def _raise(signum, frame):  # noqa: ANN001
        raise TimeoutError("exceeded its budget")

    try:
        previous = signal.signal(signal.SIGALRM, _raise)
    except ValueError:
        # Not the main thread: signals are unavailable here.
        yield False
        return

    try:
        signal.setitimer(signal.ITIMER_REAL, limit)
        yield True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _bounded_simplify(expr: Any) -> Any:
    """``sympy.simplify(expr)``, or the expression unchanged if it takes too long.

    Returning the input on timeout is the same fallback the callers already use
    when ``simplify`` raises -- an unsimplified expression is mathematically
    identical and merely reads less tidily.

    The bound covers ``simplify`` only -- by the time it is called the
    expression already exists, so it does nothing about the cost of *building*
    one.  ``_CONVERT_TIMEOUT_S`` bounds that phase separately.
    """
    import sympy as sp

    limit = _budget_seconds("NSR_SIMPLIFY_TIMEOUT_S", _SIMPLIFY_TIMEOUT_S)
    try:
        with _time_budget(limit):
            return sp.simplify(expr)
    except TimeoutError:
        return expr


def _to_sympy_affine(
    tokens: list[str],
    b0: float,
    b1: float,
    feat_mean: dict[str, float] | None = None,
    feat_std: dict[str, float] | None = None,
    feat_mode: str | None = None,
) -> tuple[str, Any] | None:
    """Convert a prefix token list to ``b0 + b1 * expr`` in *raw* feature terms.

    If ``feat_mean``/``feat_std`` are given (feature scaling), every feature
    symbol ``f`` is substituted with ``(f - mean) / std`` so the returned
    formula is expressed against the original columns.  ``feat_mode`` names the
    scale mode so the log-space ones invert correctly: ``"log"`` substitutes
    ``(log(f) - mean) / std`` and ``"geometric"`` wraps that in ``exp``.
    """
    try:
        import sympy as sp
    except ImportError:
        return None

    pos_ref = [0]
    umap = _sympy_unary_map(sp)

    def _rec() -> Any:
        if pos_ref[0] >= len(tokens):
            return None
        tok = tokens[pos_ref[0]]
        pos_ref[0] += 1

        if tok in ("+", "-", "*", "/"):
            l, r = _rec(), _rec()
            if l is None or r is None:
                return None
            if tok == "+":
                return l + r
            if tok == "-":
                return l - r
            if tok == "*":
                return l * r
            return l / (r + sp.Float(1e-9))

        if tok in umap:
            a = _rec()
            return None if a is None else umap[tok](a)

        if tok in _CONST_VALUES:
            return sp.Float(_CONST_VALUES[tok])

        sym = sp.Symbol(tok)
        if feat_mean is not None and tok in feat_mean:
            std = feat_std[tok] if feat_std is not None else 1.0
            std = std if abs(std) > 1e-12 else 1.0
            base = sp.log(sym) if feat_mode in _LOG_SCALE_MODES else sym
            scaled = (base - sp.Float(feat_mean[tok])) / sp.Float(std)
            return sp.exp(scaled) if feat_mode == "geometric" else scaled
        return sym

    # Building the expression is where a pathological candidate actually burns
    # its time -- sympy evaluates as it constructs, exponentially in unary
    # nesting depth -- so the build gets its own bound.  Returning None on
    # timeout drops the candidate, which `_assemble_front` already handles: it
    # is the same outcome as a candidate that fails to convert at all.
    limit = _budget_seconds("NSR_CONVERT_TIMEOUT_S", _CONVERT_TIMEOUT_S)
    try:
        with _time_budget(limit):
            expr = _rec()
            if expr is None:
                return None
            final = sp.Float(b0) + sp.Float(b1) * expr
    except TimeoutError:
        print(
            f"[nsr] warning: skipped candidate whose sympy conversion exceeded "
            f"{limit:g}s: {' '.join(tokens)}",
            flush=True,
        )
        return None

    try:
        simplified = _bounded_simplify(final)
    except Exception:
        simplified = final
    # `str` walks the expression too, so it is bounded on the same grounds.
    try:
        with _time_budget(limit):
            eq_str = str(simplified)
    except Exception:
        return None
    return eq_str, simplified


def _to_sympy(tokens: list[str]) -> tuple[str, Any] | None:
    """Convert a prefix token list to a sympy expression."""
    try:
        import sympy as sp
    except ImportError:
        return None

    pos_ref = [0]
    umap = _sympy_unary_map(sp)

    def _rec() -> Any:
        if pos_ref[0] >= len(tokens):
            return None
        tok = tokens[pos_ref[0]]
        pos_ref[0] += 1

        if tok in ("+", "-", "*", "/"):
            l, r = _rec(), _rec()
            if l is None or r is None:
                return None
            if tok == "+":
                return l + r
            if tok == "-":
                return l - r
            if tok == "*":
                return l * r
            return l / (r + sp.Float(1e-9))

        if tok in umap:
            a = _rec()
            return None if a is None else umap[tok](a)

        if tok in _CONST_VALUES:
            return sp.Float(_CONST_VALUES[tok])

        return sp.Symbol(tok)

    limit = _budget_seconds("NSR_CONVERT_TIMEOUT_S", _CONVERT_TIMEOUT_S)
    try:
        with _time_budget(limit):
            expr = _rec()
    except TimeoutError:
        return None
    if expr is None:
        return None

    try:
        simplified = _bounded_simplify(expr)
    except Exception:
        simplified = expr
    try:
        with _time_budget(limit):
            eq_str = str(simplified)
    except Exception:
        return None

    return eq_str, simplified


# ---------------------------------------------------------------------------
# GRU policy
# ---------------------------------------------------------------------------

class _GRUPolicy(nn.Module):  # type: ignore[misc]
    """Single-layer GRU that emits logits over the token vocabulary step-by-step."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru_cell = nn.GRUCell(embed_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size - 1)  # -1: no start token in output
        self.hidden_dim = hidden_dim

    def step_batch(
        self,
        token_ids: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One GRU step for a whole batch. Returns (logits (B, V-1), hidden (B, H))."""
        emb = self.embed(token_ids)
        hidden = self.gru_cell(emb, hidden)
        return self.proj(hidden), hidden

    def step(
        self,
        token_id: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One GRU step for a single sequence. Returns (logits, new_hidden)."""
        logits, hidden = self.step_batch(token_id, hidden)
        return logits[0], hidden

    def init_hidden(self, device: torch.device, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)


# ---------------------------------------------------------------------------
# Token library: vocab + pre-computed arity/validity mask tensors
# ---------------------------------------------------------------------------

@dataclass
class _TokenLibrary:
    vocab: list[str]
    token_to_id: dict[str, int]
    start_id: int
    arity: "torch.Tensor"
    terminal_mask: "torch.Tensor"
    unary_mask: "torch.Tensor"
    binary_mask: "torch.Tensor"
    feature_mask: "torch.Tensor"  # (V_out,) True for feature (variable) tokens
    child_forbidden: "torch.Tensor"  # (V, V_out) True => token forbidden as child


def _make_library(
    features: list[str],
    binary_ops: tuple[str, ...],
    unary_ops: tuple[str, ...],
    const_tokens: tuple[str, ...],
    device: "torch.device",
) -> _TokenLibrary:
    vocab, token_to_id = _build_vocab(features, binary_ops, unary_ops, const_tokens)
    v_out = len(vocab) - 1
    arities = [_get_arity(vocab[i], binary_ops, unary_ops) for i in range(v_out)]
    arity = torch.tensor(arities, dtype=torch.long, device=device)

    feature_set = set(features)
    feature_mask = torch.tensor(
        [vocab[i] in feature_set for i in range(v_out)],
        dtype=torch.bool,
        device=device,
    )
    # Row per possible *previous* token (full vocab incl. start), column per
    # emittable child token.  Rows for non-unary parents (and the start token)
    # stay all-False, so the constraint only ever fires under a unary parent.
    child_forbidden = torch.zeros(len(vocab), v_out, dtype=torch.bool, device=device)
    for parent, children in _UNARY_CHILD_FORBIDDEN.items():
        p = token_to_id.get(parent)
        if p is None:
            continue
        for child in children:
            c = token_to_id.get(child)
            if c is not None and c < v_out:
                child_forbidden[p, c] = True

    return _TokenLibrary(
        vocab=vocab,
        token_to_id=token_to_id,
        start_id=token_to_id[_START_TOKEN],
        arity=arity,
        terminal_mask=(arity == 0),
        unary_mask=(arity == 1),
        binary_mask=(arity == 2),
        feature_mask=feature_mask,
        child_forbidden=child_forbidden,
    )


# ---------------------------------------------------------------------------
# Batched sequence sampling with arity-aware masking
# ---------------------------------------------------------------------------

def _allowed_mask(
    lib: _TokenLibrary,
    arity_remaining: torch.Tensor,
    prev_token_ids: torch.Tensor,
    budget: int,
    constrain: bool,
) -> torch.Tensor:
    """Boolean ``(B, V_out)`` mask of tokens emittable at the current step.

    Combines the arity/length well-formedness rules (a token may be chosen only
    if the remaining budget can still complete the tree) with the optional
    semantic grammar constraints (A3): tokens forbidden as the direct child of
    the previous unary parent are removed, unless doing so would leave a row
    with no legal token, in which case the constraint is skipped for that row.
    """
    need = arity_remaining.unsqueeze(1)
    allow = lib.terminal_mask.unsqueeze(0) | (
        lib.binary_mask.unsqueeze(0) & (budget >= need + 2)
    ) | (
        lib.unary_mask.unsqueeze(0) & (budget >= need + 1)
    )
    if constrain:
        constrained = allow & ~lib.child_forbidden[prev_token_ids]
        dead = ~constrained.any(dim=1, keepdim=True)
        allow = torch.where(dead, allow, constrained)
    return allow


def _sample_batch(
    policy: _GRUPolicy,
    lib: _TokenLibrary,
    batch_size: int,
    max_len: int,
    device: torch.device,
    constrain: bool = False,
) -> tuple[list[list[str]], torch.Tensor, torch.Tensor]:
    """Sample ``batch_size`` expressions from the policy in parallel.

    Returns ``(sequences, seq_log_probs, entropies)`` where ``seq_log_probs``
    and ``entropies`` are ``(batch_size,)`` tensors retaining the gradient
    graph for REINFORCE.  Every sequence is a complete prefix expression.
    ``constrain`` enables the A3 semantic grammar constraints.
    """
    b = batch_size
    hidden = policy.init_hidden(device, b)
    token_ids = torch.full((b,), lib.start_id, dtype=torch.long, device=device)
    arity_remaining = torch.ones(b, dtype=torch.long, device=device)
    active = torch.ones(b, dtype=torch.bool, device=device)

    seq_log_prob = torch.zeros(b, device=device)
    entropy_sum = torch.zeros(b, device=device)
    sampled_ids: list[torch.Tensor] = []
    step_active: list[torch.Tensor] = []

    for step in range(max_len):
        logits, hidden_new = policy.step_batch(token_ids, hidden)
        hidden = torch.where(active.unsqueeze(1), hidden_new, hidden)

        allow = _allowed_mask(lib, arity_remaining, token_ids, max_len - step, constrain)

        logits = logits.masked_fill(~allow, -1e9)
        log_p = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            probs = log_p.exp()
            bad = (~torch.isfinite(probs).all(dim=1)) | (probs.sum(dim=1) < 1e-30)
            if bad.any():
                probs[bad] = allow[bad].float()
            ids = torch.multinomial(probs, 1).squeeze(1)

        step_lp = log_p.gather(1, ids.unsqueeze(1)).squeeze(1)
        step_lp = torch.nan_to_num(step_lp, nan=0.0, posinf=0.0, neginf=-50.0)
        act_f = active.float()
        seq_log_prob = seq_log_prob + step_lp * act_f
        ent = -(log_p.exp() * log_p).sum(dim=1)
        entropy_sum = entropy_sum + ent * act_f

        sampled_ids.append(torch.where(active, ids, torch.zeros_like(ids)))
        step_active.append(active.clone())

        arity_remaining = arity_remaining + (lib.arity[ids] - 1) * active.long()
        token_ids = torch.where(active, ids, token_ids)
        active = active & (arity_remaining > 0)
        if not bool(active.any()):
            break

    ids_mat = torch.stack(sampled_ids, dim=1).cpu().numpy()
    act_mat = torch.stack(step_active, dim=1).cpu().numpy()
    n_steps = ids_mat.shape[1]
    sequences = [
        [lib.vocab[ids_mat[i, t]] for t in range(n_steps) if act_mat[i, t]]
        for i in range(b)
    ]
    return sequences, seq_log_prob, entropy_sum


def _sequence_log_probs(
    policy: _GRUPolicy,
    lib: _TokenLibrary,
    sequences: list[list[str]],
    max_len: int,
    device: torch.device,
    constrain: bool,
) -> torch.Tensor:
    """Teacher-forced log-prob of each given token sequence under ``policy``.

    Replays the sequences through the GRU with the *same* arity/constraint
    masking used at sampling time, so the returned ``(B,)`` log-probs are
    consistent with :func:`_sample_batch`.  Used by priority-queue training
    (A2) to raise the likelihood of the best sequences discovered so far.
    """
    b = len(sequences)
    lengths = [min(len(s), max_len) for s in sequences]
    span = max(lengths) if lengths else 0
    if span == 0:
        return torch.zeros(b, device=device)

    ids = torch.zeros(b, span, dtype=torch.long, device=device)
    active_mat = torch.zeros(b, span, dtype=torch.bool, device=device)
    for i, seq in enumerate(sequences):
        for t, tok in enumerate(seq[:span]):
            ids[i, t] = lib.token_to_id[tok]
            active_mat[i, t] = True

    hidden = policy.init_hidden(device, b)
    token_ids = torch.full((b,), lib.start_id, dtype=torch.long, device=device)
    arity_remaining = torch.ones(b, dtype=torch.long, device=device)
    logp = torch.zeros(b, device=device)

    for step in range(span):
        logits, hidden = policy.step_batch(token_ids, hidden)
        allow = _allowed_mask(lib, arity_remaining, token_ids, max_len - step, constrain)
        logits = logits.masked_fill(~allow, -1e9)
        log_p = F.log_softmax(logits, dim=-1)

        cur = ids[:, step]
        active = active_mat[:, step]
        step_lp = log_p.gather(1, cur.unsqueeze(1)).squeeze(1)
        step_lp = torch.nan_to_num(step_lp, nan=0.0, posinf=0.0, neginf=-50.0)
        logp = logp + step_lp * active.float()

        arity_remaining = arity_remaining + (lib.arity[cur] - 1) * active.long()
        token_ids = torch.where(active, cur, token_ids)

    return logp


# ---------------------------------------------------------------------------
# Candidate pool entry
# ---------------------------------------------------------------------------

@dataclass
class _OOCExpr:
    """A candidate expression held during training."""

    tokens: tuple[str, ...]
    complexity: int
    approx_mse: float


# ---------------------------------------------------------------------------
# NSR Engine
# ---------------------------------------------------------------------------

def _merge_fronts(*fronts: ParetoFront) -> ParetoFront:
    """Union fronts into one non-dominated front, de-duplicated by equation.

    Identical points do not dominate each other (dominance needs a strict
    inequality in one dimension), so the same equation reached by two fronts
    would survive twice; keyed by equation, the better-scoring copy wins.
    """
    best: dict[str, ParetoPoint] = {}
    for front in fronts:
        for pt in front.points:
            incumbent = best.get(pt.equation)
            if incumbent is None or (pt.score, pt.complexity) < (
                incumbent.score,
                incumbent.complexity,
            ):
                best[pt.equation] = pt
    return ParetoFront(list(best.values())).dominance_filter()


class _BoostingRound:
    """One boosting round: an engine that also records the front it produced.

    ``ResidualBoostedNSR`` returns one point per kept round and discards the
    rest of each round's front.  Round 1's full front is the un-boosted result,
    so it is worth keeping; this records it on the way past without widening
    the booster's own contract.
    """

    def __init__(self, engine: "NSREngine", sink: list[ParetoFront]) -> None:
        self._engine = engine
        self._sink = sink

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ParetoFront:
        front = self._engine.fit(X, y)
        self._sink.append(front)
        return front


class NSREngine:
    """Neural symbolic regression engine with PyTorch RNN + REINFORCE.

    Implements the ``SREngine`` protocol: ``fit(X, y) -> ParetoFront``.

    Parameters
    ----------
    lambda_grid:
        Explicit lambda values for the complexity-penalty sweep.  If ``None``,
        a log-spaced grid of ``n_lambda`` values in ``[lambda_min, lambda_max]``
        is used.
    n_lambda:
        Number of lambda values when auto-generating the grid.
    lambda_min, lambda_max:
        Range of the auto-generated lambda grid.
    n_iters:
        REINFORCE training iterations per lambda value.
    batch_size:
        Number of expression trees sampled per iteration.
    max_len:
        Maximum token sequence length (= maximum tree node count).
    elite_frac:
        Epsilon of the risk-seeking policy gradient: each update uses only
        samples whose reward reaches the (1-epsilon) batch quantile, with the
        quantile as baseline (Petersen et al. 2021).
    entropy_weight:
        Coefficient of the entropy bonus added to the policy loss.
    hidden_dim, embed_dim:
        GRU hidden dimension and token embedding dimension.
    lr:
        Adam learning rate.
    random_state:
        Base seed; each lambda run uses ``random_state + lambda_index``.
    cache_dir:
        Directory for caching discovered candidates (JSON per lambda).
        If ``None``, no cache is written.
    save_front:
        New in 0.9.0.  Write the front ``fit`` returns to ``front_dir`` as a
        CSV, one file per fit, named
        ``[<cache_prefix>-]front-<timestamp>-seed<random_state>.csv``.  **On by
        default**: a front is the result of a search that costs minutes to
        hours, and until 0.9.0 it existed only as the returned object, so a
        session that ended without saving it had to search again.  An existing
        file is never overwritten -- a name already taken gets ``-2``, ``-3``
        and so on.  Writing is best-effort: a failure (read-only directory, no
        space) prints a warning and returns the front unharmed, since losing a
        file is not a reason to lose the fit.  Pass ``False`` to keep ``fit``
        free of side effects.  See :meth:`~nsr_engine.ParetoFront.save` for the
        columns; ``fit`` passes the fitted ``X``/``y``, so ``fit_rmse`` and
        ``fit_r2`` are filled in.
    front_dir:
        New in 0.9.0.  Where ``save_front`` writes, created on demand.
        Relative paths resolve against the working directory; the default is
        ``nsr_pareto_front``.  Ignored when ``save_front=False``.
    standardize:
        Master switch for per-column feature scaling.  ``False`` trains on the
        raw columns and ``scale_mode`` is then irrelevant.
    scale_mode:
        New in 0.8.0.  *How* the columns are scaled when ``standardize=True``.  Stats are
        fitted on the training rows only and the returned SymPy formulas are
        converted back to raw feature terms, so the choice never leaks into the
        reported equation.

        * ``"zscore"`` (default) -- ``(x - mean) / std``.  The historical
          behaviour; it centers, so it does **not** preserve a positive domain.
        * ``"scale"`` -- ``x / rms``.  Scale-only: no centering, so a strictly
          positive column stays strictly positive and row-to-row *ratios* are
          preserved.  A monomial target ``c * x1**a * x2**b`` keeps its exact
          form under it, the rescaling being absorbed into ``c`` by the affine
          reward's slope.  For a zero-mean column it coincides with ``"zscore"``.
        * ``"minmax"`` -- affine onto ``minmax_range``.  Preserves positivity
          when the range's lower bound is positive, but it shifts, so ratios and
          power-law form are distorted; it is also the most outlier-sensitive
          of the four.
        * ``"geometric"`` -- the multiplicative analogue of ``"zscore"``:
          ``(x / GM) ** (1 / std_log)``, the z-score taken in log space and
          mapped back out of it.  It centers on the geometric mean and scales by
          the multiplicative spread — what a positive column spanning several
          orders of magnitude needs — and because of the final ``exp`` the
          scaled column is again strictly positive, with geometric mean 1.
          Requires strictly positive columns and raises otherwise.
        * ``"log"`` -- the same z-score left *in* log space,
          ``(log x - mean_log) / std_log``.  Best-conditioned of the five, but
          the scaled column is signed, so it gives up the positive domain that
          ``"geometric"`` keeps.  Nothing is lost by preferring ``"geometric"``
          while ``log`` is in the unary library: one ``log`` token recovers this
          column from that one.  Requires strictly positive columns.
    minmax_range:
        Target interval ``(lo, hi)`` of ``scale_mode="minmax"``.  The default
        ``(1e-3, 1.0)`` keeps the scaled column strictly positive; ``lo <= 0``
        is accepted but gives that up.
    score_metric:
        Accuracy metric. Supported values are ``"mse"``, ``"rmse"``,
        ``"mae"``, ``"mape"``, ``"mbd"``, ``"r2"``, and
        ``"adjusted_r2"``.
    prefilter_per_complexity:
        How many best-scoring candidates per complexity level survive to the
        sympy-conversion stage.
    prefilter_metric:
        Which score ranks the prefilter.  ``"exact"`` (default) scores candidates
        on the full dataset *before* truncating, so truncation is
        **front-preserving**: a front member is exact-optimal at its complexity,
        hence ranks first in its group and cannot be evicted.  ``"approx"`` is the
        pre-0.4 behaviour — it ranks by the noisy subsampled training score, which
        can drop the exact-best candidate at a complexity and silently perturb the
        front.  Keep ``"approx"`` only to reproduce or profile old runs.
    exact_prefilter_multiple:
        Cost guard for ``prefilter_metric="exact"``.  Exact scoring is a full pass
        over the data per candidate, and the raw pool typically runs ~10-15x larger
        than the kept set, so scoring all of it can dominate a large-dataset fit.
        With an int *m*, the pool is first cut to ``m * prefilter_per_complexity``
        per complexity by approx score, and only that set is scored exactly; the
        guarantee then holds *conditional on* the true front member surviving a cut
        that is ``m`` times looser than the final one.  ``None`` scores the entire
        pool and makes the guarantee unconditional, at proportionally higher cost.
    grammar_constraints:
        Enable the A3 semantic sampling constraints that forbid provably
        redundant unary compositions (``exp(log x)``, ``sqrt(x^2)``,
        ``abs(abs x)``, ...), saving the sample budget for useful structure.
    reinforce_all_valid:
        Enable the A1 dense learning signal: in addition to the risk-seeking
        elite term, reinforce every above-baseline sample against an EWMA
        reward baseline.  Without it the gradient comes from only the top
        ``elite_frac`` of the batch (~1-2 samples at small batch sizes).
    ewma_alpha:
        Smoothing factor of the EWMA reward baseline used by
        ``reinforce_all_valid`` (higher = faster-moving baseline).
    pqt_k:
        A2 priority-queue training: keep the best ``pqt_k`` unique sequences
        found so far and add a supervised MLE loss that raises their
        likelihood each step.  ``0`` disables PQT.
    pqt_weight:
        Target weight of the PQT MLE loss term relative to the policy-gradient
        loss (reached after the warmup, see ``pqt_warmup_frac``).  Defaults to
        ``0.5``: the PQT pull on the policy is *non-monotonic* in stability --
        a full weight of ``1.0`` can drive some seeds into a collapsed, worse
        basin, while ``0.0`` (no PQT) is also unstable, and ``0.5`` is the
        empirically robust middle ground that rescues collapse-prone runs
        without harming healthy ones.  (Prior releases defaulted to ``1.0``.)
    entropy_weight_start:
        Phase-2a exploration curriculum.  Initial entropy-bonus coefficient,
        linearly annealed to ``entropy_weight`` over each lambda's iterations, so
        the policy explores broadly early and commits late.  ``None`` disables
        annealing (constant ``entropy_weight``).
    pqt_warmup_frac:
        Phase-2a exploitation curriculum.  Fraction of a lambda's iterations over
        which the effective PQT weight ramps linearly from 0 up to ``pqt_weight``.
        The priority queue still *accumulates* the best sequences from iteration
        0; only its pull on the policy is delayed, which stops PQT from locking
        onto an early, suboptimal basin before exploration has covered ground.
        ``0.0`` applies full PQT weight immediately.
    entropy_floor:
        Stability control against policy collapse.  Clamps the (possibly annealed)
        entropy-bonus coefficient so it never drops below this value, keeping the
        sampler exploratory late in training.  Without a floor the risk-seeking and
        PQT terms can drive the policy to a near-deterministic, low-entropy
        distribution that over-commits to a poor basin; a floor counteracts that.
        ``None`` (default) imposes no floor, reproducing the pre-existing behaviour.
    restarts_per_lambda:
        Stability control against unlucky single runs.  Trains this many independent
        policies per lambda (each from a distinct seed) and pools their discovered
        expressions, keeping the best per token sequence.  Because a collapsed run's
        pool is unioned with the others', one bad restart no longer erases a lambda's
        contribution.  ``1`` (default) is the original single-run behaviour and the
        seed of that run is unchanged, so results are byte-for-byte reproducible.
        Cost scales linearly with this value.
    grad_clip_norm:
        Max global gradient-norm for policy updates (``clip_grad_norm_``).  Previously
        hard-coded to ``1.0``; exposed so the clip that bounds a single noisy REINFORCE
        step can be tightened for extra stability.  Defaults to ``1.0`` (unchanged).
    boosting:
        Accuracy layer 1, **on by default** since 0.7.0 (before it was opt-in
        through the :class:`~nsr_engine.boosting.ResidualBoostedNSR` wrapper).
        The affine reward fits ``b0 + b1*expr`` — one expression, not a sum —
        so additive terms of comparable magnitude collapse into a linear
        surrogate.  With boosting, each round fits a fresh policy on the
        previous rounds' residual and the rounds sum to
        ``intercept + sum_k b_k*expr_k``.  Round 1 is an ordinary fit whose
        full front is merged back into the result, so the front returned is
        never smaller than ``boosting=False`` would give; the cost is up to
        ``boosting_max_rounds`` fits instead of one.  Set ``False`` for a single
        fit — the pre-0.7 behaviour — when the budget is fixed or the target is
        known to be a single term.  ``fit_memmap`` is unaffected: the
        out-of-core path always runs a single fit.
    boosting_max_rounds:
        Hard cap on the number of additive terms.  ``1`` is equivalent to
        ``boosting=False``.
    boosting_min_gain:
        After round 1, a round is kept only if it cuts the training score by at
        least this relative amount, which stops terms being appended to noise.
        Measured in ``score_metric`` when that is ``"mse"`` or ``"rmse"``, and
        in MSE otherwise — and the two are not the same threshold
        (``gain_rmse = 1 - sqrt(1 - gain_mse)``).
    boosting_term_selection:
        ``"elbow"`` takes each round's elbow point, keeping terms compact;
        ``"min_mse"`` takes the round's most accurate point, recovering more per
        round when parsimony is not the priority.
    """

    def __init__(
        self,
        *,
        lambda_grid: tuple[float, ...] | list[float] | None = None,
        n_lambda: int = 10,
        lambda_min: float = 1e-4,
        lambda_max: float = 1e-1,
        n_iters: int = 200,
        batch_size: int = 64,
        max_len: int = 15,
        elite_frac: float = 0.05,
        entropy_weight: float = 0.005,
        hidden_dim: int = 128,
        embed_dim: int = 32,
        lr: float = 1e-3,
        random_state: int = 42,
        cache_dir: str | Path | None = None,
        cache_prefix: str | None = None,
        save_front: bool = True,
        front_dir: str | Path = "nsr_pareto_front",
        binary_ops: tuple[str, ...] | list[str] | None = None,
        unary_ops: tuple[str, ...] | list[str] | None = None,
        const_tokens: tuple[str, ...] | list[str] | None = None,
        device: str = "auto",
        step_subsample_size: int | None = None,
        standardize: bool = True,
        scale_mode: str = "zscore",
        minmax_range: tuple[float, float] = (1e-3, 1.0),
        affine_reward: bool = True,
        count_affine_wrapper: bool = False,
        score_metric: str = "mse",
        prefilter_per_complexity: int = 16,
        prefilter_metric: str = "exact",
        exact_prefilter_multiple: int | None = 8,
        grammar_constraints: bool = True,
        reinforce_all_valid: bool = True,
        ewma_alpha: float = 0.1,
        pqt_k: int = 10,
        pqt_weight: float = 0.5,
        entropy_weight_start: float | None = 0.02,
        pqt_warmup_frac: float = 0.3,
        entropy_floor: float | None = None,
        restarts_per_lambda: int = 1,
        grad_clip_norm: float = 1.0,
        lambda_relative: bool = True,
        refine_constants: bool = True,
        refine_max_nfev: int = 200,
        boosting: bool = True,
        boosting_max_rounds: int = 3,
        boosting_min_gain: float = 0.02,
        boosting_term_selection: str = "elbow",
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch is required for NSREngine.  Install with: pip install torch"
            )
        if not 0.0 < ewma_alpha <= 1.0:
            raise ValueError(f"ewma_alpha must be in (0, 1] (got {ewma_alpha})")
        if pqt_k < 0:
            raise ValueError(f"pqt_k must be >= 0 (got {pqt_k})")
        if pqt_weight < 0.0:
            raise ValueError(f"pqt_weight must be >= 0 (got {pqt_weight})")
        if entropy_weight_start is not None and entropy_weight_start < 0.0:
            raise ValueError(
                f"entropy_weight_start must be >= 0 or None (got {entropy_weight_start})"
            )
        if not 0.0 <= pqt_warmup_frac <= 1.0:
            raise ValueError(f"pqt_warmup_frac must be in [0, 1] (got {pqt_warmup_frac})")
        if entropy_floor is not None and entropy_floor < 0.0:
            raise ValueError(f"entropy_floor must be >= 0 or None (got {entropy_floor})")
        if restarts_per_lambda < 1:
            raise ValueError(f"restarts_per_lambda must be >= 1 (got {restarts_per_lambda})")
        if grad_clip_norm <= 0.0:
            raise ValueError(f"grad_clip_norm must be > 0 (got {grad_clip_norm})")
        if boosting_max_rounds < 1:
            raise ValueError(
                f"boosting_max_rounds must be >= 1 (got {boosting_max_rounds})"
            )
        # Checked here rather than at fit time: the booster would otherwise only
        # reject it after round 1 has already been trained.
        from nsr_engine.boosting import _TERM_SELECTIONS

        if boosting_term_selection not in _TERM_SELECTIONS:
            supported = ", ".join(repr(sel) for sel in _TERM_SELECTIONS)
            raise ValueError(f"boosting_term_selection must be one of: {supported}")
        score_metric = score_metric.lower()
        if score_metric not in _SCORE_METRICS:
            supported = ", ".join(repr(m) for m in _SCORE_METRICS)
            raise ValueError(f"score_metric must be one of: {supported}")
        scale_mode = scale_mode.lower()
        if scale_mode not in _SCALE_MODES:
            supported = ", ".join(repr(m) for m in _SCALE_MODES)
            raise ValueError(f"scale_mode must be one of: {supported}")
        mm_lo, mm_hi = float(minmax_range[0]), float(minmax_range[1])
        if not mm_lo < mm_hi:
            raise ValueError(
                f"minmax_range must satisfy lo < hi (got {tuple(minmax_range)})"
            )
        prefilter_metric = prefilter_metric.lower()
        if prefilter_metric not in _PREFILTER_METRICS:
            supported = ", ".join(repr(m) for m in _PREFILTER_METRICS)
            raise ValueError(f"prefilter_metric must be one of: {supported}")
        if exact_prefilter_multiple is not None and exact_prefilter_multiple < 1:
            raise ValueError(
                "exact_prefilter_multiple must be >= 1 or None "
                f"(got {exact_prefilter_multiple})"
            )
        self.lambda_grid = (
            tuple(lambda_grid)
            if lambda_grid is not None
            else tuple(np.logspace(math.log10(lambda_min), math.log10(lambda_max), n_lambda))
        )
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.max_len = max_len
        self.elite_frac = elite_frac
        self.entropy_weight = entropy_weight
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.lr = lr
        self.random_state = random_state
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_prefix = cache_prefix
        self.save_front = bool(save_front)
        self.front_dir = Path(front_dir)
        # Set by `fit` to the file it wrote, or None when nothing was written.
        self.front_path_: Path | None = None
        self.binary_ops = tuple(binary_ops) if binary_ops is not None else _BINARY_OPS
        if unary_ops is not None:
            unknown = [op for op in unary_ops if op not in _SUPPORTED_UNARY_OPS]
            if unknown:
                supported = ", ".join(_SUPPORTED_UNARY_OPS)
                raise ValueError(
                    f"unsupported unary op(s): {', '.join(unknown)}. "
                    f"Supported unary ops: {supported}"
                )
        self.unary_ops = tuple(unary_ops) if unary_ops is not None else _UNARY_OPS
        self.const_tokens = tuple(const_tokens) if const_tokens is not None else _CONST_TOKENS
        self.device_str = device
        self.step_subsample_size = step_subsample_size
        self.standardize = standardize
        self.scale_mode = scale_mode
        self.minmax_range = (mm_lo, mm_hi)
        self.affine_reward = affine_reward
        self.count_affine_wrapper = count_affine_wrapper
        self.score_metric = score_metric
        self.prefilter_per_complexity = prefilter_per_complexity
        self.prefilter_metric = prefilter_metric
        self.exact_prefilter_multiple = exact_prefilter_multiple
        self.grammar_constraints = grammar_constraints
        self.reinforce_all_valid = reinforce_all_valid
        self.ewma_alpha = ewma_alpha
        self.pqt_k = pqt_k
        self.pqt_weight = pqt_weight
        self.entropy_weight_start = entropy_weight_start
        self.pqt_warmup_frac = pqt_warmup_frac
        self.entropy_floor = entropy_floor
        self.restarts_per_lambda = restarts_per_lambda
        self.lambda_relative = lambda_relative
        self.refine_constants = refine_constants
        self.refine_max_nfev = refine_max_nfev
        self.grad_clip_norm = grad_clip_norm
        self.boosting = boosting
        self.boosting_max_rounds = boosting_max_rounds
        self.boosting_min_gain = boosting_min_gain
        self.boosting_term_selection = boosting_term_selection
        self.boost_rounds_: list[dict[str, Any]] = []
        self.boost_terms_: list[tuple[Any, int]] = []
        self._feat_mean: dict[str, float] | None = None
        self._feat_std: dict[str, float] | None = None

    def _resolve_device(self) -> torch.device:
        if self.device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device_str)

    # ------------------------------------------------------------------
    # Per-feature scaling
    # ------------------------------------------------------------------

    def _require_positive(self, col: str, values: np.ndarray) -> None:
        """Guard for the log-space modes, undefined off the positive orthant."""
        if values.size and float(values.min()) <= 0.0:
            raise ValueError(
                f"scale_mode={self.scale_mode!r} requires strictly positive feature "
                f"columns; column {col!r} contains values <= 0"
            )

    def _affine_params(
        self, s1: float, s2: float, cnt: float, vmin: float, vmax: float
    ) -> tuple[float, float]:
        """Offset and scale of one column's map ``x -> (x - offset) / scale``.

        Every mode is affine in the (for ``"log"``, log-transformed) column, so
        all four share one representation — one transform, and one inverse
        substitution when the front is converted back to raw feature terms.
        """
        if cnt <= 0:
            return 0.0, 1.0
        mean = s1 / cnt
        if self.scale_mode == "zscore" or self.scale_mode in _LOG_SCALE_MODES:
            std = math.sqrt(max(s2 / cnt - mean * mean, 0.0))
            return mean, std if std > 1e-12 else 1.0
        if self.scale_mode == "scale":
            # No centering: divide by the RMS, which is exactly the z-score
            # denominator for a zero-mean column.  Row-to-row ratios survive, so
            # a monomial target keeps its form and the rescaling is absorbed by
            # the affine reward's slope.
            rms = math.sqrt(max(s2 / cnt, 0.0))
            return 0.0, rms if rms > 1e-12 else 1.0
        # minmax: map [vmin, vmax] onto [lo, hi].
        lo, hi = self.minmax_range
        span = vmax - vmin
        scale = span / (hi - lo) if span > 1e-12 else 1.0
        return vmin - lo * scale, scale

    def _set_stats_from_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        if not self.standardize:
            return
        mean: dict[str, float] = {}
        std: dict[str, float] = {}
        for col, arr in arrays.items():
            finite = arr[np.isfinite(arr)]
            if self.scale_mode in _LOG_SCALE_MODES:
                self._require_positive(col, finite)
                finite = np.log(finite.astype(np.float64))
            if finite.size == 0:
                mean[col], std[col] = 0.0, 1.0
                continue
            mean[col], std[col] = self._affine_params(
                float(finite.sum(dtype=np.float64)),
                float(np.einsum("i,i->", finite, finite, dtype=np.float64)),
                float(finite.size),
                float(finite.min()),
                float(finite.max()),
            )
        self._feat_mean, self._feat_std = mean, std

    def _set_stats_streaming(
        self, store: "MemmapDataset", lo: int, hi: int, chunk_rows: int
    ) -> None:
        from nsr_engine.memmap_store import chunk_ranges

        if not self.standardize:
            return
        cols = list(store.feature_cols)
        ncol = len(cols)
        s1 = np.zeros(ncol)
        s2 = np.zeros(ncol)
        cnt = np.zeros(ncol)
        vmin = np.full(ncol, np.inf)
        vmax = np.full(ncol, -np.inf)
        for start, stop in chunk_ranges(lo, hi, chunk_rows):
            arrays, _ = store.gather(slice(start, stop))
            for j, c in enumerate(cols):
                a = arrays[c]
                fin = a[np.isfinite(a)]
                if self.scale_mode in _LOG_SCALE_MODES:
                    self._require_positive(c, fin)
                    fin = np.log(fin.astype(np.float64))
                if fin.size == 0:
                    continue
                s1[j] += float(fin.sum(dtype=np.float64))
                s2[j] += float(np.einsum("i,i->", fin, fin, dtype=np.float64))
                cnt[j] += fin.size
                vmin[j] = min(vmin[j], float(fin.min()))
                vmax[j] = max(vmax[j], float(fin.max()))
        mean: dict[str, float] = {}
        std: dict[str, float] = {}
        for j, c in enumerate(cols):
            m, s = self._affine_params(s1[j], s2[j], cnt[j], vmin[j], vmax[j])
            mean[c], std[c] = float(m), float(s)
        self._feat_mean, self._feat_std = mean, std

    def _standardize_arrays(
        self, arrays: dict[str, np.ndarray], *, inplace: bool = False
    ) -> dict[str, np.ndarray]:
        if not self.standardize or self._feat_mean is None:
            return arrays
        mean, std = self._feat_mean, self._feat_std
        log_scaled = self.scale_mode in _LOG_SCALE_MODES
        out: dict[str, np.ndarray] = {}
        for col, arr in arrays.items():
            if col not in mean:
                out[col] = arr
            elif log_scaled:
                # Stats were fitted on the train range; rows outside it can still
                # be non-positive.  Those become NaN and are dropped by the finite
                # masks downstream rather than aborting the fit.  ``inplace`` does
                # not apply — the log needs a fresh buffer.
                with np.errstate(invalid="ignore", divide="ignore"):
                    vals = np.log(np.where(arr > 0.0, arr, np.nan))
                vals -= mean[col]
                vals /= std[col]  # type: ignore[index]
                if self.scale_mode == "geometric":
                    np.exp(vals, out=vals)
                out[col] = vals
            elif inplace:
                arr -= mean[col]
                arr /= std[col]  # type: ignore[index]
                out[col] = arr
            else:
                out[col] = (arr - mean[col]) / std[col]  # type: ignore[index]
        return out

    def _score(self, pred: np.ndarray, y: np.ndarray) -> float | None:
        if self.affine_reward:
            res = _affine_residual(pred, y)
            if res is None:
                return None
            _, b0, b1 = res
            resid = np.asarray(y, dtype=np.float64) - (
                b0 + b1 * np.asarray(pred, dtype=np.float64)
            )
        else:
            resid = np.asarray(y, dtype=np.float64) - np.asarray(pred, dtype=np.float64)
        return _metric_from_residuals(resid, self.score_metric, y=y, n_params=1)

    @staticmethod
    def _warn_negative_columns(X: pd.DataFrame) -> None:
        neg_cols = [col for col in X.columns if (X[col] < 0).any()]
        if neg_cols:
            print(
                f"[nsr] WARNING: the following input columns contain negative values "
                f"({len(neg_cols)} of {len(X.columns)}): {neg_cols}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # In-memory fit
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ParetoFront:
        """Fit and return the Pareto front.

        Residual boosting is **on by default**.  Round 1 is an ordinary fit;
        each later round trains a fresh policy on the residual its predecessors
        leave, so the result can express a *sum* of terms — something the
        affine reward (``b0 + b1*expr``, one expression) cannot reach in one
        shot.  The returned front is round 1's own front unioned with the
        boosted sums, so it never covers less than ``boosting=False`` would;
        boosting only ever adds points, at up to ``boosting_max_rounds`` times
        the cost.

        Pass ``boosting=False`` for a single fit.  ``boosting_max_rounds=1`` is
        also routed there: one round would return the round's elbow alone,
        which is a strictly smaller front than the same search un-boosted.
        """
        if not self.boosting or self.boosting_max_rounds < 2:
            front = self._fit_single(X, y)
        else:
            front = self._fit_boosted(X, y)
        self._save_front(front, X, y)
        return front

    def _fit_boosted(self, X: pd.DataFrame, y: pd.Series) -> ParetoFront:
        """Greedy additive boosting over fresh per-round engines (layer 1)."""
        from nsr_engine.boosting import _RESIDUAL_METRICS, ResidualBoostedNSR

        # The booster measures rounds in the engine's own metric wherever it can
        # drive the acceptance rule, which as of 0.9.0 is every score metric but
        # `mbd`.  That keeps a single metric across the whole front.  Under a
        # metric the booster cannot use, the rounds are scored in MSE and a
        # front mixing the two would be meaningless — so the round-1 front is
        # not merged in that case.
        metrics_agree = self.score_metric in _RESIDUAL_METRICS
        round_fronts: list[ParetoFront] = []

        booster = ResidualBoostedNSR(
            lambda round_idx: _BoostingRound(self._round_engine(round_idx), round_fronts),
            max_rounds=self.boosting_max_rounds,
            min_gain=self.boosting_min_gain,
            term_selection=self.boosting_term_selection,
            residual_metric=self.score_metric if metrics_agree else "mse",
        )
        boosted = booster.fit(X, y)
        # Kept for callers that want the terms — `joint_refit_prune` (layer 3)
        # consumes `boost_terms_` exactly as it consumes `ResidualBoostedNSR.terms_`.
        self.boost_rounds_ = booster.rounds_
        self.boost_terms_ = booster.terms_
        for record in booster.rounds_:
            print(
                f"[nsr] boost round {record['round']}: added={record['added']} "
                f"gain={record['gain']:.4f} cum_mse={record['cum_mse']:.6f} "
                f"({record['reason']})",
                flush=True,
            )

        if not round_fronts:
            return boosted
        if not metrics_agree:
            print(
                f"[nsr] note: boosted points are scored in MSE, not "
                f"{self.score_metric!r} — returning the boosted front alone",
                flush=True,
            )
            return boosted

        # Round 1 searched `y` itself, so its front is exactly what
        # `boosting=False` would have returned.  Merging it back keeps the
        # simple, low-complexity end that the booster's one-point-per-round
        # front drops.
        merged = _merge_fronts(round_fronts[0], boosted)
        print(f"[nsr] boosted front: {len(merged)} non-dominated points")
        return merged

    def _round_engine(self, round_idx: int) -> "NSREngine":
        """A fresh engine for boosting round ``round_idx``.

        ``copy.copy`` rather than a re-built constructor call, so a parameter
        added later is carried over without another place to update.  Boosting
        is switched off — the round *is* the weak learner, and a round that
        boosted again would recurse — the seed moves so rounds do not repeat
        the same search, and the cache takes a per-round prefix so a round
        cannot read a pool discovered against a different residual.
        """
        engine = copy.copy(self)
        engine.boosting = False
        engine.random_state = self.random_state + round_idx
        # One file per `fit` the caller asked for: the round fronts are merged
        # into the front this engine is about to save, so saving them too would
        # write `boosting_max_rounds` files nobody asked for.
        engine.save_front = False
        if self.cache_prefix is not None:
            engine.cache_prefix = f"{self.cache_prefix}_round{round_idx}"
        engine.boost_rounds_ = []
        engine.boost_terms_ = []
        # Standardization stats belong to the round that computes them.
        engine._feat_mean = None
        engine._feat_std = None
        return engine

    def _fit_single(self, X: pd.DataFrame, y: pd.Series) -> ParetoFront:
        """Train the NSR policy for each lambda and return the pooled Pareto front."""
        self._warn_negative_columns(X)
        features = list(X.columns)

        arrays = {
            col: X[col].to_numpy(dtype=np.float32, copy=True) for col in features
        }
        if self.standardize:
            self._set_stats_from_arrays(arrays)
            arrays = self._standardize_arrays(arrays, inplace=True)
        y_arr = y.to_numpy(dtype=np.float64)
        n_rows = len(y_arr)

        subsample_regime_ids: np.ndarray | None = None
        if self.step_subsample_size is not None and "regime_id" in X.columns:
            subsample_regime_ids = X["regime_id"].to_numpy()

        device = self._resolve_device()
        print(f"[nsr] device={device}", flush=True)
        lib = _make_library(
            features, self.binary_ops, self.unary_ops, self.const_tokens, device
        )

        subsample = (
            self.step_subsample_size is not None
            and self.step_subsample_size < n_rows
        )
        y_score_scale_full = _target_metric_scale(y_arr[np.isfinite(y_arr)], self.score_metric)

        # Per-iteration row draw.  ``np.random.choice(arr, k, replace=False)`` is
        # O(n_rows): it permutes the entire index array to take k of them.  At
        # multi-million-row scale that costs more than the expression evaluations
        # it exists to feed -- ~390 ms per draw at 6M rows, i.e. ~70 s of a
        # 180-iteration sweep spent shuffling indices.  ``Generator.choice`` uses
        # Floyd's algorithm for the same without-replacement semantics and is
        # ~200x faster.  Seeded from ``random_state`` so the draw sequence stays
        # reproducible; ``fit_memmap`` already avoided the O(n) call.
        subsample_rng = np.random.default_rng(self.random_state)

        # Stratified draws need each cell's row indices.  Recomputing them from a
        # boolean mask inside the closure would be a further O(n_rows) pass *per
        # cell per iteration*, so they are materialised once.
        regime_cell_indices: list[np.ndarray] | None = None
        if subsample and subsample_regime_ids is not None:
            regime_cell_indices = [
                np.flatnonzero(subsample_regime_ids == cell)
                for cell in np.unique(subsample_regime_ids)
            ]

        def sample_step() -> tuple[dict[str, np.ndarray], np.ndarray, float]:
            if not subsample:
                return arrays, y_arr, y_score_scale_full
            k = self.step_subsample_size or n_rows
            if regime_cell_indices is not None:
                n_per_cell = max(1, k // len(regime_cell_indices))
                parts: list[np.ndarray] = []
                for cell_idx in regime_cell_indices:
                    take = min(n_per_cell, len(cell_idx))
                    parts.append(
                        subsample_rng.choice(cell_idx, size=take, replace=False)
                    )
                idx = np.concatenate(parts)
            else:
                idx = subsample_rng.choice(n_rows, size=k, replace=False)
            step_arrays = {col: arr[idx] for col, arr in arrays.items()}
            step_y = y_arr[idx]
            finite_y = step_y[np.isfinite(step_y)]
            return step_arrays, step_y, _target_metric_scale(finite_y, self.score_metric)

        pool = self._sweep_lambdas(lib, sample_step, device)
        if not pool:
            print("[nsr] warning: no valid expressions discovered — returning empty front")
            return ParetoFront([])

        candidates = self._select_for_exact_scoring(
            list(pool.values()), per_complexity=self.prefilter_per_complexity
        )
        print(
            f"[nsr] exact full-set {self.score_metric.upper()} for {len(candidates)} candidates "
            f"({n_rows:,} rows, prefilter_metric={self.prefilter_metric}) …",
            flush=True,
        )
        exact = self._exact_eval_arrays(candidates, arrays, y_arr)
        if self.prefilter_metric == "exact":
            exact = self._truncate_exact_per_complexity(
                exact, self.prefilter_per_complexity
            )
        return self._maybe_refine(front=self._assemble_front(exact), X=X, y=y)

    # ------------------------------------------------------------------
    # Out-of-core fit (full-set training over an on-disk memmap)
    # ------------------------------------------------------------------

    def fit_memmap(
        self,
        store: "MemmapDataset",
        *,
        train_lo: int,
        train_hi: int,
        chunk_rows: int = 5_000_000,
        prefilter_per_complexity: int | None = None,
    ) -> ParetoFront:
        """Train on rows ``[train_lo, train_hi)`` of *store* without loading them.

        Each REINFORCE iteration draws a random row subsample from the train
        range (see ``step_subsample_size``) and computes rewards on it only.
        After the lambda-sweep, surviving candidates are scored exactly on the
        full train range by streaming contiguous chunks, then dominance-filtered.

        ``step_subsample_size`` must be set; it defaults to 50_000 if None.

        The front is saved like ``fit``'s (see ``save_front``), minus the
        ``fit_rmse``/``fit_r2`` columns: the train range is never in memory, and
        measuring them on the refit subsample would put a different quantity
        under the same name.  The ``score`` column still comes from exact
        scoring over every row.
        """
        from nsr_engine.memmap_store import chunk_ranges

        if train_hi - train_lo < 2:
            raise ValueError("fit_memmap: train range too small")
        step_n = self.step_subsample_size or 50_000
        step_n = min(step_n, train_hi - train_lo)
        per_complexity = (
            prefilter_per_complexity
            if prefilter_per_complexity is not None
            else self.prefilter_per_complexity
        )

        features = list(store.feature_cols)
        device = self._resolve_device()
        print(
            f"[nsr] device={device}  mode=out-of-core  "
            f"train_rows={train_hi - train_lo:,}  step_subsample={step_n:,}  "
            f"standardize={self.standardize}  scale_mode={self.scale_mode}  "
            f"affine_reward={self.affine_reward}  "
            f"score_metric={self.score_metric}",
            flush=True,
        )
        lib = _make_library(
            features, self.binary_ops, self.unary_ops, self.const_tokens, device
        )

        if self.standardize:
            print(
                f"[nsr] computing per-feature {self.scale_mode} scaling stats "
                "over train range …",
                flush=True,
            )
            self._set_stats_streaming(store, train_lo, train_hi, chunk_rows)

        def sample_step() -> tuple[dict[str, np.ndarray], np.ndarray, float]:
            idx = np.random.randint(train_lo, train_hi, size=step_n)
            step_arrays, step_y = store.gather(idx)
            step_arrays = self._standardize_arrays(step_arrays, inplace=True)
            finite_y = step_y[np.isfinite(step_y)]
            return step_arrays, step_y, _target_metric_scale(finite_y, self.score_metric)

        pool = self._sweep_lambdas(lib, sample_step, device)
        if not pool:
            print("[nsr] warning: no valid expressions discovered — empty front")
            front = ParetoFront([])
            self._save_front(front)
            return front

        candidates = self._select_for_exact_scoring(
            list(pool.values()), per_complexity=per_complexity
        )
        print(
            f"[nsr] exact full-set {self.score_metric.upper()} for {len(candidates)} candidates "
            f"by streaming {train_hi - train_lo:,} rows in chunks of {chunk_rows:,} "
            f"(prefilter_metric={self.prefilter_metric}) …",
            flush=True,
        )
        exact = self._exact_score_streaming(
            candidates, store, train_lo, train_hi, chunk_rows
        )
        if self.prefilter_metric == "exact":
            exact = self._truncate_exact_per_complexity(exact, per_complexity)
        front = self._assemble_front(exact)

        # Constant refinement needs the data in memory. Out-of-core, that means
        # a bounded random subsample of the train range -- enough to pin a
        # handful of coefficients by least squares, without giving up the
        # streaming property the whole path exists for.
        if self.refine_constants and front.points:
            n_refit = min(_REFIT_SUBSAMPLE_ROWS, train_hi - train_lo)
            idx = np.random.default_rng(self.random_state).integers(
                train_lo, train_hi, size=n_refit
            )
            sub_arrays, sub_y = store.gather(idx)
            X_sub = pd.DataFrame(
                {col: np.asarray(sub_arrays[col], dtype=np.float64)
                 for col in store.feature_cols}
            )
            front = self._maybe_refine(X=X_sub, y=pd.Series(sub_y), front=front)
        self._save_front(front)
        return front

    # ------------------------------------------------------------------
    # Shared lambda-sweep / training core
    # ------------------------------------------------------------------

    def _sweep_lambdas(
        self,
        lib: _TokenLibrary,
        sample_step: Callable[[], tuple[dict[str, np.ndarray], np.ndarray, float]],
        device: torch.device,
    ) -> dict[tuple[str, ...], _OOCExpr]:
        pool: dict[tuple[str, ...], _OOCExpr] = {}
        for i, lam in enumerate(self.lambda_grid):
            print(f"[nsr] lambda {i + 1}/{len(self.lambda_grid)} = {lam:.4g}  …", flush=True)

            cached = self._load_cache(i)
            if cached is not None:
                print(f"[nsr]   loaded {len(cached)} candidates from cache", flush=True)
                discovered = cached
            else:
                # Restart stability: train ``restarts_per_lambda`` independent policies
                # and union their pools (best approx score per token sequence), so a
                # single collapsed run cannot erase this lambda's contribution.  The
                # first restart keeps seed ``random_state + i`` exactly, so with the
                # default ``restarts_per_lambda == 1`` results are unchanged.  Restart
                # seeds are offset by whole grid-widths to stay collision-free.
                n_lam = len(self.lambda_grid)
                discovered = {}
                for r in range(self.restarts_per_lambda):
                    if self.restarts_per_lambda > 1:
                        print(
                            f"[nsr]   restart {r + 1}/{self.restarts_per_lambda}",
                            flush=True,
                        )
                    run = self._train_one_lambda(
                        lam=lam,
                        lib=lib,
                        sample_step=sample_step,
                        device=device,
                        seed=self.random_state + i + r * n_lam,
                    )
                    for key, expr in run.items():
                        if key not in discovered or expr.approx_mse < discovered[key].approx_mse:
                            discovered[key] = expr
                print(
                    f"[nsr]   found {len(discovered)} unique valid expressions",
                    flush=True,
                )
                self._save_cache(i, lam, list(discovered.values()))

            for key, expr in discovered.items():
                if key not in pool or expr.approx_mse < pool[key].approx_mse:
                    pool[key] = expr
        print(f"[nsr] pool size: {len(pool):,} unique token sequences", flush=True)
        return pool

    def _train_one_lambda(
        self,
        *,
        lam: float,
        lib: _TokenLibrary,
        sample_step: Callable[[], tuple[dict[str, np.ndarray], np.ndarray, float]],
        device: torch.device,
        seed: int,
    ) -> dict[tuple[str, ...], _OOCExpr]:
        from nsr_engine._logging import Heartbeat

        torch.manual_seed(seed)
        np.random.seed(seed)

        policy = _GRUPolicy(len(lib.vocab), self.embed_dim, self.hidden_dim).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr)
        discovered: dict[tuple[str, ...], _OOCExpr] = {}
        hb = Heartbeat(f"nsr-train lambda={lam:.4g}", interval_s=30.0)

        # Learning is frozen when lr == 0 (the random-search control): sampling
        # still fills the candidate pool, but the whole policy-gradient/PQT block
        # is a no-op, so skip it entirely for speed.
        learning = self.lr > 0.0
        # A1 EWMA reward baseline and A2 priority queue, both reset per lambda.
        ewma_baseline: float | None = None
        pqt_heap: list[tuple[float, int, tuple[str, ...]]] = []
        pqt_seen: set[tuple[str, ...]] = set()
        pqt_counter = 0

        for iteration in range(self.n_iters):
            step_arrays, step_y, step_score_scale = sample_step()
            step_n = len(step_y)
            step_y_finite = np.isfinite(step_y)

            sequences, seq_log_probs, entropies = _sample_batch(
                policy, lib, self.batch_size, self.max_len, device,
                constrain=self.grammar_constraints,
            )

            iter_rewards: dict[tuple[str, ...], float] = {}
            rewards: list[float] = []
            for tokens in sequences:
                key = tuple(tokens)
                if key in iter_rewards:
                    rewards.append(iter_rewards[key])
                    continue

                r = -1.0
                pred = _eval_prefix_numpy(tokens, step_arrays, step_n)
                if pred is not None:
                    valid_mask = step_y_finite & np.isfinite(pred)
                    if valid_mask.sum() >= 2:
                        score_val = self._score(pred[valid_mask], step_y[valid_mask])
                        if score_val is not None:
                            score_loss = _metric_to_loss(score_val, self.score_metric)
                            normalized_score = score_loss / step_score_scale
                            penalty = lam * len(tokens)
                            if self.lambda_relative:
                                penalty /= max(1, self.max_len)
                            r = 1.0 / (1.0 + normalized_score) - penalty
                            if key not in discovered or score_loss < discovered[key].approx_mse:
                                discovered[key] = _OOCExpr(
                                    tokens=key,
                                    complexity=len(tokens),
                                    approx_mse=score_loss,
                                )
                iter_rewards[key] = r
                rewards.append(r)

            # A2: refresh the priority queue with this batch's unique, valid
            # sequences, keeping the best ``pqt_k`` by reward.
            if learning and self.pqt_k > 0:
                for key, r in iter_rewards.items():
                    if r <= -0.5 or key in pqt_seen:
                        continue
                    if len(pqt_heap) < self.pqt_k:
                        heapq.heappush(pqt_heap, (r, pqt_counter, key))
                        pqt_seen.add(key)
                        pqt_counter += 1
                    elif r > pqt_heap[0][0]:
                        _, _, evicted = heapq.heappushpop(
                            pqt_heap, (r, pqt_counter, key)
                        )
                        pqt_seen.discard(evicted)
                        pqt_seen.add(key)
                        pqt_counter += 1

            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            valid = rewards_t > -0.5
            if learning and int(valid.sum()) >= 2:
                # Phase-2a curriculum: anneal entropy from ``entropy_weight_start``
                # down to ``entropy_weight`` (explore early, commit late), and ramp
                # the PQT weight up from 0 over the first ``pqt_warmup_frac`` of the
                # run (exploit only once exploration has covered ground).
                progress = iteration / max(1, self.n_iters - 1)
                if self.entropy_weight_start is None:
                    ent_w = self.entropy_weight
                else:
                    ent_w = self.entropy_weight_start + (
                        self.entropy_weight - self.entropy_weight_start
                    ) * progress
                if self.entropy_floor is not None:
                    ent_w = max(ent_w, self.entropy_floor)
                if self.pqt_warmup_frac > 0.0:
                    pqt_w = self.pqt_weight * min(1.0, progress / self.pqt_warmup_frac)
                else:
                    pqt_w = self.pqt_weight

                loss = torch.zeros((), device=device)

                # Risk-seeking term (Petersen et al. 2021): reinforce the elite
                # (1 - elite_frac) quantile against the quantile baseline.
                q = torch.quantile(rewards_t[valid], 1.0 - self.elite_frac)
                elite = valid & (rewards_t >= q)
                adv_q = (rewards_t - q).detach()
                loss = loss - (adv_q[elite] * seq_log_probs[elite]).mean()

                # A1 dense term: reinforce every above-baseline sample against an
                # EWMA reward baseline, normalized, so the update is not driven by
                # a single elite sample at small batch sizes.
                if self.reinforce_all_valid:
                    batch_mean = float(rewards_t[valid].mean())
                    if ewma_baseline is None:
                        ewma_baseline = batch_mean
                    else:
                        ewma_baseline = (
                            (1.0 - self.ewma_alpha) * ewma_baseline
                            + self.ewma_alpha * batch_mean
                        )
                    adv_b = (rewards_t - ewma_baseline).detach()
                    std = adv_b[valid].std()
                    if torch.isfinite(std) and float(std) > 1e-6:
                        adv_b = adv_b / (std + 1e-6)
                    loss = loss - (adv_b[valid] * seq_log_probs[valid]).mean()

                # Entropy bonus (annealed).
                loss = loss - ent_w * entropies[valid].mean()

                # A2 PQT: MLE toward the best sequences discovered so far
                # (weight ramped in over the warmup).
                if self.pqt_k > 0 and pqt_heap and pqt_w > 0.0:
                    pqt_seqs = [list(key) for _, _, key in pqt_heap]
                    pqt_logp = _sequence_log_probs(
                        policy, lib, pqt_seqs, self.max_len, device,
                        constrain=self.grammar_constraints,
                    )
                    loss = loss + pqt_w * (-pqt_logp.mean())

                if loss.requires_grad and torch.isfinite(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip_norm)
                    optimizer.step()

            valid_np = np.array(rewards)
            valid_r = valid_np[valid_np > -0.5]
            mean_r = valid_r.mean() if len(valid_r) else float("nan")
            hb.beat(
                f"iter {iteration + 1}/{self.n_iters}  "
                f"pool={len(discovered)}  mean_reward={mean_r:.4f}",
                force=(iteration == 0),
            )

        return discovered

    # ------------------------------------------------------------------
    # Exact scoring + front assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _prefilter_candidates(
        cands: list[_OOCExpr], per_complexity: int
    ) -> list[_OOCExpr]:
        """Keep the ``per_complexity`` best candidates per complexity by *approx* score.

        ``_OOCExpr.approx_mse`` holds a subsampled *loss* (already passed through
        ``_metric_to_loss``), so ascending order is correct for every metric.

        This ranking is noisy: two candidates are compared on different random row
        subsamples, so the exact-best candidate at a complexity can be evicted here.
        It is therefore only used as the coarse cap ahead of exact scoring (see
        ``_select_for_exact_scoring``), or as the whole prefilter under the legacy
        ``prefilter_metric="approx"`` path.
        """
        by_c: dict[int, list[_OOCExpr]] = {}
        for c in cands:
            by_c.setdefault(c.complexity, []).append(c)
        kept: list[_OOCExpr] = []
        for group in by_c.values():
            group.sort(key=lambda e: e.approx_mse)
            kept.extend(group[:per_complexity])
        return kept

    def _select_for_exact_scoring(
        self, cands: list[_OOCExpr], per_complexity: int
    ) -> list[_OOCExpr]:
        """Choose the candidate set that gets scored exactly on the full dataset.

        Under ``prefilter_metric="approx"`` this is the noisy top-K per complexity
        (legacy behaviour).  Under ``"exact"`` it is the whole pool, or — when
        ``exact_prefilter_multiple`` is set — an approx-ranked cap of
        ``multiple * per_complexity`` per complexity, which bounds the cost of the
        exact pass on very large datasets.  See the class docstring for what the cap
        does to the front-preserving guarantee.
        """
        if self.prefilter_metric == "approx":
            return self._prefilter_candidates(cands, per_complexity=per_complexity)
        if self.exact_prefilter_multiple is None:
            return cands
        return self._prefilter_candidates(
            cands, per_complexity=per_complexity * self.exact_prefilter_multiple
        )

    def _truncate_exact_per_complexity(
        self,
        exact: list[tuple[_OOCExpr, float, float, float]],
        per_complexity: int,
    ) -> list[tuple[_OOCExpr, float, float, float]]:
        """Keep the ``per_complexity`` best-scoring candidates per complexity, exactly ranked.

        Front-preserving: a front member is by definition exact-optimal at its own
        complexity, so it ranks first in its group and cannot be dropped for any
        ``per_complexity >= 1``.  Truncating here (rather than before exact scoring)
        only bounds the cost of the sympy-conversion stage.
        """
        by_c: dict[int, list[tuple[_OOCExpr, float, float, float]]] = {}
        for row in exact:
            by_c.setdefault(row[0].complexity, []).append(row)
        kept: list[tuple[_OOCExpr, float, float, float]] = []
        for group in by_c.values():
            group.sort(
                key=lambda r: (
                    math.inf
                    if not math.isfinite(r[1])
                    else _metric_to_loss(r[1], self.score_metric)
                )
            )
            kept.extend(group[:per_complexity])
        return kept

    def _exact_eval_arrays(
        self,
        candidates: list[_OOCExpr],
        arrays: dict[str, np.ndarray],
        y: np.ndarray,
    ) -> list[tuple[_OOCExpr, float, float, float]]:
        n = len(y)
        y_finite = np.isfinite(y)
        out: list[tuple[_OOCExpr, float, float, float]] = []
        for cand in candidates:
            pred = _eval_prefix_numpy(list(cand.tokens), arrays, n)
            if pred is None:
                out.append((cand, float("nan"), 0.0, 1.0))
                continue
            mask = y_finite & np.isfinite(pred)
            if int(mask.sum()) < 2:
                out.append((cand, float("nan"), 0.0, 1.0))
                continue
            pm = pred[mask]
            ym = y[mask]
            if self.affine_reward:
                fit = _affine_residual(pm, ym)
                if fit is None:
                    out.append((cand, float("nan"), 0.0, 1.0))
                else:
                    _, b0, b1 = fit
                    resid = np.asarray(ym, dtype=np.float64) - (
                        b0 + b1 * np.asarray(pm, dtype=np.float64)
                    )
                    out.append(
                        (
                            cand,
                            _metric_from_residuals(
                                resid, self.score_metric, y=ym, n_params=1
                            ),
                            b0,
                            b1,
                        )
                    )
            else:
                resid = np.asarray(ym, dtype=np.float64) - np.asarray(pm, dtype=np.float64)
                out.append(
                    (
                        cand,
                        _metric_from_residuals(
                            resid, self.score_metric, y=ym, n_params=1
                        ),
                        0.0,
                        1.0,
                    )
                )
        return out

    def _exact_score_streaming(
        self,
        candidates: list[_OOCExpr],
        store: "MemmapDataset",
        lo: int,
        hi: int,
        chunk_rows: int,
    ) -> list[tuple[_OOCExpr, float, float, float]]:
        from nsr_engine._logging import Heartbeat
        from nsr_engine.memmap_store import chunk_ranges

        k_n = len(candidates)
        sp_ = np.zeros(k_n)
        sy = np.zeros(k_n)
        spp = np.zeros(k_n)
        spy = np.zeros(k_n)
        syy = np.zeros(k_n)
        cnt = np.zeros(k_n, dtype=np.int64)
        token_lists = [list(c.tokens) for c in candidates]

        ranges = chunk_ranges(lo, hi, chunk_rows)
        hb = Heartbeat(f"nsr-exact-{self.score_metric}", interval_s=20.0)
        for ci, (start, stop) in enumerate(ranges):
            arrays, y = store.gather(slice(start, stop))
            arrays = self._standardize_arrays(arrays, inplace=True)
            n = stop - start
            y_finite = np.isfinite(y)
            for k, toks in enumerate(token_lists):
                pred = _eval_prefix_numpy(toks, arrays, n)
                if pred is None:
                    continue
                mask = y_finite & np.isfinite(pred)
                m = int(mask.sum())
                if not m:
                    continue
                pm = np.asarray(pred[mask], dtype=np.float64)
                ym = y[mask]
                sp_[k] += float(pm.sum())
                sy[k] += float(ym.sum())
                spp[k] += float(np.dot(pm, pm))
                spy[k] += float(np.dot(pm, ym))
                syy[k] += float(np.dot(ym, ym))
                cnt[k] += m
            hb.beat(f"chunk {ci + 1}/{len(ranges)}  rows<= {stop:,}", force=(ci == 0))

        out: list[tuple[_OOCExpr, float, float, float]] = []
        for k, cand in enumerate(candidates):
            n = int(cnt[k])
            if n < 2:
                out.append((cand, float("nan"), 0.0, 1.0))
                continue
            mean_p = sp_[k] / n
            mean_y = sy[k] / n
            var_p = spp[k] - sp_[k] * sp_[k] / n
            cov = spy[k] - sp_[k] * sy[k] / n
            syy_c = syy[k] - sy[k] * sy[k] / n
            if not self.affine_reward:
                resid_sum = sy[k] - sp_[k]
                mse = max((syy[k] - 2.0 * spy[k] + spp[k]) / n, 0.0)
                if self.score_metric == "rmse":
                    score = math.sqrt(mse)
                elif self.score_metric == "mbd":
                    score = abs(-resid_sum / n)
                elif self.score_metric in {"r2", "adjusted_r2"}:
                    score = _r2_from_sse(mse * n, syy_c, n, self.score_metric)
                else:
                    score = mse
                out.append((cand, score, 0.0, 1.0))
            elif var_p < 1e-18:
                resid_sum = sy[k] - n * mean_y
                mse = max(syy_c, 0.0) / n
                if self.score_metric == "rmse":
                    score = math.sqrt(mse)
                elif self.score_metric == "mbd":
                    score = abs(-resid_sum / n)
                elif self.score_metric in {"r2", "adjusted_r2"}:
                    score = _r2_from_sse(mse * n, syy_c, n, self.score_metric)
                else:
                    score = mse
                out.append((cand, score, mean_y, 0.0))
            else:
                b1 = cov / var_p
                b0 = mean_y - b1 * mean_p
                resid_sum = sy[k] - (n * b0 + b1 * sp_[k])
                mse = max(syy_c - cov * cov / var_p, 0.0) / n
                if self.score_metric == "rmse":
                    score = math.sqrt(mse)
                elif self.score_metric == "mbd":
                    score = abs(-resid_sum / n)
                elif self.score_metric in {"r2", "adjusted_r2"}:
                    score = _r2_from_sse(mse * n, syy_c, n, self.score_metric)
                else:
                    score = mse
                out.append((cand, score, b0, b1))

        if self.score_metric not in {"mae", "mape"}:
            return out

        metric_sum = np.zeros(k_n)
        hb = Heartbeat(f"nsr-exact-{self.score_metric}-pass2", interval_s=20.0)
        params = [(b0, b1) for _, _, b0, b1 in out]
        ranges = chunk_ranges(lo, hi, chunk_rows)
        for ci, (start, stop) in enumerate(ranges):
            arrays, y = store.gather(slice(start, stop))
            arrays = self._standardize_arrays(arrays, inplace=True)
            n_rows = stop - start
            y_finite = np.isfinite(y)
            for k, toks in enumerate(token_lists):
                if cnt[k] < 2:
                    continue
                pred = _eval_prefix_numpy(toks, arrays, n_rows)
                if pred is None:
                    continue
                mask = y_finite & np.isfinite(pred)
                if not int(mask.sum()):
                    continue
                b0, b1 = params[k]
                resid = np.asarray(y[mask], dtype=np.float64) - (
                    b0 + b1 * np.asarray(pred[mask], dtype=np.float64)
                )
                if self.score_metric == "mae":
                    metric_sum[k] += float(np.abs(resid).sum())
                else:
                    denom = np.maximum(
                        np.abs(np.asarray(y[mask], dtype=np.float64)), _METRIC_EPS
                    )
                    metric_sum[k] += float((np.abs(resid) / denom).sum() * 100.0)
            hb.beat(f"chunk {ci + 1}/{len(ranges)}  rows<= {stop:,}", force=(ci == 0))

        return [
            (cand, metric_sum[k] / int(cnt[k]) if int(cnt[k]) >= 2 else float("nan"), b0, b1)
            for k, (cand, _, b0, b1) in enumerate(out)
        ]


    def _maybe_refine(
        self, *, front: "ParetoFront", X: "pd.DataFrame", y: "pd.Series"
    ) -> "ParetoFront":
        """Refit each front point's constants by least squares.

        The affine reward fits ``b0 + b1*expr`` -- a single *global* scale and
        offset -- so for a multi-term expression it cannot set the terms'
        relative weights.  The search routinely finds the right shape with the
        wrong coefficients: ``0.937*x0*x1 + 1.07*x2`` where the target is
        ``x0*x1 + x2``.  Refitting the constants afterwards is what turns those
        near-misses into exact recoveries; measured on this repo's benchmark
        suite it lifted near-exact recovery from 21% to 59% of cells.

        Every other engine in the field does the equivalent inside its search
        (Operon runs Levenberg-Marquardt per candidate; PySINDy's fit *is* a
        linear solve over its basis), so this closes a structural gap rather
        than adding a trick.

        ``optimize_front`` keeps a refit only when it improves that point's
        score, so the returned front is never worse than the input.  Refinement
        needs the optional ``scipy``/``scikit-learn`` extras; without them the
        original front is returned unchanged.
        """
        if not self.refine_constants or not front.points:
            return front
        try:
            from nsr_engine.refinement import optimize_front
        except Exception:
            return front
        try:
            return optimize_front(
                front, X, y, max_nfev=self.refine_max_nfev, seed=self.random_state
            )
        except Exception as exc:      # never let a refit failure lose the front
            print(f"[nsr] constant refinement skipped: {type(exc).__name__}: {exc}",
                  flush=True)
            return front

    def _assemble_front(
        self, exact: list[tuple[_OOCExpr, float, float, float]]
    ) -> ParetoFront:
        by_equation: dict[str, ParetoPoint] = {}
        for cand, score_val, b0, b1 in exact:
            if not math.isfinite(score_val):
                continue
            converted = _to_sympy_affine(
                list(cand.tokens),
                b0,
                b1,
                self._feat_mean,
                self._feat_std,
                feat_mode=self.scale_mode if self.standardize else None,
            )
            if converted is None:
                continue
            eq_str, sympy_expr = converted
            complexity = cand.complexity
            if self.count_affine_wrapper and self.affine_reward:
                # Charge the affine wrapper b0 + b1*(.) the tree nodes a GP engine
                # would spend on it: +1 for a non-unit slope, +1 for a non-zero
                # intercept. Restores strict cross-engine complexity comparability.
                if abs(b1 - 1.0) > 1e-9:
                    complexity += 1
                if abs(b0) > 1e-9:
                    complexity += 1
            point = ParetoPoint(
                equation=eq_str,
                sympy_expr=sympy_expr,
                complexity=complexity,
                mse=score_val,
                score_metric=self.score_metric,
            )
            # Redundant token sequences ("a", "+ a 0", "* a 1", ...) collapse to the
            # same equation with an identical score, so ties here are common: ~6% of
            # equations in a typical pool arise at several complexities with zero
            # score spread.  Breaking those ties on complexity keeps the *simplest*
            # representation and, critically, makes the result independent of
            # candidate ordering -- without it the assembled front depends on which
            # duplicate happens to be visited first, so truncating the pool could
            # change the front even when truncation preserved the best candidate at
            # every complexity.
            incumbent = by_equation.get(eq_str)
            if incumbent is None or (point.score, point.complexity) < (
                incumbent.score,
                incumbent.complexity,
            ):
                by_equation[eq_str] = point

        if not by_equation:
            print("[nsr] warning: no candidate survived exact evaluation — empty front")
            return ParetoFront([])

        front = ParetoFront(list(by_equation.values())).dominance_filter()
        print(f"[nsr] Pareto front: {len(front)} non-dominated points")
        return front

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _front_path(self) -> Path:
        """A fresh file under ``front_dir``; never clobbers an existing one."""
        import time

        stamp = time.strftime("%Y%m%d-%H%M%S")
        prefix = f"{self.cache_prefix}-" if self.cache_prefix else ""
        stem = f"{prefix}front-{stamp}-seed{self.random_state}"
        path = self.front_dir / f"{stem}.csv"
        n = 2
        while path.exists():
            path = self.front_dir / f"{stem}-{n}.csv"
            n += 1
        return path

    def _save_front(
        self,
        front: ParetoFront,
        X: pd.DataFrame | None = None,
        y: pd.Series | None = None,
    ) -> None:
        """Write ``front`` to ``front_dir`` unless the caller opted out.

        ``X``/``y`` fill in the per-point ``fit_*`` columns; out-of-core they
        are left out rather than measured on a subsample, which would report a
        different quantity under the same column name.
        """
        self.front_path_ = None
        if not self.save_front:
            return
        try:
            path = front.save(self._front_path(), X=X, y=y)
        except Exception as exc:    # a file is never worth failing a fit over
            print(
                f"[nsr] warning: could not save the Pareto front to "
                f"{self.front_dir}: {exc!r}",
                flush=True,
            )
            return
        self.front_path_ = path
        print(f"[nsr] Pareto front saved: {path}", flush=True)

    def _cache_path(self, lambda_idx: int) -> Path | None:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{self.cache_prefix}_" if self.cache_prefix else ""
        return self.cache_dir / f"{prefix}nsr-lambda-{lambda_idx:03d}.json"

    def _save_cache(
        self, lambda_idx: int, lam: float, discovered: list[_OOCExpr]
    ) -> None:
        path = self._cache_path(lambda_idx)
        if path is None:
            return
        data = {
            "version": 3,
            "lambda": lam,
            "score_metric": self.score_metric,
            "affine_reward": self.affine_reward,
            "candidates": [
                {
                    "tokens": list(d.tokens),
                    "complexity": d.complexity,
                    "approx_mse": d.approx_mse,
                }
                for d in discovered
            ],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_cache(self, lambda_idx: int) -> dict[tuple[str, ...], _OOCExpr] | None:
        path = self._cache_path(lambda_idx)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            version = data.get("version")
            if version == 2:
                if self.score_metric != "mse" or not self.affine_reward:
                    return None
            elif version == 3:
                if data.get("score_metric") != self.score_metric:
                    return None
                if bool(data.get("affine_reward")) != self.affine_reward:
                    return None
            else:
                return None
            result: dict[tuple[str, ...], _OOCExpr] = {}
            for entry in data.get("candidates", []):
                key = tuple(entry["tokens"])
                result[key] = _OOCExpr(
                    tokens=key,
                    complexity=int(entry["complexity"]),
                    approx_mse=float(entry["approx_mse"]),
                )
            return result
        except Exception:
            return None
