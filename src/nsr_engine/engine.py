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
Unary ops   : square abs log  (arity 1)
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
  once after the lambda-sweep, for a small pre-filtered set (top-K per
  complexity).
* Expression evaluation runs in float32 (halving temporary-array RAM); the
  affine least-squares scoring casts the masked prediction vector to float64
  so accumulated statistics stay accurate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

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
_CONST_VALUES: dict[str, float] = {t: float(t) for t in _CONST_TOKENS}

_START_TOKEN = "<s>"  # sentinel fed at step 0
_SCORE_METRICS: tuple[str, ...] = ("mse", "rmse", "mae")


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
        if tok == "square":
            a = _rec()
            return None if a is None else a ** 2
        if tok == "abs":
            a = _rec()
            return None if a is None else np.abs(a)
        if tok == "log":
            a = _rec()
            return None if a is None else np.log(np.abs(a) + 1e-10)
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


def _metric_from_residuals(resid: np.ndarray, metric: str) -> float:
    resid = np.asarray(resid, dtype=np.float64)
    if metric == "mse":
        return float(np.mean(resid * resid))
    if metric == "rmse":
        return float(math.sqrt(float(np.mean(resid * resid))))
    if metric == "mae":
        return float(np.mean(np.abs(resid)))
    raise ValueError(f"unsupported score_metric={metric!r}")


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
    else:
        raise ValueError(f"unsupported score_metric={metric!r}")
    return max(scale, 1e-10)


# ---------------------------------------------------------------------------
# Sympy conversion (runs only for surviving candidates, never in training)
# ---------------------------------------------------------------------------

def _to_sympy_affine(
    tokens: list[str],
    b0: float,
    b1: float,
    feat_mean: dict[str, float] | None = None,
    feat_std: dict[str, float] | None = None,
) -> tuple[str, Any] | None:
    """Convert a prefix token list to ``b0 + b1 * expr`` in *raw* feature terms.

    If ``feat_mean``/``feat_std`` are given (standardization), every feature
    symbol ``f`` is substituted with ``(f - mean) / std`` so the returned
    formula is expressed against the original columns.
    """
    try:
        import sympy as sp
    except ImportError:
        return None

    pos_ref = [0]

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

        if tok == "square":
            a = _rec()
            return None if a is None else a ** 2
        if tok == "abs":
            a = _rec()
            return None if a is None else sp.Abs(a)
        if tok == "log":
            a = _rec()
            return None if a is None else sp.log(sp.Abs(a) + sp.Float(1e-10))

        if tok in _CONST_VALUES:
            return sp.Float(_CONST_VALUES[tok])

        sym = sp.Symbol(tok)
        if feat_mean is not None and tok in feat_mean:
            std = feat_std[tok] if feat_std is not None else 1.0
            std = std if abs(std) > 1e-12 else 1.0
            return (sym - sp.Float(feat_mean[tok])) / sp.Float(std)
        return sym

    expr = _rec()
    if expr is None:
        return None
    final = sp.Float(b0) + sp.Float(b1) * expr
    try:
        simplified = sp.simplify(final)
        eq_str = str(simplified)
    except Exception:
        simplified = final
        eq_str = str(final)
    return eq_str, simplified


def _to_sympy(tokens: list[str]) -> tuple[str, Any] | None:
    """Convert a prefix token list to a sympy expression."""
    try:
        import sympy as sp
    except ImportError:
        return None

    pos_ref = [0]

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

        if tok == "square":
            a = _rec()
            return None if a is None else a ** 2
        if tok == "abs":
            a = _rec()
            return None if a is None else sp.Abs(a)
        if tok == "log":
            a = _rec()
            return None if a is None else sp.log(sp.Abs(a) + sp.Float(1e-10))

        if tok in _CONST_VALUES:
            return sp.Float(_CONST_VALUES[tok])

        return sp.Symbol(tok)

    expr = _rec()
    if expr is None:
        return None
    try:
        simplified = sp.simplify(expr)
        eq_str = str(simplified)
    except Exception:
        eq_str = str(expr)
        simplified = expr

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
    return _TokenLibrary(
        vocab=vocab,
        token_to_id=token_to_id,
        start_id=token_to_id[_START_TOKEN],
        arity=arity,
        terminal_mask=(arity == 0),
        unary_mask=(arity == 1),
        binary_mask=(arity == 2),
    )


# ---------------------------------------------------------------------------
# Batched sequence sampling with arity-aware masking
# ---------------------------------------------------------------------------

def _sample_batch(
    policy: _GRUPolicy,
    lib: _TokenLibrary,
    batch_size: int,
    max_len: int,
    device: torch.device,
) -> tuple[list[list[str]], torch.Tensor, torch.Tensor]:
    """Sample ``batch_size`` expressions from the policy in parallel.

    Returns ``(sequences, seq_log_probs, entropies)`` where ``seq_log_probs``
    and ``entropies`` are ``(batch_size,)`` tensors retaining the gradient
    graph for REINFORCE.  Every sequence is a complete prefix expression.
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

        budget = max_len - step
        need = arity_remaining.unsqueeze(1)
        allow = lib.terminal_mask.unsqueeze(0) | (
            lib.binary_mask.unsqueeze(0) & (budget >= need + 2)
        ) | (
            lib.unary_mask.unsqueeze(0) & (budget >= need + 1)
        )

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
    score_metric:
        Accuracy metric to minimize. Supported values are ``"mse"``,
        ``"rmse"``, and ``"mae"``.
    prefilter_per_complexity:
        How many lowest-approx-score candidates per complexity level survive to
        the exact scoring + sympy-conversion stage.
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
        binary_ops: tuple[str, ...] | list[str] | None = None,
        unary_ops: tuple[str, ...] | list[str] | None = None,
        const_tokens: tuple[str, ...] | list[str] | None = None,
        device: str = "auto",
        step_subsample_size: int | None = None,
        standardize: bool = True,
        affine_reward: bool = True,
        score_metric: str = "mse",
        prefilter_per_complexity: int = 16,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch is required for NSREngine.  Install with: pip install torch"
            )
        score_metric = score_metric.lower()
        if score_metric not in _SCORE_METRICS:
            supported = ", ".join(repr(m) for m in _SCORE_METRICS)
            raise ValueError(f"score_metric must be one of: {supported}")
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
        self.binary_ops = tuple(binary_ops) if binary_ops is not None else _BINARY_OPS
        self.unary_ops = tuple(unary_ops) if unary_ops is not None else _UNARY_OPS
        self.const_tokens = tuple(const_tokens) if const_tokens is not None else _CONST_TOKENS
        self.device_str = device
        self.step_subsample_size = step_subsample_size
        self.standardize = standardize
        self.affine_reward = affine_reward
        self.score_metric = score_metric
        self.prefilter_per_complexity = prefilter_per_complexity
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
    # Per-feature standardization
    # ------------------------------------------------------------------

    def _set_stats_from_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        if not self.standardize:
            return
        mean: dict[str, float] = {}
        std: dict[str, float] = {}
        for col, arr in arrays.items():
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                mean[col], std[col] = 0.0, 1.0
                continue
            m = float(finite.mean(dtype=np.float64))
            s = float(finite.std(dtype=np.float64))
            mean[col] = m
            std[col] = s if s > 1e-12 else 1.0
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
        for start, stop in chunk_ranges(lo, hi, chunk_rows):
            arrays, _ = store.gather(slice(start, stop))
            for j, c in enumerate(cols):
                a = arrays[c]
                fin = a[np.isfinite(a)]
                s1[j] += float(fin.sum(dtype=np.float64))
                s2[j] += float(np.einsum("i,i->", fin, fin, dtype=np.float64))
                cnt[j] += fin.size
        mean: dict[str, float] = {}
        std: dict[str, float] = {}
        for j, c in enumerate(cols):
            if cnt[j] == 0:
                mean[c], std[c] = 0.0, 1.0
                continue
            m = s1[j] / cnt[j]
            var = max(s2[j] / cnt[j] - m * m, 0.0)
            s = math.sqrt(var)
            mean[c] = float(m)
            std[c] = float(s) if s > 1e-12 else 1.0
        self._feat_mean, self._feat_std = mean, std

    def _standardize_arrays(
        self, arrays: dict[str, np.ndarray], *, inplace: bool = False
    ) -> dict[str, np.ndarray]:
        if not self.standardize or self._feat_mean is None:
            return arrays
        mean, std = self._feat_mean, self._feat_std
        out: dict[str, np.ndarray] = {}
        for col, arr in arrays.items():
            if col in mean:
                if inplace:
                    arr -= mean[col]
                    arr /= std[col]  # type: ignore[index]
                    out[col] = arr
                else:
                    out[col] = (arr - mean[col]) / std[col]  # type: ignore[index]
            else:
                out[col] = arr
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
        return _metric_from_residuals(resid, self.score_metric)

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
        all_indices = np.arange(n_rows)
        y_score_scale_full = _target_metric_scale(y_arr[np.isfinite(y_arr)], self.score_metric)

        def sample_step() -> tuple[dict[str, np.ndarray], np.ndarray, float]:
            if not subsample:
                return arrays, y_arr, y_score_scale_full
            k = self.step_subsample_size or n_rows
            if subsample_regime_ids is not None:
                cells = np.unique(subsample_regime_ids)
                n_per_cell = max(1, k // len(cells))
                parts: list[np.ndarray] = []
                for cell in cells:
                    cell_idx = all_indices[subsample_regime_ids == cell]
                    take = min(n_per_cell, len(cell_idx))
                    parts.append(np.random.choice(cell_idx, size=take, replace=False))
                idx = np.concatenate(parts)
            else:
                idx = np.random.choice(all_indices, size=k, replace=False)
            step_arrays = {col: arr[idx] for col, arr in arrays.items()}
            step_y = y_arr[idx]
            finite_y = step_y[np.isfinite(step_y)]
            return step_arrays, step_y, _target_metric_scale(finite_y, self.score_metric)

        pool = self._sweep_lambdas(lib, sample_step, device)
        if not pool:
            print("[nsr] warning: no valid expressions discovered — returning empty front")
            return ParetoFront([])

        candidates = self._prefilter_candidates(
            list(pool.values()), per_complexity=self.prefilter_per_complexity
        )
        print(
            f"[nsr] exact full-set {self.score_metric.upper()} for {len(candidates)} candidates "
            f"({n_rows:,} rows) …",
            flush=True,
        )
        exact = self._exact_eval_arrays(candidates, arrays, y_arr)
        return self._assemble_front(exact)

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
            f"standardize={self.standardize}  affine_reward={self.affine_reward}  "
            f"score_metric={self.score_metric}",
            flush=True,
        )
        lib = _make_library(
            features, self.binary_ops, self.unary_ops, self.const_tokens, device
        )

        if self.standardize:
            print("[nsr] computing per-feature standardization stats over train range …", flush=True)
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
            return ParetoFront([])

        candidates = self._prefilter_candidates(
            list(pool.values()), per_complexity=per_complexity
        )
        print(
            f"[nsr] exact full-set {self.score_metric.upper()} for {len(candidates)} candidates "
            f"by streaming {train_hi - train_lo:,} rows in chunks of {chunk_rows:,} …",
            flush=True,
        )
        exact = self._exact_score_streaming(
            candidates, store, train_lo, train_hi, chunk_rows
        )
        return self._assemble_front(exact)

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
                discovered = self._train_one_lambda(
                    lam=lam,
                    lib=lib,
                    sample_step=sample_step,
                    device=device,
                    seed=self.random_state + i,
                )
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

        for iteration in range(self.n_iters):
            step_arrays, step_y, step_score_scale = sample_step()
            step_n = len(step_y)
            step_y_finite = np.isfinite(step_y)

            sequences, seq_log_probs, entropies = _sample_batch(
                policy, lib, self.batch_size, self.max_len, device
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
                            normalized_score = score_val / step_score_scale
                            r = 1.0 / (1.0 + normalized_score) - lam * len(tokens)
                            if key not in discovered or score_val < discovered[key].approx_mse:
                                discovered[key] = _OOCExpr(
                                    tokens=key,
                                    complexity=len(tokens),
                                    approx_mse=score_val,
                                )
                iter_rewards[key] = r
                rewards.append(r)

            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            valid = rewards_t > -0.5
            if int(valid.sum()) >= 2:
                q = torch.quantile(rewards_t[valid], 1.0 - self.elite_frac)
                elite = valid & (rewards_t >= q)
                advantage = (rewards_t - q).detach()
                pg_loss = -(advantage[elite] * seq_log_probs[elite]).mean()
                ent_loss = -entropies[valid].mean()
                loss = pg_loss + self.entropy_weight * ent_loss
                if loss.requires_grad and torch.isfinite(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
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
        by_c: dict[int, list[_OOCExpr]] = {}
        for c in cands:
            by_c.setdefault(c.complexity, []).append(c)
        kept: list[_OOCExpr] = []
        for group in by_c.values():
            group.sort(key=lambda e: e.approx_mse)
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
                    out.append((cand, _metric_from_residuals(resid, self.score_metric), b0, b1))
            else:
                resid = np.asarray(ym, dtype=np.float64) - np.asarray(pm, dtype=np.float64)
                out.append((cand, _metric_from_residuals(resid, self.score_metric), 0.0, 1.0))
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
                mse = max((syy[k] - 2.0 * spy[k] + spp[k]) / n, 0.0)
                score = math.sqrt(mse) if self.score_metric == "rmse" else mse
                out.append((cand, score, 0.0, 1.0))
            elif var_p < 1e-18:
                mse = max(syy_c, 0.0) / n
                score = math.sqrt(mse) if self.score_metric == "rmse" else mse
                out.append((cand, score, mean_y, 0.0))
            else:
                b1 = cov / var_p
                b0 = mean_y - b1 * mean_p
                mse = max(syy_c - cov * cov / var_p, 0.0) / n
                score = math.sqrt(mse) if self.score_metric == "rmse" else mse
                out.append((cand, score, b0, b1))

        if self.score_metric != "mae":
            return out

        abs_sum = np.zeros(k_n)
        hb = Heartbeat("nsr-exact-mae-pass2", interval_s=20.0)
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
                abs_sum[k] += float(np.abs(resid).sum())
            hb.beat(f"chunk {ci + 1}/{len(ranges)}  rows<= {stop:,}", force=(ci == 0))

        return [
            (cand, abs_sum[k] / int(cnt[k]) if int(cnt[k]) >= 2 else float("nan"), b0, b1)
            for k, (cand, _, b0, b1) in enumerate(out)
        ]

    def _assemble_front(
        self, exact: list[tuple[_OOCExpr, float, float, float]]
    ) -> ParetoFront:
        by_equation: dict[str, ParetoPoint] = {}
        for cand, score_val, b0, b1 in exact:
            if not math.isfinite(score_val):
                continue
            converted = _to_sympy_affine(
                list(cand.tokens), b0, b1, self._feat_mean, self._feat_std
            )
            if converted is None:
                continue
            eq_str, sympy_expr = converted
            if eq_str not in by_equation or score_val < by_equation[eq_str].score:
                by_equation[eq_str] = ParetoPoint(
                    equation=eq_str,
                    sympy_expr=sympy_expr,
                    complexity=cand.complexity,
                    mse=score_val,
                    score_metric=self.score_metric,
                )

        if not by_equation:
            print("[nsr] warning: no candidate survived exact evaluation — empty front")
            return ParetoFront([])

        front = ParetoFront(list(by_equation.values())).dominance_filter()
        print(f"[nsr] Pareto front: {len(front)} non-dominated points")
        return front

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

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
