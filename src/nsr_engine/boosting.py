"""Accuracy layer 1: residual boosting.

The engine's affine-invariant reward fits ``b0 + b1*expr`` — it can scale and
shift *one* expression, but it cannot fit a linear combination of several.  When
the target is a sum of additive terms of comparable magnitude, the one-shot fit
locks onto whichever single expression correlates best and collapses the rest
into a linear surrogate.

:class:`ResidualBoostedNSR` removes that limit without touching the reward or
the policy: each round runs a *fresh* engine on the residual left by the
previous rounds, so round ``k`` only has to explain one more term.  The affine
reward solves each single-term subproblem, and the sum of the rounds is the
multi-term formula ``intercept + sum_k b_k*expr_k`` that a one-shot fit cannot
express.  Each term carries its own ``b0``/``b1``, so intercepts and scales
compose correctly.

This is greedy — orthogonal matching pursuit, deliberately not a joint
least-squares fit.  A joint fit over a fixed library (SINDy-style) needs a
pre-enumerated basis, whereas boosting lets NSR *discover* each basis function
in turn.  The cost is that a term chosen early is never revised, which
:func:`~nsr_engine.refinement.joint_refit_prune` (layer 3) then corrects.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from nsr_engine._expr import eval_sympy_on, mse_of
from nsr_engine.engine import _metric_from_residuals, _metric_to_loss
from nsr_engine.pareto import ParetoFront, ParetoPoint

__all__ = ["ResidualBoostedNSR"]

_TERM_SELECTIONS = ("elbow", "min_mse")

# Every ``NSREngine.score_metric`` except ``"mbd"``.  See the ``residual_metric``
# docstring for why that one is excluded rather than supported.
_RESIDUAL_METRICS = ("mse", "rmse", "mae", "mape", "r2", "adjusted_r2")

# Rejected with an explanation rather than silently mixed into the gain rule.
_UNSUPPORTED_RESIDUAL_METRICS = {
    "mbd": (
        "mbd measures signed bias, which the affine reward's intercept already "
        "drives to ~0 in round 1; the gain rule would then divide by ~0 and "
        "accept arbitrary later rounds"
    ),
}


class ResidualBoostedNSR:
    """Greedy additive boosting where each weak learner is a full NSR run.

    Conforms to the ``SREngine`` protocol: ``fit(X, y) -> ParetoFront``.

    Parameters
    ----------
    engine_factory:
        ``factory(round_idx) -> engine`` with ``engine.fit(X, y) -> ParetoFront``,
        called once per round with the 1-based round index.  Must return a
        **fresh** engine, and should vary its seed with ``round_idx`` so rounds
        do not repeat the same search.  Engine-agnostic by design.
    max_rounds:
        Hard cap on the number of additive terms.
    min_gain:
        After round 1, a round is kept only if it cuts training MSE by at least
        this relative amount.  Guards against appending terms that fit noise.
    term_refiner:
        Optional ``f(expr, X, residual) -> expr`` hook, applied to each picked
        term **before** it is subtracted, so later rounds fit a cleaner
        residual.  Pass :func:`~nsr_engine.refinement.optimize_constants` here
        to run layer 2 inside layer 1.
    term_selection:
        ``"elbow"`` keeps each term compact; ``"min_mse"`` takes the most
        accurate point of the round's front, recovering more per round when
        parsimony is not the priority.
    residual_metric:
        Objective the round-acceptance rule is measured in: ``"mse"`` (default,
        the historical behaviour), ``"rmse"``, ``"mae"``, ``"mape"``, ``"r2"``
        or ``"adjusted_r2"``.  It sets what ``min_gain`` is a fraction *of* and
        what the emitted points report as their score, so a caller running the
        weak learner under a given ``score_metric`` can make the booster agree
        with it instead of mixing the two.

        New in 0.9.0.  Before, only ``"mse"`` and ``"rmse"`` were accepted, so
        any other ``score_metric`` left the weak learner optimising one
        objective while the acceptance rule measured another.  The list is now
        every ``NSREngine.score_metric`` but one, and the values are computed by
        the engine's own metric code, so the two cannot disagree about what a
        metric *means*.

        ``"mbd"`` is rejected rather than supported.  It measures signed bias,
        which the affine reward's intercept already drives to ~0 in round 1;
        the relative gain rule would then be dividing by ~0 and would accept
        essentially any later round.  It is a diagnostic, not an objective, so
        asking for it raises rather than silently falling back to MSE.

        Two design points, since ``mape`` and ``r2`` are defined against the
        *target* rather than a residual vector:

        * The rule scores the cumulative model against the original ``y``, not
          the round's residual, so the target is in hand and both are
          well-defined.  Rows on which the model is undefined are dropped from
          the residual and the target together, matching how the engine scores
          a candidate on its finite subset.
        * ``min_gain`` is applied to the metric's *loss* form -- the metric
          itself for the error metrics, and ``1 - r2`` for ``"r2"`` and
          ``"adjusted_r2"``, which are maximised and may be negative.  A
          relative threshold needs a non-negative quantity that falls as the
          model improves; the raw ``r2`` is neither.  Because ``1 - r2`` is
          proportional to MSE for a fixed target, gains under ``"r2"`` equal
          gains under ``"mse"`` exactly, while the front still reports ``r2``.
          ``"adjusted_r2"`` additionally charges the model its term count, so a
          round must beat the parameter penalty it adds.

        Metrics are monotonically related in pairs but not equally scaled, so
        the *same* ``min_gain`` means different things under each:
        ``gain_rmse = 1 - sqrt(1 - gain_mse)``.  A ``min_gain`` of 0.02 in MSE
        is 0.01 in RMSE; passing the same number under both metrics tightens
        the threshold rather than preserving it.

    Attributes
    ----------
    rounds_:
        Per-round diagnostics: round index, whether the term was added, the
        relative gain, the term, and the cumulative training MSE.
    terms_:
        ``(sympy_expr, complexity)`` of each kept term — the input to layer 3.
    """

    def __init__(
        self,
        engine_factory: Callable[[int], Any],
        max_rounds: int = 3,
        min_gain: float = 0.02,
        term_refiner: Callable[..., Any] | None = None,
        *,
        term_selection: str = "elbow",
        residual_metric: str = "mse",
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        if term_selection not in _TERM_SELECTIONS:
            supported = ", ".join(repr(s) for s in _TERM_SELECTIONS)
            raise ValueError(f"term_selection must be one of: {supported}")
        residual_metric = residual_metric.lower()
        if residual_metric in _UNSUPPORTED_RESIDUAL_METRICS:
            why = _UNSUPPORTED_RESIDUAL_METRICS[residual_metric]
            supported = ", ".join(repr(s) for s in _RESIDUAL_METRICS)
            raise ValueError(
                f"residual_metric={residual_metric!r} cannot drive the round "
                f"acceptance rule: {why}. Supported: {supported}."
            )
        if residual_metric not in _RESIDUAL_METRICS:
            supported = ", ".join(repr(s) for s in _RESIDUAL_METRICS)
            raise ValueError(f"residual_metric must be one of: {supported}")
        self.engine_factory = engine_factory
        self.max_rounds = max_rounds
        self.min_gain = min_gain
        self.term_refiner = term_refiner
        self.term_selection = term_selection
        self.residual_metric = residual_metric
        self.rounds_: list[dict[str, Any]] = []
        self.terms_: list[tuple[Any, int]] = []

    def _score_of(
        self,
        residual: np.ndarray,
        y: np.ndarray,
        n_terms: int,
    ) -> float:
        """Residual objective, in whichever metric ``residual_metric`` names.

        Delegates to the engine's own ``_metric_from_residuals`` so the booster
        and its weak learner compute a given metric identically -- the whole
        point of letting ``residual_metric`` follow ``score_metric``.

        Rows where the model is undefined are dropped before scoring, mirroring
        the engine, which scores each candidate on its finite subset.  ``y`` is
        masked alongside the residual because ``mape`` and ``r2`` are defined
        against the target, not the residual alone.

        ``n_terms`` is the number of additive terms in the model being scored;
        it is the parameter count ``adjusted_r2`` penalises.
        """
        residual = np.asarray(residual, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(residual) & np.isfinite(y)
        if int(mask.sum()) < 2:
            return float("nan")
        return _metric_from_residuals(
            residual[mask],
            self.residual_metric,
            y=y[mask],
            n_params=max(1, n_terms),
        )

    def _loss_of(self, score: float) -> float:
        """The round-acceptance rule's positive loss for a metric value.

        ``min_gain`` is a *relative* threshold, so it needs a quantity that is
        non-negative and decreases as the model improves.  The error metrics
        already are one; ``r2``/``adjusted_r2`` are maximised, so the engine's
        ``_metric_to_loss`` maps them to ``1 - r2`` -- the unexplained variance
        fraction, which for a fixed target is proportional to MSE.  Gains under
        ``r2`` therefore equal gains under ``mse`` exactly, while the points the
        front reports stay in ``r2``.
        """
        return _metric_to_loss(score, self.residual_metric)

    def _pick_term(self, front: ParetoFront) -> ParetoPoint | None:
        """Best MSE-drop-per-complexity term of the round's front."""
        points = [p for p in front.points if p.sympy_expr is not None]
        if not points:
            return None
        if self.term_selection == "min_mse":
            return min(points, key=lambda p: p.score)
        return ParetoFront(points).elbow()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ParetoFront:
        """Boost for up to ``max_rounds`` rounds; return the cumulative front."""
        y_arr = np.asarray(
            y.to_numpy() if isinstance(y, pd.Series) else y, dtype=np.float64
        )
        index = X.index

        self.rounds_ = []
        self.terms_ = []

        model_pred = np.zeros(y_arr.size, dtype=np.float64)
        model_expr: Any = None
        residual = y_arr.copy()
        points: list[ParetoPoint] = []

        for k in range(1, self.max_rounds + 1):
            resid_series = pd.Series(residual, index=index)
            front = self.engine_factory(k).fit(X, resid_series)

            term = self._pick_term(front) if len(front) else None
            if term is None:
                self.rounds_.append(
                    {
                        "round": k,
                        "added": False,
                        "reason": "empty front",
                        "gain": 0.0,
                        "term": None,
                        "cum_mse": mse_of(residual),
                        "cum_score": self._score_of(
                            residual, y_arr, len(self.terms_)
                        ),
                    }
                )
                break

            expr = term.sympy_expr
            if self.term_refiner is not None:
                refined = self.term_refiner(expr, X, resid_series)
                if refined is not None:
                    expr = refined

            # A term may be undefined on some rows (`log` of a negative feature)
            # and still be a legitimate front point: the engine scores such
            # candidates on their finite subset.  Mask the same way rather than
            # rejecting the term, but insist on enough finite rows to score.
            pred = eval_sympy_on(expr, X)
            if pred is None or int(np.isfinite(pred).sum()) < 2:
                self.rounds_.append(
                    {
                        "round": k,
                        "added": False,
                        "reason": "term evaluated finitely on fewer than 2 rows",
                        "gain": 0.0,
                        "term": expr,
                        "cum_mse": mse_of(residual),
                        "cum_score": self._score_of(
                            residual, y_arr, len(self.terms_)
                        ),
                    }
                )
                break

            prev_score = self._score_of(residual, y_arr, len(self.terms_))
            new_pred = model_pred + pred
            new_resid = y_arr - new_pred
            new_score = self._score_of(new_resid, y_arr, len(self.terms_) + 1)

            # `min_gain` is relative, so it is measured on the metric's loss
            # form, never on the metric itself: under `r2` the raw value grows
            # towards 1 and can be negative, which would invert the rule's sign.
            # For every error metric the loss *is* the metric, so `mse`/`rmse`
            # keep their historical arithmetic exactly.
            prev_loss = self._loss_of(prev_score)
            new_loss = self._loss_of(new_score)
            gain = (
                (prev_loss - new_loss) / prev_loss
                if np.isfinite(prev_loss) and prev_loss > 0.0
                else 0.0
            )

            if not np.isfinite(new_score):
                self.rounds_.append(
                    {
                        "round": k,
                        "added": False,
                        "reason": "model scored no finite rows",
                        "gain": 0.0,
                        "term": expr,
                        "cum_mse": mse_of(residual),
                        "cum_score": prev_score,
                    }
                )
                break

            # Round 1 is plain NSR and is always kept, so the boosted front is
            # never worse than the unboosted one.  Later rounds must earn it.
            if k > 1 and gain < self.min_gain:
                self.rounds_.append(
                    {
                        "round": k,
                        "added": False,
                        "reason": f"gain {gain:.4f} < min_gain {self.min_gain}",
                        "gain": gain,
                        "term": expr,
                        "cum_mse": mse_of(residual),
                        "cum_score": prev_score,
                    }
                )
                break

            model_pred = new_pred
            residual = new_resid
            model_expr = expr if model_expr is None else model_expr + expr
            self.terms_.append((expr, int(term.complexity)))

            # Complexity of the *sum*: per-term complexities plus one `+` node
            # per join, keeping boosted models on the same axis as every other
            # method.
            complexity = sum(c for _, c in self.terms_) + (len(self.terms_) - 1)
            points.append(
                ParetoPoint(
                    equation=str(model_expr),
                    sympy_expr=model_expr,
                    complexity=complexity,
                    mse=new_score,
                    score_metric=self.residual_metric,
                )
            )
            self.rounds_.append(
                {
                    "round": k,
                    "added": True,
                    "reason": "kept",
                    "gain": gain,
                    "term": expr,
                    "cum_mse": mse_of(new_resid),
                    "cum_score": new_score,
                }
            )

        # Non-dominated by construction: complexity strictly increases and the
        # training residual metric strictly decreases on every kept round.
        #
        # Every `break` above leaves `points` untouched, so a booster whose
        # first round is rejected returns an empty front.  Carry the reason the
        # round was rejected into it: `elbow()` quotes it, which turns an
        # opaque failure deep in the caller into the actual cause ("model
        # scored no finite rows").
        reason: str | None = None
        if not points and self.rounds_:
            last = self.rounds_[-1]
            reason = f"boosting round {last['round']} rejected: {last['reason']}"
        return ParetoFront(points, empty_reason=reason)
