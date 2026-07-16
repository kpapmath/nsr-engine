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
from nsr_engine.pareto import ParetoFront, ParetoPoint

__all__ = ["ResidualBoostedNSR"]

_TERM_SELECTIONS = ("elbow", "min_mse")


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
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        if term_selection not in _TERM_SELECTIONS:
            supported = ", ".join(repr(s) for s in _TERM_SELECTIONS)
            raise ValueError(f"term_selection must be one of: {supported}")
        self.engine_factory = engine_factory
        self.max_rounds = max_rounds
        self.min_gain = min_gain
        self.term_refiner = term_refiner
        self.term_selection = term_selection
        self.rounds_: list[dict[str, Any]] = []
        self.terms_: list[tuple[Any, int]] = []

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
                    }
                )
                break

            prev_mse = mse_of(residual)
            new_pred = model_pred + pred
            # mse_of ignores non-finite residuals, so rows where the model is
            # undefined drop out of the score, as they do in the engine.
            new_mse = mse_of(y_arr - new_pred)
            gain = (
                (prev_mse - new_mse) / prev_mse
                if np.isfinite(prev_mse) and prev_mse > 0.0
                else 0.0
            )

            if not np.isfinite(new_mse):
                self.rounds_.append(
                    {
                        "round": k,
                        "added": False,
                        "reason": "model scored no finite rows",
                        "gain": 0.0,
                        "term": expr,
                        "cum_mse": prev_mse,
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
                        "cum_mse": prev_mse,
                    }
                )
                break

            model_pred = new_pred
            residual = y_arr - model_pred
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
                    mse=new_mse,
                    score_metric="mse",
                )
            )
            self.rounds_.append(
                {
                    "round": k,
                    "added": True,
                    "reason": "kept",
                    "gain": gain,
                    "term": expr,
                    "cum_mse": new_mse,
                }
            )

        # Non-dominated by construction: complexity strictly increases and
        # training MSE strictly decreases on every kept round.
        return ParetoFront(points)
