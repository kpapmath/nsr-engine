# Accuracy Layers

Three composable, post-hoc passes over the engine's output. None of them touches
the RNN policy or the reward — they operate on the sympy expressions the engine
emits, so they work with any engine that yields `ParetoFront` / `ParetoPoint`
objects.

Every layer is **accept-if-better**: it can never return a worse training fit
than it was given, so enabling one is always safe.

## Why they exist

The engine scores each candidate with an **affine-invariant reward**: it fits
`b0 + b1*expr` by closed-form least squares and rewards the best achievable fit.
That is what lets a feature on any scale compete on correlation alone (a raw
price near 1e5 against an imbalance in [-1, 1]). It has two structural limits:

| Limit | Symptom | Layer that fixes it |
| --- | --- | --- |
| Fits **one** expression, not a sum `b0 + Σ bₖ·exprₖ` | Two additive terms of comparable magnitude collapse into a linear surrogate | 1 — residual boosting, then 3 — joint refit |
| Constants are **quantized** (`-1, -0.5, 0.5, 1, 2`) plus the affine `b0, b1` | Interior weights (the `1.5` in `1.5*log(x4)`) are unreachable, so the search substitutes operator surrogates (`log` → a decaying `exp`) | 2 — constant optimization |

## How they compose

```
                   ┌─────────────── Layer 1: Residual boosting ────────────────┐
   NSR engine      │ round k: fit residual → pick term → (Layer 2 refines its  │
 fit(X, y) → Front │ constants) → subtract. Returns a cumulative Pareto front  │
                   │ plus the discovered terms, terms_                         │
                   └──────────────────────────┬───────────────────────────────-┘
                                              │ terms_
                                              ▼
                   ┌──────── Layer 3: Joint refit + LASSO prune ───────────────┐
                   │ Re-weight terms_ as a fixed basis → drop redundant →      │
                   │ polish with Layer 2                                       │
                   └──────────────────────────────────────────────────────────-┘
```

Layer 2 is a leaf utility: Layer 1 uses it on each picked term, Layer 3 uses it
as a final polish, and it also works stand-alone on any front.

## Install

Layer 1 needs `sympy`; Layer 2 adds `scipy`; Layer 3 adds `scikit-learn`. The
`refine` extra pulls in all three:

```bash
pip install "nsr-engine[refine]"
```

## Layer 1 — Residual boosting

Greedy additive boosting in which each weak learner is a *full NSR run on the
residual*:

```
residual ← y;  model ← 0;  terms ← []
for round k = 1 … max_rounds:
    front  ← engine_factory(k).fit(X, residual)   # fresh engine, seed varies with k
    term   ← pick_elbow(front)                    # best MSE-drop-per-complexity point
    term   ← term_refiner(term, X, residual)      # optional Layer 2
    gain   ← (mse(residual) − mse(residual − term(X))) / mse(residual)
    if k > 1 and gain < min_gain: break           # the term only fits noise
    model ← model + term;  residual ← y − model(X);  terms.append(term)
```

Because round *k* fits the residual left by rounds `1…k-1`, it only has to
explain one more term. The affine reward solves each single-term subproblem, and
the sum of the rounds is the multi-term formula `intercept + Σₖ bₖ·exprₖ` that a
one-shot fit cannot express. Each term carries its own `b0`/`b1`, so intercepts
and scales compose correctly.

This is orthogonal matching pursuit, deliberately *not* a joint least-squares
fit: a joint fit over a fixed library (SINDy-style) needs a pre-enumerated basis,
whereas boosting lets NSR **discover** each basis function in turn. The cost is
greediness — a term chosen early is never revised, which Layer 3 corrects.

```python
from nsr_engine import NSREngine, ResidualBoostedNSR

def engine_factory(round_idx: int) -> NSREngine:
    return NSREngine(
        n_lambda=4, n_iters=150, batch_size=64, max_len=17,
        unary_ops=("square", "abs", "log", "exp", "sqrt"),
        random_state=42 + round_idx,   # a fresh engine, and a seed that moves
        device="cpu",
    )

booster = ResidualBoostedNSR(engine_factory, max_rounds=3, min_gain=0.02)
front = booster.fit(X_train, y_train)
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `engine_factory` | — | `factory(round_idx) -> engine` with `engine.fit(X, y) -> ParetoFront`, called once per round with the **1-based** round index. Must return a **fresh** engine; vary its seed with `round_idx`. |
| `max_rounds` | `3` | Hard cap on additive terms. |
| `min_gain` | `0.02` | After round 1, keep a round only if it cuts training MSE by at least this relative amount. Guards against appending noise-fitting terms. |
| `term_refiner` | `None` | Optional `f(expr, X, residual) -> expr` hook applied to each picked term **before** it is subtracted, so later rounds fit a cleaner residual. Pass `optimize_constants` here to run Layer 2 inside Layer 1. |
| `term_selection` | `"elbow"` | `"elbow"` keeps each term compact; `"min_mse"` takes the round's most accurate point, recovering more per round when parsimony is not the priority. |

After `fit`, two attributes are populated:

- `rounds_` — per-round diagnostics: `round`, `added`, `reason`, `gain`, `term`, `cum_mse`.
- `terms_` — `(sympy_expr, complexity)` per kept term; this is Layer 3's input.

**Return value.** `fit` returns a `ParetoFront` whose point *k* is the summed
model after *k* terms. Its `complexity` is the token count of the **sum**:
per-term complexities plus one `+` node per join (`c₁`, then `c₁+c₂+1`, then
`c₁+c₂+c₃+2`, …), which keeps boosted models on the same complexity axis as
every other method. Its `mse` is the cumulative **training** MSE after *k* terms.
Points are non-dominated by construction — complexity strictly increases and
training MSE strictly decreases on each kept round — so no `dominance_filter()`
is needed.

Round 1 is plain NSR and is always kept, so a boosted front is never worse than
an unboosted one. On a genuinely single-term or pure-noise target, early stopping
halts after round 1 and the result equals plain NSR.

A term may be undefined on some rows — `log` of a feature that goes negative —
and still be a legitimate front point, since the engine scores candidates on
their finite subset. Boosting masks the same way: non-finite rows drop out of
the cumulative MSE, and the model is simply undefined there. A term finite on
fewer than two rows is rejected and the round stops.

> The boosted front is scored in **MSE** even when the engine's `score_metric` is
> something else. Terms are still selected using the engine's own metric.

## Layer 2 — Constant optimization

Refits every floating-point constant numerically — what PySR does in its inner
loop.

```python
from nsr_engine import optimize_constants, optimize_front

expr = optimize_constants(expr, X, y, max_nfev=200, seed=0)
front = optimize_front(front, X, y)   # apply to every point of a front
```

1. **Parametrize.** Walk the tree and replace **every `Float` occurrence** with a
   fresh symbol, recording its value as the initial guess. `Integer` nodes are
   left alone, which keeps a `square`'s `**2` exponent fixed — a free fractional
   exponent is numerically unstable. Replacement is per *occurrence*, so two
   equal literals become independent parameters and can separate.
2. **Guard rails.** Zero constants, or more than `max_free_consts` (default 12),
   returns `expr` unchanged: too many free parameters is slow and overfits.
3. **Sub-sample.** Fit on at most `fit_subsample` rows (default 8000) drawn from
   a seeded RNG; `0` uses every row. Optimised constants generalise, and the fit
   stays fast.
4. **Least squares.** `lambdify` over the feature and constant symbols, then
   minimise the residual with `scipy.optimize.least_squares(method="trf",
   loss="soft_l1")`, warm-started from the recorded values, for at most
   `max_nfev` evaluations. A non-finite prediction is replaced by a large finite
   sentinel so that an `exp` overflow cannot abort the solve. The sentinel is
   scaled by the target's magnitude: a fixed `1e6` would sit on top of a target
   that happens to be near `1e6`, making an overflowing row read as a *perfect*
   fit instead of a rejected one.
5. **Accept-if-better.** Substitute the solution back as `Float`s and return the
   result **only if** it lowers the sub-sampled training MSE, else return the
   original.

Complexity is preserved — constants are substituted in place, so no node is
added or removed. The pass is deterministic: the same `(expr, X, y, seed)` gives
the same output.

`optimize_front` keeps each point's `score_metric` and accepts a refit only when
it improves that point's score in its own metric.

## Layer 3 — Joint refit + LASSO prune

Boosting scales each term once, against the residual of its own round only, and
never revisits it. Given the discovered terms, re-weight them **jointly** and
drop the redundant ones.

```python
from nsr_engine import joint_refit_prune

refined = joint_refit_prune(booster.terms_, X, y, coef_rel_tol=1e-3, seed=0)
# -> (expr, complexity, train_mse), or None if nothing usable remains
```

1. **Build the basis.** Evaluate each term into a column, dropping any that is
   non-finite. Stack into `Φ` (n × K).
2. **Sparse refit.** Fit `y ≈ w0 + Σ wₖ·Φₖ` with `LassoCV(cv=3)` for K ≥ 2, or
   `LinearRegression` for K = 1. LASSO's sparse weights drive collinear or
   redundant terms to `wₖ ≈ 0`. The estimator sees at most `fit_subsample` rows
   (default 8000; `0` uses every row) — the weights of a handful of basis
   columns are well determined long before every row is used, and `LassoCV` over
   millions of rows is the expensive step in this layer.
3. **Prune.** Keep terms with `|wₖ| > coef_rel_tol · maxₖ|wₖ|`. If none survive,
   return `None`.
4. **Reassemble.** `expr = w0 + Σ_kept wₖ·tₖ`, recomputing complexity on the
   **surviving** terms with the boosting convention (`Σ cₖ + (n_kept − 1)` joins),
   so it stays on the same axis and pruning lowers it.
5. **Polish.** Run Layer 2 on the reassembled sum, under the same `fit_subsample`
   cap.

The returned `train_mse` is always computed on **every** row, whatever
`fit_subsample` is set to, so the point stays comparable to the boosted front's
points.

Returns `None` — not an exception — when every term evaluates non-finite. Pass
`estimator="ols"` to skip the sparsity penalty, or `polish=None` to skip step 5.

## Composing all three

```python
from nsr_engine import (
    NSREngine, ParetoFront, ParetoPoint, ResidualBoostedNSR,
    joint_refit_prune, optimize_constants,
)

# Layer 1, with Layer 2 as the per-term refiner:
booster = ResidualBoostedNSR(
    engine_factory, max_rounds=3, min_gain=0.02, term_refiner=optimize_constants
)
front = booster.fit(X, y)

# Layer 3 over the discovered terms:
refined = joint_refit_prune(booster.terms_, X, y)
points = list(front.points)
if refined is not None:
    expr, complexity, mse = refined
    points.append(
        ParetoPoint(equation=str(expr), sympy_expr=expr, complexity=complexity, mse=mse)
    )
final = ParetoFront(points).dominance_filter()
```

Because every layer conforms to the front/point contract, downstream elbow
selection and out-of-sample scoring are unchanged.

`examples/accuracy_layers.py` runs exactly this progression and prints the test
R² of each stage.

## From the CLI

The layers are off by default. See the
[CLI reference](cli_reference.md#accuracy-layer-arguments) for every flag.

```bash
# Layer 1 only
python main.py --boosting --boosting-max-rounds 3 --boosting-min-gain 0.02

# All three
python main.py --boosting --constant-opt --joint-refit --validation-mode sequential
```

`--joint-refit` requires `--boosting`, since it consumes the boosted terms.
`--constant-opt` works on its own: without `--boosting` it refines every point of
the ordinary front.

## Ordering, determinism, and cost

- **Order:** 1 → 2 → 3. Layer 2 runs inside Layer 1 (per picked term, before the
  residual is formed) and again inside Layer 3 (final polish). Layer 3 consumes
  Layer 1's `terms_`.
- **Determinism:** thread a seed through the `engine_factory`, the Layer 2
  sub-sample RNG, and `LassoCV`; fixed seeds give reproducible output.
- **Cost:** Layer 1 dominates — up to `max_rounds` × one NSR fit, though early
  stopping makes single-term problems one round. Layer 2 is a small warm-started
  least-squares solve per expression; Layer 3 is one `LassoCV` plus one polish.
  Neither 2 nor 3 adds an NSR training run. Both cap the rows they fit on at
  `fit_subsample`, so their cost is bounded by that rather than by the dataset;
  what still scales with the row count is evaluating the expressions themselves,
  which is needed for the reported score.

## Measured results

> **Provenance.** These figures were measured on the reference implementation
> these layers were ported from, not re-measured against the layers as shipped
> here. Treat them as the expected shape of the result — which layer pays off
> where — rather than as a reproduction. `examples/accuracy_layers.py` runs the
> progression end to end if you want current numbers on your own data.

Reference grid of 30k observations, elbow test R² progressing
`NSR → +boost → +B (const-opt) → +B+A (joint refit)`:

| Formula (ideal complexity) | Noise | R² ceiling | NSR | +boost | +B | +B+A |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| `exp` (2) | low/med/high | 0.98/0.90/0.49 | 0.96/0.88/0.49 | 0.98/0.90/0.49 | = boost | = boost |
| `log` (2) | low/med/high | 0.98/0.90/0.50 | 0.98/0.88/0.51 | 0.98/0.89/0.51 | = boost | = boost |
| `sqrt_inter` (4) | low/med/high | 0.98/0.90/0.50 | 0.90/0.83/0.49 | 0.93/0.86/0.50 | = boost | = boost / +0.006 |
| **`exp_log`** (7) | low/med/high | 0.98/0.90/0.50 | 0.72/0.66/0.36 | 0.913/0.835/0.465 | = boost | **0.959/0.879/0.494** |
| `complex` (13) | low/med/high | 0.98/0.90/0.51 | 0.92/0.84/0.47 | 0.95/0.89/0.48 | = boost | 0.951/0.893/0.482 |

Three findings, one of them negative:

1. **Layer 1 is the big win.** The largest gain lands exactly where the affine
   reward fails: `exp_log` improves **+0.18 to +0.19**, recovering a real second
   transcendental term instead of a linear surrogate. Gains elsewhere are smaller
   but consistent (`complex` +0.04–0.05), and it never hurts.
2. **Layer 3 supplies the remaining lift on additive multi-term targets.**
   `exp_log` gains a further **+0.03 to +0.045** over boosting, reaching 0.959 at
   low noise against a 0.985 ceiling. Boosting finds the right *terms*; joint
   refit finds their right *combination*.
3. **Layer 2 was a no-op on this grid.** `+B` equalled `+boost` in all 15 cells:
   the affine reward plus folded standardization already extract near-optimal
   constants, leaving nothing for least squares to tighten. It ships anyway —
   cheap, accept-if-better, never harmful — and it pays off on problems with
   genuinely free interior constants, but do not expect it to move this
   benchmark. **Pruning also never triggered**, since every boosted term
   contributes: on these formulas the layers buy accuracy, not parsimony.

## Known limitations

- **Early stopping uses training MSE**, not held-out MSE. Training MSE always
  improves, so a term that only fits noise is caught by `min_gain` rather than by
  a validation signal.
- **Greediness is only partly undone.** Layer 3 re-weights the discovered terms
  but does not re-discover them. A full backfitting pass — re-optimising each
  term against the residual of all *others* — would cost another *k* NSR fits.
- **Round caches are not keyed by the residual.** Each boosting round is a full
  NSR fit, and `--cache-dir` caches each round under its own prefix
  (`<cache-prefix>_round<k>`). But round *k*'s target is a residual derived from
  the earlier rounds, and the cache key does not capture it: change the data or
  a setting that alters round 1's term, and round 2 will reuse a pool discovered
  against a *different* residual. Cached candidates are always re-scored exactly
  against the current data, so scores stay correct and only the search pool is
  stale — but use a fresh `--cache-prefix` when the target changes. A cache keyed
  by `(dataset_hash, round, seed)` would fix this properly.
- **The complexity metric charges every operator equally.** Charging `exp`,
  `log`, and `sqrt` more than `+` or `*` would push the search toward a true
  operator instead of a transcendental surrogate, reducing the churn Layer 2 has
  to clean up.
- **LASSO penalises on the terms' natural scale.** Terms are not standardized
  before the joint refit, so a large-magnitude term attracts a small coefficient
  and is penalised less. `coef_rel_tol` is relative to the largest weight, which
  mitigates but does not eliminate this.
