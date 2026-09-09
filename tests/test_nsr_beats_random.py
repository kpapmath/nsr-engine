"""Guard for the NSR-beats-Random claim (see benchmarks/run_nsr_beats_random.py).

The full 840-job sweep lives under ``benchmarks/``; this is a fast, reduced
version of the *same mechanism* wired as a pytest gate. It fits NSR and its
Random-search control (the identical engine with ``lr=0``) on one target in the
policy-favoring regime — a large shared distractor operator library — and
asserts the learned policy is, on average across a few seeds, at least as
accurate as blind random search over the same grammar.

The target is ``bilinear_plus_linear`` (``x0*x1 + x2``), the formula with the
largest and most reliable NSR advantage in the full sweep (mean ΔR² ≈ +0.14
pooled, ≈ +0.27 at low noise). The aggregate win in the sweep is driven by the
medium tier; the hardest targets (e.g. ``poly_trig_mix``) can actually favour
random search, so the guard deliberately picks a formula that isolates the
learned-policy effect rather than one where even NSR struggles.

Marked ``slow`` because it runs real ``NSREngine`` fits (like
``test_accuracy_layers.test_boosting_accepts_a_real_engine``); run with::

    pytest tests/test_nsr_beats_random.py -m slow -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sympy")

# Make the in-repo benchmark harness importable (it adds ``src/`` to sys.path).
_BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from equations import get_policy_suite, make_split  # noqa: E402
from runners import Budget, run_nsr, run_random  # noqa: E402

# A target whose search space the distractors genuinely enlarge and where the
# learned policy has a large, reliable edge, plus a small-but-real budget so the
# policy has room to learn without the test being slow. Kept in one place so the
# guard is easy to retune alongside the sweep.
_EQUATION = "bilinear_plus_linear"    # x0*x1 + x2
_ROWS = 5_000
_NOISE = 0.01                          # clean signal => informative reward
_SEEDS = (0, 1, 2)
_DISTRACTORS = ("cos", "tanh", "arctan", "cube", "reciprocal", "sign")


def _budget() -> Budget:
    return Budget(
        n_iters=60, n_lambda=3, batch_size=48, max_len=12,
        hidden_dim=64, embed_dim=24, prefilter_per_complexity=4,
        exact_prefilter_multiple=3, distractors=_DISTRACTORS,
    )


@pytest.mark.slow
def test_nsr_beats_random_on_hard_search_space():
    eq = next(e for e in get_policy_suite() if e.name == _EQUATION)
    budget = _budget()

    nsr_r2, rnd_r2 = [], []
    for seed in _SEEDS:
        split = make_split(eq, rows=_ROWS, noise=_NOISE, seed=1000 + seed)
        n = run_nsr(eq, split, budget, seed=seed)
        r = run_random(eq, split, budget, seed=seed)
        assert n.status == "ok", n.detail
        assert r.status == "ok", r.detail
        nsr_r2.append(n.test_r2)
        rnd_r2.append(r.test_r2)

    nsr_mean = float(np.nanmean(nsr_r2))
    rnd_mean = float(np.nanmean(rnd_r2))

    # Sanity: the learned policy actually fits this target well.
    assert nsr_mean > 0.5, f"NSR failed to fit (mean R2={nsr_mean:.3f})"
    # The claim: on this hard search space, learning is at least as good as
    # blind random search over the same grammar.
    assert nsr_mean >= rnd_mean, (
        f"NSR ({nsr_mean:.4f}) did not beat Random ({rnd_mean:.4f}); "
        f"per-seed NSR={np.round(nsr_r2, 4)} Random={np.round(rnd_r2, 4)}"
    )
