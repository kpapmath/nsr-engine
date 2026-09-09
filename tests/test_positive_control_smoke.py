"""R2 smoke test: the Nguyen positive-control harness runs end to end.

A fast wiring check (one easy Nguyen cell, tight budget, both methods), marked
``slow`` because it runs real ``NSREngine`` fits. It asserts the harness produces
well-formed recovery records and that the learned policy reaches a sane fit — it
does *not* gate on policy-beats-random, which is the job of the full
``benchmarks/nguyen.py`` sweep (that comparison is genuine and seed-sensitive at
a tight budget, so asserting it here would be flaky).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sympy")

_BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from nguyen import BY_NAME, _fit_cell  # noqa: E402

_BUDGET = dict(rows=400, n_lambda=2, n_iters=40, batch_size=48,
               hidden_dim=64, embed_dim=24, prefilter_per_complexity=6)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["policy", "random"])
def test_nguyen_control_runs_end_to_end(method):
    rec = _fit_cell("nguyen1", method, seed=0, budget=_BUDGET)
    assert rec["status"] == "ok", rec["detail"]
    assert np.isfinite(rec["best_test_r2"])
    assert rec["n_front"] >= 1
    assert rec["recovered"] in (0, 1)


@pytest.mark.slow
def test_policy_reaches_a_sane_fit():
    rec = _fit_cell("nguyen1", "policy", seed=0, budget=_BUDGET)
    assert rec["status"] == "ok", rec["detail"]
    # nguyen1 = x^3 + x^2 + x; even at a tight budget the policy should explain
    # most of the variance. A loose floor keeps the test robust across machines.
    assert rec["best_test_r2"] > 0.5


@pytest.mark.slow
def test_benchmark_registry_is_complete():
    # All twelve Nguyen benchmarks are registered and single-/two-variable.
    assert len(BY_NAME) == 12
    assert all(BY_NAME[f"nguyen{i}"].n_vars in (1, 2) for i in range(1, 13))
