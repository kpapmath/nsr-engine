"""R1 guard: the affine-fit reward *is* Keijzer (2003) linear scaling.

Pins the identity the paper relies on: the residual of the optimal affine fit
``b0 + b1*pred`` obeys ``mse* = Var(y)*(1 - r^2)``, so the inverse-normalised-MSE
reward reduces to ``R = 1/(2 - r^2)`` -- a strictly monotone function of the
squared Pearson correlation -- and ``(b0, b1)`` equal ``numpy.polyfit(pred, y, 1)``.

If a future edit changes the reward's scale/offset invariance, this fails.
"""

import numpy as np
import pytest

from nsr_engine.engine import _affine_residual


def _reward(pred, y):
    """Inverse-normalised-MSE reward built from the affine residual."""
    res = _affine_residual(pred, y)
    assert res is not None
    resid_mse, b0, b1 = res
    var_y = float(np.var(np.asarray(y, dtype=np.float64)))  # population variance
    return resid_mse / var_y, b0, b1  # normalised residual, plus the fit


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_reward_equals_one_over_two_minus_r2(seed):
    rng = np.random.default_rng(seed)
    p = rng.standard_normal(500)
    # y is an affine function of p plus noise, so r is neither 0 nor 1.
    y = 2.5 * p - 0.7 + 0.5 * rng.standard_normal(500)

    normalised, _, _ = _reward(p, y)
    R = 1.0 / (1.0 + normalised)

    r = np.corrcoef(p, y)[0, 1]
    R_identity = 1.0 / (2.0 - r * r)

    # normalised residual == 1 - r^2
    assert normalised == pytest.approx(1.0 - r * r, abs=1e-6)
    # reward == 1 / (2 - r^2)
    assert R == pytest.approx(R_identity, abs=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_affine_coefficients_match_polyfit(seed):
    rng = np.random.default_rng(seed)
    p = rng.standard_normal(400)
    y = -1.3 * p + 4.0 + 0.3 * rng.standard_normal(400)

    _, b0, b1 = _reward(p, y)
    b1_np, b0_np = np.polyfit(p, y, 1)  # returns (slope, intercept)

    assert b1 == pytest.approx(b1_np, abs=1e-6)
    assert b0 == pytest.approx(b0_np, abs=1e-6)


def test_reward_is_scale_and_offset_invariant():
    """Linear scaling => rescaling/shifting pred leaves the reward unchanged."""
    rng = np.random.default_rng(7)
    p = rng.standard_normal(300)
    y = 0.8 * p + 2.0 + 0.4 * rng.standard_normal(300)

    n0, _, _ = _reward(p, y)
    n1, _, _ = _reward(5.0 * p + 3.0, y)  # affine transform of pred
    assert n0 == pytest.approx(n1, abs=1e-9)
