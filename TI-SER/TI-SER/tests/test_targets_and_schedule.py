"""Weak-target construction (Eq. 28) and LR schedule (Appendix A)."""
import numpy as np
import pytest

from ti_ser.config import INITIAL_LR, VA_PROTOTYPES
from ti_ser.targets import make_reference_trajectories


def test_reference_trajectory_is_envelope_times_prototype():
    energy = np.zeros((1, 10), dtype=np.float32)
    energy[0, :6] = [0.0, 0.2, 0.4, 1.0, 0.5, 0.1]
    z, i_ref, mask = make_reference_trajectories(np.array([1]), np.array([6]), energy, VA_PROTOTYPES)
    env = (energy[0, :6] - 0.0) / (1.0 + 1e-8)
    np.testing.assert_allclose(z[0, :6], env[:, None] * VA_PROTOTYPES[1][None, :], atol=1e-6)
    np.testing.assert_allclose(i_ref[0, :6], np.linalg.norm(z[0, :6], axis=-1), atol=1e-6)
    assert mask[0].tolist() == [1.0] * 6 + [0.0] * 4
    assert np.all(z[0, 6:] == 0)


def test_constant_energy_does_not_degenerate():
    energy = np.full((1, 8), 0.3, dtype=np.float32)
    z, _, _ = make_reference_trajectories(np.array([0]), np.array([8]), energy, VA_PROTOTYPES)
    np.testing.assert_allclose(z[0], np.tile(VA_PROTOTYPES[0], (8, 1)))


def test_lr_schedule_warmup_then_cosine():
    pytest.importorskip("tensorflow")  # training.py imports the TF stack
    from ti_ser.training import _learning_rate
    # 5-epoch warmup from 0.1*lr to lr, then cosine decay to 0 over the rest.
    assert _learning_rate(0, 20) == pytest.approx(INITIAL_LR * (0.1 + 0.9 / 5))
    assert _learning_rate(4, 20) == pytest.approx(INITIAL_LR)
    assert _learning_rate(5, 20) == pytest.approx(INITIAL_LR)
    assert _learning_rate(19, 20) < 0.02 * INITIAL_LR
