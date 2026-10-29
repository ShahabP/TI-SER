"""Prototype x RMS-envelope weak reference trajectories (Eq. 28)."""

from typing import Tuple

from .compat import np


def make_reference_trajectories(y: np.ndarray, lengths: np.ndarray, energy: np.ndarray,
                                 prototypes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create framewise VA weak targets from a class prototype and the utterance RMS envelope."""
    n, t = energy.shape
    z_ref = np.zeros((n, t, 2), dtype=np.float32)
    i_ref = np.zeros((n, t), dtype=np.float32)
    mask = np.zeros((n, t), dtype=np.float32)
    for i, (label, length) in enumerate(zip(y, lengths)):
        length = int(max(1, min(int(length), t)))
        e = energy[i, :length].astype(np.float32)
        span = float(np.ptp(e))
        if span > 1e-8:
            env = (e - float(e.min())) / (span + 1e-8)
        else:
            # Avoid a degenerate zero trajectory for almost-constant RMS frames.
            env = np.ones(length, dtype=np.float32)
        env = np.clip(env, 0.0, 1.0)
        proto = prototypes[int(label)]
        z_ref[i, :length] = env[:, None] * proto[None, :]
        i_ref[i, :length] = np.linalg.norm(z_ref[i, :length], axis=-1)
        mask[i, :length] = 1.0
    return z_ref, i_ref, mask
