"""Training-only feature normalisation (statistics never see val/test data)."""

from __future__ import annotations
import hashlib
import json
from typing import Dict, Iterable
import numpy as np


class RunningStats:
    """Frame-level mean/std accumulated over *training* utterances only."""

    def __init__(self, dim: int):
        self.n = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sumsq = np.zeros(dim, dtype=np.float64)

    def update(self, feat: np.ndarray):
        feat = feat.astype(np.float64)
        self.n += feat.shape[0]
        self.sum += feat.sum(0)
        self.sumsq += (feat ** 2).sum(0)

    def finalize(self):
        mean = self.sum / max(self.n, 1)
        var = np.maximum(self.sumsq / max(self.n, 1) - mean ** 2, 0.0)
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def compute_train_stats(features_iter: Iterable[np.ndarray], dim: int = 120):
    rs = RunningStats(dim)
    for f in features_iter:
        rs.update(f)
    return rs.finalize()


def stats_fingerprint(mean: np.ndarray, std: np.ndarray) -> str:
    return hashlib.sha256(
        json.dumps([np.round(mean, 6).tolist(), np.round(std, 6).tolist()]).encode()
    ).hexdigest()[:16]
