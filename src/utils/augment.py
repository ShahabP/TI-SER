"""SpecAugment-style masking and training-set feature normalisation.

Usage per split (see ``src/pipeline.py``):
    train: Compose([FeatureNormalizer(mean, std), SpecAugment()])
    val / test: FeatureNormalizer(mean, std)        # never augmented
"""

from __future__ import annotations
import numpy as np


class SpecAugment:
    """Random frequency/time masking applied to *training* samples only.

    Masks are drawn from a private Generator re-seeded from torch's per-worker
    seed so DataLoader workers do not replay identical masks, while runs stay
    reproducible under a fixed seed.
    """

    def __init__(self, time_mask_width=20, freq_mask_width=10, n_time_masks=2,
                 n_freq_masks=2, p=0.5):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks
        self.p = p
        self._calls = 0

    def _rng(self):
        try:
            import torch
            base = int(torch.initial_seed())
        except ImportError:  # torch-free unit tests
            base = 0
        self._calls += 1
        return np.random.default_rng((base + self._calls) % (2 ** 32))

    def __call__(self, sample):
        rng = self._rng()
        if rng.random() > self.p:
            return sample
        feat = sample["features"].copy()
        T, F = feat.shape
        for _ in range(self.n_freq_masks):
            f = int(rng.integers(0, self.freq_mask_width + 1))
            f0 = int(rng.integers(0, max(1, F - f)))
            feat[:, f0:f0 + f] = 0.0
        for _ in range(self.n_time_masks):
            t = int(rng.integers(0, self.time_mask_width + 1))
            t0 = int(rng.integers(0, max(1, T - t)))
            feat[t0:t0 + t, :] = 0.0
        sample["features"] = feat
        return sample


class FeatureNormalizer:
    """Per-dimension z-score using statistics computed on the training split."""

    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def __call__(self, sample):
        feat = sample["features"]
        if self.mean is not None and self.std is not None:
            feat = (feat - self.mean) / (self.std + 1e-6)
        sample["features"] = feat.astype(np.float32)
        return sample


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
