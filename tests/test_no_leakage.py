"""Pipeline-level leakage checks (need torch, no audio): augmentation and
normalisation are attached per split, never to the shared dataset."""
import inspect
import numpy as np
import pytest

pytest.importorskip("torch")

from src.data.datasets import SERDataset, IndexedDataset  # noqa: E402
from src.data.normalization import compute_train_stats  # noqa: E402
from src.utils.augment import SpecAugment, FeatureNormalizer, Compose  # noqa: E402


class FakeBase:
    def __init__(self, n=30, T=60, F=8):
        rng = np.random.RandomState(0)
        self.items = [{"features": (rng.randn(T, F) + 3).astype(np.float32),
                       "energy": np.abs(rng.randn(T)).astype(np.float32),
                       "label": i % 3, "emotion": "x", "path": f"{i}.wav"} for i in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.items[i].items()}


def test_dataset_has_no_transform_argument():
    assert "transform" not in inspect.signature(SERDataset.__init__).parameters


def test_augmentation_only_in_train_view():
    base = FakeBase()
    train_idx, val_idx = list(range(0, 20)), list(range(20, 30))
    mean, std = compute_train_stats((base[i]["features"] for i in train_idx), dim=8)
    norm = FeatureNormalizer(mean, std)
    train = IndexedDataset(base, train_idx, Compose([norm, SpecAugment(p=1.0)]))
    val = IndexedDataset(base, val_idx, norm)
    # Training samples are masked (contain exact zeros after z-scoring); val never are.
    assert any((train[i]["features"] == 0).any() for i in range(len(train)))
    for i in range(len(val)):
        expected = (base[val_idx[i]]["features"] - mean) / (std + 1e-6)
        assert np.allclose(val[i]["features"], expected, atol=1e-5)
    # Reading the training view must not mutate the shared base data.
    assert not (base.items[0]["features"] == 0).any()


def test_val_stats_not_used_for_normalisation():
    base = FakeBase()
    mean_train, _ = compute_train_stats((base[i]["features"] for i in range(20)), dim=8)
    mean_all, _ = compute_train_stats((base[i]["features"] for i in range(30)), dim=8)
    assert not np.allclose(mean_train, mean_all)
