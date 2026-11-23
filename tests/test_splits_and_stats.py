"""Torch-free checks: stratification, disjointness, speaker separation, normalisation stats."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


splits = _load("splits", "src/data/splits.py")
norm = _load("normalization", "src/data/normalization.py")


def _labels(n_per=(192, 192, 192, 192, 192, 192, 96)):
    return np.repeat(np.arange(len(n_per)), n_per)


def test_stratified_preserves_class_proportions():
    y = _labels()
    s = splits.stratified_split(y, 0.65, 0.15, seed=1)
    splits.assert_disjoint_and_complete(s, len(y))
    for name, frac in (("train", 0.65), ("val", 0.15), ("test", 0.20)):
        counts = np.bincount(y[s[name]], minlength=7)
        expected = np.array([192] * 6 + [96]) * frac
        assert np.all(np.abs(counts - expected) <= 1.5), (name, counts)


def test_not_plain_permutation():
    # With imbalanced labels a plain permutation would not keep the minority share.
    y = np.array([0] * 900 + [1] * 100)
    s = splits.stratified_split(y, 0.65, 0.15, seed=0)
    for name in ("train", "val", "test"):
        share = (y[s[name]] == 1).mean()
        assert abs(share - 0.1) < 0.01


def test_split_is_deterministic_and_seed_dependent():
    y = _labels()
    a = splits.stratified_split(y, seed=3)
    b = splits.stratified_split(y, seed=3)
    c = splits.stratified_split(y, seed=4)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["train"], c["train"])


def test_combined_strata_keep_each_corpus_balanced():
    corpus = np.array(["ravdess"] * 600 + ["tess"] * 400)
    y = np.concatenate([np.tile(np.arange(6), 100), np.tile(np.arange(4), 100)])
    strata = [f"{c}_{l}" for c, l in zip(corpus, y)]
    s = splits.stratified_split(strata, 0.65, 0.15, seed=0)
    for name in ("train", "val", "test"):
        share = (corpus[s[name]] == "tess").mean()
        assert abs(share - 0.4) < 0.02


def test_speaker_disjoint():
    spk = np.repeat([f"a{i}" for i in range(24)], 60)
    s = splits.speaker_disjoint_split(spk, 0.65, 0.15, seed=0)
    splits.assert_disjoint_and_complete(s, len(spk))
    sets = {k: set(spk[v]) for k, v in s.items()}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])


def test_fingerprint_changes_with_split():
    y = _labels()
    paths = [f"f{i}.wav" for i in range(len(y))]
    a = splits.split_fingerprint(paths, splits.stratified_split(y, seed=1))
    b = splits.split_fingerprint(paths, splits.stratified_split(y, seed=2))
    assert a != b


def test_normalisation_uses_training_data_only():
    rng = np.random.RandomState(0)
    train = [rng.randn(50, 4).astype(np.float32) for _ in range(20)]
    test = [(rng.randn(50, 4) * 10 + 5).astype(np.float32) for _ in range(5)]   # very different
    mean, std = norm.compute_train_stats(train, dim=4)
    allm, alls = norm.compute_train_stats(train + test, dim=4)
    ref = np.concatenate(train)
    assert np.allclose(mean, ref.mean(0), atol=1e-4) and np.allclose(std, ref.std(0), atol=1e-4)
    assert not np.allclose(mean, allm, atol=0.1)      # test data would have shifted the stats
