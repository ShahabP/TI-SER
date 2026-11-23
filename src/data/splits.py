"""Deterministic, stratified (or speaker-disjoint) train/val/test splits.

Pure NumPy so it can be unit-tested without torch. Splits are returned as
index arrays and can be saved to / loaded from JSON so that every compared
configuration (hybrid vs categorical, ablations, ...) uses *identical* data.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np


def _split_counts(n: int, train_ratio: float, val_ratio: float):
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    return n_train, n_val


def stratified_split(
    strata: Sequence, train_ratio: float = 0.65, val_ratio: float = 0.15, seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Split indices so that every stratum is divided ~train/val/test.

    ``strata`` holds one hashable key per sample (class label, or
    ``(corpus, label)`` for combined data). Within each stratum indices are
    shuffled with ``RandomState(seed)`` and cut by ratio.
    """
    strata = np.asarray([str(s) for s in strata])
    rng = np.random.RandomState(seed)
    parts: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    for key in sorted(set(strata.tolist())):
        idx = np.where(strata == key)[0]
        idx = idx[rng.permutation(len(idx))]
        n_tr, n_va = _split_counts(len(idx), train_ratio, val_ratio)
        parts["train"].append(idx[:n_tr])
        parts["val"].append(idx[n_tr:n_tr + n_va])
        parts["test"].append(idx[n_tr + n_va:])
    return {k: np.sort(np.concatenate(v)).astype(int) for k, v in parts.items()}


def speaker_disjoint_split(
    speakers: Sequence, train_ratio: float = 0.65, val_ratio: float = 0.15, seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Split by speaker so no speaker appears in more than one subset."""
    speakers = np.asarray([str(s) for s in speakers])
    uniq = sorted(set(speakers.tolist()))
    rng = np.random.RandomState(seed)
    order = [uniq[i] for i in rng.permutation(len(uniq))]
    n_tr, n_va = _split_counts(len(order), train_ratio, val_ratio)
    groups = {"train": order[:n_tr], "val": order[n_tr:n_tr + n_va], "test": order[n_tr + n_va:]}
    return {k: np.where(np.isin(speakers, v))[0].astype(int) for k, v in groups.items()}


def make_split(strata, speakers=None, mode: str = "stratified",
               train_ratio=0.65, val_ratio=0.15, seed=42) -> Dict[str, np.ndarray]:
    if mode == "stratified":
        return stratified_split(strata, train_ratio, val_ratio, seed)
    if mode == "speaker_disjoint":
        if speakers is None:
            raise ValueError("speaker_disjoint split needs speaker ids")
        return speaker_disjoint_split(speakers, train_ratio, val_ratio, seed)
    raise ValueError(f"unknown split mode: {mode}")


def assert_disjoint_and_complete(split: Dict[str, np.ndarray], n: int):
    seen = set()
    for name in ("train", "val", "test"):
        s = set(split[name].tolist())
        if seen & s:
            raise AssertionError(f"overlap between subsets involving '{name}'")
        seen |= s
    if seen != set(range(n)):
        raise AssertionError("split does not cover every sample exactly once")


def split_fingerprint(paths: Sequence[str], split: Dict[str, np.ndarray]) -> str:
    """SHA-256 over the sorted file lists of each subset (order independent)."""
    h = hashlib.sha256()
    for name in ("train", "val", "test"):
        h.update(name.encode())
        for p in sorted(paths[i] for i in split[name]):
            h.update(p.encode())
    return h.hexdigest()


def save_split(path: Path, paths: Sequence[str], split: Dict[str, np.ndarray], meta: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "fingerprint": split_fingerprint(paths, split),
        "indices": {k: split[k].tolist() for k in ("train", "val", "test")},
        "files": {k: [paths[i] for i in split[k]] for k in ("train", "val", "test")},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    return payload["fingerprint"]


def load_split(path: Path) -> Dict[str, np.ndarray]:
    with open(path) as f:
        payload = json.load(f)
    return {k: np.asarray(v, dtype=int) for k, v in payload["indices"].items()}
