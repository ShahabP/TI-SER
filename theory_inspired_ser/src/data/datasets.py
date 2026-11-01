"""RAVDESS, TESS and CREMA-D dataset loaders."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from .features import extract_mfcc_stack
from .prototypes import EMOTION_LABELS

RAVDESS_EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fear", "07": "disgust", "08": "surprise",
}
TESS_EMOTION_MAP = {
    "angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy",
    "neutral": "neutral", "ps": "surprise", "sad": "sad",
}
CREMA_EMOTION_MAP = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}


def parse_ravdess(path: Path, seven_class: bool = True) -> Optional[Tuple[str, int]]:
    parts = path.stem.split("-")
    if len(parts) < 3:
        return None
    emotion = RAVDESS_EMOTION_MAP.get(parts[2])
    if emotion is None:
        return None
    if seven_class and emotion == "calm":
        emotion = "neutral"
    if emotion not in EMOTION_LABELS:
        return None
    return emotion, EMOTION_LABELS[emotion]


def parse_tess(path: Path) -> Optional[Tuple[str, int]]:
    name = path.stem.lower()
    for key, emo in TESS_EMOTION_MAP.items():
        if name.endswith("_" + key) or name.endswith(key):
            if emo in EMOTION_LABELS:
                return emo, EMOTION_LABELS[emo]
    return None


def parse_crema(path: Path) -> Optional[Tuple[str, int]]:
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    emotion = CREMA_EMOTION_MAP.get(parts[2])
    if emotion is None or emotion not in EMOTION_LABELS:
        return None
    return emotion, EMOTION_LABELS[emotion]


class SERDataset(Dataset):
    def __init__(
        self, root: str | Path, corpus: str = "ravdess", seven_class: bool = True,
        max_frames: Optional[int] = 174, cache_dir: Optional[str | Path] = None,
        sr: int = 16000, transform=None,
    ):
        self.root = Path(root)
        self.corpus = corpus.lower()
        self.seven_class = seven_class
        self.max_frames = max_frames
        self.sr = sr
        self.transform = transform
        if cache_dir is None:
            cache_dir = self.root / f".cache_{corpus}"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.samples = self._index()
        if not self.samples:
            raise RuntimeError(f"No valid audio under {self.root} for corpus={corpus}.")

    def _index(self) -> List[Dict]:
        parser = {
            "ravdess": lambda p: parse_ravdess(p, self.seven_class),
            "tess": parse_tess, "crema": parse_crema,
        }[self.corpus]
        files = []
        for pat in ("**/*.wav", "**/*.mp3", "**/*.flac"):
            files.extend(self.root.glob(pat))
        samples = []
        for path in sorted(set(files)):
            parsed = parser(path)
            if parsed is None:
                continue
            emotion, label = parsed
            samples.append({"path": str(path), "emotion": emotion, "label": label})
        return samples

    def __len__(self):
        return len(self.samples)

    def _cache_path(self, idx: int) -> Path:
        return self.cache_dir / f"{Path(self.samples[idx]['path']).stem}.npz"

    def __getitem__(self, idx: int) -> Dict:
        meta = self.samples[idx]
        cache_file = self._cache_path(idx)
        if cache_file.exists():
            data = np.load(cache_file)
            features, energy = data["features"], data["energy"]
        else:
            import librosa
            y, _ = librosa.load(meta["path"], sr=self.sr, mono=True)
            features, energy = extract_mfcc_stack(y, sr=self.sr)
            np.savez_compressed(cache_file, features=features, energy=energy)
        if self.max_frames is not None and features.shape[0] > self.max_frames:
            features, energy = features[:self.max_frames], energy[:self.max_frames]
        sample = {
            "features": features, "energy": energy, "label": meta["label"],
            "emotion": meta["emotion"], "path": meta["path"],
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def build_splits(dataset: SERDataset, train_ratio=0.65, val_ratio=0.15, seed=42):
    from torch.utils.data import Subset
    n = len(dataset)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        Subset(dataset, indices[:n_train].tolist()),
        Subset(dataset, indices[n_train:n_train + n_val].tolist()),
        Subset(dataset, indices[n_train + n_val:].tolist()),
    )


class CombinedDataset(Dataset):
    def __init__(self, datasets: List[SERDataset]):
        self.datasets = datasets
        self.offsets = [0]
        for d in datasets[:-1]:
            self.offsets.append(self.offsets[-1] + len(d))

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, idx: int):
        for offset, ds in zip(self.offsets, self.datasets):
            if idx < offset + len(ds):
                return ds[idx - offset]
        raise IndexError(idx)
