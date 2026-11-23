"""RAVDESS, TESS and CREMA-D dataset loaders."""

from __future__ import annotations
import os
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
    # TESS file names look like OAF_back_angry.wav / YAF_date_ps.wav: the emotion
    # is the last underscore-separated token (exact match, no loose suffix test).
    token = path.stem.lower().rsplit("_", 1)[-1]
    emo = TESS_EMOTION_MAP.get(token)
    if emo is not None and emo in EMOTION_LABELS:
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


def speaker_id(corpus: str, path: Path) -> str:
    """Speaker identifier parsed from the file name (used for speaker-disjoint splits)."""
    parts = path.stem.split("-") if corpus == "ravdess" else path.stem.split("_")
    if corpus == "ravdess":      # 03-01-05-01-02-01-12.wav -> actor 12
        return f"ravdess_{parts[6]}" if len(parts) >= 7 else "ravdess_unknown"
    if corpus == "tess":         # OAF_* / YAF_*  (two speakers)
        return f"tess_{parts[0].upper()}"
    return f"crema_{parts[0]}"   # CREMA-D: 1001_DFA_ANG_XX.wav -> actor 1001


class SERDataset(Dataset):
    """Raw (un-normalised, un-augmented) corpus.

    The dataset itself NEVER applies augmentation or normalisation. Those are
    attached per split through ``IndexedDataset`` so that validation/test data
    can never be augmented and training statistics never leak.
    """

    def __init__(
        self, root: str | Path, corpus: str = "ravdess", seven_class: bool = True,
        max_frames: Optional[int] = 174, cache_dir: Optional[str | Path] = None,
        sr: int = 16000,
    ):
        self.root = Path(root)
        self.corpus = corpus.lower()
        self.seven_class = seven_class
        self.max_frames = max_frames
        self.sr = sr
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
            samples.append({
                "path": str(path), "emotion": emotion, "label": label,
                "speaker": speaker_id(self.corpus, path), "corpus": self.corpus,
            })
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
            tmp = cache_file.with_name(f".tmp{os.getpid()}_{cache_file.name}")
            np.savez_compressed(tmp, features=features, energy=energy)
            os.replace(tmp, cache_file)   # atomic: safe with several DataLoader workers
        if self.max_frames is not None and features.shape[0] > self.max_frames:
            features, energy = features[:self.max_frames], energy[:self.max_frames]
        return {
            "features": features, "energy": energy, "label": meta["label"],
            "emotion": meta["emotion"], "path": meta["path"],
        }

    # metadata helpers used by the split code
    @property
    def labels(self) -> List[int]:
        return [m["label"] for m in self.samples]

    @property
    def speakers(self) -> List[str]:
        return [m["speaker"] for m in self.samples]

    @property
    def corpora(self) -> List[str]:
        return [m["corpus"] for m in self.samples]

    def rel_paths(self) -> List[str]:
        return [str(Path(m["path"]).relative_to(self.root)) if self.root in Path(m["path"]).parents
                else m["path"] for m in self.samples]

    def raw_features(self, idx: int):
        """Features without any transform (used for training-set statistics)."""
        return self[idx]["features"]


class MultiCorpusDataset(Dataset):
    """Concatenation of several corpora with a single global index space."""

    def __init__(self, datasets: List[SERDataset]):
        self.datasets = datasets
        self.offsets = [0]
        for d in datasets[:-1]:
            self.offsets.append(self.offsets[-1] + len(d))
        self.samples = [m for d in datasets for m in d.samples]
        self.root = datasets[0].root

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, idx: int):
        for offset, ds in zip(reversed(self.offsets), reversed(self.datasets)):
            if idx >= offset:
                return ds[idx - offset]
        raise IndexError(idx)

    @property
    def labels(self):
        return [m["label"] for m in self.samples]

    @property
    def speakers(self):
        return [m["speaker"] for m in self.samples]

    @property
    def corpora(self):
        return [m["corpus"] for m in self.samples]

    def rel_paths(self):
        out = []
        for d in self.datasets:
            out.extend(f"{d.corpus}/{p}" for p in d.rel_paths())
        return out

    def raw_features(self, idx: int):
        return self[idx]["features"]


class IndexedDataset(Dataset):
    """View on a base dataset restricted to ``indices`` with its OWN transform.

    One instance per split: the training view carries
    ``Compose([normalise, SpecAugment])`` while validation and test views carry
    ``normalise`` only (statistics from the training split).
    """

    def __init__(self, base, indices, transform=None):
        self.base = base
        self.indices = [int(i) for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        sample = self.base[self.indices[i]]
        if self.transform is not None:
            sample = self.transform(dict(sample))
        return sample
