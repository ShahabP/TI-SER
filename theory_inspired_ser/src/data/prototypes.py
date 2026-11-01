"""Weak VA / PAD prototype targets (theory-derived engineering anchors)."""

from __future__ import annotations
from typing import Dict, Tuple
import torch

EMOTION_LABELS = {
    "angry": 0, "happy": 1, "sad": 2, "neutral": 3,
    "fear": 4, "disgust": 5, "surprise": 6,
}
IDX_TO_EMOTION = {v: k for k, v in EMOTION_LABELS.items()}

PROTOTYPE_VA: Dict[str, Tuple[float, float]] = {
    "angry": (-0.5, 0.7), "happy": (0.8, 0.7), "sad": (-0.7, 0.3),
    "neutral": (0.0, 0.5), "fear": (-0.6, 0.8), "disgust": (-0.6, 0.6),
    "surprise": (0.5, 0.9),
}

PROTOTYPE_PAD: Dict[str, Tuple[float, float, float]] = {
    "angry": (-0.5, 0.7, 0.6), "happy": (0.8, 0.7, 0.7), "sad": (-0.7, 0.3, 0.3),
    "neutral": (0.0, 0.5, 0.5), "fear": (-0.6, 0.8, 0.4), "disgust": (-0.6, 0.6, 0.4),
    "surprise": (0.5, 0.9, 0.6),
}


def get_prototype_va(label: int | str, device: torch.device | None = None) -> torch.Tensor:
    name = IDX_TO_EMOTION[label] if isinstance(label, int) else label.lower()
    t = torch.tensor(PROTOTYPE_VA[name], dtype=torch.float32)
    return t.to(device) if device is not None else t


def get_prototype_pad(label: int | str, device: torch.device | None = None) -> torch.Tensor:
    name = IDX_TO_EMOTION[label] if isinstance(label, int) else label.lower()
    t = torch.tensor(PROTOTYPE_PAD[name], dtype=torch.float32)
    return t.to(device) if device is not None else t


def get_prototype_batch(
    labels: torch.Tensor, mode: str = "va", device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = labels.device
    fn = get_prototype_va if mode == "va" else get_prototype_pad
    return torch.stack([fn(int(l.item()), device) for l in labels])
