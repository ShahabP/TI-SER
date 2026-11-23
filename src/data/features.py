"""MFCC stack (D=120) and energy envelope for prototype modulation."""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import torch

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


def extract_mfcc_stack(
    y: np.ndarray, sr: int = 16000, n_mfcc: int = 40,
    n_fft: int = 512, hop_length: int = 160, win_length: int = 400, n_mels: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    if not _HAS_LIBROSA:
        raise ImportError("librosa is required. Install with: pip install librosa soundfile")
    if y.ndim > 1:
        y = librosa.to_mono(y)
    y = y.astype(np.float32)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
        hop_length=hop_length, win_length=win_length, n_mels=n_mels,
    )
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    stack = np.concatenate([mfcc, delta, delta2], axis=0).T.astype(np.float32)
    rms = librosa.feature.rms(y=y, frame_length=win_length, hop_length=hop_length)[0].astype(np.float32)
    T = min(stack.shape[0], rms.shape[0])
    return stack[:T], rms[:T]


def extract_proxy_features(
    y: np.ndarray, sr: int = 16000, hop_length: int = 160, win_length: int = 400,
) -> Dict[str, np.ndarray]:
    if not _HAS_LIBROSA:
        raise ImportError("librosa is required for proxy feature extraction.")
    if y.ndim > 1:
        y = librosa.to_mono(y)
    y = y.astype(np.float32)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length, n_fft=win_length)[0]
    rms = librosa.feature.rms(y=y, frame_length=win_length, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length, n_fft=win_length)[0]
    T = min(len(centroid), len(rms), len(bandwidth))
    centroid, rms, bandwidth = centroid[:T], rms[:T], bandwidth[:T]

    def _norm(x):
        mn, mx = x.min(), x.max()
        return np.zeros_like(x) if mx - mn < 1e-8 else (x - mn) / (mx - mn)

    v, a, d = _norm(centroid), _norm(rms), _norm(bandwidth)
    return {
        "valence": v.astype(np.float32), "arousal": a.astype(np.float32),
        "dominance": d.astype(np.float32), "pleasure": (0.5 * (v + a)).astype(np.float32),
        "energy": rms.astype(np.float32),
    }


def energy_envelope(energy: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mn, mx = energy.min(), energy.max()
    if mx - mn < eps:
        return np.ones_like(energy, dtype=np.float32) * 0.5
    return ((energy - mn) / (mx - mn + eps)).astype(np.float32)


def collate_variable_length(batch: list, max_len: Optional[int] = None, pad_value: float = 0.0):
    features_list = [b["features"] for b in batch]
    energy_list = [b["energy"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    lengths = [f.shape[0] for f in features_list]
    max_len = max(lengths) if max_len is None else min(max_len, max(lengths))
    B, F = len(batch), features_list[0].shape[1]
    padded_feat = torch.full((B, max_len, F), pad_value, dtype=torch.float32)
    padded_energy = torch.full((B, max_len), pad_value, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, (feat, en, L) in enumerate(zip(features_list, energy_list, lengths)):
        L = min(L, max_len)
        padded_feat[i, :L] = torch.from_numpy(feat[:L])
        padded_energy[i, :L] = torch.from_numpy(en[:L])
        mask[i, :L] = True
    return {
        "features": padded_feat, "energy": padded_energy, "mask": mask,
        "labels": labels, "lengths": torch.tensor(lengths, dtype=torch.long),
    }
