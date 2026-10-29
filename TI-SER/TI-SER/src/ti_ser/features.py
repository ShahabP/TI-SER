"""MFCC+delta+delta-delta extraction, RMS envelope, scaling and SpecAugment-style augmentation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .compat import np, librosa, StandardScaler
from .config import D_FEATURES, MAX_FRAMES, N_MFCC, SR
from .corpora import class_names_for_corpus, label_from_path
from .state import _RAW_FEATURE_CACHE


def _cache_key(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}"


def extract_mfcc_and_rms(path: Path | str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return padded MFCC+delta+delta-delta features, RMS target proxy, valid length."""
    if librosa is None:
        raise RuntimeError("librosa is required for audio feature extraction")
    p = Path(path)
    key = _cache_key(p)
    if key in _RAW_FEATURE_CACHE:
        x, e, length = _RAW_FEATURE_CACHE[key]
        return x.copy(), e.copy(), int(length)

    audio, sr = librosa.load(str(p), sr=SR, mono=True)
    if audio.size < 32:
        raise ValueError(f"Audio is empty or too short: {p}")
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
    # Choose a legal odd delta width for unusually short recordings.
    width = min(9, mfcc.shape[1] if mfcc.shape[1] % 2 == 1 else mfcc.shape[1] - 1)
    if width < 3:
        d1 = np.gradient(mfcc, axis=1)
        d2 = np.gradient(d1, axis=1)
    else:
        d1 = librosa.feature.delta(mfcc, width=width, mode="nearest")
        d2 = librosa.feature.delta(mfcc, order=2, width=width, mode="nearest")
    features = np.concatenate([mfcc, d1, d2], axis=0).T.astype(np.float32)
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512, center=True)[0].astype(np.float32)
    common = min(features.shape[0], rms.shape[0])
    features = features[:common]
    rms = rms[:common]
    length = min(common, MAX_FRAMES)
    out_x = np.zeros((MAX_FRAMES, D_FEATURES), dtype=np.float32)
    out_e = np.zeros((MAX_FRAMES,), dtype=np.float32)
    out_x[:length] = features[:length]
    out_e[:length] = rms[:length]
    _RAW_FEATURE_CACHE[key] = (out_x.copy(), out_e.copy(), length)
    return out_x, out_e, length


def load_partition(files: Sequence[Path], corpus: str) -> Dict[str, Any]:
    feats, energies, lengths, labels = [], [], [], []
    kept_files: List[Path] = []
    for path in files:
        try:
            x, energy, length = extract_mfcc_and_rms(path)
            if length < 3:
                raise ValueError("fewer than three valid feature frames")
            feats.append(x)
            energies.append(energy)
            lengths.append(length)
            labels.append(label_from_path(path, corpus))
            kept_files.append(path)
        except Exception as exc:
            logging.warning("Skipping unreadable audio %s: %s", path, exc)
    if not feats:
        raise RuntimeError(f"No usable features could be extracted for corpus {corpus}")
    return {
        "X": np.stack(feats).astype(np.float32),
        "energy": np.stack(energies).astype(np.float32),
        "lengths": np.asarray(lengths, dtype=np.int32),
        "y": np.asarray(labels, dtype=np.int64),
        "files": kept_files,
        "class_names": class_names_for_corpus(corpus),
        "corpus": corpus,
    }


def spec_augment(x: np.ndarray, valid_length: int, rng: np.random.Generator) -> np.ndarray:
    """Apply small time/frequency masks to one scaled training utterance only."""
    out = x.copy()
    valid_length = int(max(1, min(valid_length, out.shape[0])))
    t, d = out.shape
    for _ in range(2):
        max_tw = max(1, int(0.08 * valid_length))
        tw = int(rng.integers(0, max_tw + 1))
        if tw:
            start = int(rng.integers(0, max(1, valid_length - tw + 1)))
            out[start:start + tw, :] = 0.0
        max_fw = max(1, int(0.08 * d))
        fw = int(rng.integers(0, max_fw + 1))
        if fw:
            start = int(rng.integers(0, max(1, d - fw + 1)))
            out[:valid_length, start:start + fw] = 0.0
    # Keep padding frames at exactly zero after standardization/augmentation.
    out[valid_length:, :] = 0.0
    return out


def fit_scaler_on_valid_train_frames(partitions: Sequence[Dict[str, Any]]) -> StandardScaler:
    chunks = []
    for part in partitions:
        for x, length in zip(part["X"], part["lengths"]):
            if length > 0:
                chunks.append(x[:int(length)])
    if not chunks:
        raise RuntimeError("Cannot fit scaler: no valid training frames")
    scaler = StandardScaler()
    scaler.fit(np.concatenate(chunks, axis=0))
    return scaler


def transform_partition(part: Dict[str, Any], scaler: StandardScaler) -> Dict[str, Any]:
    out = dict(part)
    transformed = np.zeros_like(part["X"], dtype=np.float32)
    for i, (x, length) in enumerate(zip(part["X"], part["lengths"])):
        length = int(length)
        if length:
            transformed[i, :length] = scaler.transform(x[:length]).astype(np.float32)
    out["X"] = transformed
    return out


def augment_training_partition(part: Dict[str, Any], times: int, seed: int) -> Dict[str, Any]:
    if times <= 0 or len(part["X"]) == 0:
        return part
    rng = np.random.default_rng(seed)
    xs = [part["X"]]
    ys = [part["y"]]
    lengths = [part["lengths"]]
    energies = [part["energy"]]
    files = list(part["files"])
    for _ in range(times):
        aug = np.stack([
            spec_augment(x, int(length), rng)
            for x, length in zip(part["X"], part["lengths"])
        ]).astype(np.float32)
        xs.append(aug)
        ys.append(part["y"].copy())
        lengths.append(part["lengths"].copy())
        # Targets are based on the source utterance's observed RMS envelope.
        energies.append(part["energy"].copy())
        files.extend(part["files"])
    result = dict(part)
    result.update(
        X=np.concatenate(xs, axis=0),
        y=np.concatenate(ys, axis=0),
        lengths=np.concatenate(lengths, axis=0),
        energy=np.concatenate(energies, axis=0),
        files=files,
        original_train_count=len(part["X"]),
    )
    return result
