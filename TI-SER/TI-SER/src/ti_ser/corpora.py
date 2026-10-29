"""Corpus discovery and filename -> 7-class label mapping (RAVDESS, TESS, CREMA-D)."""

import logging
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

from .compat import np
from .config import CREMAD_EMOTIONS, CREMA_MAP, EMOTIONS, RAV_MAP, TESS_MAP
from .utils import list_wavs, try_kaggle_download


def _tess_label(stem: str) -> Optional[int]:
    name = stem.lower()
    if "pleasant_surprise" in name:
        return TESS_MAP["pleasant_surprise"]
    token = name.split("_")[-1]
    if token in TESS_MAP:
        return TESS_MAP[token]
    # A few mirrors append a numeric suffix to the final emotion token.
    for key, value in TESS_MAP.items():
        if re.search(rf"(?:^|_){re.escape(key)}(?:_|\d|$)", name):
            return value
    return None


def label_from_path(path: Path | str, corpus: str) -> int:
    p = Path(path)
    stem = p.stem.lower()
    corpus = corpus.lower()
    if corpus == "tess":
        label = _tess_label(stem)
        if label is not None:
            return label
    elif corpus == "ravdess":
        pieces = stem.split("-")
        if len(pieces) >= 3:
            try:
                emotion_code = int(pieces[2])
                if emotion_code in RAV_MAP:
                    return RAV_MAP[emotion_code]
            except ValueError:
                pass
    elif corpus == "cremad":
        tokens = re.split(r"[_\-.]", stem)
        for token in tokens:
            if token in CREMA_MAP:
                return CREMA_MAP[token]
    else:
        raise ValueError(f"Unknown corpus: {corpus}")
    raise ValueError(f"Could not determine a valid {corpus} emotion label from filename: {p.name}")


def _is_ravdess_speech(path: Path) -> bool:
    """RAVDESS filenames encode vocal channel as field 2: 01=speech, 02=song."""
    pieces = path.stem.split("-")
    return len(pieces) >= 3 and pieces[1] == "01"


def corpus_dir(corpus: str, data_root: Path) -> Path:
    names = {"ravdess": "RAVDESS", "tess": "TESS", "cremad": "CREMA-D"}
    if corpus not in names:
        raise ValueError(f"Unsupported corpus: {corpus}")
    return data_root / names[corpus]


def ensure_corpus_files(corpus: str, data_root: Path, allow_kaggle: bool = True) -> Tuple[List[Path], np.ndarray]:
    root = corpus_dir(corpus, data_root)
    clean = root / "clean"
    files = list_wavs(clean) if clean.exists() else list_wavs(root)
    if not files and allow_kaggle and corpus in {"tess", "ravdess"}:
        slug = (
            "ejlok1/toronto-emotional-speech-set-tess" if corpus == "tess"
            else "uwrfkaggler/ravdess-emotional-speech-audio"
        )
        try_kaggle_download(slug, root)
        files = list_wavs(clean) if clean.exists() else list_wavs(root)

    if corpus == "ravdess":
        files = [p for p in files if _is_ravdess_speech(p)]

    valid_files: List[Path] = []
    labels: List[int] = []
    invalid: List[str] = []
    for path in files:
        try:
            label = label_from_path(path, corpus)
        except ValueError:
            invalid.append(path.name)
            continue
        valid_files.append(path)
        labels.append(label)

    if invalid:
        logging.warning("%s: skipped %d WAV(s) whose labels could not be parsed; examples=%s", corpus, len(invalid), invalid[:4])
    if not valid_files:
        expected = root / "clean"
        raise FileNotFoundError(
            f"No usable WAV files found for {corpus}. Put the corpus WAVs under {expected}."
        )
    label_array = np.asarray(labels, dtype=np.int64)
    names = EMOTIONS if corpus != "cremad" else CREMAD_EMOTIONS
    observed = sorted(set(label_array.tolist()))
    expected = list(range(len(names)))
    if observed != expected:
        logging.warning("%s does not contain all expected emotion classes. Observed=%s expected=%s", corpus, observed, expected)
    logging.info("%s: %d usable original WAVs; class counts=%s", corpus, len(valid_files), dict(Counter(label_array.tolist())))
    return valid_files, label_array


def class_names_for_corpus(corpus: str) -> List[str]:
    return CREMAD_EMOTIONS.copy() if corpus == "cremad" else EMOTIONS.copy()
