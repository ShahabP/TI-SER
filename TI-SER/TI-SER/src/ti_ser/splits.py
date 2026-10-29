"""Original-file splits, leakage guards, combined and leave-corpus-out data preparation."""

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .compat import np, GroupShuffleSplit, train_test_split
from .config import AUGMENT_TIMES, EMOTIONS, VA_PROTOTYPES
from .corpora import class_names_for_corpus, ensure_corpus_files
from .features import augment_training_partition, fit_scaler_on_valid_train_frames, load_partition, transform_partition
from .state import SPLIT_MANIFEST


def speaker_id(path: Path | str, corpus: str) -> str:
    p = Path(path)
    if corpus == "ravdess":
        return p.parent.name if p.parent.name.lower().startswith("actor_") else "unknown"
    if corpus == "tess":
        prefix = p.stem.split("_")[0].upper()
        return prefix if prefix in {"OAF", "YAF"} else "unknown"
    if corpus == "cremad":
        return p.stem.split("_")[0]
    return "unknown"


def _standard_file_split(files: Sequence[Path], labels: np.ndarray, seed: int,
                         test_size: float = 0.20, val_size: float = 0.15) -> Tuple[List[int], List[int], List[int]]:
    idx = np.arange(len(files))
    if len(idx) < 10:
        raise RuntimeError("Too few original utterances for train/validation/test split")
    counts = Counter(labels.tolist())
    if min(counts.values()) < 3:
        raise RuntimeError(f"Stratified train/val/test split requires at least 3 samples per class; counts={dict(counts)}")
    trainval_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=labels, random_state=seed
    )
    relative_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=relative_val,
        stratify=labels[trainval_idx],
        random_state=seed,
    )
    return sorted(train_idx.tolist()), sorted(val_idx.tolist()), sorted(test_idx.tolist())


def _speaker_disjoint_split(files: Sequence[Path], labels: np.ndarray, corpus: str, seed: int,
                            test_size: float = 0.20, val_size: float = 0.15) -> Tuple[List[int], List[int], List[int]]:
    groups = np.asarray([speaker_id(f, corpus) for f in files])
    if len(set(groups)) < 5:
        raise RuntimeError(f"Speaker-disjoint train/validation/test split is not reliable for {corpus}; only {len(set(groups))} speaker groups found")
    all_classes = set(range(len(class_names_for_corpus(corpus))))
    first = None
    for offset in range(100):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + offset)
        trainval_idx, test_idx = next(splitter.split(np.zeros(len(files)), labels, groups))
        if set(labels[test_idx].tolist()) != all_classes or set(groups[trainval_idx]).intersection(groups[test_idx]):
            continue
        second = GroupShuffleSplit(n_splits=1, test_size=val_size / (1.0 - test_size), random_state=seed + 1000 + offset)
        train_rel, val_rel = next(second.split(np.zeros(len(trainval_idx)), labels[trainval_idx], groups[trainval_idx]))
        train_idx, val_idx = trainval_idx[train_rel], trainval_idx[val_rel]
        if set(labels[train_idx].tolist()) == all_classes and set(labels[val_idx].tolist()) == all_classes:
            first = (sorted(train_idx.tolist()), sorted(val_idx.tolist()), sorted(test_idx.tolist()))
            break
    if first is None:
        raise RuntimeError(
            f"Could not find a speaker-disjoint split with every class in train/val/test for {corpus}. "
            "Check the corpus organisation and consider reporting the limitation explicitly."
        )
    return first


def _record_split(corpus: str, seed: int, strategy: str, splits: Dict[str, Sequence[Path]], labels_by_path: Dict[str, int]) -> None:
    key = (corpus, int(seed), strategy)
    if key in SPLIT_MANIFEST:
        return
    rows = []
    for split_name, files in splits.items():
        for f in files:
            rows.append({"corpus": corpus, "seed": int(seed), "split_strategy": strategy,
                         "split": split_name, "file": str(f), "label": int(labels_by_path[str(f)])})
    SPLIT_MANIFEST[key] = rows


def check_disjoint(files_train: Sequence[Path], files_val: Sequence[Path], files_test: Sequence[Path]) -> None:
    sets = [set(map(str, xs)) for xs in (files_train, files_val, files_test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("Leakage guard failed: original file paths overlap between splits")


def _raw_splits(files: Sequence[Path], labels: np.ndarray, corpus: str, seed: int,
                split_strategy: str = "utterance") -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if split_strategy == "speaker":
        train_idx, val_idx, test_idx = _speaker_disjoint_split(files, labels, corpus, seed)
    else:
        train_idx, val_idx, test_idx = _standard_file_split(files, labels, seed)
    file_arrays = {
        "train": [files[i] for i in train_idx],
        "val": [files[i] for i in val_idx],
        "test": [files[i] for i in test_idx],
    }
    check_disjoint(file_arrays["train"], file_arrays["val"], file_arrays["test"])
    label_by_path = {str(p): int(y) for p, y in zip(files, labels)}
    _record_split(corpus, seed, split_strategy, file_arrays, label_by_path)
    return tuple(load_partition(file_arrays[name], corpus) for name in ("train", "val", "test"))  # type: ignore[return-value]


def _finalize_partitions(train: Dict[str, Any], val: Dict[str, Any], test: Dict[str, Any],
                         prototypes: np.ndarray, seed: int, augment: bool = True) -> Dict[str, Any]:
    scaler = fit_scaler_on_valid_train_frames([train])
    train = transform_partition(train, scaler)
    val = transform_partition(val, scaler)
    test = transform_partition(test, scaler)
    train = augment_training_partition(train, AUGMENT_TIMES if augment else 0, seed)
    out = {"corpus": train.get("corpus", "unknown"), "prototypes": prototypes.copy(),
           "class_names": train["class_names"], "scaler": scaler}
    for split, part in (("train", train), ("val", val), ("test", test)):
        for key in ("X", "y", "lengths", "energy", "files"):
            out[f"{key}_{split}"] = part[key]
    logging.info("Prepared %s: originals train/val/test=%d/%d/%d; augmented train=%d",
                 out["corpus"], train.get("original_train_count", len(train["X_train"]) if "X_train" in train else len(train["X"])),
                 len(val["X"]), len(test["X"]), len(train["X"]))
    return out


def prepare_corpus(corpus: str, data_root: Path, seed: int, prototypes: Optional[np.ndarray] = None,
                   split_strategy: str = "utterance") -> Dict[str, Any]:
    files, labels = ensure_corpus_files(corpus, data_root)
    train_raw, val_raw, test_raw = _raw_splits(files, labels, corpus, seed, split_strategy)
    proto = prototypes.copy() if prototypes is not None else VA_PROTOTYPES.copy()
    return _finalize_partitions(train_raw, val_raw, test_raw, proto, seed)


def _concat_parts(parts: Sequence[Dict[str, Any]], corpus_name: str, class_names: List[str]) -> Dict[str, Any]:
    return {
        "X": np.concatenate([p["X"] for p in parts], axis=0),
        "y": np.concatenate([p["y"] for p in parts], axis=0),
        "lengths": np.concatenate([p["lengths"] for p in parts], axis=0),
        "energy": np.concatenate([p["energy"] for p in parts], axis=0),
        "files": sum([p["files"] for p in parts], []),
        "corpus_codes": np.concatenate([
            np.full(len(p["X"]), 0 if p["corpus"] == "ravdess" else 1, dtype=np.int8)
            for p in parts
        ]),
        "class_names": class_names,
        "corpus": corpus_name,
    }


def prepare_combined(data_root: Path, seed: int, prototypes: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Split each corpus by original file, fit one shared scaler, then augment train only."""
    raw_by_corpus: Dict[str, Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = {}
    for corpus in ("ravdess", "tess"):
        files, labels = ensure_corpus_files(corpus, data_root)
        raw_by_corpus[corpus] = _raw_splits(files, labels, corpus, seed, "utterance")

    train_parts = [raw_by_corpus[c][0] for c in ("ravdess", "tess")]
    scaler = fit_scaler_on_valid_train_frames(train_parts)
    transformed: Dict[str, List[Dict[str, Any]]] = {s: [] for s in ("train", "val", "test")}
    files_manifest = []
    for corpus in ("ravdess", "tess"):
        for split_name, part in zip(("train", "val", "test"), raw_by_corpus[corpus]):
            transformed[split_name].append(transform_partition(part, scaler))
    class_names = EMOTIONS.copy()
    merged = {s: _concat_parts(transformed[s], "combined", class_names) for s in transformed}
    train = augment_training_partition(merged["train"], AUGMENT_TIMES, seed)
    proto = prototypes.copy() if prototypes is not None else VA_PROTOTYPES.copy()
    out: Dict[str, Any] = {"corpus": "combined", "class_names": class_names, "prototypes": proto, "scaler": scaler}
    for split, part in (("train", train), ("val", merged["val"]), ("test", merged["test"])):
        for key in ("X", "y", "lengths", "energy", "files"):
            out[f"{key}_{split}"] = part[key]
    out["corpus_codes_test"] = merged["test"]["corpus_codes"]
    logging.info("Prepared combined dataset: originals train/val/test=%d/%d/%d; augmented train=%d",
                 len(merged["train"]["X"]), len(merged["val"]["X"]), len(merged["test"]["X"]), len(train["X"]))
    return out


def prepare_leave_corpus_out(source: str, target: str, data_root: Path, seed: int,
                             prototypes: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Train/validation split source files; fit scaler on source train; test full target zero-shot."""
    source_files, source_labels = ensure_corpus_files(source, data_root)
    idx = np.arange(len(source_files))
    if min(Counter(source_labels.tolist()).values()) < 2:
        raise RuntimeError(f"Not enough samples per class in source corpus {source}")
    train_idx, val_idx = train_test_split(idx, test_size=0.15, stratify=source_labels, random_state=seed)
    src_train_files = [source_files[i] for i in sorted(train_idx)]
    src_val_files = [source_files[i] for i in sorted(val_idx)]
    src_train = load_partition(src_train_files, source)
    src_val = load_partition(src_val_files, source)
    tgt_files, tgt_labels = ensure_corpus_files(target, data_root)
    tgt_test = load_partition(tgt_files, target)
    source_label_map = {str(p): int(y) for p, y in zip(source_files, source_labels)}
    target_label_map = {str(p): int(y) for p, y in zip(tgt_files, tgt_labels)}
    _record_split(source, seed, f"transfer_source_to_{target}",
                  {"train": src_train_files, "val": src_val_files, "test_target_zero_shot": tgt_files},
                  {**source_label_map, **target_label_map})
    # Target is a fixed zero-shot evaluation set; it is never used in fitting.
    scaler = fit_scaler_on_valid_train_frames([src_train])
    src_train = transform_partition(src_train, scaler)
    src_val = transform_partition(src_val, scaler)
    tgt_test = transform_partition(tgt_test, scaler)
    src_train = augment_training_partition(src_train, AUGMENT_TIMES, seed)
    proto = prototypes.copy() if prototypes is not None else VA_PROTOTYPES.copy()
    out: Dict[str, Any] = {"corpus": f"{source}_to_{target}", "class_names": EMOTIONS.copy(), "prototypes": proto, "scaler": scaler}
    for split, part in (("train", src_train), ("val", src_val), ("test", tgt_test)):
        for key in ("X", "y", "lengths", "energy", "files"):
            out[f"{key}_{split}"] = part[key]
    logging.info("Prepared zero-shot %s→%s: source train/val=%d/%d, target test=%d",
                 source, target, len(src_train["X"]), len(src_val["X"]), len(tgt_test["X"]))
    return out
