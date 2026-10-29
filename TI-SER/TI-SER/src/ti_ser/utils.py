"""Logging, dependency checks, seeding, JSON/CSV writers and file helpers."""

import csv
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .compat import np, plt, librosa, train_test_split, TSNE, StandardScaler, tf


def configure_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )


def ensure_dependencies() -> None:
    missing = []
    if librosa is None:
        missing.append("librosa")
    if tf is None:
        missing.append("tensorflow")
    if plt is None:
        missing.append("matplotlib")
    if train_test_split is None or StandardScaler is None or TSNE is None:
        missing.append("scikit-learn")
    if missing:
        raise RuntimeError(
            "Missing required dependencies: " + ", ".join(sorted(set(missing))) +
            ". Install them with: pip install numpy librosa scikit-learn matplotlib tensorflow"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if tf is not None:
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.write_text(json.dumps(obj, indent=2, default=convert), encoding="utf-8")


def write_csv(rows: Sequence[Dict[str, Any]], path: Path, fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in fieldnames})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(value, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def list_wavs(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")


def try_kaggle_download(dataset_slug: str, out_dir: Path) -> bool:
    """Optional download helper; uses argv rather than shell=True."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Attempting Kaggle download: %s", dataset_slug)
        import subprocess
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(out_dir), "--unzip"],
            check=True,
        )
        return True
    except Exception as exc:
        logging.warning("Kaggle download was not completed (%s): %s", dataset_slug, exc)
        return False


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text)).strip("_").lower()
