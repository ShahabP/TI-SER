#!/usr/bin/env python3
"""
Download helpers for RAVDESS, TESS (and CREMA-D documentation).

RAVDESS Speech subset is freely available from Zenodo.
TESS is available from the Toronto Emotional Speech Set page / Kaggle / Zenodo.
CREMA-D is not redistributed here; download it manually from the official source
(https://github.com/CheyneyComputerScience/CREMA-D or the project website) and
place the AudioWAV files under data/CREMA-D/ (or the path given by --out).

Usage
-----
  python scripts/download_data.py --corpus ravdess --out data/RAVDESS
  python scripts/download_data.py --corpus tess    --out data/TESS
  python scripts/download_data.py --corpus crema   --out data/CREMA-D
  python scripts/download_data.py --corpus all
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


RAVDESS_URL = (
    "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
)
# TESS official mirror (Kaggle requires auth; Zenodo community upload used as fallback)
TESS_ZENODO = "https://zenodo.org/records/1402523/files/TESS%20Toronto%20emotional%20speech%20set%20data.zip?download=1"


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[skip] {dest} already exists")
        return
    print(f"Downloading {url}")
    print(f"  → {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def unzip(zip_path: Path, extract_to: Path):
    print(f"Extracting {zip_path} → {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def download_ravdess(out_dir: Path):
    zip_path = out_dir / "Audio_Speech_Actors_01-24.zip"
    download_file(RAVDESS_URL, zip_path)
    unzip(zip_path, out_dir)
    print("RAVDESS ready at", out_dir)


def download_tess(out_dir: Path):
    """
    TESS download. The Zenodo link may change; if it fails the user can
    place the files manually under out_dir following the expected layout.
    """
    zip_path = out_dir / "TESS.zip"
    try:
        download_file(TESS_ZENODO, zip_path)
        unzip(zip_path, out_dir)
    except Exception as e:
        print(
            f"[warn] Automatic TESS download failed: {e}\n"
            "Please download TESS manually from\n"
            "  https://tspace.library.utoronto.ca/handle/1807/24487\n"
            "or Kaggle and place the emotion folders under",
            out_dir,
        )
    print("TESS target directory:", out_dir)


def download_crema(out_dir: Path):
    """
    CREMA-D is not auto-downloaded (licence / registration). Document the
    expected layout so the rest of the pipeline can load it.
    Expected: out_dir contains (or has a subdirectory with) *.wav files named
    like 1001_DFA_ANG_XX.wav (actor_sentence_emotion_intensity).
    Official source: https://github.com/CheyneyComputerScience/CREMA-D
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_wav = len(list(out_dir.rglob("*.wav")))
    if n_wav > 0:
        print(f"CREMA-D appears present under {out_dir} ({n_wav} wav files).")
    else:
        print(
            f"[manual] CREMA-D is not distributed by this script.\n"
            f"  1. Obtain the dataset from the official CREMA-D project\n"
            f"     (https://github.com/CheyneyComputerScience/CREMA-D).\n"
            f"  2. Place the AudioWAV (or equivalent) files under:\n"
            f"       {out_dir}\n"
            f"  File names should follow the pattern ACTOR_SENTENCE_EMO_XX.wav\n"
            f"  (e.g. 1001_DFA_ANG_XX.wav). Six emotion classes are used\n"
            f"  (angry, disgust, fear, happy, neutral, sad)."
        )
    print("CREMA-D target directory:", out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        choices=["ravdess", "tess", "crema", "all"],
        default="all",
    )
    parser.add_argument("--out", type=str, default="data")
    args = parser.parse_args()

    root = Path(args.out)
    if args.corpus in ("ravdess", "all"):
        download_ravdess(root / "RAVDESS")
    if args.corpus in ("tess", "all"):
        download_tess(root / "TESS")
    if args.corpus in ("crema", "all"):
        download_crema(root / "CREMA-D")


if __name__ == "__main__":
    main()