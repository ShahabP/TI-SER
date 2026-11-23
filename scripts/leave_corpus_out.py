#!/usr/bin/env python3
"""Leave-corpus-out with MATCHED baselines (Section 7.3).

For every (model, direction, seed) the categorical-only baseline and the hybrid
model use the identical source train/val split, training-only normalisation,
initial weights, mini-batch order and training budget. The whole target corpus
is the test set.

  python scripts/leave_corpus_out.py --models psyCNN,psyTransformer --seeds 42,43,44,45,46
  python scripts/leave_corpus_out.py --pairs ravdess:crema,tess:crema   # extra corpora
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments import plan, execute  # noqa: E402
from src.pipeline import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--models", default="psyCNN,psyTransformer")
    ap.add_argument("--pairs", default="ravdess:tess,tess:ravdess")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else cfg["experiment"]["seeds"]
    execute(cfg, plan("leave", a.models.split(","), [], seeds, a.pairs.split(",")),
            device=a.device, force=a.force, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
