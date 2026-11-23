#!/usr/bin/env python3
"""Train and evaluate ONE configuration (writes a run record under results/).

  python scripts/train.py --corpus ravdess --model psyCNN --variant full --seed 42
  python scripts/train.py --corpus ravdess --model psyCNN --categorical-only      # baseline
  python scripts/train.py --corpus combined --model psyTransformer
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import load_config, run_experiment, VARIANTS, VARIANT_ALIASES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Train theory-inspired SER models")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--corpus", default="ravdess", choices=["ravdess", "tess", "crema", "combined"])
    ap.add_argument("--test-corpus", default=None, choices=[None, "ravdess", "tess", "crema"],
                    help="leave-corpus-out: train on --corpus, test on this corpus")
    ap.add_argument("--model", default=None, help="psyCNN | psyTransformer")
    ap.add_argument("--variant", default="full",
                    help=f"loss variant: {sorted(VARIANTS)} (aliases: {sorted(VARIANT_ALIASES)})")
    ap.add_argument("--categorical-only", action="store_true", help="alias for --variant categorical")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--group", default="single")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
    run_experiment(
        cfg, model_name=a.model or cfg["model"]["name"],
        variant="categorical" if a.categorical_only else a.variant, corpus=a.corpus,
        test_corpus=a.test_corpus, seed=a.seed if a.seed is not None else cfg["train"]["seed"],
        group=a.group, device=a.device, force=a.force,
    )


if __name__ == "__main__":
    main()
