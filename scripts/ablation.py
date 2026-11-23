#!/usr/bin/env python3
"""Matched, repeated-seed loss-component ablation (Section 7).

Runs the full factorial over the auxiliary terms {dim (valence-arousal), int
(intensity), phase (phase-aware)} on top of the shared CE+focal classification
loss, plus a prototype-shuffle control:

  categorical | dim | int | phase | dim_int | dim_phase | int_phase | full | full_shuffled_proto

All variants use the SAME config/epochs as the main experiments and, per seed,
the same split, normalisation, initial weights and batch order. Use
``scripts/aggregate_results.py`` for paired statistics against ``categorical``.

  python scripts/ablation.py --models psyCNN,psyTransformer --corpora ravdess --seeds 42,43,44,45,46
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments import plan, execute, ABLATION_VARIANTS  # noqa: E402
from src.pipeline import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--models", default="psyCNN,psyTransformer")
    ap.add_argument("--corpora", default="ravdess")
    ap.add_argument("--variants", default=",".join(ABLATION_VARIANTS))
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="leave unset so ablations match the main runs (config train.epochs)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else cfg["experiment"]["seeds"]
    execute(cfg, plan("ablation", a.models.split(","), a.corpora.split(","), seeds, [],
                      variants=a.variants.split(",")),
            device=a.device, force=a.force, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
