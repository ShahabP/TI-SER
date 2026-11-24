#!/usr/bin/env python3
"""Full experimental protocol, every comparison matched on split/seed/config.

Stages
  within    RAVDESS / TESS, categorical-only vs hybrid, both backbones (primary, 30 epochs)
  combined  RAVDESS+TESS,   categorical-only vs hybrid, both backbones
  leave     RAVDESS->TESS and TESS->RAVDESS, categorical-only vs hybrid, both backbones
  ablation  loss-component ablations (paper protocol: CNN 3 seeds both corpora; Transformer 1 seed RAVDESS; 15 epochs)
  crema     CREMA-D actor-disjoint (six classes, seeds 42–44, 12 epochs, speaker-disjoint splits)

Finished runs (same config hash) are skipped, so an interrupted job can simply be restarted.

  python scripts/run_experiments.py --only all
  python scripts/run_experiments.py --only within --models psyCNN --corpora ravdess --seeds 42,43 --epochs 2
  python scripts/run_experiments.py --dry-run
"""

from __future__ import annotations
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments import plan, execute  # noqa: E402
from src.pipeline import load_config  # noqa: E402


def csv(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--models", default="psyCNN,psyTransformer")
    ap.add_argument("--corpora", default="ravdess,tess")
    ap.add_argument("--pairs", default="ravdess:tess,tess:ravdess", help="leave-corpus-out src:tgt pairs")
    ap.add_argument("--ablation-corpora", default="ravdess,tess")
    ap.add_argument("--seeds", default=None, help="default: experiment.seeds in the config")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--only", choices=["within", "combined", "leave", "ablation", "crema", "all"],
                    default="all")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
    seeds = [int(s) for s in csv(a.seeds)] if a.seeds else cfg["experiment"]["seeds"]
    models, corpora, pairs = csv(a.models), csv(a.corpora), csv(a.pairs)
    stages = (["within", "combined", "leave", "ablation", "crema"]
              if a.only == "all" else [a.only])

    for stage in stages:
        print("\n" + "=" * 70 + f"\nSTAGE: {stage}\n" + "=" * 70)
        if stage == "crema":
            # CREMA-D actor-disjoint: six classes, seeds 42–44, 12 epochs, matched splits
            ccfg = copy.deepcopy(cfg)
            ccfg["train"]["epochs"] = 12
            ccfg["data"]["split_mode"] = "speaker_disjoint"
            ccfg["data"]["seven_class"] = False  # six CREMA emotions (no surprise)
            ccfg["model"]["num_classes"] = 6
            crema_seeds = [42, 43, 44]
            execute(ccfg, plan("within", models, ["crema"], crema_seeds, []),
                    device=a.device, force=a.force, dry_run=a.dry_run)
            continue

        if stage == "ablation":
            # Paper protocol: CNN, three seeds (42–44), both RAVDESS and TESS;
            # Transformer, one seed (42), RAVDESS; 15 epochs.
            acfg = copy.deepcopy(cfg)
            acfg["train"]["epochs"] = 15
            ab_corpora = csv(a.ablation_corpora)
            # CNN on both corpora, seeds 42–44
            execute(acfg, plan("ablation", ["psyCNN"], ab_corpora, [42, 43, 44], []),
                    device=a.device, force=a.force, dry_run=a.dry_run)
            # Transformer on RAVDESS only, seed 42
            execute(acfg, plan("ablation", ["psyTransformer"], ["ravdess"], [42], []),
                    device=a.device, force=a.force, dry_run=a.dry_run)
            continue

        cs = corpora
        execute(cfg, plan(stage, models, cs, seeds, pairs), device=a.device,
                force=a.force, dry_run=a.dry_run)

    print("\nDone. Aggregate with: python scripts/aggregate_results.py")


if __name__ == "__main__":
    main()
