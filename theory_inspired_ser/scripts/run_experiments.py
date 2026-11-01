#!/usr/bin/env python3
"""
Orchestrate the full experimental protocol from the paper:

  1. Within-dataset (RAVDESS, TESS) – PsyCNN & PsyTransformer, hybrid + categorical
  2. Combined-dataset
  3. Leave-corpus-out (train on one, test on the other)
  4. Loss-component ablations (optional)

Results are written under results/ and summarised in results/summary.json.

Usage
-----
  # Full protocol (requires data already downloaded)
  python scripts/run_experiments.py --config configs/default.yaml

  # Only within-corpus PsyCNN on RAVDESS, 2 seeds
  python scripts/run_experiments.py --only within --models psyCNN --corpora ravdess --seeds 42,43

  # Quick smoke test with fewer epochs
  python scripts/run_experiments.py --epochs 2 --seeds 42 --only within --corpora ravdess
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]):
    print("\n>>>", " ".join(cmd))
    ret = subprocess.call(cmd, cwd=str(ROOT))
    if ret != 0:
        print(f"[error] command failed with code {ret}")
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", default="psyCNN,psyTransformer")
    parser.add_argument("--corpora", default="ravdess,tess")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--only",
        choices=["within", "combined", "leave", "ablation", "all"],
        default="all",
    )
    parser.add_argument("--skip-categorical", action="store_true")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    corpora = [c.strip() for c in args.corpora.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    python = sys.executable
    train_script = str(ROOT / "scripts" / "train.py")

    def base_cmd(model, corpus, seed, categorical=False, name=None):
        cmd = [
            python,
            train_script,
            "--config",
            args.config,
            "--model",
            model,
            "--corpus",
            corpus,
            "--seed",
            str(seed),
        ]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        if categorical:
            cmd += ["--categorical-only"]
        if name:
            cmd += ["--name", name]
        return cmd

    # ------------------------------------------------------------------
    # 1. Within-dataset
    # ------------------------------------------------------------------
    if args.only in ("within", "all"):
        print("\n" + "=" * 60)
        print("WITHIN-DATASET EXPERIMENTS")
        print("=" * 60)
        for model in models:
            for corpus in corpora:
                for seed in seeds:
                    # Hybrid
                    run(
                        base_cmd(
                            model,
                            corpus,
                            seed,
                            categorical=False,
                            name=f"{model}_{corpus}_hybrid_seed{seed}",
                        )
                    )
                    # Categorical baseline
                    if not args.skip_categorical:
                        run(
                            base_cmd(
                                model,
                                corpus,
                                seed,
                                categorical=True,
                                name=f"{model}_{corpus}_categorical_seed{seed}",
                            )
                        )

    # ------------------------------------------------------------------
    # 2. Combined-dataset
    # ------------------------------------------------------------------
    if args.only in ("combined", "all"):
        print("\n" + "=" * 60)
        print("COMBINED-DATASET EXPERIMENTS")
        print("=" * 60)
        for model in models:
            for seed in seeds:
                run(
                    base_cmd(
                        model,
                        "combined",
                        seed,
                        categorical=False,
                        name=f"{model}_combined_hybrid_seed{seed}",
                    )
                )

    # ------------------------------------------------------------------
    # 3. Leave-corpus-out (handled by a dedicated script)
    # ------------------------------------------------------------------
    if args.only in ("leave", "all"):
        print("\n" + "=" * 60)
        print("LEAVE-CORPUS-OUT EXPERIMENTS")
        print("=" * 60)
        leave_script = str(ROOT / "scripts" / "leave_corpus_out.py")
        for model in models:
            for seed in seeds:
                run(
                    [
                        python,
                        leave_script,
                        "--config",
                        args.config,
                        "--model",
                        model,
                        "--seed",
                        str(seed),
                    ]
                    + (["--epochs", str(args.epochs)] if args.epochs else [])
                )

    # ------------------------------------------------------------------
    # 4. Loss-component ablations (PsyCNN, RAVDESS, 3 seeds)
    # ------------------------------------------------------------------
    if args.only in ("ablation", "all"):
        print("\n" + "=" * 60)
        print("LOSS-COMPONENT ABLATIONS")
        print("=" * 60)
        ablation_script = str(ROOT / "scripts" / "ablation.py")
        for seed in seeds[:3]:
            run(
                [
                    python,
                    ablation_script,
                    "--config",
                    args.config,
                    "--seed",
                    str(seed),
                ]
                + (["--epochs", str(args.epochs)] if args.epochs else [])
            )

    print("\nAll requested experiments finished.")
    print("Results are under the results/ directory.")


if __name__ == "__main__":
    main()