#!/usr/bin/env python3
"""
Leave-corpus-out protocol (Section 7.3):
  Train on the entire source corpus, evaluate on the entire target corpus.
  Two directions: RAVDESS → TESS and TESS → RAVDESS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import SERDataset
from src.data.features import collate_variable_length
from src.data.prototypes import IDX_TO_EMOTION
from src.losses.hybrid_loss import HybridLoss
from src.models import PsyCNN, PsyTransformer
from src.training.trainer import Trainer
from src.training.metrics import evaluate_model
from src.utils.schedulers import warmup_cosine_scheduler
from scripts.train import set_seed, load_config, build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="psyCNN")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["model"]["name"] = args.model
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    set_seed(args.seed)

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    data_cfg = cfg["data"]
    max_frames = data_cfg.get("max_frames", 174)

    directions = [
        ("ravdess", "tess", data_cfg["ravdess_root"], data_cfg["tess_root"]),
        ("tess", "ravdess", data_cfg["tess_root"], data_cfg["ravdess_root"]),
    ]

    target_names = [IDX_TO_EMOTION[i] for i in range(cfg["model"]["num_classes"])]
    results = {}

    for src_name, tgt_name, src_root, tgt_root in directions:
        print(f"\n=== Leave-corpus-out: train={src_name}  test={tgt_name} ===")
        if not Path(src_root).exists() or not Path(tgt_root).exists():
            print(f"[skip] missing data roots {src_root} or {tgt_root}")
            continue

        train_ds = SERDataset(
            root=src_root,
            corpus=src_name,
            seven_class=True,
            max_frames=max_frames,
        )
        # Use a small hold-out from source as validation
        from src.data.datasets import build_splits

        train_sub, val_sub, _ = build_splits(train_ds, train_ratio=0.85, val_ratio=0.15, seed=args.seed)

        test_ds = SERDataset(
            root=tgt_root,
            corpus=tgt_name,
            seven_class=True,
            max_frames=max_frames,
        )

        bs = cfg["train"]["batch_size"]
        nw = data_cfg.get("num_workers", 4)

        train_loader = DataLoader(
            train_sub, batch_size=bs, shuffle=True, num_workers=nw,
            collate_fn=collate_variable_length, pin_memory=True,
        )
        val_loader = DataLoader(
            val_sub, batch_size=bs, shuffle=False, num_workers=nw,
            collate_fn=collate_variable_length, pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=bs, shuffle=False, num_workers=nw,
            collate_fn=collate_variable_length, pin_memory=True,
        )

        model = build_model(cfg, device)
        loss_fn = HybridLoss(
            num_classes=cfg["model"]["num_classes"],
            lambda_dim=cfg["loss"]["lambda_dim"],
            lambda_int=cfg["loss"]["lambda_int"],
            lambda_phase=cfg["loss"]["lambda_phase"],
            mode=cfg["loss"].get("mode", "va"),
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
        )
        scheduler = warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=cfg["train"]["warmup_epochs"],
            total_epochs=cfg["train"]["epochs"],
        )

        exp_name = f"{args.model}_leave_{src_name}_to_{tgt_name}_seed{args.seed}"
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=cfg["train"]["epochs"],
            output_dir=cfg["experiment"]["output_dir"],
            experiment_name=exp_name,
            target_names=target_names,
        )
        trainer.fit()
        trainer.load_best()

        test_metrics = evaluate_model(model, test_loader, device, target_names=target_names)
        print(f"  Target ({tgt_name}) accuracy: {test_metrics['accuracy']*100:.2f}%")
        results[f"{src_name}_to_{tgt_name}"] = test_metrics

        with open(trainer.output_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

    # Summary
    out = Path(cfg["experiment"]["output_dir"]) / f"leave_corpus_summary_seed{args.seed}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nLeave-corpus-out summary written to {out}")


if __name__ == "__main__":
    main()