#!/usr/bin/env python3
"""
Loss-component ablation study (Section 7).

Trains PsyCNN on RAVDESS under four settings:
  1. categorical only
  2. categorical + dimensional (no phase)
  3. categorical + dimensional + intensity
  4. full hybrid (categorical + dim + int + phase)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import SERDataset, build_splits
from src.data.features import collate_variable_length
from src.data.prototypes import IDX_TO_EMOTION
from src.losses.hybrid_loss import HybridLoss
from src.models import PsyCNN
from src.training.trainer import Trainer
from src.training.metrics import evaluate_model
from src.utils.schedulers import warmup_cosine_scheduler
from scripts.train import set_seed, load_config


ABLATIONS = {
    "categorical": dict(lambda_dim=0.0, lambda_int=0.0, lambda_phase=0.0),
    "cat_dim": dict(lambda_dim=0.5, lambda_int=0.0, lambda_phase=0.0),
    "cat_dim_int": dict(lambda_dim=0.5, lambda_int=0.3, lambda_phase=0.0),
    "full_hybrid": dict(lambda_dim=0.5, lambda_int=0.3, lambda_phase=0.2),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    data_cfg = cfg["data"]
    root = data_cfg["ravdess_root"]
    if not Path(root).exists():
        print(f"[skip] RAVDESS not found at {root}")
        return

    full = SERDataset(
        root=root,
        corpus="ravdess",
        seven_class=True,
        max_frames=data_cfg.get("max_frames", 174),
    )
    train_ds, val_ds, test_ds = build_splits(full, seed=args.seed)

    bs = cfg["train"]["batch_size"]
    nw = data_cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw,
        collate_fn=collate_variable_length, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw,
        collate_fn=collate_variable_length, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, num_workers=nw,
        collate_fn=collate_variable_length, pin_memory=True,
    )

    target_names = [IDX_TO_EMOTION[i] for i in range(7)]
    summary = {}

    for name, lambdas in ABLATIONS.items():
        print(f"\n=== Ablation: {name} ===")
        model = PsyCNN(input_dim=120, num_classes=7).to(device)
        loss_fn = HybridLoss(num_classes=7, mode="va", **lambdas)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
        )
        scheduler = warmup_cosine_scheduler(
            optimizer, warmup_epochs=3, total_epochs=args.epochs
        )

        exp_name = f"ablation_{name}_seed{args.seed}"
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            output_dir=cfg["experiment"]["output_dir"],
            experiment_name=exp_name,
            target_names=target_names,
        )
        trainer.fit()
        trainer.load_best()
        metrics = evaluate_model(model, test_loader, device, target_names=target_names)
        summary[name] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }
        print(f"  Test acc={metrics['accuracy']*100:.2f}%  macro-F1={metrics['macro_f1']*100:.2f}%")

    out = Path(cfg["experiment"]["output_dir"]) / f"ablation_summary_seed{args.seed}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAblation summary → {out}")


if __name__ == "__main__":
    main()