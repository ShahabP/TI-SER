#!/usr/bin/env python3
"""
Train a single PsyCNN or PsyTransformer run.

Examples
--------
  # Within-corpus RAVDESS with PsyCNN (hybrid objective)
  python scripts/train.py --config configs/default.yaml --corpus ravdess --model psyCNN

  # Categorical-only baseline
  python scripts/train.py --config configs/default.yaml --corpus ravdess --model psyCNN --categorical-only

  # PsyTransformer on TESS
  python scripts/train.py --config configs/default.yaml --corpus tess --model psyTransformer

  # Combined RAVDESS+TESS
  python scripts/train.py --config configs/default.yaml --corpus combined --model psyCNN
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Make src importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import SERDataset, CombinedDataset, build_splits
from src.data.features import collate_variable_length
from src.data.prototypes import IDX_TO_EMOTION
from src.losses.hybrid_loss import HybridLoss
from src.models import PsyCNN, PsyTransformer
from src.training.trainer import Trainer
from src.utils.augment import SpecAugment, Compose, FeatureNormalizer
from src.utils.schedulers import warmup_cosine_scheduler


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device):
    name = cfg["model"]["name"].lower()
    if name == "psycnn":
        model = PsyCNN(
            input_dim=cfg["model"]["input_dim"],
            num_classes=cfg["model"]["num_classes"],
        )
    elif name == "psytransformer":
        model = PsyTransformer(
            input_dim=cfg["model"]["input_dim"],
            d_model=cfg["model"].get("d_model", 128),
            n_layers=cfg["model"].get("n_layers", 3),
            n_heads=cfg["model"].get("n_heads", 4),
            d_ff=cfg["model"].get("d_ff", 256),
            num_classes=cfg["model"]["num_classes"],
            max_len=cfg["model"].get("max_len", 174),
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return model.to(device)


def make_loaders(cfg: dict, corpus: str, seed: int):
    data_cfg = cfg["data"]
    max_frames = data_cfg.get("max_frames", 174)

    transform_train = None
    if cfg["train"].get("augmentation", True):
        transform_train = SpecAugment()

    def _load(name: str, root_key: str):
        root = data_cfg[root_key]
        if not Path(root).exists():
            raise FileNotFoundError(
                f"Dataset root not found: {root}\n"
                "Run: python scripts/download_data.py --corpus {name}"
            )
        return SERDataset(
            root=root,
            corpus=name,
            seven_class=data_cfg.get("seven_class", True),
            max_frames=max_frames,
            transform=transform_train,
        )

    if corpus == "ravdess":
        full = _load("ravdess", "ravdess_root")
        train_ds, val_ds, test_ds = build_splits(
            full,
            train_ratio=data_cfg["train_ratio"],
            val_ratio=data_cfg["val_ratio"],
            seed=seed,
        )
    elif corpus == "tess":
        full = _load("tess", "tess_root")
        train_ds, val_ds, test_ds = build_splits(
            full,
            train_ratio=data_cfg["train_ratio"],
            val_ratio=data_cfg["val_ratio"],
            seed=seed,
        )
    elif corpus == "combined":
        rav = _load("ravdess", "ravdess_root")
        tes = _load("tess", "tess_root")
        # Build splits independently then combine train/val/test
        tr1, va1, te1 = build_splits(rav, seed=seed)
        tr2, va2, te2 = build_splits(tes, seed=seed)

        class _ListDataset(torch.utils.data.Dataset):
            def __init__(self, subsets):
                self.items = []
                for s in subsets:
                    self.items.extend([s[i] for i in range(len(s))])

            def __len__(self):
                return len(self.items)

            def __getitem__(self, i):
                return self.items[i]

        train_ds = _ListDataset([tr1, tr2])
        val_ds = _ListDataset([va1, va2])
        test_ds = _ListDataset([te1, te2])
    else:
        raise ValueError(f"Unknown corpus: {corpus}")

    # Validation / test should not use augmentation
    # (Subsets share the underlying dataset which may have a transform;
    #  for simplicity we accept train-only augmentation via the transform
    #  that is already attached at construction time for the train portion.)

    bs = cfg["train"]["batch_size"]
    nw = data_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        collate_fn=collate_variable_length,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        collate_fn=collate_variable_length,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        collate_fn=collate_variable_length,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Train theory-inspired SER models")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--corpus", default="ravdess", choices=["ravdess", "tess", "combined"])
    parser.add_argument("--model", default=None, help="Override model name (psyCNN|psyTransformer)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--categorical-only", action="store_true",
                        help="Disable dimensional / phase losses (baseline)")
    parser.add_argument("--name", default=None, help="Experiment name suffix")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["name"] = args.model
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    seed = args.seed if args.seed is not None else cfg["train"]["seed"]
    set_seed(seed)

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Build data
    train_loader, val_loader, test_loader = make_loaders(cfg, args.corpus, seed)

    # Model
    model = build_model(cfg, device)

    # Loss
    loss_cfg = cfg["loss"].copy()
    if args.categorical_only:
        loss_cfg["lambda_dim"] = 0.0
        loss_cfg["lambda_int"] = 0.0
        loss_cfg["lambda_phase"] = 0.0
        exp_tag = "categorical"
    else:
        exp_tag = "hybrid"

    loss_fn = HybridLoss(
        num_classes=cfg["model"]["num_classes"],
        **{k: v for k, v in loss_cfg.items() if k != "mode"},
        mode=loss_cfg.get("mode", "va"),
    )

    # Optimizer + schedule
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

    # Experiment name
    model_name = cfg["model"]["name"]
    exp_name = args.name or f"{model_name}_{args.corpus}_{exp_tag}_seed{seed}"

    target_names = [IDX_TO_EMOTION[i] for i in range(cfg["model"]["num_classes"])]

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
        use_amp=cfg["train"].get("use_amp", False),
    )

    trainer.fit()

    # Final test evaluation with best checkpoint
    trainer.load_best()
    from src.training.metrics import evaluate_model

    test_metrics = evaluate_model(model, test_loader, device, target_names=target_names)
    print("\n=== Test set results ===")
    print(f"Accuracy : {test_metrics['accuracy']*100:.2f}%")
    print(f"Macro-F1 : {test_metrics['macro_f1']*100:.2f}%")

    import json
    with open(trainer.output_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()