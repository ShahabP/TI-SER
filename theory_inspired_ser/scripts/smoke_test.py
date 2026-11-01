#!/usr/bin/env python3
"""
End-to-end smoke test with synthetic data (no real audio required).

Verifies:
  - feature shapes
  - model forward passes
  - hybrid loss computation
  - one training step for both backbones
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.features import collate_variable_length, energy_envelope
from src.data.prototypes import get_prototype_batch, EMOTION_LABELS
from src.losses.hybrid_loss import HybridLoss
from src.models import PsyCNN, PsyTransformer


class SyntheticSER(Dataset):
    def __init__(self, n=32, max_t=80, f=120, n_classes=7):
        self.n = n
        self.max_t = max_t
        self.f = f
        self.n_classes = n_classes
        rng = np.random.RandomState(0)
        self.data = []
        for i in range(n):
            T = rng.randint(30, max_t + 1)
            feat = rng.randn(T, f).astype(np.float32)
            energy = np.abs(rng.randn(T)).astype(np.float32)
            label = int(i % n_classes)
            self.data.append({"features": feat, "energy": energy, "label": label})

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.data[i]


def main():
    print("=== Smoke test: theory-inspired SER framework ===\n")
    device = torch.device("cpu")
    ds = SyntheticSER()
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_variable_length)

    batch = next(iter(loader))
    features = batch["features"]
    energy = batch["energy"]
    mask = batch["mask"]
    labels = batch["labels"]
    print(f"Batch features : {features.shape}")
    print(f"Batch energy   : {energy.shape}")
    print(f"Batch mask     : {mask.shape}")
    print(f"Batch labels   : {labels.tolist()}")

    # Prototypes
    protos = get_prototype_batch(labels, mode="va")
    print(f"VA prototypes  : {protos.shape}  e.g. {protos[0].tolist()}")

    loss_fn = HybridLoss(num_classes=7, lambda_dim=0.5, lambda_int=0.3, lambda_phase=0.2)

    for name, model in [
        ("PsyCNN", PsyCNN(input_dim=120, num_classes=7)),
        ("PsyTransformer", PsyTransformer(input_dim=120, num_classes=7, max_len=80)),
    ]:
        print(f"\n--- {name} ({model.count_parameters():,} params) ---")
        model = model.to(device)
        model.train()
        out = model(features, mask=mask)
        print(f"  logits    : {out['logits'].shape}")
        print(f"  va        : {out['va'].shape}")
        print(f"  intensity : {out['intensity'].shape}")

        loss_dict = loss_fn(
            logits=out["logits"],
            va=out["va"],
            intensity=out["intensity"],
            labels=labels,
            energy=energy,
            mask=mask,
        )
        print(
            f"  loss={loss_dict['loss'].item():.4f}  "
            f"cls={loss_dict['L_cls'].item():.4f}  "
            f"dim={loss_dict['L_dim'].item():.4f}  "
            f"int={loss_dict['L_int'].item():.4f}  "
            f"phase={loss_dict['L_phase'].item():.4f}"
        )

        # One optimizer step
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_dict["loss"].backward()
        opt.step()
        print("  backward + step OK")

    print("\n=== Smoke test passed ===")


if __name__ == "__main__":
    main()