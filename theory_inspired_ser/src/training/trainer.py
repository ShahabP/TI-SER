"""Training loop for PsyCNN / PsyTransformer."""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from src.losses.hybrid_loss import HybridLoss
from src.training.metrics import evaluate_model


class _NullWriter:
    def add_scalar(self, *a, **k): pass
    def close(self): pass


class Trainer:
    def __init__(
        self, model, train_loader, val_loader, device, loss_fn, optimizer,
        scheduler=None, epochs=20, output_dir="results", experiment_name="run",
        target_names=None, log_interval=20, use_amp=False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_names = target_names
        self.log_interval = log_interval
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb")) if _HAS_TB else _NullWriter()
        self.best_val_acc = -1.0
        self.history = {"train_loss": [], "val_acc": [], "val_macro_f1": [], "lr": []}

    def _train_epoch(self, epoch):
        self.model.train()
        totals = {"loss": 0.0, "L_cls": 0.0, "L_dim": 0.0, "L_int": 0.0, "L_phase": 0.0}
        n = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [train]", leave=False)
        for step, batch in enumerate(pbar):
            features = batch["features"].to(self.device)
            energy = batch["energy"].to(self.device)
            mask = batch["mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                out = self.model(features, mask=mask)
                loss_dict = self.loss_fn(
                    logits=out["logits"], va=out["va"], intensity=out["intensity"],
                    labels=labels, energy=energy, mask=mask,
                )
                loss = loss_dict["loss"]
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            for k in totals:
                totals[k] += loss_dict[k].item() if k != "loss" else loss.item()
            n += 1
            if step % self.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.3f}"})
        return {k: v / max(n, 1) for k, v in totals.items()}

    @torch.no_grad()
    def _validate(self):
        return evaluate_model(self.model, self.val_loader, self.device, self.target_names)

    def fit(self):
        print(f"Training {self.model.__class__.__name__} for {self.epochs} epochs")
        print(f"  Device : {self.device}  Params : {self.model.count_parameters():,}")
        print(f"  Output : {self.output_dir}")
        for epoch in range(self.epochs):
            t0 = time.time()
            train_stats = self._train_epoch(epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
            val_metrics = self._validate()
            self.history["train_loss"].append(train_stats["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_macro_f1"].append(val_metrics["macro_f1"])
            self.history["lr"].append(lr)
            self.writer.add_scalar("train/loss", train_stats["loss"], epoch)
            self.writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
            self.writer.add_scalar("val/macro_f1", val_metrics["macro_f1"], epoch)
            print(
                f"Epoch {epoch+1:02d}/{self.epochs}  loss={train_stats['loss']:.4f}  "
                f"val_acc={val_metrics['accuracy']*100:.2f}%  "
                f"macro_f1={val_metrics['macro_f1']*100:.2f}%  "
                f"lr={lr:.2e}  ({time.time()-t0:.1f}s)"
            )
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                torch.save({
                    "epoch": epoch, "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_accuracy": self.best_val_acc, "val_metrics": val_metrics,
                }, self.output_dir / "best_model.pt")
                with open(self.output_dir / "best_metrics.json", "w") as f:
                    json.dump(val_metrics, f, indent=2)
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        self.writer.close()
        print(f"Best validation accuracy: {self.best_val_acc*100:.2f}%")
        return {"best_val_acc": self.best_val_acc, "history": self.history}

    def load_best(self):
        ckpt = torch.load(self.output_dir / "best_model.pt", map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        return ckpt
