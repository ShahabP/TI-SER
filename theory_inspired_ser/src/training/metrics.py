"""Evaluation metrics for categorical SER."""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import torch

try:
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, precision_recall_fscore_support,
    )
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _fallback_metrics(y_true, y_pred):
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return {
        "accuracy": acc, "macro_f1": acc, "weighted_f1": acc,
        "per_class_precision": [], "per_class_recall": [], "per_class_f1": [],
        "support": [], "classification_report": {}, "confusion_matrix": [],
    }


def compute_metrics(y_true, y_pred, labels=None, target_names=None) -> Dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if not _HAS_SKLEARN:
        return _fallback_metrics(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names,
        zero_division=0, output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(acc), "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1),
        "per_class_precision": precision.tolist(), "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(), "support": support.tolist(),
        "classification_report": report, "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def evaluate_model(model, dataloader, device, target_names=None) -> Dict:
    model.eval()
    all_preds, all_labels = [], []
    for batch in dataloader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)
        preds = model(features, mask=mask)["logits"].argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return compute_metrics(all_labels, all_preds, target_names=target_names)
