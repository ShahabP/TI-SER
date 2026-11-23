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
    """Accuracy, macro-F1 and per-class statistics.

    ``labels`` is the full label set (e.g. 0..6) so reports have a fixed shape
    even if a class is absent from a corpus (CREMA-D has no 'surprise').
    Macro-F1 averages over classes present in ``y_true``.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if not _HAS_SKLEARN:
        return _fallback_metrics(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    present = sorted(set(y_true.tolist()))
    macro_f1 = f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
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
def predict_model(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    for batch in dataloader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        out = model(features, mask=mask)["logits"].argmax(dim=-1)
        preds.extend(out.cpu().tolist())
        labels.extend(batch["labels"].tolist())
    return labels, preds


@torch.no_grad()
def evaluate_model(model, dataloader, device, target_names=None, return_predictions=False) -> Dict:
    y_true, y_pred = predict_model(model, dataloader, device)
    labels = list(range(len(target_names))) if target_names is not None else None
    metrics = compute_metrics(y_true, y_pred, labels=labels, target_names=target_names)
    if return_predictions:
        return metrics, y_true, y_pred
    return metrics
