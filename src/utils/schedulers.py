"""Warmup + cosine decay learning-rate schedule."""

from __future__ import annotations
import math
import torch
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_scheduler(optimizer, warmup_epochs=5, total_epochs=20, min_lr_ratio=0.01):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)
