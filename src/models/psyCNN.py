"""PsyCNN – 1-D residual CNN with frame-level VA and utterance classification heads."""

from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, downsample=True, dropout=0.25):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.downsample = downsample
        self.proj = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False), nn.BatchNorm1d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        residual = self.proj(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            out = F.max_pool1d(out, 2, 2)
            residual = F.max_pool1d(residual, 2, 2)
        return self.dropout(F.relu(out + residual))


class PsyCNN(nn.Module):
    def __init__(self, input_dim=120, num_classes=7, dropout_cls=0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, 48, 5, padding=2, bias=False),
            nn.BatchNorm1d(48), nn.ReLU(inplace=True),
        )
        self.block1 = ResidualBlock1D(48, 80, downsample=True, dropout=0.25)
        self.block2 = ResidualBlock1D(80, 112, downsample=True, dropout=0.30)
        self.block3 = ResidualBlock1D(112, 160, downsample=True, dropout=0.35)
        self.va_head = nn.Sequential(
            nn.Conv1d(160, 64, 1), nn.ReLU(inplace=True), nn.Conv1d(64, 2, 1),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(160, 192), nn.ReLU(inplace=True),
            nn.Dropout(dropout_cls), nn.Linear(192, num_classes),
        )

    def forward(self, x, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        h = self.stem(x.transpose(1, 2))
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        va = self.va_head(h).transpose(1, 2)
        intensity = torch.linalg.norm(va, ord=2, dim=-1)
        embedding = h.mean(dim=-1)
        return {
            "logits": self.cls_head(embedding),
            "va": va, "intensity": intensity, "embedding": embedding,
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
