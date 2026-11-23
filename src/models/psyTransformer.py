"""PsyTransformer – compact Transformer with frame-level VA head."""

from __future__ import annotations
from typing import Dict, Optional
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        return x + self.pe(pos)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        return self.norm2(x + self.ffn(x))


class PsyTransformer(nn.Module):
    def __init__(
        self, input_dim=120, d_model=128, n_layers=3, n_heads=4, d_ff=256,
        num_classes=7, max_len=174, dropout=0.1, dropout_cls=0.4,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 192), nn.ReLU(inplace=True),
            nn.Dropout(dropout_cls), nn.Linear(192, num_classes),
        )
        self.va_head = nn.Linear(d_model, 2)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        h = self.input_dropout(self.pos_enc(self.input_proj(x) * math.sqrt(self.d_model)))
        key_padding_mask = ~mask if mask is not None else None
        for layer in self.layers:
            h = layer(h, key_padding_mask=key_padding_mask)
        va = self.va_head(h)
        intensity = torch.linalg.norm(va, ord=2, dim=-1)
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            embedding = (h * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
        else:
            embedding = h.mean(1)
        return {
            "logits": self.cls_head(embedding),
            "va": va, "intensity": intensity, "embedding": embedding,
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
