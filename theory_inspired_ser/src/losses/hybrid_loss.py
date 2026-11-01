"""Hybrid multi-task objective: classification + VA regression + intensity + phase."""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data.prototypes import get_prototype_batch


def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    loss = alpha * (1 - pt) ** gamma * ce
    return loss.mean() if reduction == "mean" else loss.sum() if reduction == "sum" else loss


def energy_modulated_targets(labels, energy, mask, mode="va"):
    B, T = energy.shape
    device = energy.device
    protos = get_prototype_batch(labels, mode=mode, device=device)
    eta = torch.zeros_like(energy)
    for b in range(B):
        valid = mask[b]
        if valid.any():
            e = energy[b, valid]
            mn, mx = e.min(), e.max()
            eta[b, valid] = (e - mn) / (mx - mn) if mx - mn > 1e-8 else 0.5
    z_star = eta.unsqueeze(-1) * protos.unsqueeze(1)
    I_star = torch.linalg.norm(z_star, ord=2, dim=-1)
    return z_star, I_star


def extract_phase_boundaries(I_star, mask, eps=0.1, delta=0.05):
    B, T = I_star.shape
    device = I_star.device
    tau_on = torch.zeros(B, device=device)
    tau_pk = torch.zeros(B, device=device)
    tau_re = torch.ones(B, device=device)
    for b in range(B):
        valid = mask[b]
        if not valid.any():
            continue
        I = I_star[b, valid]
        n = I.shape[0]
        if n < 2:
            continue
        tau = torch.linspace(0, 1, n, device=device)
        thr = eps * I.max().clamp(min=1e-6)
        above = (I >= thr).nonzero(as_tuple=False)
        if above.numel() > 0:
            tau_on[b] = tau[above[0]]
        pk_idx = I.argmax()
        tau_pk[b] = tau[pk_idx]
        below = (I[pk_idx:] <= thr).nonzero(as_tuple=False)
        if below.numel() > 0:
            tau_re[b] = tau[pk_idx + below[0]]
    return {"onset": tau_on, "peak": tau_pk, "recovery": tau_re}


def phase_summary(z, mask, tau_s, tau_e):
    B, T, q = z.shape
    device = z.device
    mu = torch.zeros(B, q, device=device)
    delta = torch.zeros(B, q, device=device)
    sigma = torch.zeros(B, device=device)
    for b in range(B):
        valid = mask[b]
        if not valid.any():
            continue
        idx = valid.nonzero(as_tuple=False).squeeze(-1)
        n = idx.numel()
        if n < 2:
            continue
        tau = torch.linspace(0, 1, n, device=device)
        z_b = z[b, idx]
        inside = (tau >= tau_s[b]) & (tau <= tau_e[b])
        if not inside.any():
            continue
        z_ph = z_b[inside]
        mu[b] = z_ph.mean(0)
        delta[b] = z_ph[-1] - z_ph[0]
        if z_ph.shape[0] > 1:
            dz = z_ph[1:] - z_ph[:-1]
            sigma[b] = dz.pow(2).sum(-1).mean()
    return mu, delta, sigma


class HybridLoss(nn.Module):
    def __init__(
        self, num_classes=7, lambda_dim=0.5, lambda_int=0.3, lambda_phase=0.2,
        lambda_ce=1.0, lambda_focal=0.5, focal_alpha=0.25, focal_gamma=2.0,
        phase_delta=0.05, phase_eps=0.1, omega_build=1.0, omega_peak=1.0,
        omega_rec=1.0, beta_phase=0.5, gamma_phase=0.3, mode="va",
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.lambda_dim = lambda_dim
        self.lambda_int = lambda_int
        self.lambda_phase = lambda_phase
        self.lambda_ce = lambda_ce
        self.lambda_focal = lambda_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.phase_delta = phase_delta
        self.phase_eps = phase_eps
        self.omega = {"B": omega_build, "P": omega_peak, "R": omega_rec}
        self.beta_phase = beta_phase
        self.gamma_phase = gamma_phase
        self.mode = mode
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(num_classes),
        )

    def forward(self, logits, va, intensity, labels, energy, mask):
        device = logits.device
        ce = F.cross_entropy(logits, labels, weight=self.class_weights.to(device))
        fl = focal_loss(logits, labels, alpha=self.focal_alpha, gamma=self.focal_gamma)
        L_cls = self.lambda_ce * ce + self.lambda_focal * fl

        z_star_full, I_star_full = energy_modulated_targets(labels, energy, mask, mode=self.mode)
        T_prime = va.shape[1]
        z_star = F.interpolate(
            z_star_full.transpose(1, 2), size=T_prime, mode="linear", align_corners=True
        ).transpose(1, 2)
        I_star = F.interpolate(
            I_star_full.unsqueeze(1), size=T_prime, mode="linear", align_corners=True
        ).squeeze(1)
        mask_prime = F.interpolate(
            mask.float().unsqueeze(1), size=T_prime, mode="nearest"
        ).squeeze(1) > 0.5

        diff = (va - z_star) ** 2 * mask_prime.unsqueeze(-1).float()
        n_valid = mask_prime.float().sum().clamp(min=1.0)
        L_dim = diff.sum() / n_valid
        L_int = ((intensity - I_star) ** 2 * mask_prime.float()).sum() / n_valid

        boundaries = extract_phase_boundaries(
            I_star, mask_prime, eps=self.phase_eps, delta=self.phase_delta
        )
        tau_on, tau_pk, tau_re = boundaries["onset"], boundaries["peak"], boundaries["recovery"]
        intervals = {
            "B": (tau_on, tau_pk),
            "P": ((tau_pk - self.phase_delta).clamp(0), (tau_pk + self.phase_delta).clamp(max=1)),
            "R": (tau_pk, tau_re),
        }
        L_phase = torch.tensor(0.0, device=device)
        for name, (ts, te) in intervals.items():
            mu_p, d_p, s_p = phase_summary(va, mask_prime, ts, te)
            mu_r, d_r, s_r = phase_summary(z_star, mask_prime, ts, te)
            L_phase = L_phase + self.omega[name] * (
                F.mse_loss(mu_p, mu_r)
                + self.beta_phase * F.mse_loss(d_p, d_r)
                + self.gamma_phase * F.mse_loss(s_p, s_r)
            )

        total = L_cls + self.lambda_dim * L_dim + self.lambda_int * L_int + self.lambda_phase * L_phase
        return {
            "loss": total,
            "L_cls": L_cls.detach(), "L_dim": L_dim.detach(),
            "L_int": L_int.detach(), "L_phase": L_phase.detach(),
        }
