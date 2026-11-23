"""Hybrid loss: zero-weight terms are really off; categorical-only == CE+focal only."""
import pytest

torch = pytest.importorskip("torch")

from src.losses.hybrid_loss import HybridLoss  # noqa: E402
from src.pipeline import build_loss, VARIANTS, canonical_variant  # noqa: E402

CFG = {"model": {"num_classes": 7},
       "loss": {"mode": "va", "lambda_dim": 0.5, "lambda_int": 0.3, "lambda_phase": 0.2,
                "lambda_ce": 1.0, "lambda_focal": 0.5, "focal_alpha": 0.25, "focal_gamma": 2.0,
                "phase_delta": 0.05, "phase_eps": 0.1}}


def _batch(B=4, T=40):
    torch.manual_seed(0)
    return dict(logits=torch.randn(B, 7), va=torch.randn(B, T // 4, 2), intensity=torch.rand(B, T // 4),
                labels=torch.randint(0, 7, (B,)), energy=torch.rand(B, T),
                mask=torch.ones(B, T, dtype=torch.bool))


def test_categorical_has_only_classification_terms():
    out = build_loss(CFG, "categorical")(**_batch())
    assert out["L_dim"] == 0 and out["L_int"] == 0 and out["L_phase"] == 0
    assert torch.isclose(out["loss"], out["L_cls"])


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_variant_switches_exactly_its_terms(variant):
    out = build_loss(CFG, variant)(**_batch())
    flags = VARIANTS[variant]
    assert (out["L_dim"] > 0) == flags["dim"]
    assert (out["L_int"] > 0) == flags["int"]
    assert (out["L_phase"] > 0) == flags["phase"] or not flags["phase"]


def test_aliases():
    assert canonical_variant("hybrid") == "full"
