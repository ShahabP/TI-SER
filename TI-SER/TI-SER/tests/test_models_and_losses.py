"""Architecture and loss checks; skipped when TensorFlow is unavailable."""
import numpy as np
import pytest

pytest.importorskip("tensorflow")

from ti_ser.config import D_FEATURES, MAX_FRAMES, VA_PROTOTYPES
from ti_ser.losses import classification_loss, hybrid_loss_fn
from ti_ser.models import build_ti_cnn, build_ti_transformer
from ti_ser.targets import make_reference_trajectories


def test_cnn_shapes_and_param_scale():
    m = build_ti_cnn(MAX_FRAMES, D_FEATURES, 7)
    cls, va_frame, va_utt = m.output
    assert cls.shape[-1] == 7 and va_frame.shape[-1] == 2 and va_utt.shape[-1] == 2
    assert va_frame.shape[1] == MAX_FRAMES // 8            # three /2 poolings (Table 4)
    assert 0.25e6 < m.count_params() < 0.45e6              # paper: ~0.33M


def test_transformer_shapes_and_param_scale():
    m = build_ti_transformer(MAX_FRAMES, D_FEATURES, 7)
    cls, va_frame, _ = m.output
    assert cls.shape[-1] == 7 and va_frame.shape[1] == MAX_FRAMES
    assert 0.35e6 < m.count_params() < 0.6e6               # paper: ~0.46M


def test_hybrid_loss_is_finite_and_reduces_to_cls_when_weights_zero():
    rng = np.random.default_rng(0)
    n, t = 4, MAX_FRAMES
    y = rng.integers(0, 7, n)
    energy = rng.random((n, t)).astype(np.float32)
    lengths = np.full(n, t)
    z, i_ref, mask = make_reference_trajectories(y, lengths, energy, VA_PROTOTYPES)
    y_oh = np.eye(7, dtype=np.float32)[y]
    p_cls = np.full((n, 7), 1 / 7, dtype=np.float32)
    p_frame = rng.normal(size=(n, t // 8, 2)).astype(np.float32)
    full = float(hybrid_loss_fn(y_oh, p_cls, z, p_frame, i_ref, mask))
    cls_only = float(hybrid_loss_fn(y_oh, p_cls, z, p_frame, i_ref, mask,
                                    lambda_dim=0, lambda_int=0, lambda_phase=0))
    assert np.isfinite(full) and full > cls_only
    assert cls_only == pytest.approx(float(classification_loss(y_oh, p_cls)), rel=1e-5)
