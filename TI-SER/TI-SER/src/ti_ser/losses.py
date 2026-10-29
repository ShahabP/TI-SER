"""Hybrid objective: CE+focal, dimensional, intensity and phase-aware losses (Eqs. 25-33)."""


from .compat import tf, K
from .config import FOCAL_ALPHA, FOCAL_GAMMA, LAMBDA_CE, LAMBDA_DIM, LAMBDA_FOCAL, LAMBDA_INT, LAMBDA_PHASE, PHASE_CURVATURE_WEIGHT, PHASE_SLOPE_WEIGHT, PHASE_WINDOW


def classification_loss(y_true, y_pred, lambda_ce: float = LAMBDA_CE,
                        lambda_focal: float = LAMBDA_FOCAL,
                        alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
    ce_each = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
    focal_each = alpha * tf.pow(1.0 - p_t, gamma) * ce_each
    return lambda_ce * tf.reduce_mean(ce_each) + lambda_focal * tf.reduce_mean(focal_each)


def _weighted_mean(values, weights):
    weights = tf.cast(weights, values.dtype)
    return tf.math.divide_no_nan(tf.reduce_sum(values * weights), tf.reduce_sum(weights))


def hybrid_loss_fn(y_true, y_pred_cls, z_ref, y_pred_frame, i_ref, mask_ref,
                   lambda_ce: float = LAMBDA_CE, lambda_focal: float = LAMBDA_FOCAL,
                   lambda_dim: float = LAMBDA_DIM, lambda_int: float = LAMBDA_INT,
                   lambda_phase: float = LAMBDA_PHASE):
    """Four-term objective; masks padded frames and uses target-derived phase boundaries."""
    loss_cls = classification_loss(y_true, y_pred_cls, lambda_ce, lambda_focal)
    z_ref = tf.cast(z_ref, tf.float32)
    y_pred_frame = tf.cast(y_pred_frame, tf.float32)
    i_ref = tf.cast(i_ref, tf.float32)
    mask_ref = tf.cast(mask_ref, tf.float32)
    t_pred = tf.shape(y_pred_frame)[1]
    z_resized = tf.image.resize(z_ref[..., None], tf.stack([t_pred, tf.constant(2)]), method="bilinear")[..., 0]
    i_resized = tf.image.resize(i_ref[..., None, None], tf.stack([t_pred, tf.constant(1)]), method="bilinear")[..., 0, 0]
    mask_resized = tf.image.resize(mask_ref[..., None, None], tf.stack([t_pred, tf.constant(1)]), method="nearest")[..., 0, 0]

    per_frame_dim = tf.reduce_mean(tf.square(y_pred_frame - z_resized), axis=-1)
    loss_dim = _weighted_mean(per_frame_dim, mask_resized)
    i_pred = tf.norm(y_pred_frame, axis=-1)
    loss_int = _weighted_mean(tf.square(i_pred - i_resized), mask_resized)

    # Find each reference trajectory's peak. For the zero-prototype control, use
    # the temporal midpoint rather than treating frame 0 as a meaningful peak.
    valid_count = tf.maximum(tf.reduce_sum(mask_resized, axis=1), 1.0)
    peak_idx = tf.argmax(i_resized * mask_resized, axis=1, output_type=tf.int32)
    peak_tau = tf.cast(peak_idx, tf.float32) / tf.maximum(valid_count - 1.0, 1.0)
    max_ref = tf.reduce_max(i_resized * mask_resized, axis=1)
    peak_tau = tf.where(max_ref < 1e-6, tf.fill(tf.shape(peak_tau), 0.5), peak_tau)
    positions = tf.cast(tf.range(t_pred)[None, :], tf.float32)
    tau = positions / tf.maximum(valid_count[:, None] - 1.0, 1.0)
    valid = mask_resized > 0.5
    delta = PHASE_WINDOW
    phase_masks = [
        valid & (tau < peak_tau[:, None] - delta),
        valid & (tf.abs(tau - peak_tau[:, None]) <= delta),
        valid & (tau > peak_tau[:, None] + delta),
    ]

    # Phase terms: mean intensity match, first-difference (slope), and second-difference
    # (curvature) within build-up / peak / recovery windows derived from the reference envelope.
    # Matches the phase-aware regularizer in the manuscript (peak window delta, slope/curvature weights).
    phase_terms = []
    for pm in phase_masks:
        pm_f = tf.cast(pm, tf.float32)
        mean_pred = tf.math.divide_no_nan(tf.reduce_sum(i_pred * pm_f, axis=1), tf.reduce_sum(pm_f, axis=1))
        mean_ref = tf.math.divide_no_nan(tf.reduce_sum(i_resized * pm_f, axis=1), tf.reduce_sum(pm_f, axis=1))
        phase_terms.append(tf.reduce_mean(tf.square(mean_pred - mean_ref)))
        # Slope and curvature use dynamic length so graph-mode shapes remain valid.
        dp = i_pred[:, 1:] - i_pred[:, :-1]
        dr = i_resized[:, 1:] - i_resized[:, :-1]
        dm = pm[:, 1:] & pm[:, :-1]
        dm_f = tf.cast(dm, tf.float32)
        mean_dp = tf.math.divide_no_nan(tf.reduce_sum(dp * dm_f, axis=1), tf.reduce_sum(dm_f, axis=1))
        mean_dr = tf.math.divide_no_nan(tf.reduce_sum(dr * dm_f, axis=1), tf.reduce_sum(dm_f, axis=1))
        phase_terms.append(PHASE_SLOPE_WEIGHT * tf.reduce_mean(tf.square(mean_dp - mean_dr)))
        ddp = i_pred[:, 2:] - 2.0 * i_pred[:, 1:-1] + i_pred[:, :-2]
        ddr = i_resized[:, 2:] - 2.0 * i_resized[:, 1:-1] + i_resized[:, :-2]
        sm = pm[:, 2:] & pm[:, 1:-1] & pm[:, :-2]
        sm_f = tf.cast(sm, tf.float32)
        mean_ddp = tf.math.divide_no_nan(tf.reduce_sum(ddp * sm_f, axis=1), tf.reduce_sum(sm_f, axis=1))
        mean_ddr = tf.math.divide_no_nan(tf.reduce_sum(ddr * sm_f, axis=1), tf.reduce_sum(sm_f, axis=1))
        phase_terms.append(PHASE_CURVATURE_WEIGHT * tf.reduce_mean(tf.square(mean_ddp - mean_ddr)))
    loss_phase = tf.add_n(phase_terms) / float(len(phase_terms)) if phase_terms else tf.constant(0.0)
    return loss_cls + lambda_dim * loss_dim + lambda_int * loss_int + lambda_phase * loss_phase
