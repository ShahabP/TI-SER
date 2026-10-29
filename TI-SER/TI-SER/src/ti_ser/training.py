"""Training loop (Adam, 5-epoch warmup + cosine decay) and test evaluation."""

import logging
from typing import Any, Dict, Iterable, List

from .compat import np, accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, tf, Adam, to_categorical
from .config import BATCH_SIZE, INITIAL_LR, LAMBDA_CE, LAMBDA_DIM, LAMBDA_FOCAL, LAMBDA_INT, LAMBDA_PHASE
from .losses import classification_loss, hybrid_loss_fn
from .models import build_ti_cnn, build_ti_transformer
from .targets import make_reference_trajectories
from .utils import set_seed


def _learning_rate(epoch: int, total_epochs: int, warmup_epochs: int = 5) -> float:
    if epoch < warmup_epochs:
        return INITIAL_LR * (0.1 + 0.9 * (epoch + 1) / warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return INITIAL_LR * 0.5 * (1.0 + float(np.cos(np.pi * progress)))


def _batch_indices(n: int, batch_size: int, seed: int, shuffle: bool) -> Iterable[np.ndarray]:
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]


def train_one(data: Dict[str, Any], model_type: str, epochs: int, seed: int,
              objective: str = "hybrid", lambda_ce: float = LAMBDA_CE,
              lambda_focal: float = LAMBDA_FOCAL, lambda_dim: float = LAMBDA_DIM,
              lambda_int: float = LAMBDA_INT, lambda_phase: float = LAMBDA_PHASE,
              verbose: bool = True) -> Dict[str, Any]:
    if tf is None:
        raise RuntimeError("TensorFlow is required to train a model")
    if objective not in {"hybrid", "categorical"}:
        raise ValueError(f"Unknown objective: {objective}")
    set_seed(seed)
    tf.keras.backend.clear_session()
    x_train, y_train = data["X_train"], data["y_train"]
    x_val, y_val = data["X_val"], data["y_val"]
    x_test, y_test = data["X_test"], data["y_test"]
    names = data["class_names"]
    n_classes = len(names)
    prototypes = data["prototypes"]
    t, d = x_train.shape[1:]

    if model_type == "cnn":
        model = build_ti_cnn(t, d, n_classes)
    elif model_type == "transformer":
        model = build_ti_transformer(t, d, n_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    y_train_oh = to_categorical(y_train, n_classes).astype(np.float32)
    y_val_oh = to_categorical(y_val, n_classes).astype(np.float32)
    z_train, i_train, m_train = make_reference_trajectories(y_train, data["lengths_train"], data["energy_train"], prototypes)
    z_val, i_val, m_val = make_reference_trajectories(y_val, data["lengths_val"], data["energy_val"], prototypes)
    optimizer = Adam(learning_rate=INITIAL_LR)

    @tf.function(reduce_retracing=True)
    def train_step(xb, yb, zb, ib, mb):
        with tf.GradientTape() as tape:
            p_cls, p_frame, _ = model(xb, training=True)
            if objective == "hybrid":
                loss = hybrid_loss_fn(yb, p_cls, zb, p_frame, ib, mb,
                                      lambda_ce, lambda_focal, lambda_dim, lambda_int, lambda_phase)
            else:
                loss = classification_loss(yb, p_cls, lambda_ce, lambda_focal)
            if model.losses:
                loss += tf.add_n(model.losses)
        grads = tape.gradient(loss, model.trainable_variables)
        grad_var = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
        optimizer.apply_gradients(grad_var)
        return loss, p_cls

    @tf.function(reduce_retracing=True)
    def eval_step(xb, yb, zb, ib, mb):
        p_cls, p_frame, _ = model(xb, training=False)
        if objective == "hybrid":
            loss = hybrid_loss_fn(yb, p_cls, zb, p_frame, ib, mb,
                                  lambda_ce, lambda_focal, lambda_dim, lambda_int, lambda_phase)
        else:
            loss = classification_loss(yb, p_cls, lambda_ce, lambda_focal)
        if model.losses:
            loss += tf.add_n(model.losses)
        return loss, p_cls

    history: Dict[str, List[float]] = {"loss": [], "val_loss": [], "acc": [], "val_acc": [], "lr": []}
    best_val_acc = -np.inf
    best_weights = None
    for epoch in range(epochs):
        lr = _learning_rate(epoch, epochs)
        optimizer.learning_rate.assign(lr)
        tr_loss_sum, tr_n = 0.0, 0
        tr_true, tr_pred = [], []
        for batch_no, idx in enumerate(_batch_indices(len(x_train), BATCH_SIZE, seed + epoch, True)):
            loss, p_cls = train_step(
                tf.convert_to_tensor(x_train[idx], dtype=tf.float32),
                tf.convert_to_tensor(y_train_oh[idx], dtype=tf.float32),
                tf.convert_to_tensor(z_train[idx], dtype=tf.float32),
                tf.convert_to_tensor(i_train[idx], dtype=tf.float32),
                tf.convert_to_tensor(m_train[idx], dtype=tf.float32),
            )
            bs = len(idx)
            tr_loss_sum += float(loss.numpy()) * bs
            tr_n += bs
            tr_true.extend(y_train[idx].tolist())
            tr_pred.extend(np.argmax(p_cls.numpy(), axis=1).tolist())
        val_loss_sum, val_n = 0.0, 0
        va_true, va_pred = [], []
        for idx in _batch_indices(len(x_val), BATCH_SIZE, seed, False):
            loss, p_cls = eval_step(
                tf.convert_to_tensor(x_val[idx], dtype=tf.float32),
                tf.convert_to_tensor(y_val_oh[idx], dtype=tf.float32),
                tf.convert_to_tensor(z_val[idx], dtype=tf.float32),
                tf.convert_to_tensor(i_val[idx], dtype=tf.float32),
                tf.convert_to_tensor(m_val[idx], dtype=tf.float32),
            )
            bs = len(idx)
            val_loss_sum += float(loss.numpy()) * bs
            val_n += bs
            va_true.extend(y_val[idx].tolist())
            va_pred.extend(np.argmax(p_cls.numpy(), axis=1).tolist())
        tr_acc = float(accuracy_score(tr_true, tr_pred))
        val_acc = float(accuracy_score(va_true, va_pred))
        history["loss"].append(tr_loss_sum / max(tr_n, 1))
        history["val_loss"].append(val_loss_sum / max(val_n, 1))
        history["acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(float(lr))
        if verbose:
            logging.info("%s %s seed=%d epoch %d/%d lr=%.6g loss=%.4f acc=%.4f val_loss=%.4f val_acc=%.4f",
                         data["corpus"], model_type, seed, epoch + 1, epochs, lr,
                         history["loss"][-1], tr_acc, history["val_loss"][-1], val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.get_weights()

    if best_weights is not None:
        model.set_weights(best_weights)
    pred_probs, _, _ = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(pred_probs, axis=1)
    labels = list(range(n_classes))
    report = classification_report(y_test, y_pred, labels=labels, target_names=names,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    result = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, labels=labels, average="weighted", zero_division=0)),
        "report": report,
        "cm": cm,
        "history": history,
        "model": model,
        "y_true": y_test.copy(),
        "y_pred": y_pred,
        "y_prob": pred_probs,
        "n_test": int(len(y_test)),
        "best_val_accuracy": float(best_val_acc),
        "parameter_count": int(model.count_params()),
    }
    if "corpus_codes_test" in data:
        subgroup_metrics = {}
        corpus_codes = np.asarray(data["corpus_codes_test"])
        for code, subgroup_name in ((0, "ravdess"), (1, "tess")):
            use = corpus_codes == code
            if np.any(use):
                subgroup_metrics[f"{subgroup_name}_n"] = int(np.sum(use))
                subgroup_metrics[f"{subgroup_name}_accuracy"] = float(accuracy_score(y_test[use], y_pred[use]))
                subgroup_metrics[f"{subgroup_name}_macro_f1"] = float(
                    f1_score(y_test[use], y_pred[use], labels=labels, average="macro", zero_division=0)
                )
        result["subgroup_metrics"] = subgroup_metrics
    logging.info("TEST %s | %s | objective=%s | seed=%d | n=%d | accuracy=%.4f macro-F1=%.4f balanced-accuracy=%.4f",
                 data["corpus"], model_type, objective, seed, len(y_test), result["accuracy"], result["macro_f1"], result["balanced_accuracy"])
    return result
