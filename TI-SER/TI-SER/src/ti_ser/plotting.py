"""Figures: training curves, confusion matrices, acoustic proxies, t-SNE, ablations."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .compat import np, plt, librosa, TSNE, Model
from .config import BATCH_SIZE, MAX_FRAMES, SR
from .corpora import class_names_for_corpus, ensure_corpus_files
from .utils import _slug, write_csv, write_json


def save_run_artifacts(result: Dict[str, Any], data: Dict[str, Any], meta: Dict[str, Any], run_dir: Path) -> None:
    run_key = "__".join(_slug(meta[k]) for k in ("protocol", "corpus", "model", "objective")) + f"__seed{meta['seed']}"
    detail_dir = run_dir / "runs" / run_key
    detail_dir.mkdir(parents=True, exist_ok=True)
    write_json({k: v for k, v in meta.items()}, detail_dir / "metadata.json")
    history_rows = []
    for i, vals in enumerate(zip(result["history"]["loss"], result["history"]["val_loss"], result["history"]["acc"], result["history"]["val_acc"], result["history"]["lr"]), start=1):
        history_rows.append({"epoch": i, "train_loss": vals[0], "val_loss": vals[1], "train_accuracy": vals[2], "val_accuracy": vals[3], "learning_rate": vals[4]})
    write_csv(history_rows, detail_dir / "training_history.csv")
    class_names = data["class_names"]
    write_csv(
        [{"true_emotion": class_names[i], **{class_names[j]: int(result["cm"][i, j]) for j in range(len(class_names))}}
         for i in range(len(class_names))], detail_dir / "confusion_matrix.csv"
    )
    write_json(result["report"], detail_dir / "classification_report.json")


def plot_training_curves(history: Dict[str, List[float]], title: str, path: Path) -> None:
    if plt is None or not history.get("loss"):
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ep = np.arange(1, len(history["loss"]) + 1)
    axes[0].plot(ep, np.asarray(history["acc"]) * 100, label="Training")
    axes[0].plot(ep, np.asarray(history["val_acc"]) * 100, "--", label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy")
    axes[1].plot(ep, history["loss"], label="Training")
    axes[1].plot(ep, history["val_loss"], "--", label="Validation")
    axes[1].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved figure: %s", path)


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], title: str, path: Path, normalize: bool = False) -> None:
    if plt is None:
        return
    matrix = cm.astype(float)
    if normalize:
        den = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, den, out=np.zeros_like(matrix), where=den > 0)
    fig, ax = plt.subplots(figsize=(7.4, 6.5))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("True emotion")
    ax.set_title(title + (" (row-normalized)" if normalize else " (counts)"))
    threshold = float(np.nanmax(matrix)) / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            s = f"{matrix[i, j]:.2f}" if normalize else str(int(matrix[i, j]))
            ax.text(j, i, s, ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] > threshold else "black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved figure: %s", path)


def _minmax_trajectory(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    span = float(np.ptp(x)) if x.size else 0.0
    if span < 1e-8:
        return np.zeros_like(x)
    return (x - float(x.min())) / (span + 1e-8)


def _resample_trajectory(x: np.ndarray, target_len: int = MAX_FRAMES) -> np.ndarray:
    if len(x) < 2:
        return np.zeros(target_len, dtype=np.float32)
    old_t = np.linspace(0.0, 1.0, len(x))
    new_t = np.linspace(0.0, 1.0, target_len)
    return np.interp(new_t, old_t, x).astype(np.float32)


def plot_acoustic_proxy_trajectories(corpus: str, data_root: Path, path: Path,
                                    max_per_class: int = 80, seed: int = 42) -> None:
    """Plot manuscript-style spectral-centroid/RMS/bandwidth acoustic proxies.

    These curves are descriptive acoustic proxies, not human ratings or model
    predictions. Each utterance's proxy is min-max normalized before averaging.
    """
    if plt is None or librosa is None:
        return
    files, labels = ensure_corpus_files(corpus, data_root)
    rng = np.random.default_rng(seed)
    selected: List[Tuple[Path, int]] = []
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        if len(indices) > max_per_class:
            indices = np.sort(rng.choice(indices, size=max_per_class, replace=False))
        selected.extend((files[int(i)], int(label)) for i in indices)
    by_class: Dict[int, Dict[str, List[np.ndarray]]] = {
        i: {k: [] for k in ("valence", "arousal", "dominance", "pleasure")}
        for i in range(len(class_names_for_corpus(corpus)))
    }
    for path_in, label in selected:
        try:
            y, sr = librosa.load(str(path_in), sr=SR, mono=True)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512)[0]
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512, center=True)[0]
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=2048, hop_length=512)[0]
            n = min(len(centroid), len(rms), len(bandwidth))
            val = _resample_trajectory(_minmax_trajectory(centroid[:n]))
            aro = _resample_trajectory(_minmax_trajectory(rms[:n]))
            dom = _resample_trajectory(_minmax_trajectory(bandwidth[:n]))
            by_class[label]["valence"].append(val)
            by_class[label]["arousal"].append(aro)
            by_class[label]["dominance"].append(dom)
            by_class[label]["pleasure"].append((val + aro) / 2.0)
        except Exception as exc:
            logging.warning("Could not create acoustic proxies for %s: %s", path_in, exc)

    dims = [("valence", "Spectral centroid proxy (normalized)"),
            ("arousal", "RMS energy proxy (normalized)"),
            ("dominance", "Spectral bandwidth proxy (normalized)"),
            ("pleasure", "Mean of valence and arousal proxies")]
    names = class_names_for_corpus(corpus)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    tau = np.linspace(0, 1, MAX_FRAMES)
    palette = plt.get_cmap("tab10")
    for ax, (key, ylabel) in zip(axes.flat, dims):
        for label, emotion in enumerate(names):
            seqs = by_class[label][key]
            if not seqs:
                continue
            ax.plot(tau, np.mean(np.stack(seqs, axis=0), axis=0), color=palette(label), label=emotion, linewidth=1.7)
        ax.set_title(key.capitalize())
        ax.set_xlabel("Normalized utterance time")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{corpus.upper()}: class-averaged acoustic proxy trajectories\nDescriptive proxies, not psychological measurements")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved figure: %s", path)


def plot_tsne_embeddings(model: Model, x: np.ndarray, y: np.ndarray,
                         class_names: Sequence[str], title: str, path: Path,
                         seed: int, max_points: int = 1500) -> None:
    if plt is None or TSNE is None or len(x) < 4:
        logging.warning("Skipping t-SNE: insufficient samples or plotting dependencies unavailable")
        return
    rng = np.random.default_rng(seed)
    if len(x) > max_points:
        chosen: List[int] = []
        # Stratified subsampling so rare classes remain visible.
        for cls in sorted(set(y.tolist())):
            ids = np.flatnonzero(y == cls)
            cap = max(1, int(max_points * len(ids) / len(y)))
            chosen.extend(rng.choice(ids, size=min(len(ids), cap), replace=False).tolist())
        chosen = sorted(chosen[:max_points])
        x, y = x[chosen], y[chosen]
    emb_model = Model(model.input, model.get_layer("classifier_hidden").output)
    embedding = emb_model.predict(x, batch_size=BATCH_SIZE, verbose=0)
    n = len(embedding)
    perplexity = max(1.0, min(30.0, float((n - 1) // 3)))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed)
    coords = tsne.fit_transform(embedding)
    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    palette = plt.get_cmap("tab10")
    for cls, name in enumerate(class_names):
        m = y == cls
        if np.any(m):
            ax.scatter(coords[m, 0], coords[m, 1], s=13, alpha=0.75, color=palette(cls), label=name)
    ax.set_title(title + "\nQualitative diagnostic only")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, frameon=False, markerscale=1.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved figure: %s", path)


def plot_ablation_curves(histories: Dict[str, Dict[str, List[float]]], title: str, path: Path) -> None:
    if plt is None or not histories:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, h in histories.items():
        ep = np.arange(1, len(h.get("acc", [])) + 1)
        if not len(ep):
            continue
        axes[0].plot(ep, np.asarray(h["acc"]) * 100, label=f"{name} train")
        axes[0].plot(ep, np.asarray(h["val_acc"]) * 100, "--", label=f"{name} validation")
        axes[1].plot(ep, h["loss"], label=f"{name} train")
        axes[1].plot(ep, h["val_loss"], "--", label=f"{name} validation")
    axes[0].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy")
    axes[1].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved figure: %s", path)


def plot_comparison(summary_rows: Sequence[Dict[str, Any]], path: Path) -> None:
    if plt is None or not summary_rows:
        return
    rows = [r for r in summary_rows if r.get("protocol") in {"within", "combined"} and r.get("objective") in {"hybrid", "categorical"}]
    if not rows:
        return
    labels, acc, f1, err = [], [], [], []
    for r in rows:
        labels.append(f"{r['corpus']}\n{r['model']}\n{r['objective']}")
        acc.append(float(r["accuracy_mean"]) * 100)
        f1.append(float(r["macro_f1_mean"]) * 100)
        err.append(float(r["accuracy_std"]) * 100)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(11, len(labels) * 0.6), 5.5))
    ax.bar(x - 0.18, acc, width=0.36, yerr=err, capsize=2, label="Accuracy")
    ax.bar(x + 0.18, f1, width=0.36, label="Macro-F1")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_title("Recomputed held-out performance (mean ± SD across seeds)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_weight_sensitivity(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    if plt is None or not rows:
        return
    parameters = [p for p in ("lambda_ce", "lambda_dim", "lambda_focal")
                  if any(r.get("parameter") == p for r in rows)]
    if not parameters:
        return
    fig, axes = plt.subplots(1, len(parameters), figsize=(5 * len(parameters), 4.2), squeeze=False)
    for ax, parameter in zip(axes.flat, parameters):
        subset = sorted([r for r in rows if r.get("parameter") == parameter], key=lambda r: float(r["value"]))
        x = [float(r["value"]) for r in subset]
        y = [100.0 * float(r["accuracy_mean"]) for r in subset]
        err = [100.0 * float(r["accuracy_std"]) for r in subset]
        ax.errorbar(x, y, yerr=err, marker="o", capsize=3)
        ax.set_xlabel(parameter)
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(parameter)
        ax.grid(True, alpha=0.25)
    fig.suptitle("RAVDESS PsyCNN one-at-a-time loss-weight sensitivity\nMean ± SD across configured seeds")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved figure: %s", path)
