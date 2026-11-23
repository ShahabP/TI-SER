"""Single source of truth for the experimental pipeline.

Every script (train / run_experiments / ablation / leave_corpus_out) goes
through ``run_experiment`` so that all compared conditions share:

  * identical, stratified (or speaker-disjoint) splits saved to disk,
  * normalisation statistics computed on the training split ONLY,
  * SpecAugment on the training split ONLY,
  * identical seeds -> identical initial weights and mini-batch order,
  * a machine-readable ``run_record.json`` per run (config, split fingerprint,
    seed, metrics, environment) plus per-sample test predictions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.datasets import SERDataset, MultiCorpusDataset, IndexedDataset
from src.data.features import collate_variable_length
from src.data.normalization import compute_train_stats, stats_fingerprint
from src.data.prototypes import IDX_TO_EMOTION
from src.data.splits import (
    make_split, assert_disjoint_and_complete, split_fingerprint, save_split, load_split,
)
from src.losses.hybrid_loss import HybridLoss
from src.models import PsyCNN, PsyTransformer
from src.training.metrics import evaluate_model
from src.training.trainer import Trainer
from src.utils.augment import SpecAugment, FeatureNormalizer, Compose
from src.utils.schedulers import warmup_cosine_scheduler

PIPELINE_VERSION = "2.0.0"   # records without this tag are historical, not reproduced

ROOT_KEYS = {"ravdess": "ravdess_root", "tess": "tess_root", "crema": "crema_root"}

# Full factorial over the three auxiliary terms (+ a prototype-shuffle control).
# Weights are taken from cfg["loss"]; a variant only switches terms on/off.
VARIANTS: Dict[str, Dict[str, bool]] = {
    "categorical": dict(dim=False, int=False, phase=False),   # CE + focal only
    "dim":         dict(dim=True,  int=False, phase=False),
    "int":         dict(dim=False, int=True,  phase=False),
    "phase":       dict(dim=False, int=False, phase=True),
    "dim_int":     dict(dim=True,  int=True,  phase=False),
    "dim_phase":   dict(dim=True,  int=False, phase=True),
    "int_phase":   dict(dim=False, int=True,  phase=True),
    "full":        dict(dim=True,  int=True,  phase=True),    # = "hybrid" in the paper
    "full_shuffled_proto": dict(dim=True, int=True, phase=True, shuffle=True),
}
VARIANT_ALIASES = {"hybrid": "full", "cat_dim": "dim", "cat_dim_int": "dim_int",
                   "full_hybrid": "full"}
SHUFFLE_PERMUTATION = [(i + 3) % 7 for i in range(7)]   # fixed derangement of the 7 classes


def canonical_variant(name: str) -> str:
    name = VARIANT_ALIASES.get(name, name)
    if name not in VARIANTS:
        raise ValueError(f"unknown variant '{name}'. Choose from {sorted(VARIANTS)}")
    return name


# --------------------------------------------------------------------------- setup
def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device):
    m = cfg["model"]
    name = m["name"].lower()
    if name == "psycnn":
        model = PsyCNN(input_dim=m["input_dim"], num_classes=m["num_classes"])
    elif name == "psytransformer":
        model = PsyTransformer(
            input_dim=m["input_dim"], d_model=m.get("d_model", 128), n_layers=m.get("n_layers", 3),
            n_heads=m.get("n_heads", 4), d_ff=m.get("d_ff", 256), num_classes=m["num_classes"],
            max_len=m.get("max_len", 174),
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return model.to(device)


def build_loss(cfg: dict, variant: str) -> HybridLoss:
    variant = canonical_variant(variant)
    flags = VARIANTS[variant]
    lc = {k: v for k, v in cfg["loss"].items() if k != "mode"}
    lc["lambda_dim"] = lc["lambda_dim"] if flags["dim"] else 0.0
    lc["lambda_int"] = lc["lambda_int"] if flags["int"] else 0.0
    lc["lambda_phase"] = lc["lambda_phase"] if flags["phase"] else 0.0
    if flags.get("shuffle"):
        lc["proto_permutation"] = SHUFFLE_PERMUTATION
    return HybridLoss(num_classes=cfg["model"]["num_classes"],
                      mode=cfg["loss"].get("mode", "va"), **lc)


# --------------------------------------------------------------------------- data
def _load_corpus(cfg: dict, name: str) -> SERDataset:
    root = cfg["data"][ROOT_KEYS[name]]
    if not Path(root).exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}. Run: python scripts/download_data.py --corpus {name}")
    return SERDataset(root=root, corpus=name, seven_class=cfg["data"].get("seven_class", True),
                      max_frames=cfg["data"].get("max_frames", 174),
                      sr=cfg["data"].get("sample_rate", 16000))


def load_base(cfg: dict, corpus: str):
    """``corpus`` is 'ravdess' | 'tess' | 'crema' | 'combined' (RAVDESS+TESS)."""
    if corpus == "combined":
        return MultiCorpusDataset([_load_corpus(cfg, "ravdess"), _load_corpus(cfg, "tess")])
    return _load_corpus(cfg, corpus)


def _effective_split_mode(cfg: dict, corpus: str) -> str:
    mode = cfg["data"].get("split_mode", "stratified")
    # TESS has two speakers and 'combined' mixes corpora: speaker-disjoint is only meaningful
    # for corpora with many speakers.
    if mode == "speaker_disjoint" and corpus not in ("ravdess", "crema"):
        return "stratified"
    return mode


def get_split(cfg: dict, base, tag: str, corpus: str, seed: int, split_dir: Path,
              train_ratio: float, val_ratio: float) -> Dict[str, np.ndarray]:
    """Create (or load and verify) the split for (tag, seed). Saved as JSON so every
    compared configuration reuses exactly the same utterances."""
    mode = _effective_split_mode(cfg, corpus)
    paths = base.rel_paths()
    path = Path(split_dir) / f"{tag}_{mode}_seed{seed}.json"
    if path.exists():
        split = load_split(path)
        with open(path) as f:
            saved = json.load(f)
        for k in ("train", "val", "test"):
            if [paths[i] for i in split[k]] != saved["files"][k]:
                raise RuntimeError(f"{path} no longer matches the data on disk; delete it to regenerate")
    else:
        strata = [f"{c}_{l}" for c, l in zip(base.corpora, base.labels)]
        split = make_split(strata, speakers=base.speakers, mode=mode,
                           train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
        assert_disjoint_and_complete(split, len(base))
        save_split(path, paths, split, dict(tag=tag, mode=mode, seed=seed,
                                            train_ratio=train_ratio, val_ratio=val_ratio))
    return split


@dataclass
class Data:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    split: Dict[str, np.ndarray]
    split_fingerprint: str
    norm_fingerprint: str
    norm_mean: np.ndarray
    norm_std: np.ndarray
    split_mode: str
    sizes: Dict[str, int]


def _loader(ds, bs, shuffle, nw, gen):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw,
                      collate_fn=collate_variable_length,
                      pin_memory=torch.cuda.is_available(), generator=gen)


def prepare_data(cfg: dict, corpus: str, seed: int, split_dir: Path,
                 test_corpus: Optional[str] = None) -> Data:
    """Build loaders.

    within/combined: stratified train/val/test split of ``corpus``.
    leave-corpus-out (``test_corpus`` given): stratified train/val split of the
    source corpus; the ENTIRE target corpus is the test set.
    """
    d = cfg["data"]
    base = load_base(cfg, corpus)
    if test_corpus is None:
        tag = corpus
        split = get_split(cfg, base, tag, corpus, seed, split_dir, d["train_ratio"], d["val_ratio"])
        eval_base, test_idx = base, split["test"]
        mode = _effective_split_mode(cfg, corpus)
    else:
        lco_val = cfg.get("leave_corpus_out", {}).get("val_ratio", 0.15)
        tag = f"lco_{corpus}_to_{test_corpus}"
        split = get_split(cfg, base, tag, corpus, seed, split_dir, 1.0 - lco_val, lco_val)
        eval_base = load_base(cfg, test_corpus)
        test_idx = np.arange(len(eval_base))
        split = {**split, "test": test_idx}
        mode = _effective_split_mode(cfg, corpus)

    # Normalisation statistics from TRAINING utterances only.
    mean, std = compute_train_stats((base[i]["features"] for i in split["train"]),
                                    dim=cfg["model"]["input_dim"])
    norm = FeatureNormalizer(mean, std)
    train_tf = Compose([norm, SpecAugment()]) if cfg["train"].get("augmentation", True) else norm

    train_ds = IndexedDataset(base, split["train"], train_tf)
    val_ds = IndexedDataset(base, split["val"], norm)        # never augmented
    test_ds = IndexedDataset(eval_base, test_idx, norm)      # never augmented

    gen = torch.Generator()
    gen.manual_seed(seed)
    bs, nw = cfg["train"]["batch_size"], d.get("num_workers", 4)
    fp = split_fingerprint(base.rel_paths(), {**split, "test": split["test"] if test_corpus is None
                                              else np.array([], dtype=int)})
    return Data(
        train=_loader(train_ds, bs, True, nw, gen), val=_loader(val_ds, bs, False, nw, None),
        test=_loader(test_ds, bs, False, nw, None), split=split, split_fingerprint=fp,
        norm_fingerprint=stats_fingerprint(mean, std), norm_mean=mean, norm_std=std,
        split_mode=mode, sizes={"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
    )


# --------------------------------------------------------------------------- records
def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=Path(__file__).resolve().parents[1]).decode().strip()
    except Exception:
        return None


def environment_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
            "git_commit": _git_commit()}
    for mod in ("librosa", "sklearn"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    return info


def config_hash(cfg: dict, variant: str, corpus: str, test_corpus: Optional[str]) -> str:
    payload = json.dumps({"model": cfg["model"], "loss": cfg["loss"], "train": cfg["train"],
                          "data": cfg["data"], "variant": variant, "corpus": corpus,
                          "test_corpus": test_corpus}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run_experiment(cfg: dict, *, model_name: str, variant: str, corpus: str, seed: int,
                   group: str, test_corpus: Optional[str] = None, device: Optional[str] = None,
                   output_dir: Optional[str] = None, force: bool = False) -> dict:
    """Train one model under one condition and write its run record."""
    cfg = copy.deepcopy(cfg)
    cfg["model"]["name"] = model_name
    variant = canonical_variant(variant)
    out_root = Path(output_dir or cfg["experiment"]["output_dir"])
    split_dir = out_root / "splits"
    corpus_tag = corpus if test_corpus is None else f"lco_{corpus}_to_{test_corpus}"
    run_id = f"{model_name}_{corpus_tag}_{variant}_seed{seed}"
    run_dir = out_root / group / run_id
    record_path = run_dir / "run_record.json"
    chash = config_hash(cfg, variant, corpus, test_corpus)

    if record_path.exists() and not force:
        with open(record_path) as f:
            rec = json.load(f)
        if rec.get("config_hash") == chash and rec.get("pipeline_version") == PIPELINE_VERSION:
            print(f"[resume] {run_id} already done")
            return rec

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(seed)
    data = prepare_data(cfg, corpus, seed, split_dir, test_corpus)

    set_seed(seed)                      # identical initial weights across variants
    model = build_model(cfg, dev)
    loss_fn = build_loss(cfg, variant)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                                 weight_decay=cfg["train"]["weight_decay"])
    scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs=cfg["train"]["warmup_epochs"],
                                        total_epochs=cfg["train"]["epochs"])
    target_names = [IDX_TO_EMOTION[i] for i in range(cfg["model"]["num_classes"])]

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model_summary.txt").write_text(str(model))
    t0 = time.time()
    trainer = Trainer(model=model, train_loader=data.train, val_loader=data.val, device=dev,
                      loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
                      epochs=cfg["train"]["epochs"], output_dir=str(out_root / group),
                      experiment_name=run_id, target_names=target_names,
                      use_amp=cfg["train"].get("use_amp", False))
    fit = trainer.fit()
    trainer.load_best()                 # checkpoint chosen on validation accuracy only
    test_metrics, y_true, y_pred = evaluate_model(model, data.test, dev, target_names,
                                                  return_predictions=True)
    with open(run_dir / "test_predictions.csv", "w") as f:
        f.write("index,y_true,y_pred\n")
        for i, (a, b) in enumerate(zip(y_true, y_pred)):
            f.write(f"{i},{a},{b}\n")

    record = {
        "pipeline_version": PIPELINE_VERSION, "run_id": run_id, "group": group,
        "model": model_name, "variant": variant, "corpus": corpus, "test_corpus": test_corpus,
        "protocol": ("leave-corpus-out" if test_corpus else
                     "combined" if corpus == "combined" else "within-corpus"),
        "seed": seed, "config_hash": chash,
        "effective_loss_weights": {"lambda_dim": loss_fn.lambda_dim, "lambda_int": loss_fn.lambda_int,
                                   "lambda_phase": loss_fn.lambda_phase,
                                   "lambda_ce": loss_fn.lambda_ce, "lambda_focal": loss_fn.lambda_focal,
                                   "prototype_shuffled": bool(VARIANTS[variant].get("shuffle"))},
        "config": cfg,
        "split": {"mode": data.split_mode, "fingerprint": data.split_fingerprint,
                  "sizes": data.sizes, "file": f"splits/ (see {corpus_tag})"},
        "normalization": {"fit_on": "training split only", "fingerprint": data.norm_fingerprint},
        "augmentation": ("SpecAugment, training split only"
                         if cfg["train"].get("augmentation", True) else "none"),
        "n_parameters": model.count_parameters(),
        "best_epoch": fit["best_epoch"], "best_val_accuracy": fit["best_val_acc"],
        "test": {"accuracy": test_metrics["accuracy"], "macro_f1": test_metrics["macro_f1"],
                 "weighted_f1": test_metrics["weighted_f1"],
                 "per_class_f1": test_metrics["per_class_f1"],
                 "confusion_matrix": test_metrics["confusion_matrix"]},
        "train_seconds": round(time.time() - t0, 1),
        "environment": environment_info(),
    }
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[done] {run_id}: acc={record['test']['accuracy']*100:.2f}% "
          f"macroF1={record['test']['macro_f1']*100:.2f}%")
    return record
