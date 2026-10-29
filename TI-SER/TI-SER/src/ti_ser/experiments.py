"""Scenario orchestration: seeds, ablations, loss-weight sensitivity, metric rows."""

import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .compat import np
from .config import LAMBDA_CE, LAMBDA_DIM, LAMBDA_FOCAL, LAMBDA_INT, LAMBDA_PHASE
from .plotting import plot_confusion_matrix, plot_training_curves, plot_tsne_embeddings, save_run_artifacts
from .splits import prepare_combined, prepare_corpus
from .state import RUN_CLASS_METRICS, RUN_METRICS
from .training import train_one
from .utils import _slug


def metric_row(result: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(meta)
    row.update({
        "n_test": result["n_test"],
        "accuracy": result["accuracy"],
        "balanced_accuracy": result["balanced_accuracy"],
        "macro_f1": result["macro_f1"],
        "weighted_f1": result["weighted_f1"],
        "best_val_accuracy": result["best_val_accuracy"],
        "parameter_count": result["parameter_count"],
    })
    row.update(result.get("subgroup_metrics", {}))
    for class_name in meta["class_names"]:
        report = result["report"].get(class_name, {})
        prefix = _slug(class_name)
        row[f"{prefix}_precision"] = report.get("precision", 0.0)
        row[f"{prefix}_recall"] = report.get("recall", 0.0)
        row[f"{prefix}_f1"] = report.get("f1-score", 0.0)
        row[f"{prefix}_support"] = report.get("support", 0)
    return row


def run_scenario(corpus: str, protocol: str, model_type: str, objective: str,
                 seeds: Sequence[int], epochs: int, data_root: Path, run_dir: Path,
                 prototypes: Optional[np.ndarray] = None, split_strategy: str = "utterance",
                 data_builder=None, extra_meta: Optional[Dict[str, Any]] = None,
                 figure_path: Optional[Path] = None, make_tsne: bool = False,
                 make_confusion: bool = True, make_history: bool = True,
                 sensitivity: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    per_seed: List[Dict[str, Any]] = []
    representative_result = None
    representative_data = None
    for seed in seeds:
        logging.info("START protocol=%s corpus=%s model=%s objective=%s seed=%d", protocol, corpus, model_type, objective, seed)
        if data_builder is not None:
            data = data_builder(seed)
        elif corpus == "combined":
            data = prepare_combined(data_root, seed, prototypes=prototypes)
        else:
            data = prepare_corpus(corpus, data_root, seed, prototypes=prototypes, split_strategy=split_strategy)
        meta = {
            "protocol": protocol,
            "corpus": corpus,
            "model": model_type,
            "objective": objective,
            "seed": int(seed),
            "epochs": int(epochs),
            "split_strategy": split_strategy,
            "class_names": data["class_names"],
            "lambda_ce": float((sensitivity or {}).get("lambda_ce", LAMBDA_CE)),
            "lambda_focal": float((sensitivity or {}).get("lambda_focal", LAMBDA_FOCAL)),
            "lambda_dim": float((sensitivity or {}).get("lambda_dim", LAMBDA_DIM)),
            "lambda_int": float((sensitivity or {}).get("lambda_int", LAMBDA_INT)),
            "lambda_phase": float((sensitivity or {}).get("lambda_phase", LAMBDA_PHASE)),
            "prototype_control": "null" if prototypes is not None and np.allclose(prototypes, 0) else "canonical",
        }
        if extra_meta:
            meta.update(extra_meta)
        cfg = sensitivity or {}
        result = train_one(
            data, model_type=model_type, epochs=epochs, seed=seed, objective=objective,
            lambda_ce=float(cfg.get("lambda_ce", LAMBDA_CE)),
            lambda_focal=float(cfg.get("lambda_focal", LAMBDA_FOCAL)),
            lambda_dim=float(cfg.get("lambda_dim", LAMBDA_DIM)),
            lambda_int=float(cfg.get("lambda_int", LAMBDA_INT)),
            lambda_phase=float(cfg.get("lambda_phase", LAMBDA_PHASE)),
            verbose=False,
        )
        row = metric_row(result, meta)
        RUN_METRICS.append(row)
        per_seed.append(row)
        for class_name in data["class_names"]:
            report = result["report"].get(class_name, {})
            RUN_CLASS_METRICS.append({
                **{k: meta.get(k) for k in ("protocol", "corpus", "model", "objective", "seed", "epochs", "lambda_ce", "lambda_focal", "lambda_dim", "lambda_int", "lambda_phase", "prototype_control")},
                "emotion": class_name,
                "precision": float(report.get("precision", 0.0)),
                "recall": float(report.get("recall", 0.0)),
                "f1": float(report.get("f1-score", 0.0)),
                "support": int(report.get("support", 0)),
            })
        save_run_artifacts(result, data, meta, run_dir)
        cm_dir = run_dir / "figures" / "supplementary"
        if make_confusion:
            plot_confusion_matrix(result["cm"], data["class_names"],
                                  f"{corpus} {model_type} {objective} seed {seed}",
                                  cm_dir / f"cm_{_slug(protocol)}_{_slug(corpus)}_{model_type}_{objective}_seed{seed}.png")
            plot_confusion_matrix(result["cm"], data["class_names"],
                                  f"{corpus} {model_type} {objective} seed {seed}",
                                  cm_dir / f"cmnorm_{_slug(protocol)}_{_slug(corpus)}_{model_type}_{objective}_seed{seed}.png", normalize=True)
        if make_history and figure_path is not None and seed == seeds[0]:
            plot_training_curves(result["history"], f"{corpus} — {model_type} — {objective} — seed {seed}", figure_path)
        if make_tsne and seed == seeds[0]:
            plot_tsne_embeddings(result["model"], data["X_test"], data["y_test"], data["class_names"],
                                 f"{corpus.upper()} {model_type} embeddings (test set)",
                                 run_dir / "figures" / f"figure_{'03' if corpus == 'ravdess' else '04'}_{corpus}_tsne.png", seed)
        if seed == seeds[0]:
            representative_result = result
            representative_data = data
        else:
            del result, data
            gc.collect()
    summary = summarise_rows(per_seed)
    summary.update({"protocol": protocol, "corpus": corpus, "model": model_type, "objective": objective,
                    "prototype_control": "null" if prototypes is not None and np.allclose(prototypes, 0) else "canonical"})
    if extra_meta:
        summary.update(extra_meta)
    if sensitivity:
        summary.update(sensitivity)
    return {"summary": summary, "per_seed": per_seed,
            "representative_result": representative_result, "representative_data": representative_data}


def summarise_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    out: Dict[str, Any] = {"n_seeds": len(rows)}
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    out["n_test_mean"] = float(np.mean([r["n_test"] for r in rows]))
    for key in ("ravdess_accuracy", "ravdess_macro_f1", "tess_accuracy", "tess_macro_f1"):
        available = [float(r[key]) for r in rows if r.get(key) is not None]
        if available:
            values = np.asarray(available, dtype=np.float64)
            out[f"{key}_mean"] = float(values.mean())
            out[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    for key in ("ravdess_n", "tess_n"):
        available = [float(r[key]) for r in rows if r.get(key) is not None]
        if available:
            out[f"{key}_mean"] = float(np.mean(available))
    return out


def _run_ablation_suite(corpus: str, model_type: str, seeds: Sequence[int], epochs: int,
                        data_root: Path, run_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, List[float]]]]:
    configs = {
        "full_hybrid": {"objective": "hybrid"},
        "no_dim": {"objective": "hybrid", "lambda_dim": 0.0},
        "no_intensity": {"objective": "hybrid", "lambda_int": 0.0},
        "no_phase": {"objective": "hybrid", "lambda_phase": 0.0},
        "categorical_only": {"objective": "categorical"},
    }
    rows: List[Dict[str, Any]] = []
    representative_histories: Dict[str, Dict[str, List[float]]] = {}
    for seed in seeds:
        data = prepare_corpus(corpus, data_root, seed)
        for name, config in configs.items():
            objective = config["objective"]
            params = {k: v for k, v in config.items() if k != "objective"}
            res = train_one(data, model_type, epochs, seed, objective=objective, **params, verbose=False)
            meta = {"protocol": "ablation", "corpus": corpus, "model": model_type,
                    "objective": name, "seed": seed, "epochs": epochs,
                    "split_strategy": "utterance", "class_names": data["class_names"],
                    "lambda_ce": LAMBDA_CE, "lambda_focal": LAMBDA_FOCAL,
                    "lambda_dim": params.get("lambda_dim", LAMBDA_DIM),
                    "lambda_int": params.get("lambda_int", LAMBDA_INT),
                    "lambda_phase": params.get("lambda_phase", LAMBDA_PHASE),
                    "prototype_control": "canonical"}
            row = metric_row(res, meta)
            RUN_METRICS.append(row)
            rows.append(row)
            for class_name in data["class_names"]:
                report = res["report"].get(class_name, {})
                RUN_CLASS_METRICS.append({
                    **{k: meta.get(k) for k in ("protocol", "corpus", "model", "objective", "seed", "epochs", "lambda_ce", "lambda_focal", "lambda_dim", "lambda_int", "lambda_phase", "prototype_control")},
                    "emotion": class_name, "precision": report.get("precision", 0.0),
                    "recall": report.get("recall", 0.0), "f1": report.get("f1-score", 0.0), "support": report.get("support", 0),
                })
            save_run_artifacts(res, data, meta, run_dir)
            if seed == seeds[0]:
                representative_histories[name] = res["history"]
            del res
            gc.collect()
        del data
        gc.collect()
    return rows, representative_histories


def _run_weight_sensitivity(seeds: Sequence[int], epochs: int, data_root: Path, run_dir: Path) -> List[Dict[str, Any]]:
    """One-at-a-time sensitivity on the RAVDESS PsyCNN hybrid model."""
    tests: List[Tuple[str, float, Dict[str, float]]] = []
    for v in [0.0, 0.25, 0.5, 1.0, 2.0]:
        tests.append(("lambda_ce", v, {"lambda_ce": v}))
    for v in [0.0, 0.1, 0.3, 0.6, 1.0]:
        tests.append(("lambda_dim", v, {"lambda_dim": v}))
    for v in [0.0, 0.25, 0.5, 1.0]:
        tests.append(("lambda_focal", v, {"lambda_focal": v}))
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        data = prepare_corpus("ravdess", data_root, seed)
        for parameter, value, params in tests:
            res = train_one(data, "cnn", epochs, seed, objective="hybrid", **params, verbose=False)
            meta = {"protocol": "weight_sensitivity", "corpus": "ravdess", "model": "cnn",
                    "objective": "hybrid", "seed": seed, "epochs": epochs,
                    "split_strategy": "utterance", "class_names": data["class_names"],
                    "lambda_ce": params.get("lambda_ce", LAMBDA_CE),
                    "lambda_focal": params.get("lambda_focal", LAMBDA_FOCAL),
                    "lambda_dim": params.get("lambda_dim", LAMBDA_DIM),
                    "lambda_int": LAMBDA_INT, "lambda_phase": LAMBDA_PHASE,
                    "sensitivity_parameter": parameter, "sensitivity_value": value,
                    "prototype_control": "canonical"}
            row = metric_row(res, meta)
            RUN_METRICS.append(row)
            rows.append(row)
            del res
            gc.collect()
        del data
        gc.collect()
    return rows
