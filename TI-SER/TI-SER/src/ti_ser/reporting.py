"""Aggregation, article tables, run README and manifest."""

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .compat import np, librosa, tf
from .config import AUGMENT_TIMES, EMOTIONS, LAMBDA_CE, LAMBDA_DIM, LAMBDA_FOCAL, LAMBDA_INT, LAMBDA_PHASE, MAX_FRAMES, PHASE_CURVATURE_WEIGHT, PHASE_SLOPE_WEIGHT, PHASE_WINDOW, VA_PROTOTYPES
from .experiments import summarise_rows
from .plotting import plot_weight_sensitivity
from .state import RUN_CLASS_METRICS, RUN_METRICS, SPLIT_MANIFEST
from .utils import write_csv, write_json


def _aggregate_summary(metrics: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = ["protocol", "corpus", "model", "objective", "prototype_control", "sensitivity_parameter", "sensitivity_value", "lambda_ce", "lambda_focal", "lambda_dim", "lambda_int", "lambda_phase"]
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        grouped[tuple(row.get(k) for k in keys)].append(row)
    out = []
    for key, rows in grouped.items():
        record = {k: v for k, v in zip(keys, key) if v is not None}
        record.update(summarise_rows(rows))
        out.append(record)
    return sorted(out, key=lambda r: (str(r.get("protocol", "")), str(r.get("corpus", "")), str(r.get("model", "")), str(r.get("objective", "")), str(r.get("sensitivity_parameter", "")), float(r.get("sensitivity_value", -1))))


def _aggregate_class_metrics(class_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = ["protocol", "corpus", "model", "objective", "prototype_control", "emotion"]
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in class_rows:
        # For sensitivity tables, retain the sensitivity identity in the summary elsewhere;
        # the regular per-class table focuses on core protocol conditions/ablations.
        grouped[tuple(row.get(k) for k in keys)].append(row)
    out = []
    for key, rows in grouped.items():
        record = {k: v for k, v in zip(keys, key)}
        record["n_seeds"] = len(set(r.get("seed") for r in rows))
        for metric in ("precision", "recall", "f1", "support"):
            vals = np.asarray([float(r.get(metric, 0)) for r in rows])
            record[f"{metric}_mean"] = float(vals.mean())
            record[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out.append(record)
    return sorted(out, key=lambda r: (str(r["protocol"]), str(r["corpus"]), str(r["model"]), str(r["objective"]), str(r["emotion"])))


def _filter(rows: Sequence[Dict[str, Any]], **equals: Any) -> List[Dict[str, Any]]:
    return [r for r in rows if all(r.get(k) == v for k, v in equals.items())]


def _write_article_tables(run_dir: Path) -> None:
    tables = run_dir / "tables"
    summary = _aggregate_summary(RUN_METRICS)
    classes = _aggregate_class_metrics(RUN_CLASS_METRICS)
    write_csv(RUN_METRICS, tables / "all_runs_metrics.csv")
    write_csv(RUN_CLASS_METRICS, tables / "all_runs_class_metrics.csv")
    write_csv(summary, tables / "summary_mean_std.csv")
    write_csv(classes, tables / "class_metrics_mean_std.csv")

    # Manuscript table sequence. All generated metric values come from this run.
    named = [
        ("table_07_ravdess_cnn_categorical_per_class.csv", _filter(classes, protocol="within", corpus="ravdess", model="cnn", objective="categorical")),
        ("table_08_tess_cnn_categorical_per_class.csv", _filter(classes, protocol="within", corpus="tess", model="cnn", objective="categorical")),
        ("table_09_ravdess_cnn_hybrid_per_class.csv", _filter(classes, protocol="within", corpus="ravdess", model="cnn", objective="hybrid")),
        ("table_10_tess_cnn_hybrid_per_class.csv", _filter(classes, protocol="within", corpus="tess", model="cnn", objective="hybrid")),
        ("table_12_combined_cnn_hybrid_per_class.csv", _filter(classes, protocol="combined", corpus="combined", model="cnn", objective="hybrid")),
        ("table_15_combined_transformer_categorical_per_class.csv", _filter(classes, protocol="combined", corpus="combined", model="transformer", objective="categorical")),
        ("table_19_ravdess_transformer_ablation_per_class.csv", _filter(classes, protocol="ablation", corpus="ravdess", model="transformer")),
        ("table_21_cremad_hybrid_per_class.csv", _filter(classes, protocol="cremad_speaker_independent", corpus="cremad", objective="hybrid")),
    ]
    for filename, rows in named:
        if rows:
            write_csv(rows, tables / filename)

    # Overall tables for baseline/hybrid comparisons and cross-corpus transfer.
    for filename, protocol, corpus, model, objective in [
        ("table_14_within_transformer_categorical.csv", "within", None, "transformer", "categorical"),
        ("table_16_combined_transformer_categorical.csv", "combined", "combined", "transformer", "categorical"),
        ("table_17_within_transformer_hybrid.csv", "within", None, "transformer", "hybrid"),
        ("table_18_combined_transformer_hybrid.csv", "combined", "combined", "transformer", "hybrid"),
        ("table_20_leave_corpus_out.csv", "leave_corpus_out", None, None, "hybrid"),
        ("table_21_cremad_overall.csv", "cremad_speaker_independent", "cremad", None, None),
    ]:
        rows = [r for r in summary if r.get("protocol") == protocol and (corpus is None or r.get("corpus") == corpus)
                and (model is None or r.get("model") == model) and (objective is None or r.get("objective") == objective)]
        if rows:
            write_csv(rows, tables / filename)

    # Table 19: aggregate scores for each Transformer loss ablation.
    table19 = [r for r in summary if r.get("protocol") == "ablation" and r.get("corpus") == "ravdess" and r.get("model") == "transformer"]
    if table19:
        write_csv(table19, tables / "table_19_ravdess_transformer_ablation_overall.csv")

    # Table 11 loss-weight sensitivity: only use computed sensitivity results.
    sens = [r for r in RUN_METRICS if r.get("protocol") == "weight_sensitivity"]
    if sens:
        sens_summary: Dict[Tuple[str, float], List[Dict[str, Any]]] = defaultdict(list)
        for r in sens:
            sens_summary[(str(r.get("sensitivity_parameter")), float(r.get("sensitivity_value")))].append(r)
        sens_rows = []
        for (parameter, value), group in sorted(sens_summary.items()):
            s = summarise_rows(group)
            sens_rows.append({"parameter": parameter, "value": value, **s})
        write_csv(sens_rows, tables / "table_11_ravdess_cnn_weight_sensitivity.csv")
        plot_weight_sensitivity(sens_rows, run_dir / "figures" / "supplementary" / "weight_sensitivity.png")

    # Paired, same-seed comparisons for the objective effect. A paired difference
    # is more informative than comparing metrics from unrelated splits.
    grouped_by_pair: Dict[Tuple[str, str, str, int], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in RUN_METRICS:
        if row.get("protocol") in {"within", "combined", "cremad_speaker_independent"} and row.get("objective") in {"hybrid", "categorical"}:
            grouped_by_pair[(str(row.get("protocol")), str(row.get("corpus")), str(row.get("model")), int(row.get("seed")))][str(row.get("objective"))] = row
    paired = []
    for (protocol, corpus, model, seed), pair in sorted(grouped_by_pair.items()):
        if "hybrid" in pair and "categorical" in pair:
            paired.append({"protocol": protocol, "corpus": corpus, "model": model, "seed": seed,
                           "hybrid_minus_categorical_accuracy": pair["hybrid"]["accuracy"] - pair["categorical"]["accuracy"],
                           "hybrid_minus_categorical_macro_f1": pair["hybrid"]["macro_f1"] - pair["categorical"]["macro_f1"]})
    if paired:
        write_csv(paired, tables / "paired_hybrid_vs_categorical_differences.csv")

    # Benchmark table retains only published values cited in the supplied manuscript
    # as context; it explicitly avoids preserving the old, unrecomputed PsyCNN value.
    current_ravdess = [r for r in summary if r.get("protocol") == "within" and r.get("corpus") == "ravdess" and r.get("model") == "cnn" and r.get("objective") == "hybrid"]
    benchmark_rows = [
        {"method": "EmoHRNet", "metric": "accuracy_percent", "reported_value": 92.45, "source_type": "published value cited in submitted manuscript", "comparison_note": "Protocol/class mapping differs; contextual only"},
        {"method": "AMH-Net", "metric": "accuracy_percent", "reported_value": 92.10, "source_type": "published value cited in submitted manuscript", "comparison_note": "Protocol/class mapping differs; contextual only"},
        {"method": "Frozen wav2vec 2.0 embeddings", "metric": "unweighted_average_recall_percent", "reported_value": 84.30, "source_type": "published value cited in submitted manuscript", "comparison_note": "Different metric and protocol; not directly comparable"},
    ]
    for r in current_ravdess:
        benchmark_rows.append({"method": "PsyCNN (recomputed by this runner)", "metric": "accuracy_percent", "reported_value": 100 * r["accuracy_mean"], "reported_std": 100 * r["accuracy_std"], "n_seeds": r.get("n_seeds", ""), "source_type": "current corrected run", "comparison_note": "Seven-class mapping; utterance-level split; report only with its protocol limitations"})
    write_csv(benchmark_rows, tables / "table_13_benchmark_context.csv")


def _write_results_readme(run_dir: Path, seeds: Sequence[int], epochs: int, args: argparse.Namespace) -> None:
    figure_map = [
        ("figure_01_ravdess_acoustic_proxies.png", "RAVDESS class-averaged acoustic proxies"),
        ("figure_02_tess_acoustic_proxies.png", "TESS class-averaged acoustic proxies"),
        ("figure_03_ravdess_tsne.png", "RAVDESS hybrid CNN test embeddings (t-SNE)"),
        ("figure_04_tess_tsne.png", "TESS hybrid CNN test embeddings (t-SNE)"),
        ("figure_05_ravdess_cnn_categorical_curves.png", "RAVDESS categorical CNN training/validation curves"),
        ("figure_06_tess_cnn_categorical_curves.png", "TESS categorical CNN training/validation curves"),
        ("figure_07_ravdess_cnn_ablation_curves.png", "RAVDESS CNN loss ablation curves"),
        ("figure_08_tess_cnn_ablation_curves.png", "TESS CNN loss ablation curves"),
        ("figure_09_ravdess_transformer_categorical_curves.png", "RAVDESS categorical Transformer curves"),
        ("figure_10_tess_transformer_categorical_curves.png", "TESS categorical Transformer curves"),
        ("figure_11_combined_transformer_categorical_curves.png", "Combined-data categorical Transformer curves"),
        ("figure_12_ravdess_transformer_hybrid_curves.png", "RAVDESS hybrid Transformer curves"),
        ("figure_13_tess_transformer_hybrid_curves.png", "TESS hybrid Transformer curves"),
        ("figure_14_combined_transformer_hybrid_curves.png", "Combined-data hybrid Transformer curves"),
    ]
    table_map = [
        ("table_07_ravdess_cnn_categorical_per_class.csv", "RAVDESS CNN categorical per-class results"),
        ("table_08_tess_cnn_categorical_per_class.csv", "TESS CNN categorical per-class results"),
        ("table_09_ravdess_cnn_hybrid_per_class.csv", "RAVDESS CNN hybrid per-class results"),
        ("table_10_tess_cnn_hybrid_per_class.csv", "TESS CNN hybrid per-class results"),
        ("table_11_ravdess_cnn_weight_sensitivity.csv", "RAVDESS CNN one-at-a-time loss-weight sensitivity"),
        ("table_12_combined_cnn_hybrid_per_class.csv", "Combined CNN hybrid per-class results"),
        ("table_13_benchmark_context.csv", "Contextual literature comparison; not a controlled benchmark"),
        ("table_14_within_transformer_categorical.csv", "Within-corpus categorical Transformer results"),
        ("table_15_combined_transformer_categorical_per_class.csv", "Combined categorical Transformer per-class results"),
        ("table_16_combined_transformer_categorical.csv", "Combined categorical Transformer overall results"),
        ("table_17_within_transformer_hybrid.csv", "Within-corpus hybrid Transformer results"),
        ("table_18_combined_transformer_hybrid.csv", "Combined hybrid Transformer results"),
        ("table_19_ravdess_transformer_ablation_overall.csv", "RAVDESS Transformer loss ablation results"),
        ("table_20_leave_corpus_out.csv", "Zero-shot leave-corpus-out results"),
        ("table_21_cremad_overall.csv", "CREMA-D speaker-independent results (if data were available)"),
    ]
    lines = [
        "# TI-SER experiment outputs",
        "",
        f"Seeds: {', '.join(map(str, seeds))}",
        f"Primary epochs: {epochs}",
        f"Data root: `{Path(args.data_root).resolve()}`",
        "",
        "## Required manuscript figures, in manuscript order",
        "",
    ]
    for filename, description in figure_map:
        exists = (run_dir / "figures" / filename).exists()
        lines.append(f"- {'[x]' if exists else '[ ]'} `figures/{filename}` — {description}{'' if exists else ' (not generated; check skipped experiment block or data availability)'}")
    lines += ["", "## Table outputs", ""]
    for filename, description in table_map:
        exists = (run_dir / "tables" / filename).exists()
        lines.append(f"- {'[x]' if exists else '[ ]'} `tables/{filename}` — {description}{'' if exists else ' (not generated; check skipped experiment block or data availability)'}")
    lines += [
        "",
        "## Other important files",
        "",
        "- `tables/all_runs_metrics.csv`: one row per evaluated model/seed/configuration.",
        "- `tables/all_runs_class_metrics.csv`: per-class precision, recall, F1 and support per run.",
        "- `tables/summary_mean_std.csv`: mean and standard deviation across seeds.",
        "- `tables/paired_hybrid_vs_categorical_differences.csv`: matched-seed hybrid-minus-baseline differences.",
        "- `tables/split_manifest_original_files.csv`: original-file identities and split memberships.",
        "- `run_manifest.json`: effective configuration, package versions and protocol notes.",
        "- `run.log`: chronological run log.",
        "- `runs/`: per-run histories, confusion matrices, reports and metadata.",
        "",
        "## Submission cautions",
        "",
        "- These are newly calculated results from this run, not a reproduction of earlier summary values unless the setup and data match.",
        "- Review any unchecked figure/table above before using the PDF for submission.",
        "- Within-corpus results use utterance-level splits, not unseen-speaker splits.",
        "- Use the same class inventory, prototypes, phase-loss settings, and split manifests described in the final manuscript.",
        "- Model inputs are MFCC+delta+delta-delta only; RMS is a target-envelope proxy and centroid/bandwidth support descriptive figures.",
        "- The literature comparison is contextual because published protocols and metrics differ.",
        "- CREMA-D results are produced only when the files are present and a speaker-disjoint split can be formed.",
    ]
    (run_dir / "README_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(run_dir: Path, args: argparse.Namespace, status: str,
                    effective_seeds: Sequence[int], effective_epochs: int) -> None:
    write_csv([row for rows in SPLIT_MANIFEST.values() for row in rows], run_dir / "tables" / "split_manifest_original_files.csv")
    manifest = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "tensorflow_version": getattr(tf, "__version__", None),
        "numpy_version": np.__version__,
        "librosa_version": getattr(librosa, "__version__", None),
        "data_root": str(Path(args.data_root).resolve()),
        "output_dir": str(run_dir.resolve()),
        "seeds": list(effective_seeds),
        "epochs": int(effective_epochs),
        "augmentation_times": AUGMENT_TIMES,
        "feature_definition": "Model input: 40 MFCC + delta + delta-delta; 120 dimensions per frame. RMS is target-only for the weak trajectory envelope; spectral centroid/RMS/bandwidth are used for descriptive figures.",
        "frame_length": MAX_FRAMES,
        "split": "65% train / 15% validation / 20% test for within/combined; original files split before augmentation",
        "scaler": "StandardScaler fit on valid frames in training partition only; common scaler for combined-corpus experiments",
        "augmentation": "SpecAugment-style masking after split; training partition only",
        "label_inventory": EMOTIONS,
        "va_prototypes": VA_PROTOTYPES.tolist(),
        "loss_weights": {"lambda_ce": LAMBDA_CE, "lambda_focal": LAMBDA_FOCAL, "lambda_dim": LAMBDA_DIM, "lambda_int": LAMBDA_INT, "lambda_phase": LAMBDA_PHASE},
        "phase_loss_internal_weights": {"peak_window_delta": PHASE_WINDOW, "slope_weight": PHASE_SLOPE_WEIGHT, "curvature_weight": PHASE_CURVATURE_WEIGHT},
        "notes": [
            "All saved performance values were computed by this run; old table values are not copied into model-result tables.",
            "Within-corpus experiments use utterance-level splits and are not speaker-independent.",
            "TESS contains two speakers; very high random-utterance split results must not be treated as unseen-speaker generalization.",
            "Acoustic proxy figures are descriptive signals, not continuous human affect ratings.",
            "CREMA-D is run only if its WAV files are present. Its split is speaker-disjoint.",
        ],
    }
    write_json(manifest, run_dir / "run_manifest.json")
