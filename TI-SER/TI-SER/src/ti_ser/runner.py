"""Top-level experiment suite and CLI argument parser."""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .compat import np
from .config import DEFAULT_ABLATION_EPOCHS, DEFAULT_ABLATION_SEEDS, DEFAULT_EPOCHS, DEFAULT_SEEDS, DEFAULT_SENSITIVITY_EPOCHS, DEFAULT_SENSITIVITY_SEEDS, VA_PROTOTYPES
from .corpora import corpus_dir
from .experiments import _run_ablation_suite, _run_weight_sensitivity, run_scenario, summarise_rows
from .plotting import plot_ablation_curves, plot_acoustic_proxy_trajectories, plot_comparison, plot_training_curves
from .reporting import _aggregate_summary, _write_article_tables, _write_manifest, _write_results_readme
from .splits import prepare_leave_corpus_out
from .state import RUN_METRICS
from .utils import configure_logging, ensure_dependencies, list_wavs, write_csv, write_json


def run_experiments(args: argparse.Namespace) -> Path:
    ensure_dependencies()
    data_root = Path(args.data_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"The requested run directory is not empty: {run_dir}. "
                "Choose a new directory to avoid mixing outputs from different runs."
            )
    else:
        run_dir = output_root / f"TI_SER_run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(run_dir)
    figs = run_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)

    seeds = list(args.seeds)
    epochs = int(args.epochs)
    ablation_seeds = list(args.ablation_seeds)
    sensitivity_seeds = list(args.sensitivity_seeds)
    if args.quick:
        seeds = [seeds[0] if seeds else 42]
        ablation_seeds = []
        sensitivity_seeds = []
        epochs = min(2, epochs)
        args.skip_combined = True
        args.skip_leave_corpus_out = True
        args.skip_null_prototype = True
        args.skip_ablation = True
        args.skip_sensitivity = True
        args.skip_cremad = True
        logging.warning("QUICK MODE: smoke-test outputs are not manuscript-ready evidence.")

    logging.info("Output directory: %s", run_dir.resolve())
    logging.info("Seeds=%s epochs=%d", seeds, epochs)
    write_json({"args": vars(args), "effective_seeds": seeds, "effective_epochs": epochs}, run_dir / "invocation.json")

    # Figure 1-2: descriptive acoustic proxies.
    for i, corpus in enumerate(("ravdess", "tess"), start=1):
        try:
            plot_acoustic_proxy_trajectories(corpus, data_root, figs / f"figure_{i:02d}_{corpus}_acoustic_proxies.png")
        except Exception as exc:
            logging.exception("Could not generate acoustic-proxy figure for %s: %s", corpus, exc)

    # Small local helper keeps every within-corpus condition on the same split
    # manifest for a given corpus/seed and writes its run summary immediately.
    def run_within(corpus: str, model: str, objective: str,
                   figure_path: Optional[Path] = None, make_tsne: bool = False) -> Dict[str, Any]:
        result = run_scenario(
            corpus=corpus, protocol="within", model_type=model, objective=objective,
            seeds=seeds, epochs=epochs, data_root=data_root, run_dir=run_dir,
            figure_path=figure_path, make_tsne=make_tsne,
            extra_meta={"split_strategy": "utterance"},
        )
        write_json(result["summary"], run_dir / "tables" / f"summary_within_{corpus}_{model}_{objective}.json")
        if objective == "hybrid" and model == "cnn" and result["representative_result"] is not None:
            plot_training_curves(result["representative_result"]["history"],
                                 f"{corpus.upper()} CNN hybrid training curves (seed {seeds[0]})",
                                 figs / "supplementary" / f"training_within_{corpus}_cnn_hybrid.png")
        return result

    # Figure 3-4: first train the hybrid CNN and then visualize its test embeddings.
    # Fig. 1-2 above are acoustic summaries, so the visual outputs are now emitted
    # in the same order as the manuscript's figure numbering.
    for corpus in ("ravdess", "tess"):
        run_within(corpus, "cnn", "hybrid", make_tsne=True)

    # Figure 5-6: categorical CNN learning curves.
    for corpus, figure_no in (("ravdess", 5), ("tess", 6)):
        run_within(corpus, "cnn", "categorical",
                   figs / f"figure_{figure_no:02d}_{corpus}_cnn_categorical_curves.png")

    # Figure 7-8 and Transformer ablation table: run matched ablations before the
    # Transformer comparison figures. Ablations reuse a single split per seed.
    if not args.skip_ablation:
        ablation_sets = [
            ("ravdess", "cnn", figs / "figure_07_ravdess_cnn_ablation_curves.png"),
            ("tess", "cnn", figs / "figure_08_tess_cnn_ablation_curves.png"),
            ("ravdess", "transformer", figs / "supplementary" / "ravdess_transformer_ablation_curves.png"),
        ]
        for corpus, model, figure_path in ablation_sets:
            rows, histories = _run_ablation_suite(
                corpus, model, ablation_seeds or seeds, args.ablation_epochs, data_root, run_dir
            )
            write_csv(rows, run_dir / "tables" / f"ablation_runs_{corpus}_{model}.csv")
            ablation_summary = []
            for name in sorted(set(r["objective"] for r in rows)):
                group = [r for r in rows if r["objective"] == name]
                ablation_summary.append({"corpus": corpus, "model": model, "ablation": name, **summarise_rows(group)})
            write_csv(ablation_summary, run_dir / "tables" / f"ablation_summary_{corpus}_{model}.csv")
            plot_ablation_curves(
                histories,
                f"{corpus.upper()} {model.upper()} loss-component ablation (seed {(ablation_seeds or seeds)[0]})",
                figure_path,
            )

    # Figure 9-10: categorical Transformer within-corpus curves.
    for corpus, figure_no in (("ravdess", 9), ("tess", 10)):
        run_within(corpus, "transformer", "categorical",
                   figs / f"figure_{figure_no:02d}_{corpus}_transformer_categorical_curves.png")

    # Figure 11: categorical Transformer trained on the combined corpora.
    if not args.skip_combined:
        combined_transformer_baseline = run_scenario(
            "combined", "combined", "transformer", "categorical", seeds, epochs,
            data_root, run_dir,
            figure_path=figs / "figure_11_combined_transformer_categorical_curves.png",
            extra_meta={"split_strategy": "independent-stratified-per-corpus; shared-scaler"},
        )
        write_json(combined_transformer_baseline["summary"], run_dir / "tables" / "summary_combined_transformer_categorical.json")

    # Figure 12-13: hybrid Transformer within each corpus.
    for corpus, figure_no in (("ravdess", 12), ("tess", 13)):
        run_within(corpus, "transformer", "hybrid",
                   figs / f"figure_{figure_no:02d}_{corpus}_transformer_hybrid_curves.png")

    # Figure 14: hybrid Transformer trained on the combined corpora.
    if not args.skip_combined:
        combined_transformer_hybrid = run_scenario(
            "combined", "combined", "transformer", "hybrid", seeds, epochs,
            data_root, run_dir,
            figure_path=figs / "figure_14_combined_transformer_hybrid_curves.png",
            extra_meta={"split_strategy": "independent-stratified-per-corpus; shared-scaler"},
        )
        write_json(combined_transformer_hybrid["summary"], run_dir / "tables" / "summary_combined_transformer_hybrid.json")

    # Remaining combined CNN experiments do not have a numbered main-paper curve,
    # but are needed for the CNN combined-dataset result table.
    if not args.skip_combined:
        for objective in ("categorical", "hybrid"):
            result = run_scenario(
                "combined", "combined", "cnn", objective, seeds, epochs,
                data_root, run_dir,
                extra_meta={"split_strategy": "independent-stratified-per-corpus; shared-scaler"},
            )
            write_json(result["summary"], run_dir / "tables" / f"summary_combined_cnn_{objective}.json")

    # Section 7.3: zero-shot transfer, source scaler only, no target adaptation.
    if not args.skip_leave_corpus_out:
        for source, target in (("ravdess", "tess"), ("tess", "ravdess")):
            for model in ("cnn", "transformer"):
                def builder(seed: int, s=source, t=target):
                    return prepare_leave_corpus_out(s, t, data_root, seed)
                result = run_scenario(
                    corpus=f"{source}_to_{target}", protocol="leave_corpus_out", model_type=model,
                    objective="hybrid", seeds=seeds, epochs=epochs, data_root=data_root, run_dir=run_dir,
                    data_builder=builder, extra_meta={"source_corpus": source, "target_corpus": target,
                                                       "split_strategy": "source train/validation; full target held out"},
                )
                write_json(result["summary"], run_dir / "tables" / f"summary_lco_{source}_to_{target}_{model}.json")

    # Null-prototype control; this is not a substitute for the canonical-prototype runs.
    if not args.skip_null_prototype:
        null_proto = np.zeros_like(VA_PROTOTYPES)
        for corpus in ("ravdess", "tess"):
            for model in ("cnn", "transformer"):
                result = run_scenario(corpus, "null_prototype", model, "hybrid", seeds, epochs,
                                      data_root, run_dir, prototypes=null_proto,
                                      extra_meta={"control": "all-zero VA prototypes"})
                write_json(result["summary"], run_dir / "tables" / f"summary_nullproto_{corpus}_{model}.json")

    # Table 11: one-at-a-time loss-weight sensitivity using matched splits.
    if not args.skip_sensitivity:
        sens_rows = _run_weight_sensitivity(sensitivity_seeds or seeds, args.sensitivity_epochs, data_root, run_dir)
        write_csv(sens_rows, run_dir / "tables" / "weight_sensitivity_per_seed.csv")

    # CREMA-D is optional; if present, use a speaker-disjoint six-class protocol.
    cremad_dir = corpus_dir("cremad", data_root)
    cremad_files = list_wavs(cremad_dir / "clean") if (cremad_dir / "clean").exists() else list_wavs(cremad_dir)
    if not args.skip_cremad and cremad_files:
        cremad_seeds = seeds
        for model in ("cnn", "transformer"):
            for objective in ("categorical", "hybrid"):
                run_scenario(
                    corpus="cremad", protocol="cremad_speaker_independent", model_type=model,
                    objective=objective, seeds=cremad_seeds, epochs=epochs, data_root=data_root,
                    run_dir=run_dir, prototypes=VA_PROTOTYPES[:6].copy(), split_strategy="speaker",
                    extra_meta={"split_strategy": "speaker-disjoint"},
                )
    elif not args.skip_cremad:
        logging.warning("CREMA-D files not found; CREMA-D experiment block skipped. Expected: %s", cremad_dir / "clean")

    # Supplemental aggregate plot and ordered tables.
    summary_rows = _aggregate_summary(RUN_METRICS)
    plot_comparison(summary_rows, figs / "supplementary" / "performance_comparison.png")
    _write_article_tables(run_dir)
    _write_manifest(run_dir, args, status="completed", effective_seeds=seeds, effective_epochs=epochs)
    _write_results_readme(run_dir, seeds, epochs, args)
    logging.info("Experiment suite completed. Results: %s", run_dir.resolve())
    logging.info("Main paper figures are numbered figure_01 through figure_14; supplementary figures are under figures/supplementary/.")
    logging.info("Tables and run-level CSV/JSON outputs are under %s", (run_dir / "tables").resolve())
    return run_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-aware TI-SER result and figure generator")
    parser.add_argument("--data-root", default="data", help="Parent folder containing RAVDESS/, TESS/, and optional CREMA-D/")
    parser.add_argument("--output-dir", default="results", help="Root folder for timestamped experiment output")
    parser.add_argument("--run-dir", default=None, help="Optional exact output directory; existing files may be overwritten")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Seeds for primary/baseline/transfer experiments")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Epoch budget for primary experiments")
    parser.add_argument("--ablation-seeds", nargs="+", type=int, default=DEFAULT_ABLATION_SEEDS)
    parser.add_argument("--ablation-epochs", type=int, default=DEFAULT_ABLATION_EPOCHS)
    parser.add_argument("--sensitivity-seeds", nargs="+", type=int, default=DEFAULT_SENSITIVITY_SEEDS)
    parser.add_argument("--sensitivity-epochs", type=int, default=DEFAULT_SENSITIVITY_EPOCHS)
    parser.add_argument("--quick", action="store_true", help="Smoke test only: one seed, 2 epochs, core within-corpus suite")
    parser.add_argument("--skip-combined", action="store_true")
    parser.add_argument("--skip-leave-corpus-out", action="store_true")
    parser.add_argument("--skip-null-prototype", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-cremad", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.ablation_epochs < 1 or args.sensitivity_epochs < 1:
        parser.error("Epoch counts must be positive integers")
    return args
