#!/usr/bin/env python3
"""Run the paper's PsyCNN zero-prototype control on RAVDESS and TESS.

Add this file to the repository as ``scripts/zero_prototype_control.py`` and run
from the repository root:

    python scripts/zero_prototype_control.py

By default this runs PsyCNN on RAVDESS and TESS with seeds 42--46.  It reuses
TI-SER's normal experiment pipeline, so the saved split files, training-only
normalisation, training-only augmentation, initialisation, optimiser, model
selection, checkpoints, predictions, and run records follow the repository's
existing implementation.

Control definition
------------------
Every class's VA prototype is replaced by the same [0, 0] vector.  The normal
energy-envelope calculation is retained, so z*(t) = eta(t) * [0, 0] = [0, 0]
and I*(t) = ||z*(t)|| = 0 for every frame.  The PsyCNN VA head and the
lambda_dim, lambda_int, and lambda_phase losses remain active.  This is the
zero-target control described in the manuscript; it is NOT the
categorical-only/no-auxiliary-loss condition and is NOT the shuffled-prototype
control.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import src.pipeline as pipeline  # noqa: E402
import src.losses.hybrid_loss as hybrid_loss_module  # noqa: E402

ZERO_VARIANT = "full_zero_proto"
PAPER_SEEDS = [42, 43, 44, 45, 46]
PAPER_CORPORA = ["ravdess", "tess"]
PAPER_LOSS_WEIGHTS = {
    "lambda_ce": 1.0,
    "lambda_focal": 1.0,
    "lambda_dim": 0.5,
    "lambda_int": 0.3,
    "lambda_phase": 0.2,
}

# Keep a reference to the repository's real prototype loader.  We use it only
# to obtain the correct tensor shape/device/dtype, then replace every value by
# zero.  energy_modulated_targets() itself is NOT replaced: it still computes
# eta from frame energy and applies eta * prototype, exactly as in the codebase.
_original_get_prototype_batch = hybrid_loss_module.get_prototype_batch


def _zero_prototype_batch(labels, mode="va", device=None):
    if mode != "va":
        raise ValueError(
            "The paper's zero-prototype experiment is VA-only; expected mode='va'."
        )
    real_batch = _original_get_prototype_batch(labels, mode=mode, device=device)
    return torch.zeros_like(real_batch)


def install_zero_prototype_control() -> None:
    """Register the variant and make the loss use class-independent zero VA anchors."""
    # pipeline.build_loss() looks up flags in this registry.  All auxiliary
    # terms are explicitly on; no prototype permutation/shuffle is applied.
    pipeline.VARIANTS[ZERO_VARIANT] = {
        "dim": True,
        "int": True,
        "phase": True,
        "zero_prototype": True,
    }
    hybrid_loss_module.get_prototype_batch = _zero_prototype_batch


def _configure_paper_protocol(cfg: dict[str, Any], epochs: int) -> None:
    """Pin the manuscript's within-corpus zero-control protocol."""
    cfg["model"]["name"] = "psyCNN"
    cfg["model"]["num_classes"] = 7

    cfg["data"]["seven_class"] = True
    cfg["data"]["split_mode"] = "stratified"  # utterance-level, not speaker-disjoint
    cfg["data"]["train_ratio"] = 0.65
    cfg["data"]["val_ratio"] = 0.15

    cfg["loss"]["mode"] = "va"
    cfg["loss"].update(PAPER_LOSS_WEIGHTS)

    # The manuscript places the null-prototype condition with the 15-epoch
    # PsyCNN ablations, rather than the longer primary experiments.
    cfg["train"]["epochs"] = epochs


def _self_check(cfg: dict[str, Any]) -> None:
    """Fail early if zero targets or any auxiliary loss is accidentally disabled."""
    device = torch.device("cpu")
    labels = torch.tensor([0, 1], dtype=torch.long, device=device)
    energy = torch.tensor(
        [[0.1, 0.4, 0.8, 0.3], [0.7, 0.2, 0.5, 0.9]], dtype=torch.float32
    )
    mask = torch.ones((2, 4), dtype=torch.bool)

    z_star, intensity_star = hybrid_loss_module.energy_modulated_targets(
        labels, energy, mask, mode="va"
    )
    if torch.count_nonzero(z_star).item() != 0:
        raise RuntimeError("Zero-prototype check failed: VA frame targets are not zero.")
    if torch.count_nonzero(intensity_star).item() != 0:
        raise RuntimeError("Zero-prototype check failed: intensity targets are not zero.")

    loss_fn = pipeline.build_loss(cfg, ZERO_VARIANT)
    for name in ("lambda_dim", "lambda_int", "lambda_phase"):
        if getattr(loss_fn, name) <= 0:
            raise RuntimeError(f"The {name} auxiliary loss is disabled; expected it to be active.")

    # Smoke-test the real loss call path and check that every returned term is finite.
    logits = torch.randn(2, 7, requires_grad=True)
    va = torch.randn(2, 4, 2, requires_grad=True)
    intensity = torch.rand(2, 4, requires_grad=True)
    terms = loss_fn(
        logits=logits,
        va=va,
        intensity=intensity,
        labels=labels,
        energy=energy,
        mask=mask,
    )
    for name in ("loss", "L_cls", "L_dim", "L_int", "L_phase"):
        value = terms[name]
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Loss smoke test produced a non-finite {name} value.")
    terms["loss"].backward()
    if va.grad is None or intensity.grad is None:
        raise RuntimeError("VA/intensity heads did not receive gradients in the zero-target control.")


def _parse_csv_ints(value: str) -> list[int]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers, e.g. 42,43,44,45,46") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("Seed values must be unique.")
    return parsed


def _parse_corpora(value: str) -> list[str]:
    parsed = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not parsed or any(corpus not in PAPER_CORPORA for corpus in parsed):
        raise argparse.ArgumentTypeError("Corpora must be selected from: ravdess,tess")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("Corpus names must be unique.")
    return parsed


def _write_summary(records: list[dict[str, Any]], output_dir: Path, epochs: int) -> Path:
    """Write a compact mean/sample-SD summary while retaining all per-seed records."""
    rows: list[dict[str, Any]] = []
    for corpus in PAPER_CORPORA:
        selected = [record for record in records if record["corpus"] == corpus]
        if not selected:
            continue
        accuracies = [100.0 * float(record["test"]["accuracy"]) for record in selected]
        macro_f1s = [100.0 * float(record["test"]["macro_f1"]) for record in selected]
        rows.append(
            {
                "corpus": corpus,
                "n_seeds": len(selected),
                "seeds": ";".join(str(record["seed"]) for record in selected),
                "accuracy_mean_pct": statistics.mean(accuracies),
                "accuracy_sample_sd_pct": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "macro_f1_mean_pct": statistics.mean(macro_f1s),
                "macro_f1_sample_sd_pct": statistics.stdev(macro_f1s) if len(macro_f1s) > 1 else 0.0,
                "epochs": epochs,
                "split": "stratified_65_15_20_utterance_level",
                "control": "zero_VA_prototypes_auxiliary_losses_active",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "zero_prototype_control_summary.csv"
    if rows:
        with summary_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TI-SER PsyCNN zero-prototype control on RAVDESS and TESS."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Repository YAML config")
    parser.add_argument(
        "--corpora", type=_parse_corpora, default=PAPER_CORPORA,
        help="Comma-separated corpus list (default: ravdess,tess)",
    )
    parser.add_argument(
        "--seeds", type=_parse_csv_ints, default=PAPER_SEEDS,
        help="Comma-separated seeds (default: 42,43,44,45,46)",
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Ablation training budget (default: 15, as in the manuscript's ablation protocol)",
    )
    parser.add_argument("--device", default=None, help="e.g. cpu, cuda, cuda:0; default auto")
    parser.add_argument("--output-dir", default=None, help="Override results output directory")
    parser.add_argument("--force", action="store_true", help="Rerun even if a matching run record exists")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without training")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = ROOT / config_path
    if not config_path.is_file():
        parser.error(f"Configuration file not found: {config_path}")

    cfg = pipeline.load_config(str(config_path))
    _configure_paper_protocol(cfg, args.epochs)
    if args.output_dir:
        cfg["experiment"]["output_dir"] = args.output_dir

    install_zero_prototype_control()
    _self_check(cfg)

    output_root = Path(cfg["experiment"]["output_dir"])
    jobs = [(corpus, seed) for corpus in args.corpora for seed in args.seeds]
    print("Zero-prototype PsyCNN control")
    print(f"  Config       : {config_path}")
    print(f"  Corpora      : {', '.join(args.corpora)}")
    print(f"  Seeds        : {', '.join(str(s) for s in args.seeds)}")
    print(f"  Epochs       : {args.epochs}")
    print("  Split        : stratified utterance-level 65/15/20")
    print("  VA targets   : [0, 0] for every class and frame")
    print("  Loss weights : " + ", ".join(f"{key}={value}" for key, value in PAPER_LOSS_WEIGHTS.items()))
    print("  Active terms : classification + dimensional + intensity + phase")
    print(f"  Output       : {output_root / 'ablation'}")
    print(f"  Planned runs : {len(jobs)}")

    if args.dry_run:
        for corpus, seed in jobs:
            print(f"  [dry-run] psyCNN {corpus} {ZERO_VARIANT} seed={seed}")
        return

    # Check requested datasets before beginning the first run.
    root_keys = {"ravdess": "ravdess_root", "tess": "tess_root"}
    for corpus in args.corpora:
        data_root = Path(cfg["data"][root_keys[corpus]])
        if not data_root.exists():
            raise FileNotFoundError(
                f"{corpus} data directory not found: {data_root}. "
                f"Run: python scripts/download_data.py --corpus {corpus} --out data"
            )

    completed: list[dict[str, Any]] = []
    for corpus, seed in jobs:
        record = pipeline.run_experiment(
            cfg,
            model_name="psyCNN",
            variant=ZERO_VARIANT,
            corpus=corpus,
            seed=seed,
            group="ablation",
            device=args.device,
            force=args.force,
        )

        # Make the exact control definition explicit in each machine-readable record.
        run_dir = output_root / "ablation" / record["run_id"]
        record_path = run_dir / "run_record.json"
        if record_path.exists():
            with record_path.open("r", encoding="utf-8") as stream:
                saved_record = json.load(stream)
        else:
            saved_record = record
        saved_record["control"] = {
            "name": "zero_prototype_control",
            "prototype_values": "all class VA prototypes are [0.0, 0.0]",
            "frame_targets": "z_star(t) = eta(t) * [0, 0] = [0, 0]",
            "intensity_targets": "I_star(t) = ||z_star(t)||_2 = 0",
            "classification_loss_active": True,
            "lambda_dim_active": True,
            "lambda_int_active": True,
            "lambda_phase_active": True,
            "is_categorical_only": False,
            "is_prototype_shuffle": False,
        }
        saved_record.setdefault("effective_loss_weights", {})["prototype_zero"] = True
        saved_record["effective_loss_weights"]["prototype_shuffled"] = False
        saved_record["protocol_details"] = {
            "epochs": args.epochs,
            "split": "stratified utterance-level 65/15/20",
            "seeds_requested": args.seeds,
            "dataset_mapping": "shared seven-class mapping; RAVDESS calm mapped to neutral",
        }
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with record_path.open("w", encoding="utf-8") as stream:
            json.dump(saved_record, stream, indent=2)
        completed.append(saved_record)

    summary_path = _write_summary(completed, output_root / "ablation", args.epochs)
    print("\nFinished zero-prototype control.")
    print(f"Per-seed run records : {output_root / 'ablation' / 'psyCNN_<corpus>_full_zero_proto_seed<seed>' / 'run_record.json'}")
    print(f"Summary              : {summary_path}")
    print("Keep the run_record.json files and test_predictions.csv files as the reproducibility evidence.")


if __name__ == "__main__":
    main()
