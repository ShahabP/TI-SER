#!/usr/bin/env bash
# Reproduce every reported comparison with the corrected pipeline.
# Usage: ./reproduce.sh [SEEDS]      e.g. ./reproduce.sh 42,43,44,45,46
set -euo pipefail
SEEDS="${1:-42,43,44,45,46}"

python scripts/download_data.py --corpus all --out data      # RAVDESS + TESS (+ CREMA-D docs)
python scripts/smoke_test.py                                  # shapes / loss variants
python -m pytest tests -q                                     # leakage + split checks

# Primary (within / combined / leave-corpus-out, 30 epochs), ablations (15 epochs,
# paper protocol: CNN 3 seeds both corpora; Transformer 1 seed RAVDESS), CREMA-D
# actor-disjoint (12 epochs, seeds 42–44, six classes, speaker-disjoint splits).
# Every condition is matched on split, seed, initial weights and batch order.
# Re-running skips finished runs.
python scripts/run_experiments.py --only all --seeds "$SEEDS"

# Per-seed tables, mean±SD, descriptive paired differences (Appendix B7 style).
# Pass --inferential for optional t-tests / Wilcoxon / Holm.
python scripts/aggregate_results.py --results results
echo "See results/summary/report.md, per_run.csv, paired_differences.csv and results/splits/*.json"
