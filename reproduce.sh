#!/usr/bin/env bash
# Reproduce every reported comparison with the corrected pipeline.
# Usage: ./reproduce.sh [SEEDS]      e.g. ./reproduce.sh 42,43,44,45,46
set -euo pipefail
SEEDS="${1:-42,43,44,45,46}"

python scripts/download_data.py --corpus all --out data      # RAVDESS + TESS
python scripts/smoke_test.py                                  # shapes / loss variants
python -m pytest tests -q                                     # leakage + split checks

# Main (within / combined / leave-corpus-out) and ablations; every condition is
# matched on split, seed, initial weights and batch order. Re-running skips finished runs.
python scripts/run_experiments.py --only all --seeds "$SEEDS"

# Per-seed tables, mean±SD, paired tests vs the categorical-only baseline.
python scripts/aggregate_results.py --results results
echo "See results/summary/report.md, per_run.csv, paired_tests.csv and results/splits/*.json"
