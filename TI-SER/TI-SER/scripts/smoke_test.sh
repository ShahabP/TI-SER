#!/usr/bin/env bash
# One seed, 2 epochs, within-corpus RAVDESS/TESS only.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/run_experiments.py --quick --data-root data --output-dir results_smoke
