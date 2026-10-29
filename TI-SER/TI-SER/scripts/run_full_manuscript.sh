#!/usr/bin/env bash
# Full manuscript configuration (heavy): 5 seeds x 20 epochs for primary runs,
# 3 seeds x 15 epochs for ablations and loss-weight sensitivity.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/run_experiments.py \
  --data-root data --output-dir results \
  --seeds 42 43 44 45 46 --epochs 20 \
  --ablation-seeds 42 43 44 --ablation-epochs 15 \
  --sensitivity-seeds 42 43 44 --sensitivity-epochs 15
