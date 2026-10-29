# TI-SER — Theory-Inspired Neural Networks for Prosodic Speech Emotion Recognition

Reference implementation of **PsyCNN** (1D residual CNN) and **PsyTransformer** (compact Transformer encoder),
which add auxiliary valence–arousal prototype supervision and trajectory regularisation to a categorical
SER objective. Evaluated on acted RAVDESS and TESS (within-corpus, combined, leave-corpus-out) plus an optional
CREMA-D comparison. The reported findings are descriptive and limited to acted speech; the VA targets are weak
prototype-derived engineering priors, not human affect ratings.

## Repository layout

```
TI-SER/
├── src/ti_ser/
│   ├── config.py        # constants, loss weights, VA prototypes (Table 3), label maps
│   ├── corpora.py       # file discovery + label parsing (RAVDESS / TESS / CREMA-D)
│   ├── features.py      # MFCC+Δ+ΔΔ, RMS envelope, scaling, SpecAugment-style augmentation
│   ├── splits.py        # leakage-aware splits; combined and leave-corpus-out preparation
│   ├── targets.py       # prototype × RMS-envelope reference trajectories (Eq. 28)
│   ├── models.py        # PsyCNN / PsyTransformer
│   ├── losses.py        # CE+focal, dimensional, intensity, phase-aware, joint objective
│   ├── training.py      # training loop and test evaluation
│   ├── plotting.py      # figures (curves, confusion matrices, proxies, t-SNE, ablations)
│   ├── experiments.py   # per-scenario orchestration, ablations, loss-weight sensitivity
│   ├── reporting.py     # aggregation, article tables, run README + manifest
│   ├── runner.py        # full experiment suite + CLI arguments
│   ├── state.py, utils.py, compat.py, cli.py
├── scripts/             # run_experiments.py, run_full_manuscript.sh, smoke_test.sh
├── notebooks/colab_launcher.py   # single-cell Colab launcher (original behaviour)
├── tests/               # config/label/split/target tests (+ TF model & loss tests)
├── docs/paper_to_code.md         # equation/table → module map and code-vs-paper differences
├── data/                # put RAVDESS/TESS/(CREMA-D) WAVs here, see data/README.md
└── legacy/TI-SER_original_colab_export.py   # the untouched single-file script
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.10+
pip install -e .            # or: pip install -r requirements.txt
```

## Run

```bash
# smoke test: 1 seed, 2 epochs, within-corpus RAVDESS/TESS only
bash scripts/smoke_test.sh

# full manuscript configuration (5 seeds x 20 epochs; ablation/sensitivity 3 seeds x 15 epochs) — heavy
bash scripts/run_full_manuscript.sh

# or with explicit options
ti-ser --data-root data --output-dir results --seeds 42 43 44 45 46 --epochs 20
python -m ti_ser --help
```

Useful flags: `--skip-combined`, `--skip-leave-corpus-out`, `--skip-null-prototype`, `--skip-ablation`,
`--skip-sensitivity`, `--skip-cremad`, `--run-dir`.

Each run writes a timestamped folder under `results/` with per-run metrics, class reports, confusion matrices,
split manifests, aggregate CSV tables, `figure_01…figure_14` plus supplementary figures, a manifest and a log.

## Methodology guarantees (as implemented)

* Original utterance files are split **before** augmentation; only training data is augmented.
* The feature scaler is fit on valid training frames only (one shared scaler for the combined setting;
  source-train only for leave-corpus-out).
* Within-/combined-corpus splits are stratified 65:15:20 and **not speaker-independent**; CREMA-D uses a
  speaker-disjoint split.
* Model input is MFCC(40)+Δ+ΔΔ (D=120, T=174). RMS is only the envelope for the weak prototype trajectories.

See `docs/paper_to_code.md` for the exact mapping and for the places where the code and the manuscript text
currently disagree.

## Tests

```bash
pip install -e ".[dev]" && pytest
```
TensorFlow-dependent tests are skipped automatically if TensorFlow is not installed.
