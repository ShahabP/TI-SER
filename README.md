# Theory-Inspired SER

Implementation of *Theory-Inspired Neural Networks for Prosodic Speech Emotion Recognition* (Pasha, Zhao & Ritz).

Two backbones share the same categorical + frame-level valence–arousal heads and hybrid objective:

- **PsyCNN** – 1-D residual convolutional network  
- **PsyTransformer** – compact Transformer encoder  

Training combines classification (CE + focal), weak VA prototype regression, intensity consistency, and phase-aware trajectory regularization (build-up / peak / recovery). Prototypes are fixed theory-derived anchors, not continuous human annotations.

Evaluated on acted RAVDESS and TESS (within-corpus, combined, leave-corpus-out). Results are descriptive and limited to these corpora.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

```bash
python scripts/download_data.py --corpus all --out data
```

Expected layout:

```
data/RAVDESS/Audio_Speech_Actors_01-24/...
data/TESS/...   # emotion folders (OAF_*, YAF_*)
data/CREMA-D/...  # AudioWAV/ (manual install; see scripts/download_data.py)
```

Features (40 MFCC + Δ + ΔΔ, energy) are cached on first load.

**Training budgets (matching the manuscript):** primary experiments 30 epochs; CREMA-D actor-disjoint 12 epochs; ablations 15 epochs.

## Quick start

```bash
python scripts/smoke_test.py                # synthetic data, no audio needed
python -m pytest tests -q                   # leakage / split / loss-variant checks

# one run
python scripts/train.py --corpus ravdess --model psyCNN --variant full --seed 42
python scripts/train.py --corpus ravdess --model psyCNN --categorical-only --seed 42

# everything in the paper (within, combined, leave-corpus-out, ablations) + statistics
./reproduce.sh                              # or: python scripts/run_experiments.py --only all
python scripts/aggregate_results.py         # -> results/summary/
```

Finished runs are skipped on restart (same config hash), so long jobs can be resumed.

## Evaluation protocol (what the code does)

| Item | Implementation |
|------|----------------|
| Splits | 65/15/20 train/val/test, **stratified** by class (and by corpus in `combined`); `split_mode: speaker_disjoint` is available for RAVDESS/CREMA-D. Split indices and file names are written to `results/splits/` and reused by every compared condition |
| Normalisation | per-dimension z-score, statistics from the **training split only** |
| Augmentation | SpecAugment on the **training split only** (val/test views never augment) |
| Model selection | best validation accuracy; test set touched once, after training |
| Matching | for a given (model, corpus, seed) every loss variant shares split, initial weights, batch order and training budget |
| Leave-corpus-out | train/val = stratified split of the source corpus, test = entire target corpus; categorical-only baseline trained under identical conditions |
| Ablations | full factorial over {dim, int, phase} + prototype-shuffle control, repeated seeds, same config as main runs |

Variants: `categorical` (CE + focal only), `dim`, `int`, `phase`, `dim_int`, `dim_phase`, `int_phase`,
`full` (= "hybrid"), `full_shuffled_proto`.

## Reproducibility artefacts

```
results/
  splits/<corpus>_<mode>_seed<seed>.json     split indices + file names + fingerprint
  <group>/<run_id>/run_record.json           seed, config, loss weights, fingerprints, metrics, environment
  <group>/<run_id>/test_predictions.csv      per-utterance predictions
  <group>/<run_id>/model_summary.txt         exact architecture
  summary/per_run.csv  summary.csv  paired_tests.csv  report.md  excluded_runs.txt
```

`aggregate_results.py` only uses records tagged with this pipeline version; anything else is
listed in `excluded_runs.txt` as historical. See `docs/REVIEWER_CHANGES.md` for the mapping from
reviewer comments to code changes.

## Repository layout

```
configs/default.yaml          # hyper-parameters
scripts/
  train.py                    # single run
  run_experiments.py          # full protocol
  leave_corpus_out.py         # cross-corpus transfer with matched baselines
  ablation.py                 # component ablations (factorial + control)
  aggregate_results.py        # per-seed tables, summaries, paired statistics
  download_data.py  smoke_test.py
src/
  pipeline.py                 # data prep, training, records (single source of truth)
  experiments.py              # run plans
  data/                       # prototypes, features, datasets, splits, normalisation
  models/  losses/  training/  utils/
tests/                        # leakage, split, loss-variant tests
docs/REVIEWER_CHANGES.md
```

## Method (brief)

| Item | Detail |
|------|--------|
| Features | 40 MFCC + Δ + ΔΔ (D=120); energy used only as envelope for prototypes |
| Prototypes | Fixed VA anchors (Russell / Mehrabian ordering) |
| Loss | L = L_cls + λ_dim L_dim + λ_int L_int + λ_phase L_phase, with L_cls = λ_ce·CE + λ_focal·focal |
| Schedule | Adam 1e-3, 5-epoch warmup + cosine decay; primary 30 epochs, CREMA-D 12, ablations 15 |

Default loss weights: `λ_dim=0.5`, `λ_int=0.3`, `λ_phase=0.2`. Splits are utterance-level and
therefore **not speaker-independent** unless `split_mode: speaker_disjoint` is used.

## Citation

```bibtex
@article{pasha2025theory,
  title  = {Theory-Inspired Neural Networks for Prosodic Speech Emotion Recognition},
  author = {Pasha, Shahab and Zhao, Jiahong and Ritz, Christian},
  year   = {2025}
}
```

Datasets remain under their original licences.
