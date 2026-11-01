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
```

Features (40 MFCC + Δ + ΔΔ, energy) are cached on first load.

## Quick start

```bash
# Synthetic smoke test (no audio required)
python scripts/smoke_test.py

# Train PsyCNN on RAVDESS (hybrid objective)
python scripts/train.py --corpus ravdess --model psyCNN --seed 42

# Categorical-only baseline
python scripts/train.py --corpus ravdess --model psyCNN --categorical-only

# Full protocol (within / combined / leave-corpus-out / ablations)
python scripts/run_experiments.py --config configs/default.yaml
```

Checkpoints and metrics are written under `results/`.

## Repository layout

```
configs/default.yaml          # hyper-parameters
scripts/
  train.py                    # single run
  run_experiments.py          # full protocol
  leave_corpus_out.py         # cross-corpus transfer
  ablation.py                 # loss-component ablations
  download_data.py
  smoke_test.py
src/
  data/                       # prototypes, features, datasets
  models/                     # PsyCNN, PsyTransformer
  losses/                     # hybrid objective
  training/                   # trainer, metrics
  utils/                      # scheduler, SpecAugment
```

## Method (brief)

| Item | Detail |
|------|--------|
| Features | 40 MFCC + Δ + ΔΔ (D=120); energy used only as envelope for prototypes |
| Prototypes | Fixed VA anchors (Russell / Mehrabian ordering) |
| Loss | L = L_cls + λ_dim L_dim + λ_int L_int + λ_phase L_phase |
| Schedule | Adam 1e-3, 5-epoch warmup + cosine decay, 20 epochs |
| Splits | 65/15/20 utterance-stratified (not speaker-independent) |

Default loss weights: `λ_dim=0.5`, `λ_int=0.3`, `λ_phase=0.2`.

## Citation

```bibtex
@article{pasha2025theory,
  title  = {Theory-Inspired Neural Networks for Prosodic Speech Emotion Recognition},
  author = {Pasha, Shahab and Zhao, Jiahong and Ritz, Christian},
  year   = {2025}
}
```

Datasets remain under their original licences.
