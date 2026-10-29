# -*- coding: utf-8 -*-
"""Google Colab single-cell launcher for the full manuscript experiment suite.

Usage in Colab:
    !git clone <your-repo-url> TI-SER && pip install -q -e TI-SER
    # upload data to /content/data/{RAVDESS,TESS,CREMA-D}/clean/*.wav
    %run TI-SER/notebooks/colab_launcher.py
"""
import re
from pathlib import Path

from ti_ser.config import DEFAULT_SEEDS  # noqa: F401  (kept for reference)
from ti_ser.runner import parse_args, run_experiments

# =============================================================================
# Google Colab single-cell launcher: runs all experiments, saves all outputs,
# and previews every main figure plus contact sheets for supplementary figures.
# =============================================================================
from IPython.display import display as _display, Image as _IPythonImage, Markdown as _Markdown
from PIL import Image as _PILImage, ImageDraw as _ImageDraw
import shutil as _shutil

# EDIT ONLY THESE SETTINGS if your files are stored elsewhere or you want a
# different experiment budget. Expected folders are data/RAVDESS/clean/*.wav,
# data/TESS/clean/*.wav, and optional data/CREMA-D/clean/*.wav.
COLAB_DATA_ROOT = Path('/content/data')
COLAB_OUTPUT_ROOT = Path('/content/TI_SER_results')
COLAB_SEEDS = [42, 43, 44, 45, 46]
COLAB_EPOCHS = 20
COLAB_ABLATION_SEEDS = [42, 43, 44]
COLAB_ABLATION_EPOCHS = 15
COLAB_SENSITIVITY_SEEDS = [42, 43, 44]
COLAB_SENSITIVITY_EPOCHS = 15

# Parse only our known defaults; do not parse Colab's injected -f kernel.json arg.
_COLAB_ARGS = parse_args([])
_COLAB_ARGS.data_root = str(COLAB_DATA_ROOT)
_COLAB_ARGS.output_dir = str(COLAB_OUTPUT_ROOT)
_COLAB_ARGS.run_dir = None
_COLAB_ARGS.seeds = list(COLAB_SEEDS)
_COLAB_ARGS.epochs = int(COLAB_EPOCHS)
_COLAB_ARGS.ablation_seeds = list(COLAB_ABLATION_SEEDS)
_COLAB_ARGS.ablation_epochs = int(COLAB_ABLATION_EPOCHS)
_COLAB_ARGS.sensitivity_seeds = list(COLAB_SENSITIVITY_SEEDS)
_COLAB_ARGS.sensitivity_epochs = int(COLAB_SENSITIVITY_EPOCHS)
_COLAB_ARGS.quick = False
_COLAB_ARGS.skip_combined = False
_COLAB_ARGS.skip_leave_corpus_out = False
_COLAB_ARGS.skip_null_prototype = False
_COLAB_ARGS.skip_ablation = False
_COLAB_ARGS.skip_sensitivity = False
_COLAB_ARGS.skip_cremad = False

print('TI-SER full manuscript experiment suite')
print(f'Data root: {COLAB_DATA_ROOT}')
print(f'Output root: {COLAB_OUTPUT_ROOT}')
print(f'Primary seeds: {COLAB_SEEDS}; epochs: {COLAB_EPOCHS}')
print('This is the full run, not quick/smoke-test mode. It can take a long time.')

RUN_DIR = run_experiments(_COLAB_ARGS)
FIGURE_DIR = RUN_DIR / 'figures'

# Main manuscript figures are shown inline, one at a time.
_main_figures = sorted(
    [p for p in FIGURE_DIR.rglob('*.png') if re.match(r'^figure_\d{2}_', p.name)],
    key=lambda p: (int(re.match(r'^figure_(\d{2})_', p.name).group(1)), str(p))
)
_display(_Markdown(f'## Main-paper figures ({len(_main_figures)} files)'))
for _fig_path in _main_figures:
    _display(_Markdown(f'**{_fig_path.name}**'))
    _display(_IPythonImage(filename=str(_fig_path), width=1000))

# Supplementary diagnostics can include many per-seed confusion matrices.
# Save every original PNG, and preview all of them via compact contact sheets.
_all_pngs = sorted(FIGURE_DIR.rglob('*.png'))
_main_paths = set(_main_figures)
_supp = [p for p in _all_pngs if p not in _main_paths and not p.name.startswith('preview_supplementary_')]
_sheets = []
if _supp:
    _display(_Markdown(f'## Supplementary figure previews ({len(_supp)} files; originals are in the figures folder)'))
    _cols, _rows, _thumb_w, _thumb_h, _caption_h = 3, 4, 360, 220, 38
    _per_sheet = _cols * _rows
    for _sheet_idx, _start in enumerate(range(0, len(_supp), _per_sheet), start=1):
        _batch = _supp[_start:_start + _per_sheet]
        _canvas = _PILImage.new('RGB', (_cols * _thumb_w, _rows * (_thumb_h + _caption_h)), 'white')
        _draw = _ImageDraw.Draw(_canvas)
        for _i, _path in enumerate(_batch):
            try:
                _im = _PILImage.open(_path).convert('RGB')
                _im.thumbnail((_thumb_w - 12, _thumb_h - 10))
                _x = (_i % _cols) * _thumb_w + (_thumb_w - _im.width) // 2
                _y = (_i // _cols) * (_thumb_h + _caption_h)
                _canvas.paste(_im, (_x, _y))
                _caption = str(_path.relative_to(FIGURE_DIR))
                _draw.text(((_i % _cols) * _thumb_w + 6, _y + _thumb_h + 2), _caption[:52], fill='black')
            except Exception as _exc:
                print(f'Could not preview {_path}: {_exc}')
        _sheet_path = FIGURE_DIR / f'preview_supplementary_{_sheet_idx:02d}.jpg'
        _canvas.save(_sheet_path, quality=88)
        _sheets.append(_sheet_path)
        _display(_Markdown(f'**Supplementary contact sheet {_sheet_idx}**'))
        _display(_IPythonImage(filename=str(_sheet_path), width=1100))

# Bundle everything for easy downloading from Colab.
_BUNDLE_BASE = Path('/content/TI_SER_results_bundle')
_BUNDLE_PATH = Path(_shutil.make_archive(str(_BUNDLE_BASE), 'zip', root_dir=str(RUN_DIR)))
print('\nALL EXPERIMENTS FINISHED')
print(f'Results directory: {RUN_DIR}')
print(f'Figures directory: {FIGURE_DIR}')
print(f'CSV/JSON tables directory: {RUN_DIR / "tables"}')
print(f'Run ZIP: {_BUNDLE_PATH}')
print('In Colab, use the Files pane to open /content/TI_SER_results and download TI_SER_results_bundle.zip.')
