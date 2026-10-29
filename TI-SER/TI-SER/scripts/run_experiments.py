#!/usr/bin/env python3
"""Thin wrapper so the suite can be run without installing the package.

    python scripts/run_experiments.py --data-root data --output-dir results
    python scripts/run_experiments.py --quick     # smoke test
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ti_ser.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
