"""Split / leakage-guard checks that do not need audio or TensorFlow."""
from pathlib import Path

import numpy as np
import pytest

from ti_ser.splits import _standard_file_split, check_disjoint


def test_standard_split_ratios_and_disjointness():
    n_per_class, n_classes = 100, 7
    labels = np.repeat(np.arange(n_classes), n_per_class)
    files = [Path(f"f{i}.wav") for i in range(len(labels))]
    tr, va, te = _standard_file_split(files, labels, seed=42)
    assert len(tr) + len(va) + len(te) == len(files)
    assert len(te) / len(files) == pytest.approx(0.20, abs=0.01)
    assert len(va) / len(files) == pytest.approx(0.15, abs=0.01)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    for part in (tr, va, te):
        assert set(labels[part].tolist()) == set(range(n_classes))  # stratified


def test_check_disjoint_raises_on_overlap():
    a, b = [Path("a.wav"), Path("b.wav")], [Path("b.wav")]
    with pytest.raises(RuntimeError):
        check_disjoint(a, b, [Path("c.wav")])
