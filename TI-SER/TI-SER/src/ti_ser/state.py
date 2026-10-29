"""Process-wide caches and result accumulators shared across modules (mutated in place)."""

from typing import Any, Dict, List, Tuple

from .compat import np


# Caches raw features within one process so repeated seeds/conditions do not
# repeatedly decode and featurize every waveform.
_RAW_FEATURE_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}


RUN_METRICS: List[Dict[str, Any]] = []


RUN_CLASS_METRICS: List[Dict[str, Any]] = []


SPLIT_MANIFEST: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
