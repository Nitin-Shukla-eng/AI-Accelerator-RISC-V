"""Milestone 3 — Feature Extraction.

Extracts exactly the four statistical time-domain features specified in the
PRD: RMS, Peak, Standard Deviation, and Crest Factor. Features are computed
via a small registry pattern so that additional features (e.g. kurtosis,
skewness, spectral features) can be added later without touching the
extraction driver logic (`extract_features_from_window` /
`build_feature_dataset`).
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from src.segmentation import SegmentedWindow

logger = logging.getLogger(__name__)

# Ordered registry: insertion order defines the feature vector column order
# fed to the model (must match PRD Input Layer order: RMS, Peak, STD, Crest).
_FEATURE_REGISTRY: dict[str, Callable[[np.ndarray], float]] = {}


def register_feature(name: str) -> Callable:
    """Decorator that registers a function as a named feature extractor.

    Every registered function must accept a 1-D window array and return a
    single float. New features can be added anywhere in this module (or
    imported from elsewhere and registered) without modifying the driver
    functions below.
    """

    def decorator(func: Callable[[np.ndarray], float]) -> Callable[[np.ndarray], float]:
        if name in _FEATURE_REGISTRY:
            raise ValueError(f"Feature '{name}' is already registered.")
        _FEATURE_REGISTRY[name] = func
        return func

    return decorator


@register_feature("rms")
def compute_rms(window: np.ndarray) -> float:
    """Root Mean Square — overall vibration energy of the window."""
    return float(np.sqrt(np.mean(np.square(window))))


@register_feature("peak")
def compute_peak(window: np.ndarray) -> float:
    """Peak amplitude — maximum absolute deviation in the window."""
    return float(np.max(np.abs(window)))


@register_feature("std")
def compute_std(window: np.ndarray) -> float:
    """Standard deviation — spread of the vibration signal around its mean."""
    return float(np.std(window))


@register_feature("crest_factor")
def compute_crest_factor(window: np.ndarray) -> float:
    """Crest Factor = Peak / RMS — indicates presence of sharp impacts.

    Depends on ``compute_rms`` and ``compute_peak`` rather than the registry,
    so its value is always internally consistent even if those two are
    later overridden.
    """
    rms_val = compute_rms(window)
    peak_val = compute_peak(window)
    if rms_val == 0.0:
        logger.warning("RMS is zero for a window; crest factor set to 0.0 to avoid div-by-zero.")
        return 0.0
    return peak_val / rms_val


def feature_names() -> list[str]:
    """Return the ordered list of currently registered feature names."""
    return list(_FEATURE_REGISTRY.keys())


def extract_features_from_window(window: np.ndarray) -> dict[str, float]:
    """Compute every registered feature for a single window.

    Returns
    -------
    dict mapping feature name -> value, in registry (insertion) order.
    """
    return {name: func(window) for name, func in _FEATURE_REGISTRY.items()}


def build_feature_dataset(segments: list[SegmentedWindow]) -> pd.DataFrame:
    """Convert a list of SegmentedWindow objects into a tabular feature set.

    Parameters
    ----------
    segments:
        Output of ``segmentation.segment_recordings``.

    Returns
    -------
    pd.DataFrame
        One row per window with columns: [rms, peak, std, crest_factor,
        label] plus traceability metadata (fault_type, severity_mil,
        load_hp, source_file, window_index). The metadata columns are kept
        for auditing/debugging and cross-load evaluation splits, but are
        NOT fed to the model — only the 4 PRD features + label are used
        for training.
    """
    if not segments:
        raise ValueError("segments list is empty; nothing to extract features from.")

    rows = []
    for seg in segments:
        features = extract_features_from_window(seg.window)
        rows.append(
            {
                **features,
                "label": seg.label,
                "fault_type": seg.fault_type,
                "severity_mil": seg.severity_mil,
                "load_hp": seg.load_hp,
                "rpm": seg.rpm,
                "source_file": seg.source_file,
                "window_index": seg.window_index,
            }
        )

    df = pd.DataFrame(rows)

    # Sanity checks: NaNs or infs would silently corrupt scaling/training.
    feature_cols = feature_names()
    row_is_finite = np.isfinite(df[feature_cols].to_numpy()).all(axis=1)
    n_bad = int((~row_is_finite).sum())
    if n_bad:
        logger.warning(
            "%d/%d rows contain non-finite feature values and should be reviewed.",
            n_bad,
            len(df),
        )

    logger.info(
        "Built feature dataset: %d rows x %d features %s",
        len(df),
        len(feature_cols),
        feature_cols,
    )
    return df
