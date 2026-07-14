"""Milestone 2 — Signal Segmentation.

Splits each raw vibration recording into fixed-length, optionally overlapping
windows. Segmentation is performed strictly per-recording so that no window
ever mixes samples from two different .mat files (which would leak
load/fault-condition boundaries into a single feature vector).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from config import CONFIG, SegmentationConfig
from src.load_dataset import BearingRecording

logger = logging.getLogger(__name__)


class SegmentationError(Exception):
    """Raised when a signal cannot be segmented with the given parameters."""


@dataclass(frozen=True)
class SegmentedWindow:
    """A single fixed-length window sliced from one BearingRecording."""

    window: np.ndarray  # shape (window_size,)
    label: int
    fault_type: str
    severity_mil: int | None
    load_hp: int
    rpm: int
    source_file: str
    window_index: int


def _compute_step(window_size: int, overlap: float) -> int:
    """Compute the hop size (in samples) between consecutive windows."""
    if not 0.0 <= overlap < 1.0:
        raise SegmentationError(f"overlap must be in [0, 1), got {overlap}")
    step = int(round(window_size * (1.0 - overlap)))
    if step < 1:
        raise SegmentationError(
            f"Computed step size {step} < 1 for window_size={window_size}, "
            f"overlap={overlap}. Reduce overlap."
        )
    return step


def segment_signal(
    signal: np.ndarray, window_size: int, overlap: float
) -> np.ndarray:
    """Slice a 1-D signal into overlapping fixed-length windows.

    Parameters
    ----------
    signal:
        1-D array of raw vibration samples.
    window_size:
        Number of samples per window.
    overlap:
        Fraction of each window that overlaps with the next, in [0, 1).

    Returns
    -------
    np.ndarray of shape (n_windows, window_size). Any trailing samples that
    don't fill a full window are dropped (not zero-padded), to avoid
    injecting synthetic zero-artifacts into feature statistics.

    Raises
    ------
    SegmentationError
        If the signal is shorter than one window or parameters are invalid.
    """
    if signal.ndim != 1:
        raise SegmentationError(f"Expected 1-D signal, got shape {signal.shape}")
    if signal.shape[0] < window_size:
        raise SegmentationError(
            f"Signal length {signal.shape[0]} shorter than window_size {window_size}"
        )

    step = _compute_step(window_size, overlap)
    n_windows = 1 + (signal.shape[0] - window_size) // step

    windows = np.empty((n_windows, window_size), dtype=signal.dtype)
    for i in range(n_windows):
        start = i * step
        windows[i] = signal[start : start + window_size]

    return windows


def segment_recordings(
    recordings: list[BearingRecording],
    config: SegmentationConfig = CONFIG.segmentation,
) -> list[SegmentedWindow]:
    """Segment every recording into windows, preserving per-recording labels.

    Parameters
    ----------
    recordings:
        Output of ``load_dataset.load_cwru_dataset``.
    config:
        Segmentation parameters (window_size, overlap).

    Returns
    -------
    list[SegmentedWindow]
        Flat list of all windows across all recordings.
    """
    all_windows: list[SegmentedWindow] = []
    skipped = 0

    for rec in recordings:
        try:
            windows = segment_signal(rec.signal, config.window_size, config.overlap)
        except SegmentationError as exc:
            logger.warning("Skipping '%s': %s", rec.file_path.name, exc)
            skipped += 1
            continue

        for idx in range(windows.shape[0]):
            all_windows.append(
                SegmentedWindow(
                    window=windows[idx],
                    label=rec.label,
                    fault_type=rec.fault_type,
                    severity_mil=rec.severity_mil,
                    load_hp=rec.load_hp,
                    rpm=rec.rpm,
                    source_file=rec.file_path.name,
                    window_index=idx,
                )
            )

        logger.info(
            "%s -> %d windows (window_size=%d, overlap=%.2f)",
            rec.file_path.name,
            windows.shape[0],
            config.window_size,
            config.overlap,
        )

    if not all_windows:
        raise SegmentationError("No windows produced from any recording.")

    if skipped:
        logger.warning("Skipped %d/%d recordings during segmentation.", skipped, len(recordings))

    logger.info(
        "Segmentation complete: %d total windows from %d recordings.",
        len(all_windows),
        len(recordings) - skipped,
    )
    return all_windows
