"""Milestone 1 — Automatic CWRU Bearing Dataset Loader.

Scans a directory of raw CWRU ``.mat`` files, infers the fault class,
severity, orientation, and load condition directly from each filename via
regex (no hardcoded filenames), and loads the Drive-End (DE) accelerometer
channel plus shaft RPM for every recording.

Real CWRU filenames encountered in this project (flat directory, not the
Normal/Ball/InnerRace/OuterRace sub-folder layout some CWRU mirrors use):

    Normal_0.mat .. Normal_3.mat          -> healthy baseline, load 0-3 hp
    B007_0.mat .. B021_3.mat              -> Ball fault, severity 007/014/021 mil
    IR007_0.mat .. IR021_3.mat            -> Inner Race fault, severity 007/014/021 mil
    OR0076_0.mat .. OR0216_3.mat          -> Outer Race fault, severity 007/014/021 mil
                                              + clock-position digit (e.g. 6 o'clock)

Each .mat file exposes MATLAB variables named like ``X118_DE_time`` and
``X118RPM`` where the numeric prefix is an internal CWRU experiment ID that
varies per file and must not be assumed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as sio

from config import CONFIG, DatasetConfig

logger = logging.getLogger(__name__)

# Matches: Normal_0.mat | B007_0.mat | IR014_2.mat | OR0076_3.mat
_FILENAME_PATTERN = re.compile(
    r"^(?P<fault_type>Normal|B|IR|OR)(?P<code>\d{3,4})?_(?P<load>\d)\.mat$",
    re.IGNORECASE,
)

_FAULT_TYPE_NAMES = {
    "normal": "Normal",
    "b": "Ball",
    "ir": "InnerRace",
    "or": "OuterRace",
}


class DatasetLoadError(Exception):
    """Raised when a raw .mat file cannot be parsed or loaded."""


@dataclass(frozen=True)
class BearingRecording:
    """A single loaded CWRU recording with its parsed metadata."""

    file_path: Path
    fault_type: str  # Normal | Ball | InnerRace | OuterRace
    severity_mil: Optional[int]  # fault diameter in thousandths of an inch
    orientation_oclock: Optional[int]  # OuterRace fault position, if present
    load_hp: int  # motor load condition, 0-3 hp
    rpm: int
    label: int  # binary label per DatasetConfig.label_map
    signal: np.ndarray  # 1-D float32 Drive-End vibration signal


def _parse_filename(filename: str) -> dict:
    """Extract fault type, severity, orientation, and load from a filename.

    Parameters
    ----------
    filename:
        Base filename, e.g. ``"OR0076_2.mat"``.

    Returns
    -------
    dict with keys: fault_type, severity_mil, orientation_oclock, load_hp.

    Raises
    ------
    DatasetLoadError
        If the filename does not match the expected CWRU naming convention.
    """
    match = _FILENAME_PATTERN.match(filename)
    if not match:
        raise DatasetLoadError(
            f"Filename '{filename}' does not match expected CWRU pattern "
            f"(e.g. Normal_0.mat, B007_1.mat, IR014_2.mat, OR0076_3.mat)."
        )

    raw_type = match.group("fault_type").lower()
    fault_type = _FAULT_TYPE_NAMES[raw_type]
    code = match.group("code")
    load_hp = int(match.group("load"))

    severity_mil: Optional[int] = None
    orientation_oclock: Optional[int] = None

    if fault_type == "OuterRace" and code is not None:
        # e.g. "0076" -> severity 007, orientation 6 o'clock
        severity_mil = int(code[:3])
        orientation_oclock = int(code[3:]) if len(code) > 3 else None
    elif code is not None:
        severity_mil = int(code)

    return {
        "fault_type": fault_type,
        "severity_mil": severity_mil,
        "orientation_oclock": orientation_oclock,
        "load_hp": load_hp,
    }


def _extract_channel_arrays(mat_dict: dict, config: DatasetConfig) -> tuple:
    """Locate the DE-time signal and RPM arrays inside a loaded .mat dict.

    CWRU files prefix every variable with an internal experiment ID (e.g.
    ``X118_DE_time``), so we search by suffix rather than assuming an exact
    key name.
    """
    signal_key = next(
        (k for k in mat_dict if k.endswith(config.channel_suffix)), None
    )
    rpm_key = next(
        (k for k in mat_dict if k.endswith(config.rpm_suffix) and "_" not in k),
        None,
    )

    if signal_key is None:
        raise DatasetLoadError(
            f"No key ending in '{config.channel_suffix}' found. "
            f"Available keys: {[k for k in mat_dict if not k.startswith('__')]}"
        )

    signal = np.asarray(mat_dict[signal_key], dtype=np.float32).flatten()
    rpm = int(mat_dict[rpm_key].flatten()[0]) if rpm_key is not None else -1

    return signal, rpm


def _load_single_file(
    file_path: Path, config: DatasetConfig
) -> BearingRecording:
    """Load and parse a single .mat recording into a BearingRecording."""
    metadata = _parse_filename(file_path.name)

    try:
        mat_dict = sio.loadmat(str(file_path))
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise DatasetLoadError(f"Failed to read '{file_path}': {exc}") from exc

    signal, rpm = _extract_channel_arrays(mat_dict, config)

    label = config.label_map[metadata["fault_type"]]

    return BearingRecording(
        file_path=file_path,
        fault_type=metadata["fault_type"],
        severity_mil=metadata["severity_mil"],
        orientation_oclock=metadata["orientation_oclock"],
        load_hp=metadata["load_hp"],
        rpm=rpm,
        label=label,
        signal=signal,
    )


def load_cwru_dataset(
    raw_dir: Optional[Path] = None,
    config: DatasetConfig = CONFIG.dataset,
) -> list:
    """Load every CWRU ``.mat`` recording found under ``raw_dir``.

    Parameters
    ----------
    raw_dir:
        Directory containing CWRU .mat files (flat, non-recursive). Defaults
        to ``config.raw_dir``.
    config:
        Dataset configuration (channel suffix, label mapping, etc.).

    Returns
    -------
    list[BearingRecording]
        One entry per successfully loaded .mat file. Files that fail to
        parse or load are skipped with a logged warning, not silently
        dropped.

    Raises
    ------
    DatasetLoadError
        If the directory does not exist or contains no loadable files.
    """
    directory = Path(raw_dir) if raw_dir is not None else config.raw_dir

    if not directory.is_dir():
        raise DatasetLoadError(f"Dataset directory not found: {directory}")

    mat_files = sorted(directory.glob("*.mat"))
    if not mat_files:
        raise DatasetLoadError(f"No .mat files found in {directory}")

    recordings: list = []
    skipped: list = []

    for file_path in mat_files:
        try:
            recording = _load_single_file(file_path, config)
            recordings.append(recording)
            logger.info(
                "Loaded %s -> fault=%s severity=%s load=%dhp rpm=%d samples=%d",
                file_path.name,
                recording.fault_type,
                recording.severity_mil,
                recording.load_hp,
                recording.rpm,
                recording.signal.shape[0],
            )
        except DatasetLoadError as exc:
            logger.warning("Skipping '%s': %s", file_path.name, exc)
            skipped.append(file_path.name)

    if not recordings:
        raise DatasetLoadError(
            f"All {len(mat_files)} files in {directory} failed to load."
        )

    if skipped:
        logger.warning(
            "Loaded %d/%d files; skipped: %s",
            len(recordings),
            len(mat_files),
            skipped,
        )
    else:
        logger.info("Successfully loaded all %d files.", len(recordings))

    return recordings


def build_manifest(recordings: list):
    """Build a pandas DataFrame manifest summarizing all loaded recordings.

    Kept separate from ``load_cwru_dataset`` so the loader itself has no
    pandas dependency and can be unit-tested with plain lists.
    """
    import pandas as pd  # local import: manifest is optional tooling

    rows = [
        {
            "file": rec.file_path.name,
            "fault_type": rec.fault_type,
            "severity_mil": rec.severity_mil,
            "orientation_oclock": rec.orientation_oclock,
            "load_hp": rec.load_hp,
            "rpm": rec.rpm,
            "label": rec.label,
            "num_samples": rec.signal.shape[0],
            "duration_sec": round(
                rec.signal.shape[0] / CONFIG.dataset.sample_rate_hz, 2
            ),
        }
        for rec in recordings
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    CONFIG_INSTANCE = CONFIG
    from config import ensure_directories

    ensure_directories()

    recordings = load_cwru_dataset()
    manifest = build_manifest(recordings)

    manifest_path = CONFIG.dataset.raw_dir.parent.parent.parent / "outputs" / "dataset_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"\nLoaded {len(recordings)} recordings.")
    print(manifest.groupby(["fault_type", "load_hp"]).size().unstack(fill_value=0))
    print(f"\nClass balance (0=Healthy, 1=Faulty):")
    print(manifest["label"].value_counts())
    print(f"\nManifest saved to: {manifest_path}")
