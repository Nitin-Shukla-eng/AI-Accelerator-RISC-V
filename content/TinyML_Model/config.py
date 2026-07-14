"""Central configuration for the TinyML Bearing Fault Detection pipeline.

All tunable parameters live here so that every stage (loading, segmentation,
feature extraction, training, quantization, export) reads from a single
source of truth. Nothing downstream should hardcode a path or hyperparameter
that belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATASET_RAW_DIR: Path = PROJECT_ROOT / "dataset" / "raw" / "cwru"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
MEM_DIR: Path = PROJECT_ROOT / "mem"
LOGS_DIR: Path = PROJECT_ROOT / "logs"


@dataclass(frozen=True)
class DatasetConfig:
    """Everything related to raw signal acquisition from CWRU .mat files."""

    raw_dir: Path = DATASET_RAW_DIR
    # CWRU 12 kHz Drive-End dataset. Verified against actual file durations
    # (Normal_0.mat = 243938 samples / 12000 Hz = 20.33 s; B007_0.mat =
    # 122571 samples / 12000 Hz = 10.21 s), both consistent with the
    # documented 12 kHz DE acquisition rate.
    sample_rate_hz: int = 12_000
    # Only the Drive-End accelerometer channel is used (per project scope).
    channel_suffix: str = "_DE_time"
    rpm_suffix: str = "RPM"
    # Binary task: 0 = Healthy (Normal), 1 = Faulty (Ball / InnerRace / OuterRace)
    label_map: dict = field(
        default_factory=lambda: {
            "Normal": 0,
            "Ball": 1,
            "InnerRace": 1,
            "OuterRace": 1,
        }
    )
    class_names: tuple = ("Healthy", "Faulty")


@dataclass(frozen=True)
class SegmentationConfig:
    """Sliding-window segmentation of raw vibration signals."""

    window_size: int = 1024  # samples per window (~85 ms @ 12 kHz)
    overlap: float = 0.5  # fraction of window overlapped by the next window


@dataclass(frozen=True)
class SplitConfig:
    """Train/validation/test split strategy."""

    test_size: float = 0.2
    val_size: float = 0.1  # fraction of the remaining train data
    random_seed: int = 42
    stratify: bool = True
    # Bonus robustness check: after the primary stratified split, a second
    # evaluation trains on load conditions {0, 1, 2} and tests purely on
    # held-out load condition {3}, to test generalization to an unseen
    # operating condition. Toggle via run_cross_load_eval.
    run_cross_load_eval: bool = True
    cross_load_holdout: int = 3


@dataclass(frozen=True)
class TrainConfig:
    """Model architecture and training hyperparameters."""

    input_features: int = 4  # RMS, Peak, STD, Crest Factor
    hidden_units: int = 8
    output_units: int = 2
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    random_seed: int = 42


@dataclass(frozen=True)
class QuantizationConfig:
    """INT8 post-training quantization settings."""

    weights_dtype: str = "int8"
    bias_dtype: str = "int32"
    representative_dataset_samples: int = 200


@dataclass(frozen=True)
class PipelineConfig:
    """Aggregate config object passed through the pipeline."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)


CONFIG = PipelineConfig()


def ensure_directories() -> None:
    """Create all project output directories if they do not already exist."""
    for directory in (MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, MEM_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
