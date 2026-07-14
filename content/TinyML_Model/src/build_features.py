"""Driver: Milestones 1-3 — Load -> Segment -> Extract Features.

Produces outputs/features_dataset.csv containing the 4 PRD-specified
features (rms, peak, std, crest_factor) + label for every window, plus
traceability metadata for auditing and the later cross-load robustness
evaluation.

Run with:  PYTHONPATH=. python3 src/build_features.py
"""

from __future__ import annotations

import logging

from config import CONFIG, OUTPUTS_DIR, ensure_directories
from src.feature_extraction import build_feature_dataset, feature_names
from src.load_dataset import build_manifest, load_cwru_dataset
from src.segmentation import segment_recordings

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ensure_directories()

    # Milestone 1
    recordings = load_cwru_dataset()
    manifest = build_manifest(recordings)
    manifest.to_csv(OUTPUTS_DIR / "dataset_manifest.csv", index=False)

    # Milestone 2
    windows = segment_recordings(recordings, CONFIG.segmentation)

    # Milestone 3
    feature_df = build_feature_dataset(windows)
    feature_df.to_csv(OUTPUTS_DIR / "features_dataset.csv", index=False)

    # --- Summary ---
    print(f"\nRecordings loaded : {len(recordings)}")
    print(f"Windows extracted : {len(windows)} "
          f"(window_size={CONFIG.segmentation.window_size}, "
          f"overlap={CONFIG.segmentation.overlap})")
    print(f"Features per row  : {feature_names()}")
    print(f"\nLabel balance (0=Healthy, 1=Faulty):")
    print(feature_df["label"].value_counts())
    print(f"\nFeature summary statistics:")
    print(feature_df[feature_names()].describe().T[["mean", "std", "min", "max"]])
    print(f"\nSaved: {OUTPUTS_DIR / 'features_dataset.csv'}")


if __name__ == "__main__":
    main()
