"""Preprocessing — Feature Scaling and Train/Val/Test Splitting.

Two split strategies are implemented, both requested for this project:

1. Primary split (`stratified_split`): stratified random split on the
   window-level dataset. This is the main split used for model training,
   validation, and the headline accuracy/F1/ROC metrics.

2. Bonus robustness split (`cross_load_split`): trains on motor-load
   conditions {0, 1, 2} hp and tests purely on the held-out load condition
   {3} hp, which the model never sees during training. This measures
   generalization to an unseen operating condition, which a random split
   cannot measure (random split lets windows from the same recording end up
   in both train and test, inflating apparent accuracy).

Both splits use StandardScaler fit ONLY on the training partition, then
applied to val/test, to avoid data leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import CONFIG, SplitConfig
from src.feature_extraction import feature_names

logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Raised when scaling or splitting cannot proceed as configured."""


@dataclass
class SplitData:
    """Container for a scaled train/val/test split, ready for training."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    strategy: str


def stratified_split(
    feature_df: pd.DataFrame, config: SplitConfig = CONFIG.split
) -> SplitData:
    """Primary split: stratified random split at the window level.

    Parameters
    ----------
    feature_df:
        Output of ``feature_extraction.build_feature_dataset``.
    config:
        Split configuration (test_size, val_size, random_seed, stratify).

    Returns
    -------
    SplitData with scaled train/val/test arrays and the fitted scaler.
    """
    cols = feature_names()
    X = feature_df[cols].to_numpy(dtype=np.float32)
    y = feature_df["label"].to_numpy(dtype=np.int64)

    stratify_arg = y if config.stratify else None
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_seed, stratify=stratify_arg
    )

    stratify_arg_2 = y_train_full if config.stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=config.val_size,
        random_state=config.random_seed,
        stratify=stratify_arg_2,
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    logger.info(
        "Stratified split -> train=%d val=%d test=%d (label balance train=%s)",
        len(y_train), len(y_val), len(y_test),
        dict(zip(*np.unique(y_train, return_counts=True))),
    )

    return SplitData(
        X_train=X_train_s, y_train=y_train,
        X_val=X_val_s, y_val=y_val,
        X_test=X_test_s, y_test=y_test,
        scaler=scaler, strategy="stratified",
    )


def cross_load_split(
    feature_df: pd.DataFrame, config: SplitConfig = CONFIG.split
) -> SplitData:
    """Bonus robustness split: hold out one entire load condition for test.

    Trains on every load EXCEPT ``config.cross_load_holdout``; tests only on
    that held-out load. A validation slice is carved out of the training
    loads (stratified) so early stopping doesn't peek at the test load.

    Raises
    ------
    PreprocessingError
        If the holdout load condition has no rows in ``feature_df``.
    """
    cols = feature_names()
    holdout = config.cross_load_holdout

    train_mask = feature_df["load_hp"] != holdout
    test_mask = feature_df["load_hp"] == holdout

    if test_mask.sum() == 0:
        raise PreprocessingError(
            f"No rows found for cross_load_holdout={holdout}hp; check config."
        )

    train_df = feature_df[train_mask]
    test_df = feature_df[test_mask]

    X_train_full = train_df[cols].to_numpy(dtype=np.float32)
    y_train_full = train_df["label"].to_numpy(dtype=np.int64)
    X_test = test_df[cols].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.int64)

    stratify_arg = y_train_full if config.stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=config.val_size,
        random_state=config.random_seed,
        stratify=stratify_arg,
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    logger.info(
        "Cross-load split (holdout=%dhp) -> train=%d val=%d test=%d",
        holdout, len(y_train), len(y_val), len(y_test),
    )

    return SplitData(
        X_train=X_train_s, y_train=y_train,
        X_val=X_val_s, y_val=y_val,
        X_test=X_test_s, y_test=y_test,
        scaler=scaler, strategy=f"cross_load_holdout_{holdout}hp",
    )
