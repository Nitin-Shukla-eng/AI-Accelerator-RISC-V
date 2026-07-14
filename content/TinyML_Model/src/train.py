"""Milestone 4 — Model Architecture and Training Pipeline.

Builds the PRD-specified 4-8-2 MLP (4 input features -> 8 ReLU hidden units
-> 2 softmax outputs) and trains it with EarlyStopping, ModelCheckpoint, and
full training-history logging. TensorFlow/Keras only, per PRD (no CNN/LSTM/
Transformer).
"""

from __future__ import annotations

import json
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import CONFIG, MODELS_DIR, OUTPUTS_DIR, TrainConfig, ensure_directories
from src.feature_extraction import build_feature_dataset
from src.load_dataset import load_cwru_dataset
from src.preprocessing import SplitData, stratified_split
from src.segmentation import segment_recordings

logger = logging.getLogger(__name__)


def build_model(config: TrainConfig = CONFIG.train) -> keras.Model:
    """Construct the 4-8-2 MLP specified by the PRD.

    Architecture
    ------------
    Input(4) -> Dense(8, ReLU) -> Dense(2, Softmax)

    No convolutional, recurrent, or attention layers, per PRD constraint —
    this keeps the model small enough for a hand-rolled FPGA accelerator
    (a handful of INT8 multiply-accumulates per layer).
    """
    tf.random.set_seed(config.random_seed)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(config.input_features,), name="features"),
            keras.layers.Dense(config.hidden_units, activation="relu", name="hidden"),
            keras.layers.Dense(config.output_units, activation="softmax", name="output"),
        ],
        name="bearing_fault_mlp",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_class_weights(y_train: np.ndarray) -> dict:
    """Inverse-frequency class weights to counter the Healthy/Faulty imbalance.

    Without this, a model can score ~72% accuracy by always predicting
    "Faulty" (the majority class in this dataset) while being useless for
    actually catching healthy machines correctly.
    """
    classes, counts = np.unique(y_train, return_counts=True)
    total = counts.sum()
    weights = {int(c): float(total / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}
    logger.info("Class weights (imbalance correction): %s", weights)
    return weights


def train_model(
    split: SplitData,
    config: TrainConfig = CONFIG.train,
    checkpoint_name: str = "best_model.keras",
) -> tuple:
    """Train the 4-8-2 MLP with EarlyStopping + ModelCheckpoint.

    Returns
    -------
    (model, history_dict)
    """
    ensure_directories()
    model = build_model(config)
    checkpoint_path = MODELS_DIR / checkpoint_name

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    class_weights = compute_class_weights(split.y_train)

    history = model.fit(
        split.X_train,
        split.y_train,
        validation_data=(split.X_val, split.y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    history_path = OUTPUTS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)
    logger.info("Saved training history to %s", history_path)
    logger.info("Saved best model checkpoint to %s", checkpoint_path)

    return model, history.history


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ensure_directories()

    recordings = load_cwru_dataset()
    windows = segment_recordings(recordings)
    feature_df = build_feature_dataset(windows)

    split = stratified_split(feature_df)
    model, history = train_model(split)

    final_epoch = len(history["loss"])
    print(f"\nTraining stopped after {final_epoch} epochs (EarlyStopping).")
    print(f"Final train acc: {history['accuracy'][-1]:.4f} | val acc: {history['val_accuracy'][-1]:.4f}")

    # Save fitted scaler + split arrays for the next milestone (evaluation).
    import joblib

    joblib.dump(split.scaler, MODELS_DIR / "scaler.joblib")
    np.savez(
        OUTPUTS_DIR / "test_split.npz",
        X_test=split.X_test,
        y_test=split.y_test,
    )
    print(f"Scaler saved to {MODELS_DIR / 'scaler.joblib'}")
    print(f"Test split saved to {OUTPUTS_DIR / 'test_split.npz'}")


if __name__ == "__main__":
    main()
