"""
Trains the DS-CNN keyword-spotting model on the features produced by
data_prep.py.

Run:
    python train.py --data_dir ./data/features --epochs 30
"""
import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from model import build_ds_cnn


def load_split(data_dir, split):
    X = np.load(os.path.join(data_dir, f"X_{split}.npy"))
    y = np.load(os.path.join(data_dir, f"y_{split}.npy"))
    X = X[..., np.newaxis]  # add channel dim -> (N, 49, 10, 1)
    # per-feature normalization computed on training set, applied to all splits
    return X, y


def main(args):
    X_train, y_train = load_split(args.data_dir, "train")
    X_val, y_val = load_split(args.data_dir, "val")
    X_test, y_test = load_split(args.data_dir, "test")

    mean = X_train.mean()
    std = X_train.std() + 1e-6
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    np.save(os.path.join(args.data_dir, "norm_stats.npy"), np.array([mean, std]))

    model = build_ds_cnn()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=4),
        keras.callbacks.ModelCheckpoint(args.out_path, monitor="val_accuracy", save_best_only=True),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_acc:.4f}")
    model.save(args.out_path)
    print(f"Saved trained model to {args.out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/features")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_path", default="./ds_cnn_kws.keras")
    args = parser.parse_args()
    main(args)
