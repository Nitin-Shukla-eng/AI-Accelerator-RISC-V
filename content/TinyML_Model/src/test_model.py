"""Quick test script — evaluates the trained model on the held-out test set.

Loads the saved model and test split produced by train.py, and prints
accuracy/precision/recall/F1, a confusion matrix, and sample predictions
with confidence percentages.

Run with:  PYTHONPATH=. python3 src/test_model.py
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow import keras

from config import MODELS_DIR, OUTPUTS_DIR

CLASS_NAMES = ["Healthy", "Faulty"]


def main() -> None:
    model_path = MODELS_DIR / "best_model.keras"
    test_path = OUTPUTS_DIR / "test_split.npz"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run src/train.py first.")
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found at {test_path}. Run src/train.py first.")

    model = keras.models.load_model(model_path)
    test_data = np.load(test_path)
    X_test, y_test = test_data["X_test"], test_data["y_test"]

    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    confidence_pct = np.max(probs, axis=1) * 100

    print(f"Test set size : {len(y_test)}")
    print(f"Test Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision     : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall        : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score      : {f1_score(y_test, y_pred):.4f}")

    print("\nConfusion Matrix (rows=true, cols=pred, order=[Healthy, Faulty]):")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    print("Sample predictions (true, predicted, confidence%):")
    for i in range(10):
        true_label = CLASS_NAMES[y_test[i]]
        pred_label = CLASS_NAMES[y_pred[i]]
        print(f"  true={true_label:8s} pred={pred_label:8s} confidence={confidence_pct[i]:.2f}%")


if __name__ == "__main__":
    main()
