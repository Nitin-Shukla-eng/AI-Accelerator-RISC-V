"""Milestone 5 — Evaluation.

Produces the full evaluation suite the PRD asks for (accuracy, precision,
recall, F1, ROC curve, confusion matrix, classification report), and runs it
TWICE:

1. On the primary stratified test split (the "easy" number — windows from
   the same recordings can appear in both train and test).
2. On the cross-load holdout split (the honest number — the model never
   sees load condition 3hp during training, so this measures genuine
   generalization to an unseen operating condition rather than memorized
   per-recording noise patterns).

Both models, plots, and reports are saved so the gap between the two is
visible and citable in a project report.

Run with:  PYTHONPATH=. python3 src/evaluate.py
"""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")  # headless: no display needed, just save PNGs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import CONFIG, FIGURES_DIR, OUTPUTS_DIR, ensure_directories
from src.feature_extraction import build_feature_dataset
from src.load_dataset import load_cwru_dataset
from src.preprocessing import SplitData, cross_load_split, stratified_split
from src.segmentation import segment_recordings
from src.train import train_model

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Healthy", "Faulty"]


def evaluate_split(model, split: SplitData, tag: str) -> dict:
    """Compute all PRD-required metrics for one trained model + test split.

    Saves a confusion matrix PNG and an ROC curve PNG to FIGURES_DIR, named
    with ``tag`` so the stratified and cross-load results don't overwrite
    each other.

    Returns
    -------
    dict of scalar metrics + the full classification report text, suitable
    for JSON serialization.
    """
    probs = model.predict(split.X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_score = probs[:, 1]  # probability of the "Faulty" class, for ROC

    acc = accuracy_score(split.y_test, y_pred)
    prec = precision_score(split.y_test, y_pred)
    rec = recall_score(split.y_test, y_pred)
    f1 = f1_score(split.y_test, y_pred)
    auc = roc_auc_score(split.y_test, y_score)
    cm = confusion_matrix(split.y_test, y_pred)
    report_text = classification_report(split.y_test, y_pred, target_names=CLASS_NAMES)

    logger.info(
        "[%s] acc=%.4f precision=%.4f recall=%.4f f1=%.4f auc=%.4f",
        tag, acc, prec, rec, f1, auc,
    )

    # --- Confusion matrix plot ---
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({tag})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    cm_path = FIGURES_DIR / f"confusion_matrix_{tag}.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    # --- ROC curve plot ---
    fpr, tpr, _ = roc_curve(split.y_test, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc).plot(ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title(f"ROC Curve ({tag})")
    ax.legend()
    fig.tight_layout()
    roc_path = FIGURES_DIR / f"roc_curve_{tag}.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    logger.info("Saved %s and %s", cm_path.name, roc_path.name)

    return {
        "tag": tag,
        "test_size": int(len(split.y_test)),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_text,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ensure_directories()

    recordings = load_cwru_dataset()
    windows = segment_recordings(recordings)
    feature_df = build_feature_dataset(windows)

    results = {}

    # --- 1. Primary stratified evaluation ---
    logger.info("=== Evaluation 1/2: Stratified split (primary metric) ===")
    strat_split = stratified_split(feature_df)
    strat_model, _ = train_model(strat_split, checkpoint_name="best_model.keras")
    results["stratified"] = evaluate_split(strat_model, strat_split, "stratified")

    # --- 2. Cross-load robustness evaluation ---
    logger.info("=== Evaluation 2/2: Cross-load holdout (robustness check) ===")
    cross_split = cross_load_split(feature_df)
    cross_model, _ = train_model(cross_split, checkpoint_name="best_model_crossload.keras")
    results["cross_load"] = evaluate_split(cross_model, cross_split, "crossload")

    report_path = OUTPUTS_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    # --- Summary printed to console ---
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for tag, r in results.items():
        print(f"\n[{tag}]  (test size = {r['test_size']})")
        print(f"  Accuracy : {r['accuracy']:.4f}")
        print(f"  Precision: {r['precision']:.4f}")
        print(f"  Recall   : {r['recall']:.4f}")
        print(f"  F1 Score : {r['f1_score']:.4f}")
        print(f"  ROC AUC  : {r['roc_auc']:.4f}")

    gap = results["stratified"]["accuracy"] - results["cross_load"]["accuracy"]
    print(f"\nGeneralization gap (stratified - cross_load): {gap:.4f}")
    if gap > 0.05:
        print("  -> Meaningful drop on unseen load condition; model relies partly")
        print("     on load-specific patterns. Worth discussing in your report.")
    else:
        print("  -> Small drop; model generalizes well to an unseen load condition.")

    print(f"\nFull report saved to: {report_path}")
    print(f"Plots saved to: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
