"""Milestone 6 — INT8 Quantization.

Converts the trained Keras model to a fully INT8 TensorFlow Lite model
(both weights AND activations quantized, input/output included), and
verifies that accuracy degradation versus the float32 model is minimal —
the PRD's explicit acceptance criterion for this stage.

A "representative dataset" (a sample of real, scaled training features) is
required for full-integer post-training quantization: TFLite uses it to
calibrate the activation ranges of each layer before assigning INT8 scales.

Run with:  PYTHONPATH=. python3 src/quantize.py
"""

from __future__ import annotations

import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import CONFIG, MODELS_DIR, OUTPUTS_DIR, ensure_directories
from src.feature_extraction import build_feature_dataset
from src.load_dataset import load_cwru_dataset
from src.preprocessing import stratified_split
from src.segmentation import segment_recordings

logger = logging.getLogger(__name__)


class QuantizationError(Exception):
    """Raised when quantization fails or produces an unusable model."""


def make_representative_dataset_fn(X_train: np.ndarray, n_samples: int):
    """Build the calibration generator TFLite needs for full-INT8 conversion.

    Parameters
    ----------
    X_train:
        Scaled training features (output of StandardScaler), NOT raw
        features — quantization must calibrate on the same distribution
        the model was actually trained on.
    n_samples:
        Number of calibration samples to draw (config.quantization
        .representative_dataset_samples).
    """
    rng = np.random.default_rng(CONFIG.train.random_seed)
    n_samples = min(n_samples, X_train.shape[0])
    indices = rng.choice(X_train.shape[0], size=n_samples, replace=False)
    calibration_data = X_train[indices].astype(np.float32)

    def representative_dataset():
        for row in calibration_data:
            yield [row.reshape(1, -1)]

    return representative_dataset


def quantize_model(
    keras_model: keras.Model,
    X_train: np.ndarray,
) -> bytes:
    """Convert a trained Keras model to a full-INT8 TFLite flatbuffer.

    Returns
    -------
    bytes: the serialized .tflite model.

    Raises
    ------
    QuantizationError
        If the converter cannot produce a fully-integer model (e.g. an
        unsupported op forces a float fallback).
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset_fn(
        X_train, CONFIG.quantization.representative_dataset_samples
    )
    # Force full-integer quantization: no float fallback anywhere, matching
    # the PRD's "full INT8 TensorFlow Lite" requirement and what an FPGA
    # integer datapath actually requires.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    try:
        tflite_model = converter.convert()
    except Exception as exc:  # noqa: BLE001
        raise QuantizationError(f"TFLite conversion failed: {exc}") from exc

    return tflite_model


def evaluate_tflite_model(
    tflite_bytes: bytes, X_test: np.ndarray, y_test: np.ndarray
) -> dict:
    """Run inference through the quantized TFLite model and score accuracy.

    Handles the INT8 input/output quantization manually: the caller passes
    float32 scaled features, and this function applies the model's own
    input scale/zero-point to convert them to INT8 before inference, then
    dequantizes the INT8 output logits back to compare against labels.
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_scale, in_zero_point = input_details["quantization"]
    out_scale, out_zero_point = output_details["quantization"]

    correct = 0
    for i in range(X_test.shape[0]):
        x = X_test[i : i + 1]
        x_int8 = np.round(x / in_scale + in_zero_point).astype(np.int8)

        interpreter.set_tensor(input_details["index"], x_int8)
        interpreter.invoke()
        out_int8 = interpreter.get_tensor(output_details["index"])[0]

        out_float = (out_int8.astype(np.float32) - out_zero_point) * out_scale
        pred = int(np.argmax(out_float))
        if pred == y_test[i]:
            correct += 1

    accuracy = correct / X_test.shape[0]
    return {
        "accuracy": accuracy,
        "input_scale": float(in_scale),
        "input_zero_point": int(in_zero_point),
        "output_scale": float(out_scale),
        "output_zero_point": int(out_zero_point),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ensure_directories()

    model_path = MODELS_DIR / "best_model.keras"
    if not model_path.exists():
        raise QuantizationError(f"No trained model at {model_path}. Run src/train.py first.")

    keras_model = keras.models.load_model(model_path)

    # Rebuild the exact same stratified split used to train best_model.keras
    # so the representative dataset and float-vs-int8 comparison are on the
    # same data the model actually saw / was tested on.
    recordings = load_cwru_dataset()
    windows = segment_recordings(recordings)
    feature_df = build_feature_dataset(windows)
    split = stratified_split(feature_df)

    logger.info("Converting to full-INT8 TFLite...")
    tflite_bytes = quantize_model(keras_model, split.X_train)

    tflite_path = MODELS_DIR / "model_int8.tflite"
    tflite_path.write_bytes(tflite_bytes)
    logger.info("Saved %s (%d bytes)", tflite_path, len(tflite_bytes))

    # --- Float32 baseline accuracy on the same test set ---
    float_probs = keras_model.predict(split.X_test, verbose=0)
    float_pred = np.argmax(float_probs, axis=1)
    float_accuracy = float(np.mean(float_pred == split.y_test))

    # --- INT8 accuracy on the same test set ---
    int8_results = evaluate_tflite_model(tflite_bytes, split.X_test, split.y_test)

    degradation = float_accuracy - int8_results["accuracy"]

    print("\n" + "=" * 60)
    print("QUANTIZATION RESULTS")
    print("=" * 60)
    print(f"Float32 accuracy      : {float_accuracy:.4f}")
    print(f"INT8 accuracy         : {int8_results['accuracy']:.4f}")
    print(f"Accuracy degradation  : {degradation:.4f} ({degradation*100:.2f} pp)")
    print(f"\nModel size (float32)  : {model_path.stat().st_size} bytes")
    print(f"Model size (INT8)     : {tflite_path.stat().st_size} bytes")
    print(f"Size reduction        : {(1 - tflite_path.stat().st_size / model_path.stat().st_size)*100:.1f}%")
    print(f"\nInput  quant: scale={int8_results['input_scale']:.6f}, zero_point={int8_results['input_zero_point']}")
    print(f"Output quant: scale={int8_results['output_scale']:.6f}, zero_point={int8_results['output_zero_point']}")

    if degradation > 0.02:
        print("\nWARNING: accuracy degradation exceeds 2 percentage points.")
        print("Consider increasing representative_dataset_samples in config.py.")
    else:
        print("\nAccuracy degradation is minimal — PRD acceptance criterion met.")

    print(f"\nSaved: {tflite_path}")


if __name__ == "__main__":
    main()
