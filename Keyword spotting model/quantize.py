"""
Converts the trained Keras model to a fully int8-quantized TFLite model
(the format your RISC-V baseline and FPGA-accelerated inference engines
will both run), and dumps it as a C byte array for embedding directly
in firmware.

Run:
    python quantize.py --model_path ./ds_cnn_kws.keras --data_dir ./data/features
"""
import argparse
import os

import numpy as np
import tensorflow as tf


def representative_dataset_gen(X_train, num_samples=200):
    def gen():
        idx = np.random.choice(len(X_train), size=min(num_samples, len(X_train)), replace=False)
        for i in idx:
            sample = X_train[i:i + 1].astype(np.float32)
            yield [sample]
    return gen


def convert_to_int8_tflite(model_path, X_train, out_path):
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen(X_train)
    # Force full int8: input, output, and all ops quantized -- required for
    # running on a microcontroller / RISC-V core with no float unit.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved int8 TFLite model to {out_path} ({len(tflite_model)} bytes)")
    return tflite_model


def evaluate_tflite(tflite_model, X_test, y_test):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_scale, in_zero = input_details["quantization"]
    correct = 0
    for i in range(len(X_test)):
        x = X_test[i:i + 1]
        x_q = (x / in_scale + in_zero).astype(np.int8)
        interpreter.set_tensor(input_details["index"], x_q)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
        pred = np.argmax(out)
        if pred == y_test[i]:
            correct += 1
    acc = correct / len(X_test)
    print(f"Quantized int8 model test accuracy: {acc:.4f}")
    return acc


def export_c_array(tflite_path, out_c_path, var_name="ds_cnn_kws_model"):
    with open(tflite_path, "rb") as f:
        data = f.read()
    with open(out_c_path, "w") as f:
        f.write(f"// Auto-generated from {os.path.basename(tflite_path)}\n")
        f.write(f"// {len(data)} bytes\n")
        f.write(f"#include <cstdint>\n\n")
        f.write(f"alignas(8) const unsigned char {var_name}[] = {{\n")
        for i in range(0, len(data), 12):
            chunk = data[i:i + 12]
            line = ", ".join(f"0x{b:02x}" for b in chunk)
            f.write(f"  {line},\n")
        f.write("};\n")
        f.write(f"const unsigned int {var_name}_len = {len(data)};\n")
    print(f"Saved C array header to {out_c_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./ds_cnn_kws.keras")
    parser.add_argument("--data_dir", default="./data/features")
    parser.add_argument("--out_tflite", default="./ds_cnn_kws_int8.tflite")
    parser.add_argument("--out_c_header", default="./ds_cnn_kws_int8_model.h")
    args = parser.parse_args()

    X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))[..., None]
    X_test = np.load(os.path.join(args.data_dir, "X_test.npy"))[..., None]
    y_test = np.load(os.path.join(args.data_dir, "y_test.npy"))
    mean, std = np.load(os.path.join(args.data_dir, "norm_stats.npy"))
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    tflite_model = convert_to_int8_tflite(args.model_path, X_train, args.out_tflite)
    evaluate_tflite(tflite_model, X_test, y_test)
    export_c_array(args.out_tflite, args.out_c_header)
