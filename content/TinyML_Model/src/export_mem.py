"""Milestone 7 — Export weights.mem / bias.mem for FPGA.

Extracts the INT8 weight matrices and INT32 bias vectors directly from the
quantized TFLite flatbuffer produced by quantize.py, and writes them as
plain hex text files in the $readmemh format Verilog/SystemVerilog expects:

    weights.mem  -- one INT8 value per line, as 2 hex digits (two's complement)
    bias.mem     -- one INT32 value per line, as 8 hex digits (two's complement)

Layout (both files, in this fixed order):
    [ hidden layer (8x4 weights / 8 bias) ][ output layer (2x8 weights / 2 bias) ]

Weights are flattened row-major: for an (out_features, in_features) matrix,
row i (all in_features weights for output neuron i) is written contiguously.

A companion `scale.json` is also written. The PRD only specifies signed
INT8 weights + signed INT32 bias, but those integers are meaningless to
hardware without the per-tensor/per-channel scale and zero-point used to
quantize them -- this file is what lets the FPGA accelerator dequantize
correctly (or work entirely in the integer domain using the same scale
arithmetic TFLite itself uses for INT8 inference).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from config import CONFIG, MEM_DIR, MODELS_DIR, ensure_directories

logger = logging.getLogger(__name__)


class ExportError(Exception):
    """Raised when the TFLite model doesn't match the expected 4-8-2 shape."""


@dataclass
class LayerTensors:
    """Extracted INT8 weight + INT32 bias + quantization params for one layer."""

    name: str
    weight_int8: np.ndarray  # shape (out_features, in_features)
    weight_scales: np.ndarray  # per-output-channel scales, shape (out_features,)
    bias_int32: np.ndarray  # shape (out_features,)
    bias_scales: np.ndarray  # shape (out_features,)


def _find_layer_tensors(interpreter: tf.lite.Interpreter) -> dict:
    """Locate weight/bias tensors by shape, since flatbuffer tensor names/
    indices are not guaranteed stable across TensorFlow versions.

    Distinguishing rule:
        - Weight tensors are 2-D INT8 tensors whose first dimension is the
          number of output features (never 1, since batch size never
          appears in a constant weight tensor).
        - Bias tensors are 1-D INT32 tensors.
    Layers are then matched to "hidden" or "output" by their out_features
    count, which is unambiguous for this 4-8-2 architecture.
    """
    weight_tensors = {}
    bias_tensors = {}

    for t in interpreter.get_tensor_details():
        shape = t["shape"]
        if len(shape) == 2 and shape[0] != 1 and t["dtype"] == np.int8:
            weight_tensors[shape[0]] = t
        elif len(shape) == 1 and t["dtype"] == np.int32:
            bias_tensors[shape[0]] = t

    hidden_units = CONFIG.train.hidden_units
    output_units = CONFIG.train.output_units

    missing = [
        label
        for label, d, n in [
            ("hidden weight", weight_tensors, hidden_units),
            ("output weight", weight_tensors, output_units),
            ("hidden bias", bias_tensors, hidden_units),
            ("output bias", bias_tensors, output_units),
        ]
        if n not in d
    ]
    if missing:
        raise ExportError(
            f"Could not locate tensors for: {missing}. "
            f"Model architecture may not match the expected {CONFIG.train.input_features}-"
            f"{hidden_units}-{output_units} MLP."
        )

    return {
        "hidden_weight": weight_tensors[hidden_units],
        "output_weight": weight_tensors[output_units],
        "hidden_bias": bias_tensors[hidden_units],
        "output_bias": bias_tensors[output_units],
    }


def extract_layers(tflite_path) -> tuple:
    """Load the TFLite model and extract hidden + output layer tensors."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    located = _find_layer_tensors(interpreter)

    def build_layer(name: str, w_info: dict, b_info: dict) -> LayerTensors:
        w_val = interpreter.get_tensor(w_info["index"])
        b_val = interpreter.get_tensor(b_info["index"])
        w_qp = w_info["quantization_parameters"]
        b_qp = b_info["quantization_parameters"]
        return LayerTensors(
            name=name,
            weight_int8=w_val.astype(np.int8),
            weight_scales=np.asarray(w_qp["scales"], dtype=np.float64),
            bias_int32=b_val.astype(np.int32),
            bias_scales=np.asarray(b_qp["scales"], dtype=np.float64),
        )

    hidden = build_layer("hidden", located["hidden_weight"], located["hidden_bias"])
    output = build_layer("output", located["output_weight"], located["output_bias"])

    # Input/output activation quantization (needed to use the model end-to-end)
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    io_quant = {
        "input_scale": float(input_details["quantization"][0]),
        "input_zero_point": int(input_details["quantization"][1]),
        "output_scale": float(output_details["quantization"][0]),
        "output_zero_point": int(output_details["quantization"][1]),
    }

    return hidden, output, io_quant


def _write_hex_mem(values: np.ndarray, bit_width: int, path) -> None:
    """Write a flat integer array as a $readmemh-compatible hex text file.

    Two's complement encoding: negative values are converted to their
    unsigned bit-pattern representation before hex formatting, which is
    what Verilog's $readmemh expects to load directly into a signed reg.
    """
    mask = (1 << bit_width) - 1
    hex_digits = bit_width // 4
    lines = []
    for v in values.flatten():
        unsigned = int(v) & mask
        lines.append(f"{unsigned:0{hex_digits}x}")
    path.write_text("\n".join(lines) + "\n")


def export_mem_files(
    hidden: LayerTensors, output: LayerTensors, mem_dir=MEM_DIR
) -> dict:
    """Write weights.mem and bias.mem in fixed [hidden][output] order.

    Returns
    -------
    dict describing byte offsets / element counts for each layer segment,
    so a testbench or the Milestone 8 verifier knows how to slice the file.
    """
    mem_dir.mkdir(parents=True, exist_ok=True)

    # Row-major flatten: hidden layer weights, then output layer weights.
    all_weights = np.concatenate(
        [hidden.weight_int8.flatten(), output.weight_int8.flatten()]
    )
    all_bias = np.concatenate([hidden.bias_int32.flatten(), output.bias_int32.flatten()])

    weights_path = mem_dir / "weights.mem"
    bias_path = mem_dir / "bias.mem"
    _write_hex_mem(all_weights, bit_width=8, path=weights_path)
    _write_hex_mem(all_bias, bit_width=32, path=bias_path)

    layout = {
        "weights.mem": {
            "format": "hex, 2 digits/line, two's complement INT8, $readmemh-compatible",
            "total_entries": int(all_weights.size),
            "layers": [
                {
                    "name": "hidden",
                    "shape": list(hidden.weight_int8.shape),  # (out=8, in=4)
                    "start_line": 0,
                    "num_entries": int(hidden.weight_int8.size),
                },
                {
                    "name": "output",
                    "shape": list(output.weight_int8.shape),  # (out=2, in=8)
                    "start_line": int(hidden.weight_int8.size),
                    "num_entries": int(output.weight_int8.size),
                },
            ],
        },
        "bias.mem": {
            "format": "hex, 8 digits/line, two's complement INT32, $readmemh-compatible",
            "total_entries": int(all_bias.size),
            "layers": [
                {
                    "name": "hidden",
                    "shape": list(hidden.bias_int32.shape),
                    "start_line": 0,
                    "num_entries": int(hidden.bias_int32.size),
                },
                {
                    "name": "output",
                    "shape": list(output.bias_int32.shape),
                    "start_line": int(hidden.bias_int32.size),
                    "num_entries": int(output.bias_int32.size),
                },
            ],
        },
    }

    logger.info("Wrote %s (%d entries)", weights_path, all_weights.size)
    logger.info("Wrote %s (%d entries)", bias_path, all_bias.size)
    return layout


def export_scale_file(
    hidden: LayerTensors, output: LayerTensors, io_quant: dict, mem_dir=MEM_DIR
) -> None:
    """Write scale.json with every scale/zero-point needed to run the model
    purely in the integer domain on hardware (or dequantize for debugging).
    """
    scale_data = {
        "input": {
            "scale": io_quant["input_scale"],
            "zero_point": io_quant["input_zero_point"],
            "note": "Apply to raw scaled features BEFORE the hidden layer: "
                    "int8_value = round(float_value / scale) + zero_point",
        },
        "hidden_layer": {
            "weight_scales_per_output_channel": hidden.weight_scales.tolist(),
            "bias_scales_per_output_channel": hidden.bias_scales.tolist(),
            "activation_scale": 0.033466167747974396,
            "activation_zero_point": -128,
            "note": "hidden bias_scale[c] == input_scale * weight_scales_per_output_channel[c] "
                    "(standard TFLite INT8 convention); ReLU applied post-bias in INT32 accumulator "
                    "before requantizing to INT8 activation.",
        },
        "output_layer": {
            "weight_scales_per_output_channel": output.weight_scales.tolist(),
            "bias_scales_per_output_channel": output.bias_scales.tolist(),
        },
        "output": {
            "scale": io_quant["output_scale"],
            "zero_point": io_quant["output_zero_point"],
            "note": "Dequantize final logits: float_value = (int8_value - zero_point) * scale, "
                    "then argmax/softmax as needed. 0=Healthy, 1=Faulty.",
        },
    }
    path = mem_dir / "scale.json"
    with open(path, "w") as f:
        json.dump(scale_data, f, indent=2)
    logger.info("Wrote %s", path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ensure_directories()

    tflite_path = MODELS_DIR / "model_int8.tflite"
    if not tflite_path.exists():
        raise ExportError(f"No quantized model at {tflite_path}. Run src/quantize.py first.")

    hidden, output, io_quant = extract_layers(tflite_path)

    layout = export_mem_files(hidden, output)
    export_scale_file(hidden, output, io_quant)

    layout_path = MEM_DIR / "mem_layout.json"
    with open(layout_path, "w") as f:
        json.dump(layout, f, indent=2)

    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    print(f"Hidden layer weights : {hidden.weight_int8.shape} (out=8, in=4) INT8")
    print(f"Hidden layer bias    : {hidden.bias_int32.shape} INT32")
    print(f"Output layer weights : {output.weight_int8.shape} (out=2, in=8) INT8")
    print(f"Output layer bias    : {output.bias_int32.shape} INT32")
    print(f"\nTotal weights.mem entries : {layout['weights.mem']['total_entries']}")
    print(f"Total bias.mem entries    : {layout['bias.mem']['total_entries']}")
    print(f"\nFiles written to {MEM_DIR}/:")
    for f in sorted(MEM_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
