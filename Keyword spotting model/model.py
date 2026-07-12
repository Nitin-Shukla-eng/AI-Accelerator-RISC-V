"""
DS-CNN (Depthwise-Separable CNN) for keyword spotting.

This is the MLPerf Tiny reference architecture for KWS: small enough to
run on a microcontroller / soft RISC-V core, but built almost entirely
out of conv and depthwise-conv ops -- which is exactly what makes it a
good benchmark for a conv/matmul FPGA accelerator. Most of the runtime
will be spent in these ops, so a speedup there should show up clearly
in your baseline-vs-accelerated comparison.

Input:  (49, 10, 1)  -- 49 time frames x 10 MFCC coefficients
Output: 12 classes   -- 10 keywords + _unknown_ + _silence_
"""
from tensorflow import keras
from tensorflow.keras import layers

NUM_CLASSES = 12
INPUT_SHAPE = (49, 10, 1)


def build_ds_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, num_filters=64):
    inputs = keras.Input(shape=input_shape)

    # Initial standard conv: downsamples time/freq, expands to num_filters channels
    x = layers.Conv2D(num_filters, kernel_size=(10, 4), strides=(2, 2),
                       padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 4 depthwise-separable conv blocks -- the accelerator's main workload
    for _ in range(4):
        x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(num_filters, kernel_size=(1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="ds_cnn_kws")
    return model


if __name__ == "__main__":
    m = build_ds_cnn()
    m.summary()
