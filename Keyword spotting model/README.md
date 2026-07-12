# TinyML keyword-spotting model (DS-CNN)

This is the model you'll benchmark twice: once running purely in software
on your RISC-V core, and once with the conv/depthwise-conv ops offloaded
to your FPGA accelerator. Same weights, same int8 model file, both times —
only the execution path changes, so any speedup you measure is attributable
to the accelerator.

## Why this architecture

DS-CNN (depthwise-separable CNN) is the MLPerf Tiny reference model for
keyword spotting. It's small enough for a microcontroller-class core, and
almost all of its compute is conv / depthwise-conv / 1x1-conv (matmul-like)
ops — exactly what an accelerator targets. That makes it a meaningful,
comparable benchmark rather than a toy example.

- Input: 1 second of audio -> 10-coefficient MFCC over 49 time frames
- Output: 12 classes (yes, no, up, down, left, right, on, off, stop, go,
  unknown, silence)
- Dataset: Google Speech Commands v0.02 (~105k labeled 1-second clips)

## Pipeline

```
pip install -r requirements.txt

python data_prep.py --data_dir ./data          # downloads dataset, extracts MFCC features
python train.py --data_dir ./data/features     # trains the float model
python quantize.py --model_path ./ds_cnn_kws.keras --data_dir ./data/features
```

`quantize.py` produces:
- `ds_cnn_kws_int8.tflite` — the deployable int8 model
- `ds_cnn_kws_int8_model.h` — the same model as a C byte array, ready to
  `#include` in firmware for both your baseline and accelerated builds

## Expected results

Full int8 DS-CNN-S on Speech Commands typically lands around 90-93% test
accuracy. If you see much lower, check that `data_prep.py` finished
extracting all word folders and that normalization stats loaded correctly
in `quantize.py`.

## Where this plugs into the bigger project

This model file is the one constant across your whole benchmarking setup:

1. **Baseline run** — load `ds_cnn_kws_int8_model.h` into your RISC-V
   software-only inference engine (TFLite Micro or CMSIS-NN), run
   inference, record cycles/latency/power.
2. **Accelerated run** — load the *same* model into your accelerator-aware
   inference engine, which offloads the conv/depthwise-conv layers to the
   FPGA, run inference, record the same metrics.
3. **Compare** — same model, same test inputs, same accuracy — the only
   variable is where the conv ops executed.

Next step: the software-only RISC-V inference engine (TFLite Micro build
targeting your core), so you can get baseline numbers before touching the
FPGA side.
