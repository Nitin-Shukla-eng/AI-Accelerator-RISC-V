"""
Data preparation for TinyML keyword spotting.

Downloads Google Speech Commands v0.02, computes 10-coefficient MFCC
features over 49 time frames per 1-second clip (the exact input shape
used by the MLPerf Tiny DS-CNN keyword-spotting reference model), and
saves train/val/test splits as .npy files.

12 output classes: yes, no, up, down, left, right, on, off, stop, go,
                    _unknown_, _silence_

Run:
    python data_prep.py --data_dir ./data
"""
import argparse
import hashlib
import os
import tarfile
import time
import urllib.request

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import tensorflow as tf

# Extraction is per-file (small tensors); keep TF from grabbing all GPU
# memory up front since we're intentionally forcing this step onto CPU.
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
WANTED_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABELS = WANTED_WORDS + ["_unknown_", "_silence_"]
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30
WINDOW_STRIDE_MS = 20
NUM_MFCC = 10
# (clip_ms - window_ms) / stride_ms + 1 = (1000-30)/20 + 1 = 49.5 -> 49 frames
NUM_FRAMES = 49


def download_and_extract(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    archive_path = os.path.join(data_dir, "speech_commands_v0.02.tar.gz")
    extract_dir = os.path.join(data_dir, "raw")
    if os.path.isdir(extract_dir) and os.listdir(extract_dir):
        print("Dataset already extracted, skipping download.")
        return extract_dir
    if not os.path.exists(archive_path):
        print("Downloading Google Speech Commands v0.02 (~2.4GB)...")
        urllib.request.urlretrieve(DATA_URL, archive_path)
    print("Extracting...")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(archive_path) as tar:
        tar.extractall(extract_dir)
    return extract_dir


def which_set(filename, validation_pct=10, testing_pct=10):
    """Deterministically assign a file to train/val/test using MD5 hashing,
    matching the original TensorFlow speech commands split logic so that
    a file's split assignment is stable across runs and additions."""
    base_name = os.path.basename(filename)
    hash_name = base_name.split("_nohash_")[0]
    hash_name_hashed = hashlib.sha1(hash_name.encode()).hexdigest()
    percentage_hash = (int(hash_name_hashed, 16) % 100)
    if percentage_hash < validation_pct:
        return "validation"
    elif percentage_hash < (testing_pct + validation_pct):
        return "testing"
    else:
        return "training"


def compute_mfcc(wav_path):
    sr, data = wavfile.read(wav_path)
    data = data.astype(np.float32) / 32768.0
    # Pad or truncate to exactly 1 second
    target_len = SAMPLE_RATE
    if len(data) < target_len:
        data = np.pad(data, (0, target_len - len(data)))
    else:
        data = data[:target_len]

    # Force CPU: these are tiny per-file ops (1s of audio), and GPU dispatch/
    # transfer overhead for ops this small is typically slower than just
    # running them on CPU when processing files one at a time.
    with tf.device("/CPU:0"):
        waveform = tf.constant(data, dtype=tf.float32)
        frame_length = int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000)
        frame_step = int(SAMPLE_RATE * WINDOW_STRIDE_MS / 1000)
        stft = tf.signal.stft(waveform, frame_length=frame_length, frame_step=frame_step,
                               fft_length=frame_length)
        spectrogram = tf.abs(stft)

        num_spectrogram_bins = spectrogram.shape[-1]
        mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=40, num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=SAMPLE_RATE, lower_edge_hertz=20.0, upper_edge_hertz=4000.0)
        mel_spectrogram = tf.tensordot(spectrogram, mel_matrix, 1)
        log_mel = tf.math.log(mel_spectrogram + 1e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :NUM_MFCC]

    mfccs = mfccs.numpy()
    if mfccs.shape[0] < NUM_FRAMES:
        mfccs = np.pad(mfccs, ((0, NUM_FRAMES - mfccs.shape[0]), (0, 0)))
    else:
        mfccs = mfccs[:NUM_FRAMES]
    return mfccs.astype(np.float32)


def build_dataset(raw_dir, out_dir, limit_per_word=None):
    os.makedirs(out_dir, exist_ok=True)
    label_to_idx = {l: i for i, l in enumerate(LABELS)}
    splits = {"training": [], "validation": [], "testing": []}

    word_dirs = [d for d in os.listdir(raw_dir)
                 if os.path.isdir(os.path.join(raw_dir, d)) and not d.startswith("_")]

    print(f"Found {len(word_dirs)} word folders. Scanning files...")
    all_files = []
    for word in word_dirs:
        word_path = os.path.join(raw_dir, word)
        label = word if word in WANTED_WORDS else "_unknown_"
        idx = label_to_idx[label]
        fnames = [f for f in os.listdir(word_path) if f.endswith(".wav")]
        if limit_per_word is not None:
            fnames = fnames[:limit_per_word]
        for fname in fnames:
            all_files.append((os.path.join(word_path, fname), idx))

    print(f"Found {len(all_files)} total wav files. Extracting MFCC features...")
    n_failed = 0
    n_ok = 0
    first_error_shown = False
    start_time = time.time()
    total = len(all_files)
    for i, (fpath, idx) in enumerate(tqdm(all_files, unit="file")):
        split = which_set(fpath)
        try:
            feats = compute_mfcc(fpath)
        except Exception as e:
            n_failed += 1
            if not first_error_shown:
                import traceback
                print(f"\n[FIRST FAILURE] on {fpath}:")
                traceback.print_exc()
                first_error_shown = True
            continue
        n_ok += 1
        splits[split].append((feats, idx))

        if (i + 1) % 2000 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total - (i + 1)) / rate if rate > 0 else float("inf")
            print(f"  [{i + 1}/{total}] {rate:.1f} files/sec, "
                  f"elapsed {elapsed/60:.1f} min, est. remaining {remaining/60:.1f} min", flush=True)

    print(f"\nFeature extraction done: {n_ok} succeeded, {n_failed} failed.")

    # Add silence samples: zeroed / low-noise frames, ~10% of training set size
    n_silence = max(1, len(splits["training"]) // 10)
    silence_idx = label_to_idx["_silence_"]
    for _ in range(n_silence):
        noise = (np.random.randn(NUM_FRAMES, NUM_MFCC) * 0.01).astype(np.float32)
        splits["training"].append((noise, silence_idx))

    for split_name, split_key in [("training", "train"), ("validation", "val"), ("testing", "test")]:
        data = splits[split_name]
        if len(data) == 0:
            print(f"WARNING: {split_key} split has 0 samples -- saving empty placeholder.")
            X = np.zeros((0, NUM_FRAMES, NUM_MFCC), dtype=np.float32)
            y = np.zeros((0,), dtype=np.int32)
        else:
            X = np.stack([d[0] for d in data])
            y = np.array([d[1] for d in data], dtype=np.int32)
        np.save(os.path.join(out_dir, f"X_{split_key}.npy"), X)
        np.save(os.path.join(out_dir, f"y_{split_key}.npy"), y)
        print(f"{split_key}: {X.shape[0]} samples -> X shape {X.shape}")

    with open(os.path.join(out_dir, "labels.txt"), "w") as f:
        f.write("\n".join(LABELS))
    print(f"Saved features and labels.txt to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--limit_per_word", type=int, default=None,
                         help="For fast debugging: only process this many wav files per word folder.")
    args = parser.parse_args()

    raw = download_and_extract(args.data_dir)
    build_dataset(raw, os.path.join(args.data_dir, "features"), limit_per_word=args.limit_per_word)
