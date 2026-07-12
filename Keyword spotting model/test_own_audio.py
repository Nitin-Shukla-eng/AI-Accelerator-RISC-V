"""
Test the trained (float, pre-quantization) keyword-spotting model against
your own recorded audio clip.

Record yourself saying one of: yes, no, up, down, left, right, on, off,
stop, go -- as a single ~1 second clip -- and run:

    python test_own_audio.py --wav_path ./my_clip.wav --model_path ./ds_cnn_kws.keras --data_dir ./data/features

Handles resampling and mono conversion automatically, so it's fine if your
recording isn't exactly 16kHz mono.
"""
import argparse
import os

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
import tensorflow as tf

from data_prep import SAMPLE_RATE, NUM_FRAMES, NUM_MFCC, WINDOW_SIZE_MS, WINDOW_STRIDE_MS


def load_and_prepare_audio(wav_path):
    sr, data = wavfile.read(wav_path)

    # Convert to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Convert to float32 in [-1, 1], regardless of input dtype (int16, int32, float)
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)

    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        n_samples = int(len(data) * SAMPLE_RATE / sr)
        data = resample(data, n_samples)
        print(f"Resampled from {sr}Hz to {SAMPLE_RATE}Hz")

    # Pad or truncate to exactly 1 second
    target_len = SAMPLE_RATE
    if len(data) < target_len:
        data = np.pad(data, (0, target_len - len(data)))
    else:
        data = data[:target_len]

    return data.astype(np.float32)


def compute_mfcc_from_array(data):
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


def main(args):
    with open(os.path.join(args.data_dir, "labels.txt")) as f:
        labels = [l.strip() for l in f if l.strip()]
    mean, std = np.load(os.path.join(args.data_dir, "norm_stats.npy"))

    audio = load_and_prepare_audio(args.wav_path)
    feats = compute_mfcc_from_array(audio)
    feats = (feats - mean) / std
    feats = feats[np.newaxis, ..., np.newaxis]  # -> (1, 49, 10, 1)

    model = tf.keras.models.load_model(args.model_path)
    probs = model.predict(feats, verbose=0)[0]

    top3 = np.argsort(probs)[::-1][:3]
    print(f"\nInput: {args.wav_path}")
    print("Top 3 predictions:")
    for i in top3:
        print(f"  {labels[i]:12s} {probs[i]*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_path", required=True)
    parser.add_argument("--model_path", default="./ds_cnn_kws.keras")
    parser.add_argument("--data_dir", default="./data/features")
    args = parser.parse_args()
    main(args)
