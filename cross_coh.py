#!/usr/bin/env python3
"""
High-pass filter + cross-correlation & spectrogram for 4-channel WAVs.

Usage:
  python cross_coh.py --inputs drone_data1_trim.wav drone_data2_trim.wav --cutoff 300 --order 4 --output-dir outputs
"""

import argparse
import os
from typing import Tuple
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, correlate
import matplotlib.pyplot as plt


# ---------------------- FILTER FUNCTIONS ----------------------

def design_highpass(fs: float, cutoff_hz: float, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    if cutoff_hz <= 0:
        raise ValueError("cutoff must be > 0 Hz for a high-pass filter")
    nyquist_hz = fs * 0.5
    if cutoff_hz >= nyquist_hz:
        raise ValueError("cutoff must be < Nyquist (fs/2)")
    normalized_cutoff = cutoff_hz / nyquist_hz
    b, a = butter(N=order, Wn=normalized_cutoff, btype="highpass")
    return b, a


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def convert_to_float64(data: np.ndarray) -> Tuple[np.ndarray, Tuple[str, float]]:
    original_dtype = str(data.dtype)
    scale = 1.0
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = float(max(abs(info.min), info.max))
        data_f64 = data.astype(np.float64) / scale
    else:
        data_f64 = data.astype(np.float64)
    return data_f64, (original_dtype, scale)


def convert_from_float64(data_f64: np.ndarray, original: Tuple[str, float]) -> np.ndarray:
    original_dtype, scale = original
    clipped = np.clip(data_f64, -1.0, 1.0)
    if original_dtype.startswith("int"):
        return (clipped * scale).astype(original_dtype)
    return clipped.astype(np.float32 if original_dtype != "float64" else np.float64)


def apply_highpass_per_channel(data: np.ndarray, fs: float, cutoff_hz: float, order: int) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, np.newaxis]
    b, a = design_highpass(fs, cutoff_hz, order)
    filtered = np.empty_like(data, dtype=np.float64)
    for ch in range(data.shape[1]):
        filtered[:, ch] = filtfilt(b, a, data[:, ch], axis=0)
    return filtered.squeeze()


def build_output_path(output_dir: str, input_path: str, suffix: str) -> str:
    base = os.path.basename(input_path)
    name, ext = os.path.splitext(base)
    return os.path.join(output_dir, f"{name}{suffix}{ext}")


def process_file(input_path: str, output_dir: str, cutoff_hz: float, order: int) -> Tuple[str, np.ndarray, int]:
    fs, data = wavfile.read(input_path)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(f"Expected a 4-channel WAV. Got shape {data.shape} from {input_path}")
    data_f64, original = convert_to_float64(data)
    filtered_f64 = apply_highpass_per_channel(data_f64, fs, cutoff_hz, order)
    filtered = convert_from_float64(filtered_f64, original)
    ensure_dir(output_dir)
    output_path = build_output_path(output_dir, input_path, suffix=f"_hp_{int(cutoff_hz)}Hz_o{order}")
    wavfile.write(output_path, fs, filtered)
    return output_path, filtered_f64, fs


# ---------------------- CROSS-CORRELATION + PLOTTING ----------------------

def compute_cross_correlation(sig1: np.ndarray, sig2: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    corr = correlate(sig1, sig2, mode="full")
    lags = np.arange(-len(sig1) + 1, len(sig2))
    time_lags = lags / fs
    return corr, time_lags


def plot_spectrograms(sig1: np.ndarray, sig2: np.ndarray, fs: float, output_dir: str) -> None:
    from scipy.signal import spectrogram
    eps = 1e-10

    # Spectrogram for File 1 (Ch1)
    f1, t1, Sxx1 = spectrogram(sig1, fs)
    Sxx1_db = 10 * np.log10(Sxx1 + eps)
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t1, f1, Sxx1_db, shading='gouraud', cmap='viridis')
    plt.yscale('log')
    plt.title("Spectrogram - File 1 (Ch1)")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    save_path = os.path.join(output_dir, "spectrogram_file1_log.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved File 1 log-frequency spectrogram -> {save_path}")

    # Spectrogram for File 2 (Ch1)
    f2, t2, Sxx2 = spectrogram(sig2, fs)
    Sxx2_db = 10 * np.log10(Sxx2 + eps)
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t2, f2, Sxx2_db, shading='gouraud', cmap='viridis')
    plt.yscale('log')
    plt.title("Spectrogram - File 2 (Ch1)")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    save_path = os.path.join(output_dir, "spectrogram_file2_log.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved File 2 log-frequency spectrogram -> {save_path}")

def plot_cross_correlation(corr: np.ndarray, time_lags: np.ndarray, output_dir: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(time_lags, corr, color='purple')
    plt.title("Cross-Correlation (Ch1 of both WAVs)")
    plt.xlabel("Lag (seconds)")
    plt.ylabel("Correlation amplitude")
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "cross_correlation.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved cross-correlation plot -> {save_path}")


# ---------------------- MAIN ----------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-pass filter + cross-correlation for multi-channel WAV files")
    parser.add_argument("--inputs", nargs=2, required=True, help="Two input WAV paths")
    parser.add_argument("--cutoff", type=float, default=300.0, help="High-pass cutoff frequency in Hz (default: 300)")
    parser.add_argument("--order", type=int, default=4, help="Butterworth filter order (default: 4)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to write filtered WAVs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = []
    signals = []
    fs = None

    for ip in args.inputs:
        out_path, filtered, fs = process_file(ip, args.output_dir, args.cutoff, args.order)
        outputs.append(out_path)
        signals.append(filtered[:, 0] if filtered.ndim == 2 else filtered)

    print("\nWrote filtered files:")
    for op in outputs:
        print(f"  {op}")

    # Match signal lengths
    min_len = min(len(signals[0]), len(signals[1]))
    sig1 = signals[0][:min_len]
    sig2 = signals[1][:min_len]

    # Compute cross-correlation
    corr, time_lags = compute_cross_correlation(sig1, sig2, fs)
    print(f"\nCross-correlation computed for {min_len/fs:.2f} seconds of audio.")

    # Save plots
    plot_cross_correlation(corr, time_lags, args.output_dir)
    plot_spectrograms(sig1, sig2, fs, args.output_dir)


if __name__ == "__main__":
    main()
