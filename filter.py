#!/usr/bin/env python3
"""
Enhanced High-Pass Filter Noise Reduction Script
------------------------------------------------
Removes low-frequency noise from WAV audio files using a Butterworth high-pass filter.
Includes waveform and optional spectrogram comparison for verification.

Usage:
  python noise_clean_highpass_enhanced.py \
      --inputs input1.wav input2.wav \
      --lowcut 300 --order 6 \
      --output-dir ./filtered_outputs \
      --show-spectrogram
"""

import argparse
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, spectrogram
import matplotlib.pyplot as plt

# ---------------------- FILTER FUNCTIONS ----------------------
def design_highpass(fs, lowcut, order=6):
    """Design a Butterworth high-pass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    if low <= 0 or low >= 1:
        raise ValueError("Invalid lowcut frequency. Ensure 0 < lowcut < fs/2.")
    b, a = butter(order, low, btype='high')
    return b, a

def apply_highpass_filter(data, fs, lowcut, order=6):
    """Apply high-pass filter per channel."""
    if data.ndim == 1:
        data = data[:, np.newaxis]
    b, a = design_highpass(fs, lowcut, order)
    filtered = np.zeros_like(data, dtype=np.float64)
    for ch in range(data.shape[1]):
        filtered[:, ch] = filtfilt(b, a, data[:, ch])
    return filtered.squeeze()

def normalize_audio(data):
    """Normalize audio amplitude to -1 to 1."""
    max_val = np.max(np.abs(data))
    return data / max_val if max_val > 0 else data

# ---------------------- PLOTTING ----------------------
def plot_waveform_comparison(original, filtered, fs, filename, output_dir=None, downsample=100):
    """Plot waveform comparison before and after filtering."""
    t = np.arange(len(original)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(t[::downsample], original[:, 0] if original.ndim > 1 else original[::downsample], 
             label='Original', alpha=0.6)
    plt.plot(t[::downsample], filtered[:, 0] if filtered.ndim > 1 else filtered[::downsample], 
             label='Filtered', alpha=0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"Waveform Comparison: {filename}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_waveform.png"))
    plt.close()

def plot_spectrogram_comparison(original, filtered, fs, filename, output_dir=None):
    """Plot spectrogram before and after filtering."""
    plt.figure(figsize=(10, 6))

    # Original
    plt.subplot(2, 1, 1)
    f, t, Sxx = spectrogram(original[:, 0] if original.ndim > 1 else original, fs)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    plt.yscale('log')
    plt.title(f"Original Spectrogram: {filename}")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label="Power [dB]")

    # Filtered
    plt.subplot(2, 1, 2)
    f, t, Sxx = spectrogram(filtered[:, 0] if filtered.ndim > 1 else filtered, fs)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    plt.yscale('log')
    plt.title(f"Filtered Spectrogram: {filename}")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label="Power [dB]")

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_spectrogram.png"))
    plt.close()

# ---------------------- MAIN ----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="High-pass noise reduction with visualization")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input WAV file(s)")
    parser.add_argument("--lowcut", type=float, default=300.0, help="Low cutoff frequency in Hz (default: 300)")
    parser.add_argument("--order", type=int, default=6, help="Butterworth filter order (default: 6)")
    parser.add_argument("--output-dir", type=str, default="./filtered_outputs", help="Output directory")
    parser.add_argument("--show-spectrogram", action="store_true", help="Generate spectrogram plots")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for path in args.inputs:
        print(f"\nProcessing: {path}")
        fs, data = wavfile.read(path)

        # Convert to float64
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float64) / np.iinfo(data.dtype).max
        data = normalize_audio(data)

        # Apply high-pass filter
        filtered = apply_highpass_filter(data, fs, args.lowcut, args.order)
        filtered = normalize_audio(filtered)

        # Save filtered audio
        out_path = os.path.join(args.output_dir, os.path.basename(path).replace(".wav", "_filtered.wav"))
        wavfile.write(out_path, fs, (filtered * 32767).astype(np.int16))
        print(f"✅ Saved filtered audio: {out_path}")

        # Plot waveform comparison
        plot_waveform_comparison(data, filtered, fs, os.path.basename(path), args.output_dir)

        # Optional spectrogram
        if args.show_spectrogram:
            plot_spectrogram_comparison(data, filtered, fs, os.path.basename(path), args.output_dir)

    print(f"\nAll files processed. Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
