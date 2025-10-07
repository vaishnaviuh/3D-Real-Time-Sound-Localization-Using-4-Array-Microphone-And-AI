# 3D-Real-Time-Sound-Localization-Using-4-Array-Microphone-And-AI

This repository contains a Jupyter Notebook for sound localization using signal processing and/or machine learning techniques.

## Files

- `Localized_Sound_final.ipynb`: The main notebook containing the implementation.
- `README.md`: This file.

## How to Run

1. Open the notebook in Jupyter or Google Colab.
2. Install necessary dependencies if prompted.
3. Follow the cells sequentially to replicate the results.

## Requirements

- Python 3.8+
- Jupyter Notebook
- Libraries: `numpy`, `matplotlib`, `librosa`, `scipy`, etc.

## median-filtered method
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, butter, filtfilt
from scipy.interpolate import interp1d
import os

#Bandpass filter function 


def bandpass_filter(signal, sr, lowcut=300, highcut=3000, order=4):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

#Load audio 


file = "/content/DroneSound_noMuff.wav"
y, sr = librosa.load(file, sr=None, mono=False)
if y.ndim == 1:
    y = np.expand_dims(y, axis=0)

num_channels = min(5, y.shape[0])   # Use first 4 channels
print(f"Processing {num_channels} channels.")

#STFT params 


n_fft = 2048
hop_length = 512

#Compute STFT for each channel
channel_specs = []
for i in range(num_channels):
    S = librosa.stft(y[i], n_fft=n_fft, hop_length=hop_length)
    channel_specs.append(S)
channel_specs = np.stack(channel_specs, axis=-1)

#Coherence computation


window_size = 2048
step_size = 512
nperseg = 256

all_coh_specs = []
for i in range(num_channels):
    sig1 = bandpass_filter(y[i], sr)
    for j in range(i+1, num_channels):
        sig2 = bandpass_filter(y[j], sr)

        n_windows = (len(sig1) - window_size) // step_size + 1
        coh_spec = []
        for w in range(n_windows):
            start = w * step_size
            end = start + window_size
            f, Cxy = coherence(sig1[start:end], sig2[start:end], fs=sr, nperseg=nperseg)
            coh_spec.append(Cxy)
        coh_spec = np.array(coh_spec).T
        all_coh_specs.append(coh_spec)
#Median coherence spectrum 


min_time_bins = min(spec.shape[1] for spec in all_coh_specs)
all_coh_specs_trunc = [spec[:, :min_time_bins] for spec in all_coh_specs]
median_coh_spec = np.median(np.stack(all_coh_specs_trunc, axis=-1), axis=-1)

#Square as filter 


coh_filter = median_coh_spec ** 2   # squaring enhances strong coherence, suppresses weak
coh_filter = np.clip(coh_filter, 0, 1)

#Interpolate to STFT frequencies 


freq_stft = np.linspace(0, sr/2, channel_specs.shape[0])
interp_func = interp1d(f, coh_filter, axis=0, kind='nearest', fill_value="extrapolate")
coh_interp = interp_func(freq_stft)

#Apply filter 


output_dir = "/content/results_squared_filter"
os.makedirs(output_dir, exist_ok=True)

for ch in range(num_channels):
    min_time_dim = min(channel_specs.shape[1], coh_interp.shape[1])
    S_trunc = channel_specs[:, :min_time_dim, ch]
    mask_trunc = coh_interp[:, :min_time_dim]

    # Apply mask
    S_filtered = S_trunc * mask_trunc

    # Plot comparison (original vs filtered)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(S_trunc), ref=np.max),
                             sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(label="dB")
    plt.title(f"Original Spectrogram - Ch{ch+1}")

    plt.subplot(1, 2, 2)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(S_filtered), ref=np.max),
                             sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(label="dB")
    plt.title(f"Filtered Spectrogram (Squared Coherence) - Ch{ch+1}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"channel{ch+1}_comparison.png"), dpi=300)
    plt.close()

    print(f"Saved comparison spectrogram for Channel {ch+1}")
#coherence


import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from typing import Tuple

def _pair_index_to_channels(pair_idx: int, K: int) -> Tuple[int, int]:
    """Convert linear pair index back to channel indices (i,j)."""
    i = 0
    remaining = pair_idx
    while remaining >= (K - 1 - i):
        remaining -= (K - 1 - i)
        i += 1
    j = i + 1 + remaining
    return i, j

def smooth_coherence(x: np.ndarray, y: np.ndarray, win_size=5) -> np.ndarray:
    """
    Smoothed pixel-wise coherence over a small time neighborhood.
    x, y: STFT matrices (freq_bins x time_frames)
    win_size: smoothing window over time axis
    Returns: coherence (freq_bins x time_frames) in [0,1]
    """
    eps = 1e-8
    L, M = x.shape
    coh = np.zeros((L, M))
    for l in range(L):
        for m in range(M):
            t_start = max(0, m - win_size//2)
            t_end = min(M, m + win_size//2 + 1)
            num = np.abs(np.sum(x[l, t_start:t_end] * np.conj(y[l, t_start:t_end])))**2
            den = np.sum(np.abs(x[l, t_start:t_end])**2) * np.sum(np.abs(y[l, t_start:t_end])**2) + eps
            coh[l, m] = num / den
    return np.clip(coh, 0, 1)


def process_multi_channel_coherence(
        tf_maps: np.ndarray,
        calc_coherence: callable,
        coherence_threshold: float = 0.5,
        logging: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    K, L, M = tf_maps.shape
    if K < 2:
        raise ValueError("At least 2 channels required for coherence analysis")

    num_pairs = K * (K - 1) // 2
    if logging:
        print(f"[COHERENCE] Processing {K} channels, {num_pairs} channel pairs")
        print(f"[COHERENCE] TF map shape: {L}×{M}")

    coherence_streams = np.zeros((num_pairs, L, M))
    max_coherence = np.zeros((L, M))
    best_pair_indices = np.zeros((L, M), dtype=int)

    pair_idx = 0
    for i in range(K):
        for j in range(i + 1, K):
            if logging:
                print(f"[COHERENCE] Pair ({i}, {j})")

            coherence = calc_coherence(tf_maps[i], tf_maps[j])
            coherence_streams[pair_idx] = coherence

            # --- Plot coherence map for this pair ---
            plt.figure(figsize=(14, 5))
            plt.imshow(coherence, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1)
            plt.title(f"Coherence Map for Pair {pair_idx} (Channels {i},{j})")
            plt.xlabel("Time [frames]")
            plt.ylabel("Frequency bins")
            plt.colorbar(label="Coherence")
            plt.tight_layout()
            plt.show()

            # --- Update max coherence and best pair indices ---
            better_mask = coherence > max_coherence
            max_coherence[better_mask] = coherence[better_mask]
            best_pair_indices[better_mask] = pair_idx
            pair_idx += 1

    # Threshold mask
    coherence_mask = max_coherence >= coherence_threshold

    # Average TF from the best coherent pair at each bin
    average_tf_map = np.zeros((L, M), dtype=np.complex64)
    for l in range(L):
        for m in range(M):
            if coherence_mask[l, m]:
                best_pair = best_pair_indices[l, m]
                i, j = _pair_index_to_channels(best_pair, K)
                average_tf_map[l, m] = (tf_maps[i, l, m] + tf_maps[j, l, m]) / 2.0
            else:
                average_tf_map[l, m] = 0.0

    return average_tf_map, coherence_mask, best_pair_indices


if __name__ == "__main__":
    audio_file_path = "/content/Drone1_trimmed.wav"  # <-- Update path
    y, sr = sf.read(audio_file_path)

    if y.ndim == 1:
        y = y[np.newaxis, :]
    else:
        y = y.T

    num_channels = y.shape[0]
    print(f"Loaded {num_channels} channels, {y.shape[1]} samples at {sr} Hz")

    n_fft = 4096
    hop_length = n_fft // 2

    # Compute STFT for each channel
    stft_list = [librosa.stft(y[ch], n_fft=n_fft, hop_length=hop_length) for ch in range(num_channels)]
    max_frames = max(stft.shape[1] for stft in stft_list)
    freq_bins = stft_list[0].shape[0]

    # Align all channels (pad shorter ones)
    tf_maps = np.zeros((num_channels, freq_bins, max_frames), dtype=np.complex64)
    for ch in range(num_channels):
        stft = stft_list[ch]
        tf_maps[ch, :, :stft.shape[1]] = stft

    # -------------------------------
    # Process coherence with smoothed coherence
    # -------------------------------
    avg_tf_map, coherence_mask, bm = process_multi_channel_coherence(
        tf_maps, lambda x,y: smooth_coherence(x, y, win_size=7),
        coherence_threshold=0.6, logging=True
    )

    # -------------------------------
    # Plot raw TF maps (full length)
    # -------------------------------
    for ch in range(num_channels):
        plt.figure(figsize=(14, 5))
        plt.imshow(20*np.log10(np.abs(tf_maps[ch]) + 1e-8),
                   origin='lower', aspect='auto')
        plt.title(f"Channel {ch} TF Map")
        plt.xlabel("Time [frames]")
        plt.ylabel("Frequency bins")
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
        plt.show()

    # -------------------------------
    # Summary plots
    # -------------------------------
    # Average TF
    plt.figure(figsize=(14, 5))
    avg_tf_noisy = np.mean(np.abs(tf_maps), axis=0)
    plt.imshow(20*np.log10(avg_tf_noisy + 1e-8), origin='lower', aspect='auto')
    plt.title("Average TF Map")
    plt.xlabel("Time [frames]")
    plt.ylabel("Frequency bins")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.show()

    # Noise-suppressed TF
    plt.figure(figsize=(14, 5))
    masked_tf = np.abs(avg_tf_map)
    plt.imshow(20*np.log10(masked_tf + 1e-8), origin='lower', aspect='auto')
    plt.title("Best Pair TF Map")
    plt.xlabel("Time [frames]")
    plt.ylabel("Frequency bins")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.show()

    # Best pair indices (color-coded)
    plt.figure(figsize=(14, 5))
    masked_bm = np.where(coherence_mask, bm, np.nan)
    plt.imshow(masked_bm, origin='lower', aspect='auto', cmap='tab10')
    plt.title("Best Pair Indices (masked by coherence threshold)")
    plt.xlabel("Time [frames]")
    plt.ylabel("Frequency bins")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

##In 8k hz
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from typing import Tuple


def _pair_index_to_channels(pair_idx: int, K: int) -> Tuple[int, int]:
    i = 0
    remaining = pair_idx
    while remaining >= (K - 1 - i):
        remaining -= (K - 1 - i)
        i += 1
    j = i + 1 + remaining
    return i, j

def smooth_coherence(x: np.ndarray, y: np.ndarray, win_size=5) -> np.ndarray:
    eps = 1e-8
    L, M = x.shape
    coh = np.zeros((L, M))
    for l in range(L):
        for m in range(M):
            t_start = max(0, m - win_size//2)
            t_end = min(M, m + win_size//2 + 1)
            num = np.abs(np.sum(x[l, t_start:t_end] * np.conj(y[l, t_start:t_end])))**2
            den = np.sum(np.abs(x[l, t_start:t_end])**2) * np.sum(np.abs(y[l, t_start:t_end])**2) + eps
            coh[l, m] = num / den
    return np.clip(coh, 0, 1)

def process_multi_channel_coherence(
        tf_maps: np.ndarray,
        calc_coherence: callable,
        coherence_threshold: float = 0.5,
        logging: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    K, L, M = tf_maps.shape
    if K < 2:
        raise ValueError("At least 2 channels required for coherence analysis")

    num_pairs = K * (K - 1) // 2
    if logging:
        print(f"[COHERENCE] Processing {K} channels, {num_pairs} channel pairs")
        print(f"[COHERENCE] TF map shape: {L}×{M}")

    coherence_streams = np.zeros((num_pairs, L, M))
    max_coherence = np.zeros((L, M))
    best_pair_indices = np.zeros((L, M), dtype=int)

    pair_idx = 0
    for i in range(K):
        for j in range(i + 1, K):
            if logging:
                print(f"[COHERENCE] Pair ({i}, {j})")

            coherence = calc_coherence(tf_maps[i], tf_maps[j])
            coherence_streams[pair_idx] = coherence

            # Plot coherence map for this pair
            plt.figure(figsize=(14, 5))
            plt.imshow(coherence, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1,
                       extent=[0, M*hop_length/sr, 0, sr/2])
            plt.title(f"Coherence Map for Pair {pair_idx} (Channels {i},{j})")
            plt.xlabel("Time [s]")
            plt.ylabel("Frequency [Hz]")
            plt.colorbar(label="Coherence")
            plt.tight_layout()
            plt.show()

            better_mask = coherence > max_coherence
            max_coherence[better_mask] = coherence[better_mask]
            best_pair_indices[better_mask] = pair_idx
            pair_idx += 1

    coherence_mask = max_coherence >= coherence_threshold

    average_tf_map = np.zeros((L, M), dtype=np.complex64)
    for l in range(L):
        for m in range(M):
            if coherence_mask[l, m]:
                best_pair = best_pair_indices[l, m]
                i, j = _pair_index_to_channels(best_pair, K)
                average_tf_map[l, m] = (tf_maps[i, l, m] + tf_maps[j, l, m]) / 2.0
            else:
                average_tf_map[l, m] = 0.0

    return average_tf_map, coherence_mask, best_pair_indices


if __name__ == "__main__":
    audio_file_path = "/content/Drone1_trimmed.wav"
    y, sr = sf.read(audio_file_path)

    if y.ndim == 1:
        y = y[np.newaxis, :]
    else:
        y = y.T

    num_channels = y.shape[0]
    print(f"Loaded {num_channels} channels, {y.shape[1]} samples at {sr} Hz")

    n_fft = 4096
    hop_length = n_fft // 2

    # Compute STFT for each channel
    stft_list = [librosa.stft(y[ch], n_fft=n_fft, hop_length=hop_length) for ch in range(num_channels)]
    max_frames = max(stft.shape[1] for stft in stft_list)
    freq_bins = stft_list[0].shape[0]

    tf_maps = np.zeros((num_channels, freq_bins, max_frames), dtype=np.complex64)
    for ch in range(num_channels):
        stft = stft_list[ch]
        tf_maps[ch, :, :stft.shape[1]] = stft

    # Process coherence
    avg_tf_map, coherence_mask, bm = process_multi_channel_coherence(
        tf_maps, lambda x,y: smooth_coherence(x, y, win_size=7),
        coherence_threshold=0.6, logging=True
    )

    # Frequency and time axes
    freq_axis = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    time_axis = np.arange(max_frames) * hop_length / sr  # in seconds

    # Plot raw TF maps
    for ch in range(num_channels):
        plt.figure(figsize=(14, 5))
        plt.imshow(20*np.log10(np.abs(tf_maps[ch]) + 1e-8),
                   origin='lower', aspect='auto',
                   extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
        plt.title(f"Channel {ch} TF Map")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
        plt.show()

    # Average TF
    plt.figure(figsize=(14, 5))
    avg_tf_noisy = np.mean(np.abs(tf_maps), axis=0)
    plt.imshow(20*np.log10(avg_tf_noisy + 1e-8), origin='lower', aspect='auto',
               extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
    plt.title("Average TF Map")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.show()

    # Noise-suppressed TF
    plt.figure(figsize=(14, 5))
    masked_tf = np.abs(avg_tf_map)
    plt.imshow(20*np.log10(masked_tf + 1e-8), origin='lower', aspect='auto',
               extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
    plt.title("Best Pair TF Map")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.show()

    # Best pair indices
    plt.figure(figsize=(14, 5))
    masked_bm = np.where(coherence_mask, bm, np.nan)
    plt.imshow(masked_bm, origin='lower', aspect='auto',
               extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]], cmap='tab10')
    plt.title("Best Pair Indices (masked by coherence threshold)")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
##Audio Filter
#!/usr/bin/env python3
"""
High-pass filter for multi-channel WAV files.

Usage:
  python audio_filter.py --inputs drone_data1.wav drone_data2.WAV --cutoff 300 --order 4 --output-dir outputs
"""

import argparse
import os
from typing import Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt


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
    _num_samples, num_channels = data.shape
    b, a = design_highpass(fs, cutoff_hz, order)
    filtered = np.empty_like(data, dtype=np.float64)
    for ch in range(num_channels):
        filtered[:, ch] = filtfilt(b, a, data[:, ch], axis=0)
    return filtered.squeeze()


def build_output_path(output_dir: str, input_path: str, suffix: str) -> str:
    base = os.path.basename(input_path)
    name, ext = os.path.splitext(base)
    return os.path.join(output_dir, f"{name}{suffix}{ext}")


def process_file(input_path: str, output_dir: str, cutoff_hz: float, order: int) -> str:
    fs, data = wavfile.read(input_path)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(f"Expected a 4-channel WAV. Got shape {data.shape} from {input_path}")
    data_f64, original = convert_to_float64(data)
    filtered_f64 = apply_highpass_per_channel(data_f64, fs, cutoff_hz, order)
    filtered = convert_from_float64(filtered_f64, original)
    ensure_dir(output_dir)
    output_path = build_output_path(output_dir, input_path, suffix=f"_hp_{int(cutoff_hz)}Hz_o{order}")
    wavfile.write(output_path, fs, filtered)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-pass filter for multi-channel WAV files")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input WAV paths")
    parser.add_argument("--cutoff", type=float, default=300.0, help="High-pass cutoff frequency in Hz (default: 300)")
    parser.add_argument("--order", type=int, default=4, help="Butterworth filter order (default: 4)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to write filtered WAVs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = []
    for ip in args.inputs:
        outputs.append(process_file(ip, args.output_dir, args.cutoff, args.order))
    print("Wrote:")
    for op in outputs:
        print(f"  {op}")


if __name__ == "__main__":
    main()


##Coherence for two channels
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from typing import Tuple
import soundfile as sf

# -------------------------------
# Core Functions (Unchanged)
# -------------------------------

def process_coherence_for_two_channels(
        tf_maps: np.ndarray,
        calc_coherence: callable,
        coherence_threshold: float = 0.5,
        logging: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes coherence for exactly two channels.
    tf_maps: (2, L, M) complex STFT maps, L=freq bins, M=time frames
    calc_coherence: function(x,y) -> coherence matrix (L,M)
    Returns: (suppressed_tf, mask, coherence_map)
    """
    K, L, M = tf_maps.shape
    if K != 2:
        raise ValueError("This function is designed for exactly 2 channels.")

    if logging:
        print(f"[COHERENCE] Processing 2 channels...")
        print(f"[COHERENCE] TF map shape: {L}×{M}")

    ch0_tf, ch1_tf = tf_maps[0], tf_maps[1]
    coherence_map = calc_coherence(ch0_tf, ch1_tf)
    coherence_mask = coherence_map >= coherence_threshold
    avg_tf_map = np.zeros((L, M), dtype=np.complex64)
    avg_tf_map[coherence_mask] = (ch0_tf[coherence_mask] + ch1_tf[coherence_mask]) / 2.0

    if logging:
        percent_coherent = np.sum(coherence_mask) / (L * M) * 100
        print(f"[COHERENCE] Coherent area (>{coherence_threshold}): {percent_coherent:.2f}%")

    return avg_tf_map, coherence_mask, coherence_map


def simple_coherence(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pixel-wise coherence: |X*Y| / (|X||Y|)"""
    eps = 1e-8
    return np.abs(x * np.conj(y)) / (np.abs(x) * np.abs(y) + eps)

# -------------------------------
# Main Pipeline
# -------------------------------
if __name__ == "__main__":
    # ------------------------------
    # 1. LOAD YOUR TWO SEPARATE AUDIO FILES
    # ------------------------------
    # ❗️ UPDATE THESE PATHS to your two separate audio files
    # The script will fail if these paths are not correct.
    audio_file_path_1 = "/content/drone_data1_trim.wav"
    audio_file_path_2 = "/content/drone_data2_trim.wav"
    
    print(f"Loading files:\n- {audio_file_path_1}\n- {audio_file_path_2}")
    y1, sr1 = sf.read(audio_file_path_1)
    y2, sr2 = sf.read(audio_file_path_2)

    # --- Data validation and preparation ---
    assert sr1 == sr2, f"Sample rates must match. Got {sr1} Hz and {sr2} Hz."
    sr = sr1

    if y1.ndim > 1:
        print(f"⚠️ Warning: {audio_file_path_1} is not mono. Using its first channel.")
        y1 = y1[:, 0]
    if y2.ndim > 1:
        print(f"⚠️ Warning: {audio_file_path_2} is not mono. Using its first channel.")
        y2 = y2[:, 0]

    min_len = min(len(y1), len(y2))
    if len(y1) != len(y2):
        print(f"⚠️ Warning: Files have different lengths. Truncating to {min_len} samples.")
    y1, y2 = y1[:min_len], y2[:min_len]

    y = np.array([y1, y2])
    print(f"Loaded 2 separate files, using {y.shape[1]} samples at {sr} Hz")

    # ------------------------------
    # 2. COMPUTE STFT
    # ------------------------------
    n_fft = 1024
    hop_length = 512
    tf_maps = np.array([librosa.stft(channel_data, n_fft=n_fft, hop_length=hop_length) for channel_data in y])

    # ------------------------------
    # 3. PROCESS COHERENCE
    # ------------------------------
    suppressed_tf, mask, coherence_map = process_coherence_for_two_channels(
        tf_maps, simple_coherence, coherence_threshold=0.7, logging=True
    )

    # -------------------------------
    # 4. PLOTS
    # -------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("Two-File Coherence Analysis", fontsize=16)

    db_file1 = librosa.amplitude_to_db(np.abs(tf_maps[0]), ref=np.max)
    db_file2 = librosa.amplitude_to_db(np.abs(tf_maps[1]), ref=np.max)
    db_suppressed = librosa.amplitude_to_db(np.abs(suppressed_tf), ref=np.max)
    
    # Plot 1: File 1 Spectrogram
    img1 = librosa.display.specshow(db_file1, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=axs[0, 0])
    axs[0, 0].set_title(f"File 1: {audio_file_path_1.split('/')[-1]}")
    fig.colorbar(img1, ax=axs[0, 0], format="%+2.0f dB")

    # Plot 2: File 2 Spectrogram
    img2 = librosa.display.specshow(db_file2, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=axs[0, 1])
    axs[0, 1].set_title(f"File 2: {audio_file_path_2.split('/')[-1]}")
    fig.colorbar(img2, ax=axs[0, 1], format="%+2.0f dB")

    # Plot 3: Coherence Map
    img3 = axs[1, 0].imshow(coherence_map, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1, 
                           extent=[0, coherence_map.shape[1] * hop_length / sr, 0, sr / 2])
    axs[1, 0].set_title("Coherence Map (File 1 vs File 2)")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Frequency [Hz]")
    fig.colorbar(img3, ax=axs[1, 0], label="Coherence")

    # Plot 4: Noise-Suppressed Result
    img4 = librosa.display.specshow(db_suppressed, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=axs[1, 1])
    axs[1, 1].set_title("Noise-Suppressed Result")
    fig.colorbar(img4, ax=axs[1, 1], format="%+2.0f dB")

    plt.show()
##Cross Coherence
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



## Author

Vaishnavi Hiremath
