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



## Author

Vaishnavi Hiremath
