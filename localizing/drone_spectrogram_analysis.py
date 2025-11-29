#!/usr/bin/env python3
"""
Drone spectrogram + triangulation analysis for the 20‑channel WAV recording.

This script:
 1. Loads the multi‑channel WAV file.
 2. Computes and saves a spectrogram for the most energetic channel and (if present)
    the 16–20 channel cluster to visualize the drone's harmonic signature.
 3. Reuses the existing final WAV triangulation pipeline to localize the source and
    create the 2D/3D map.

Core triangulation logic in `src/triangulation.py` is NOT modified.
"""

import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import spectrogram

from final_wav_analysis import analyze_full_wav_file, create_results_map


def _compute_loudest_channel(audio: np.ndarray) -> int:
    """
    Return index of the loudest channel by RMS energy.

    audio shape: (channels, samples)
    """
    rms = np.sqrt(np.mean(audio ** 2, axis=1))
    return int(np.argmax(rms))


def _make_spectrogram_plot(
    signal: np.ndarray,
    sample_rate: int,
    title: str,
    output_path: str,
    max_freq_hz: float = 8000.0,
) -> None:
    """
    Compute and save a spectrogram plot for a 1D signal.
    """
    # Use a reasonably sized window to highlight tonal components (drone harmonics)
    nperseg = 2048
    noverlap = nperseg // 2

    f, t, Sxx = spectrogram(signal, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

    # Limit frequency axis if desired
    if max_freq_hz is not None:
        freq_mask = f <= max_freq_hz
        f = f[freq_mask]
        Sxx = Sxx[freq_mask, :]

    # Convert to dB scale for clearer visualization
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="magma")
    plt.colorbar(label="Power (dB)")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_spectrogram_and_triangulate(
    wav_path: str,
    kml_path: str,
    spectrogram_duration_s: float = 30.0,
) -> Tuple[str, str]:
    """
    1) Analyze spectrogram of the provided multi‑channel WAV to reveal drone signature.
    2) Run existing triangulation pipeline and generate 2D/3D map.

    Returns:
        Tuple of (spectrogram_image_path, triangulation_map_path)
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV file not found: {wav_path}")
    if not os.path.exists(kml_path):
        raise FileNotFoundError(f"KML file not found: {kml_path}")

    print("🎵 Loading multi‑channel WAV for spectrogram analysis...")
    audio_data, sample_rate = sf.read(wav_path, always_2d=True)  # shape: (samples, channels)
    audio_data = audio_data.T  # shape: (channels, samples)

    total_duration_s = audio_data.shape[1] / sample_rate
    print(f"   Channels: {audio_data.shape[0]}, Duration: {total_duration_s:.1f}s, fs={sample_rate} Hz")

    # Limit to initial segment for spectrogram (to keep figure compact and fast)
    max_samples = int(min(spectrogram_duration_s, total_duration_s) * sample_rate)
    segment = audio_data[:, :max_samples]

    # 1) Loudest single channel spectrogram
    loudest_idx = _compute_loudest_channel(segment)
    loudest_signal = segment[loudest_idx]
    print(f"   🔍 Using channel {loudest_idx + 1} (loudest) for detailed spectrogram.")

    spectrogram_dir = "plots"
    ch_spec_path = os.path.join(
        spectrogram_dir, f"drone_spectrogram_channel_{loudest_idx + 1}.png"
    )
    _make_spectrogram_plot(
        loudest_signal,
        sample_rate,
        title=f"Spectrogram – Channel {loudest_idx + 1} (Drone Signature)",
        output_path=ch_spec_path,
        max_freq_hz=8000.0,
    )
    print(f"   💾 Saved single‑channel spectrogram: {ch_spec_path}")

    # 2) Cluster (channels 16–20) spectrogram if available
    cluster_spec_path = ""
    if segment.shape[0] >= 20:
        cluster_signal = np.mean(segment[15:20, :], axis=0)
        print("   🔍 Computing spectrogram for cluster (channels 16–20).")
        cluster_spec_path = os.path.join(spectrogram_dir, "drone_spectrogram_cluster_16_20.png")
        _make_spectrogram_plot(
            cluster_signal,
            sample_rate,
            title="Spectrogram – Cluster Channels 16–20 (Drone Signature)",
            output_path=cluster_spec_path,
            max_freq_hz=8000.0,
        )
        print(f"   💾 Saved cluster spectrogram: {cluster_spec_path}")
    else:
        print("   ⚠️ Less than 20 channels – skipping cluster spectrogram.")

    print("\n📡 Running existing triangulation pipeline to create 2D/3D map...")
    # Reuse existing high‑level analysis and plotting functions
    sensor_names, sensor_positions, detections = analyze_full_wav_file(
        wav_path=wav_path,
        kml_path=kml_path,
        chunk_duration=2.0,
        max_chunks=100,
    )
    map_path = create_results_map(sensor_names, sensor_positions, detections)

    print("\n✅ Completed spectrogram + triangulation analysis.")
    return (cluster_spec_path or ch_spec_path), map_path


def main() -> None:
    kml_path = "Sensor-Locations-BOP-Dharma.kml"
    wav_path = "multi-20251122-141610-627897594.wav"

    try:
        spec_path, map_path = analyze_spectrogram_and_triangulate(wav_path, kml_path)
        print("\n📊 Summary:")
        print(f"   Spectrogram image: {spec_path}")
        print(f"   Triangulation map: {map_path}")
    except Exception as e:
        print(f"❌ Spectrogram + triangulation analysis failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()


