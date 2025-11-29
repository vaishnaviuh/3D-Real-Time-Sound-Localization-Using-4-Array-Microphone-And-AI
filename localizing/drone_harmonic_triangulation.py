#!/usr/bin/env python3
"""
Drone-harmonic-aware triangulation on the 20-channel WAV (15 sensors + 1 cluster).

This script:
  - Loads the multi-channel WAV file.
  - For each analysis chunk, analyzes the spectrogram of the cluster channel INTERNALLY
    (no plots) to detect a harmonic "drone-like" signature.
  - Only when drone harmonics are present does it call the existing triangulation
    engine to estimate 3D position.
  - Uses the existing 2D/3D map creation from `final_wav_analysis.py`.

Core triangulation logic in `src/triangulation.py` is NOT modified.
"""

import os
import time
from typing import List, Dict, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import spectrogram, find_peaks

from src.kml_parser import get_sensor_positions_xyz
from src.triangulation import TriangulationEngine

# Reuse existing helpers for channel handling and plotting
from final_wav_analysis import (
    combine_cluster_channels,
    simple_bandpass_filter,
    create_results_map,
)


def _detect_drone_harmonics(
    signal: np.ndarray,
    fs: int,
    min_fund_hz: float = 80.0,
    max_fund_hz: float = 800.0,
    min_harmonics: int = 3,
    peak_prominence_db: float = 6.0,
) -> Tuple[bool, float]:
    """
    Heuristic harmonic detector for a drone-like tone series in a 1D signal.

    Returns:
        (has_harmonics, detected_fundamental_hz)
    """
    if signal.size < fs:  # require at least 1 s of data for robust spectrum
        return False, 0.0

    # Spectrogram, then average over time to get a stable spectrum
    nperseg = min(4096, max(1024, fs // 4))
    noverlap = nperseg // 2
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

    spectrum = np.mean(Sxx, axis=1)
    spectrum_db = 10.0 * np.log10(spectrum + 1e-12)

    # Focus on plausible drone fundamental band
    band_mask = (f >= min_fund_hz) & (f <= max_fund_hz)
    if not np.any(band_mask):
        return False, 0.0

    f_band = f[band_mask]
    spec_band_db = spectrum_db[band_mask]

    # Find prominent peaks in this band
    global_max = np.max(spec_band_db)
    min_height = global_max - peak_prominence_db
    peaks, peak_props = find_peaks(spec_band_db, height=min_height, distance=3)

    if peaks.size == 0:
        return False, 0.0

    candidate_freqs = f_band[peaks]

    # For each candidate fundamental, check if we see multiple harmonics
    best_fund = 0.0
    best_count = 0

    for fund in candidate_freqs:
        harmonic_indices = []
        # Look for up to ~8 harmonics within Nyquist
        max_harm = int((fs / 2) // fund)
        if max_harm < 2:
            continue

        for h in range(1, max_harm + 1):
            target = h * fund
            # tolerance proportional to fundamental
            tol = max(3.0, 0.03 * target)  # at least 3 Hz or 3%
            idx_candidates = np.where(np.abs(f - target) <= tol)[0]
            if idx_candidates.size == 0:
                continue

            # Check if any of these frequencies have high enough power
            if np.max(spectrum_db[idx_candidates]) >= global_max - 2.0 * peak_prominence_db:
                harmonic_indices.append(h)

        if len(harmonic_indices) >= min_harmonics:
            if len(harmonic_indices) > best_count:
                best_count = len(harmonic_indices)
                best_fund = float(fund)

    if best_count >= min_harmonics and best_fund > 0.0:
        return True, best_fund

    return False, 0.0


def analyze_wav_with_drone_harmonics(
    wav_path: str,
    kml_path: str,
    chunk_duration: float = 2.0,
    max_chunks: int | None = 100,
) -> Tuple[List[str], List[List[float]], List[Dict]]:
    """
    Analyze the WAV file, selecting only chunks that show drone-like harmonics
    in the cluster channel, and triangulate those chunks.

    Returns:
        (sensor_names, sensor_positions, detections)
    """
    print("🎵 Drone-harmonic-aware WAV analysis")
    print("=" * 60)

    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV file not found: {wav_path}")
    if not os.path.exists(kml_path):
        raise FileNotFoundError(f"KML file not found: {kml_path}")

    # Load multi-channel audio
    print(f"📂 Loading WAV: {wav_path}")
    audio_data, sample_rate = sf.read(wav_path, always_2d=True)  # (samples, channels)
    audio_data = audio_data.T  # (channels, samples)

    duration_s = audio_data.shape[1] / sample_rate
    print(f"   ✅ Loaded {audio_data.shape[0]} channels, {duration_s:.1f}s, {sample_rate} Hz")

    # Load sensor positions (15 sensors + 1 cluster in XYZ)
    print(f"📍 Loading sensors: {kml_path}")
    sensor_names, sensor_positions_array = get_sensor_positions_xyz(
        kml_path, add_opposite_sensors=True
    )
    print(f"   ✅ Loaded {len(sensor_names)} sensors (including cluster)")

    # Initialize triangulation engine (core logic stays untouched)
    print("🔧 Initializing triangulation engine...")
    engine = TriangulationEngine(sensor_positions_array, speed_of_sound=343.0)

    # Chunking parameters
    chunk_samples = int(chunk_duration * sample_rate)
    overlap = 0.5
    step_samples = int(chunk_samples * (1.0 - overlap))

    total_samples = audio_data.shape[1]
    total_possible_chunks = max(0, (total_samples - chunk_samples) // step_samples + 1)
    if max_chunks is None:
        chunks_to_process = total_possible_chunks
    else:
        chunks_to_process = min(max_chunks, total_possible_chunks)

    print(
        f"🔍 Processing {chunks_to_process}/{total_possible_chunks} chunks "
        f"({chunk_duration:.1f}s each, {int(overlap * 100)}% overlap)"
    )

    # Energy / activity thresholds
    energy_threshold = 0.002
    min_active_channels = 8
    confidence_threshold = 0.15

    print(
        f"   Thresholds: energy>{energy_threshold}, "
        f"active_ch>={min_active_channels}, confidence>{confidence_threshold}"
    )
    print("   Drone detection: using harmonic analysis of cluster channel only.")

    detections: List[Dict] = []
    start_time = time.time()

    for i in range(chunks_to_process):
        start_sample = i * step_samples
        end_sample = start_sample + chunk_samples
        timestamp = start_sample / sample_rate

        # Extract chunk (channels, samples)
        chunk = audio_data[:, start_sample:end_sample]
        if chunk.shape[1] < chunk_samples:
            break  # last short chunk – skip for consistency

        # Light bandpass per-chunk (reuses existing helper)
        try:
            chunk = simple_bandpass_filter(chunk, sample_rate, 300.0, 4000.0)
        except Exception:
            pass  # always keep something usable

        # Combine 16–20 into cluster
        processed = combine_cluster_channels(chunk)  # (<=16 channels, samples)

        # Basic energy metrics
        rms_levels = np.sqrt(np.mean(processed**2, axis=1))
        max_energy = float(np.max(rms_levels))
        active_channels = int(np.sum(rms_levels > energy_threshold))

        # Periodic progress log
        if i % 25 == 0:
            elapsed = time.time() - start_time
            progress = (i + 1) / max(chunks_to_process, 1) * 100.0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (chunks_to_process - i - 1) / max(rate, 1e-6)
            print(
                f"   📊 {progress:5.1f}% ({i+1:3d}/{chunks_to_process}) "
                f"| {elapsed:5.1f}s | {rate:4.1f} ch/s | ETA: {eta:4.0f}s "
                f"| Detections: {len(detections)}"
            )

        # Require basic activity first
        if max_energy <= energy_threshold or active_channels < min_active_channels:
            if i % 80 == 0:
                print(
                    f"      ⏭️  t={timestamp:6.1f}s: low energy/active sensors "
                    f"(energy={max_energy:.6f}, active={active_channels})"
                )
            continue

        # Drone harmonic detection on cluster channel (index 15 if present)
        if processed.shape[0] >= 16:
            cluster_signal = processed[15]
        else:
            # fallback: use loudest channel if we don't have a full cluster
            cluster_signal = processed[np.argmax(rms_levels)]

        has_drone, fund_hz = _detect_drone_harmonics(
            cluster_signal,
            fs=sample_rate,
            min_fund_hz=80.0,
            max_fund_hz=800.0,
            min_harmonics=3,
            peak_prominence_db=6.0,
        )

        if not has_drone:
            # Skip chunks that are loud but not harmonic (e.g., wind or broadband noise)
            if i % 40 == 0:
                print(f"      🔇 t={timestamp:6.1f}s: no clear drone harmonics detected")
            continue

        # At this point: loud, enough active channels, AND harmonic drone signature
        try:
            result = engine.triangulate_audio_chunk(
                processed.T,  # (samples, channels)
                sample_rate,
                tdoa_method="gcc_phat",
                triangulation_method="robust",
            )

            if result and result.confidence > confidence_threshold:
                detection = {
                    "timestamp": timestamp,
                    "position": result.position.copy(),
                    "confidence": float(result.confidence),
                    "residual_error": float(result.residual_error),
                    "method": result.method,
                    "max_energy": max_energy,
                    "active_channels": active_channels,
                    "num_sensors_used": int(result.num_sensors_used),
                    "detected_fundamental_hz": float(fund_hz),
                }
                detections.append(detection)

                print(
                    f"      ✅ t={timestamp:6.1f}s: "
                    f"pos=({result.position[0]:6.1f}, {result.position[1]:6.1f}, {result.position[2]:6.1f}) "
                    f"conf={result.confidence:.3f} fund≈{fund_hz:6.1f} Hz "
                    f"sensors={result.num_sensors_used}"
                )
        except Exception as e:
            if i % 50 == 0:
                print(f"      ❌ t={timestamp:6.1f}s: triangulation error: {str(e)[:60]}...")

    elapsed_total = time.time() - start_time
    print("\n🎯 Drone-harmonic analysis complete")
    print(f"   ⏱️  Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(
        f"   📊 Processed: {chunks_to_process} chunks "
        f"({chunks_to_process * chunk_duration:.0f}s of audio)"
    )
    print(f"   🎯 Detections (drone-like): {len(detections)}")

    if detections:
        confidences = [d["confidence"] for d in detections]
        fundamentals = [d["detected_fundamental_hz"] for d in detections]
        print(
            f"   📈 Confidence: avg={np.mean(confidences):.3f}, "
            f"range={np.min(confidences):.3f}-{np.max(confidences):.3f}"
        )
        print(
            f"   🎵 Fundamental (Hz): avg={np.mean(fundamentals):.1f}, "
            f"range={np.min(fundamentals):.1f}-{np.max(fundamentals):.1f}"
        )

    return sensor_names, sensor_positions_array.tolist(), detections


def main() -> None:
    kml_path = "Sensor-Locations-BOP-Dharma.kml"
    wav_path = "multi-20251122-141610-627897594.wav"

    if not os.path.exists(kml_path):
        print(f"❌ KML file not found: {kml_path}")
        return
    if not os.path.exists(wav_path):
        print(f"❌ WAV file not found: {wav_path}")
        return

    try:
        sensor_names, sensor_positions, detections = analyze_wav_with_drone_harmonics(
            wav_path=wav_path,
            kml_path=kml_path,
            chunk_duration=2.0,
            max_chunks=100,
        )

        # Reuse existing 2D/3D map rendering (15 sensors + 1 cluster)
        map_path = create_results_map(sensor_names, sensor_positions, detections)

        print("\n🎉 Drone-harmonic triangulation complete!")
        print(f"   📊 Drone-like detections: {len(detections)}")
        print(f"   🗺️  Map saved to: {map_path}")

    except Exception as e:
        print(f"❌ Drone-harmonic triangulation failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()


