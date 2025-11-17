#!/usr/bin/env python3
"""
Diagnostic script to check why sound tracking isn't working.
"""
import numpy as np
from src.config import AppConfig
from src.audio import record_multichannel
from src.doa import apply_bandpass_filter, detect_harmonics
import matplotlib.pyplot as plt

def analyze_audio():
    cfg = AppConfig()
    
    print("=" * 60)
    print("SOUND TRACKING DIAGNOSTIC")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Frequency range: {cfg.detection.min_freq_hz} - {cfg.detection.max_freq_hz} Hz")
    print(f"  Target harmonics: {cfg.detection.target_harmonic_fundamentals_hz}")
    print(f"  Min harmonics required: {cfg.detection.min_harmonics_detected}")
    print(f"  Harmonic tolerance: {cfg.detection.harmonic_tolerance_hz} Hz")
    print(f"  Min magnitude ratio: {cfg.detection.harmonic_min_magnitude_ratio}")
    
    print(f"\nRecording {cfg.audio.record_seconds} seconds of audio...")
    signals = record_multichannel(
        samplerate=cfg.audio.samplerate,
        duration_s=cfg.audio.record_seconds,
        dtype=cfg.audio.dtype,
        channels_to_use=cfg.audio.channels_to_use,
        device_query=cfg.audio.device_query,
        requested_channels=cfg.audio.requested_channels,
        blocksize=cfg.audio.blocksize,
    )
    
    print(f"Signal shape: {signals.shape}")
    print(f"Signal RMS (before filter): {np.sqrt(np.mean(signals**2)):.6f}")
    
    # Apply frequency filter
    signals_filtered = apply_bandpass_filter(
        signals=signals,
        fs=cfg.audio.samplerate,
        low_freq=cfg.detection.min_freq_hz,
        high_freq=cfg.detection.max_freq_hz,
    )
    
    print(f"Signal RMS (after filter): {np.sqrt(np.mean(signals_filtered**2)):.6f}")
    
    # Analyze frequency content
    signal = np.mean(signals_filtered, axis=1)
    n = len(signal)
    nfft = 1
    while nfft < n:
        nfft <<= 1
    
    fft_signal = np.fft.rfft(signal, n=nfft)
    magnitude = np.abs(fft_signal)
    freqs = np.fft.rfftfreq(nfft, 1.0/cfg.audio.samplerate)
    
    # Find dominant frequencies
    magnitude_norm = magnitude / np.max(magnitude) if np.max(magnitude) > 0 else magnitude
    peak_indices = np.where(magnitude_norm > 0.1)[0]
    peak_freqs = freqs[peak_indices]
    peak_mags = magnitude_norm[peak_indices]
    
    print(f"\nFrequency Analysis:")
    print(f"  Dominant frequencies (top 10):")
    if len(peak_freqs) > 0:
        top_indices = np.argsort(peak_mags)[-10:][::-1]
        for idx in top_indices:
            print(f"    {peak_freqs[idx]:.1f} Hz - magnitude: {peak_mags[idx]:.3f}")
    else:
        print("    No significant peaks found!")
    
    # Check harmonic detection
    print(f"\nHarmonic Detection Test:")
    harmonics_detected, detected_fundamental = detect_harmonics(
        signals=signals_filtered,
        fs=cfg.audio.samplerate,
        target_fundamentals_hz=cfg.detection.target_harmonic_fundamentals_hz,
        min_harmonics=cfg.detection.min_harmonics_detected,
        tolerance_hz=cfg.detection.harmonic_tolerance_hz,
        min_magnitude_ratio=cfg.detection.harmonic_min_magnitude_ratio,
    )
    
    if not cfg.detection.target_harmonic_fundamentals_hz:
        print("  ✓ Harmonic detection is DISABLED (empty list) - all sounds will be tracked")
        print("  ✓ This should work for any sound in the frequency range")
    else:
        if harmonics_detected:
            print(f"  ✓ Harmonics DETECTED! Fundamental: {detected_fundamental:.1f} Hz")
            print("  ✓ Sound tracking should work")
        else:
            print("  ✗ Harmonics NOT detected")
            print(f"  ✗ Looking for fundamentals: {cfg.detection.target_harmonic_fundamentals_hz} Hz")
            print("  ✗ Sound tracking will be SKIPPED")
            print("\n  Possible solutions:")
            print("    1. Make sure the sound contains the target harmonics")
            print("    2. Increase harmonic_tolerance_hz (currently {})".format(cfg.detection.harmonic_tolerance_hz))
            print("    3. Decrease harmonic_min_magnitude_ratio (currently {})".format(cfg.detection.harmonic_min_magnitude_ratio))
            print("    4. Set target_harmonic_fundamentals_hz = [] to disable harmonic detection")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    analyze_audio()


