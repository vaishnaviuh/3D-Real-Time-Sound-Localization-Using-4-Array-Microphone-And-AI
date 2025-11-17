import os
import json
import numpy as np

from src.config import AppConfig
from src.utils import countdown, ensure_dir, now_timestamp
from src.audio import record_multichannel
from src.doa import (
    build_mic_positions,
    estimate_doa_az_el,
    rough_distance_from_energy,
    estimate_position_from_tdoa,
    apply_bandpass_filter,
    detect_harmonics,
    auto_detect_candidate_fundamentals,
    detect_signal_activity,
)
from src.classify import classify_audio_signal
from src.plotting import plot_2d_azimuth_distance, plot_3d_direction, plot_combined_2d_3d


def run_once(cfg: AppConfig) -> None:
    if cfg.audio.countdown_seconds > 0:
        countdown(cfg.audio.countdown_seconds)

    signals = record_multichannel(
        samplerate=cfg.audio.samplerate,
        duration_s=cfg.audio.record_seconds,
        dtype=cfg.audio.dtype,
        channels_to_use=cfg.audio.channels_to_use,
        device_query=cfg.audio.device_query,
        requested_channels=cfg.audio.requested_channels,
        blocksize=cfg.audio.blocksize,
    )
    # signals shape: (N, 4)

    # Apply frequency filtering (300-4000 Hz)
    signals = apply_bandpass_filter(
        signals=signals,
        fs=cfg.audio.samplerate,
        low_freq=cfg.detection.min_freq_hz,
        high_freq=cfg.detection.max_freq_hz,
    )
    
    # CLASSIFICATION: Detect trigger sounds (drone, mechanical, clap) and filter out speech/background
    trigger_detected = False  # Default to FALSE - only trigger if explicitly detected
    predicted_class = "unknown"
    classification_confidence = 0.0
    classification_features = {}
    
    if cfg.classification.enable_classification:
        predicted_class, classification_confidence, classification_features, rule_matches = classify_audio_signal(
            signals=signals,
            fs=cfg.audio.samplerate,
            config=cfg.classification,
            channel_idx=None,  # Use average of all channels
        )
        
        # Check if this is a trigger sound
        trigger_classes = []
        if cfg.classification.drone_enabled:
            trigger_classes.append('drone')
        if cfg.classification.mechanical_enabled:
            trigger_classes.append('mechanical')
        if cfg.classification.clap_enabled:
            trigger_classes.append('clap')
        
        trigger_detected = (
            predicted_class in trigger_classes and
            classification_confidence >= cfg.classification.min_trigger_confidence
        )
        
        if trigger_detected:
            print(f"✓ TRIGGER SOUND DETECTED → Localization Active")
            print(f"  Class: {predicted_class.upper()}, Confidence: {classification_confidence:.2f}")
            if cfg.detection.enable_debug:
                print(f"  Features: centroid={classification_features['spectral_centroid']:.1f}Hz, "
                      f"bandwidth={classification_features['spectral_bandwidth']:.1f}Hz, "
                      f"harmonic_ratio={classification_features['harmonic_ratio']:.3f}, "
                      f"rms={classification_features['rms_energy']:.6f}")
        else:
            print(f"✗ Sound classified as '{predicted_class}' (confidence: {classification_confidence:.2f}) - IGNORED")
            if cfg.detection.enable_debug:
                print(f"  Not a trigger sound. Skipping localization.")
                print(f"  Features: centroid={classification_features.get('spectral_centroid', 0):.1f}Hz, "
                      f"bandwidth={classification_features.get('spectral_bandwidth', 0):.1f}Hz, "
                      f"rms={classification_features.get('rms_energy', 0):.6f}")
            return
    
    # IMPORTANT: When classification is enabled, we ONLY proceed if trigger_detected=True
    # The old harmonic detection below is only for when classification is disabled
    # Skip harmonic detection when classification is enabled (already validated)
    harmonics_detected = False
    detected_fundamental = None
    
    if not cfg.classification.enable_classification:
        # OLD DETECTION: Only run when classification is disabled
        # Determine target fundamentals (explicit or auto-detected)
        target_fundamentals = list(cfg.detection.target_harmonic_fundamentals_hz)

        if target_fundamentals:
            harmonics_detected, detected_fundamental = detect_harmonics(
                signals=signals,
                fs=cfg.audio.samplerate,
                target_fundamentals_hz=target_fundamentals,
                min_harmonics=cfg.detection.min_harmonics_detected,
                tolerance_hz=cfg.detection.harmonic_tolerance_hz,
                min_magnitude_ratio=cfg.detection.harmonic_min_magnitude_ratio,
            )
        elif cfg.detection.auto_detect_harmonics:
            candidate_fundamentals = auto_detect_candidate_fundamentals(
                signals=signals,
                fs=cfg.audio.samplerate,
                min_freq_hz=cfg.detection.min_freq_hz,
                max_freq_hz=cfg.detection.max_freq_hz,
                min_peak_ratio=cfg.detection.auto_detect_min_peak_ratio,
                max_candidates=cfg.detection.auto_detect_max_candidates,
            )
            if cfg.detection.enable_debug:
                print(f"[DEBUG] Auto-detected harmonic candidates: {[f'{f:.1f}' for f in candidate_fundamentals]}")
            for candidate in candidate_fundamentals:
                harmonics_detected, detected_fundamental = detect_harmonics(
                    signals=signals,
                    fs=cfg.audio.samplerate,
                    target_fundamentals_hz=[candidate],
                    min_harmonics=cfg.detection.min_harmonics_detected,
                    tolerance_hz=cfg.detection.harmonic_tolerance_hz,
                    min_magnitude_ratio=cfg.detection.harmonic_min_magnitude_ratio,
                )
                if harmonics_detected:
                    break
    else:
        # Classification enabled - skip harmonic detection, already validated trigger
        harmonics_detected = True  # Set to True since classification passed
        detected_fundamental = None
        if cfg.detection.enable_debug:
            print("[DEBUG] Classification validated - skipping harmonic detection")

    # When classification is enabled, we ONLY use classification - skip old detection logic
    if not cfg.classification.enable_classification:
        # OLD DETECTION LOGIC (only used when classification is disabled)
        # Assess general signal activity as a fallback for broadband sources (voice/noise)
        signal_active, activity_info = detect_signal_activity(
            signals=signals,
            min_rms=cfg.detection.min_activity_rms,
            min_peak_to_mean=cfg.detection.min_activity_peak_to_mean,
        )
        if cfg.detection.enable_debug:
            print(
                "[DEBUG] Signal activity check: "
                f"rms={activity_info['rms_energy']:.6f} "
                f"(threshold {activity_info['min_rms_threshold']:.6f}), "
                f"peak_to_mean={activity_info['peak_to_mean']:.2f} "
                f"(threshold {activity_info['min_peak_to_mean_threshold']:.2f})"
            )
        
        if not harmonics_detected:
            if cfg.detection.enable_debug:
                print(f"[DEBUG] No harmonic match (targets: {cfg.detection.target_harmonic_fundamentals_hz})")
            if not signal_active:
                print("No harmonic match and insufficient broadband activity. Skipping localization.")
                return
            if cfg.detection.require_harmonic_match:
                print("Skipping localization because require_harmonic_match=True.")
                return
            if cfg.detection.enable_debug:
                print("[DEBUG] Proceeding with DOA estimation using broadband activity trigger.")
        else:
            if cfg.detection.enable_debug and detected_fundamental is not None:
                print(f"[DEBUG] Detected harmonic series with fundamental: {detected_fundamental:.1f} Hz")
    else:
        # Classification is enabled - we already validated trigger_detected above
        # Just log that we're proceeding with localization
        if cfg.detection.enable_debug:
            print("[DEBUG] Classification passed - proceeding with DOA estimation.")

    mic_positions = build_mic_positions(
        radius_m=cfg.geometry.radius_m, angles_deg=cfg.geometry.mic_angles_deg
    )
    
    # Estimate DOA (azimuth and elevation angles)
    az_deg, el_deg, confidence = estimate_doa_az_el(
        signals=signals,
        fs=cfg.audio.samplerate,
        mic_positions_m=mic_positions,
        speed_of_sound=cfg.geometry.speed_of_sound,
        min_correlation_quality=cfg.detection.min_correlation_quality,
        min_rms_energy=cfg.detection.min_rms_energy,
        enable_debug=cfg.detection.enable_debug,
    )
    
    # Validate angle calculation
    if az_deg == 0.0 and el_deg == 0.0 and confidence == 0.0:
        if cfg.detection.enable_debug:
            print("[DEBUG] DOA calculation failed - angles are 0.0 (likely insufficient mic correlation)")
        # Still proceed but with warning
    # Estimate distance via TDOA position solver; fallback to energy proxy if needed
    est_pos = estimate_position_from_tdoa(
        signals=signals,
        fs=cfg.audio.samplerate,
        mic_positions_m=mic_positions,
        speed_of_sound=cfg.geometry.speed_of_sound,
        ref_index=0,
        min_correlation_quality=cfg.detection.min_correlation_quality,
        enable_debug=cfg.detection.enable_debug,
    )
    if est_pos is not None:
        distance_m = float(np.linalg.norm(est_pos))
    else:
        distance_m = rough_distance_from_energy(signals) if cfg.plot.show_plots else None

    if confidence < cfg.detection.min_confidence_threshold:
        print(f"No valid sound source detected (confidence: {confidence:.2f})")
    else:
        if distance_m is not None:
            print(f"Azimuth: {az_deg:.1f} deg, Elevation: {el_deg:.1f} deg, Distance~{distance_m:.2f} m, Confidence: {confidence:.2f}")
        else:
            print(f"Azimuth: {az_deg:.1f} deg, Elevation: {el_deg:.1f} deg, Distance: unknown, Confidence: {confidence:.2f}")

    if cfg.saving.enable_save_audio or cfg.saving.enable_save_results:
        ensure_dir(cfg.saving.output_dir)
        ts = now_timestamp()
        if cfg.saving.enable_save_audio:
            np.save(os.path.join(cfg.saving.output_dir, f"audio_{ts}.npy"), signals)
        if cfg.saving.enable_save_results:
            with open(os.path.join(cfg.saving.output_dir, f"doa_{ts}.json"), "w") as f:
                json.dump(
                    {"azimuth_deg": az_deg, "elevation_deg": el_deg, "distance_m": distance_m},
                    f,
                    indent=2,
                )

    if cfg.plot.show_plots:
        # Combined big view with side-by-side 2D and 3D plots
        plot_combined_2d_3d(
            azimuth_deg=az_deg,
            elevation_deg=el_deg,
            distance_m=distance_m,
            mic_positions=mic_positions,
        )


if __name__ == "__main__":
    config = AppConfig()
    run_once(config)


