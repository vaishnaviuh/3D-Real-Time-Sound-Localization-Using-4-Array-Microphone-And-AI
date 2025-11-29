from __future__ import annotations

from typing import Tuple, Optional, Dict, Any, List

import numpy as np

from src.config import AppConfig
from src.doa import (
    estimate_doa_az_el,
    estimate_position_from_tdoa,
    rough_distance_from_energy,
    detect_harmonics,
    auto_detect_candidate_fundamentals,
    detect_signal_activity,
    compute_signal_quality,
)
from src.classify import classify_audio_signal


def _evaluate_detection(signals: np.ndarray, cfg: AppConfig, enable_debug: bool) -> Tuple[bool, Dict[str, Any]]:
    detection_cfg = cfg.detection
    classification_cfg = cfg.classification

    rms_energy, peak_to_mean = compute_signal_quality(signals)
    info: Dict[str, Any] = {
        "classification": None,
        "trigger_detected": False,
        "harmonic_match": False,
        "detected_fundamental_hz": None,
        "signal_active": False,
        "activity_info": {},
        "rms_energy": float(rms_energy),
        "peak_to_mean": float(peak_to_mean),
    }

    if classification_cfg.enable_classification:
        predicted_class, classification_confidence, classification_features, rule_matches = classify_audio_signal(
            signals=signals,
            fs=cfg.audio.samplerate,
            config=classification_cfg,
            channel_idx=None,
        )

        trigger_classes = []
        if classification_cfg.drone_enabled:
            trigger_classes.append("drone")
        if classification_cfg.mechanical_enabled:
            trigger_classes.append("mechanical")
        if classification_cfg.clap_enabled:
            trigger_classes.append("clap")

        trigger_detected = (
            predicted_class in trigger_classes and classification_confidence >= classification_cfg.min_trigger_confidence
        )

        info["classification"] = {
            "predicted_class": predicted_class,
            "confidence": float(classification_confidence),
            "trigger_detected": trigger_detected,
            "features": classification_features,
            "rule_matches": rule_matches,
        }
        info["trigger_detected"] = trigger_detected

        if trigger_detected:
            if enable_debug:
                print(
                    f"✓ TRIGGER SOUND DETECTED → Localization Active | "
                    f"{predicted_class.upper()} ({classification_confidence:.2f})"
                )
            return True, info
        else:
            if enable_debug:
                print(
                    f"[DEBUG] Sound classified as '{predicted_class}' "
                    f"(confidence: {classification_confidence:.2f}) - IGNORED"
                )
                print("[DEBUG] Not a trigger sound. Skipping localization.")
            info["reason"] = "classification_rejected"
            return False, info

    # --- Harmonic / broadband detection path ---

    harmonics_detected = False
    detected_fundamental = None
    detection_cfg_targets = list(detection_cfg.target_harmonic_fundamentals_hz)

    if detection_cfg_targets:
        harmonics_detected, detected_fundamental = detect_harmonics(
            signals=signals,
            fs=cfg.audio.samplerate,
            target_fundamentals_hz=detection_cfg_targets,
            min_harmonics=detection_cfg.min_harmonics_detected,
            tolerance_hz=detection_cfg.harmonic_tolerance_hz,
            min_magnitude_ratio=detection_cfg.harmonic_min_magnitude_ratio,
        )
    elif detection_cfg.auto_detect_harmonics:
        candidate_fundamentals = auto_detect_candidate_fundamentals(
            signals=signals,
            fs=cfg.audio.samplerate,
            min_freq_hz=detection_cfg.min_freq_hz,
            max_freq_hz=detection_cfg.max_freq_hz,
            min_peak_ratio=detection_cfg.auto_detect_min_peak_ratio,
            max_candidates=detection_cfg.auto_detect_max_candidates,
        )
        if enable_debug:
            print(f"[DEBUG] Auto-detected harmonic candidates: {[f'{f:.1f}' for f in candidate_fundamentals]}")
        for candidate in candidate_fundamentals:
            harmonics_detected, detected_fundamental = detect_harmonics(
                signals=signals,
                fs=cfg.audio.samplerate,
                target_fundamentals_hz=[candidate],
                min_harmonics=detection_cfg.min_harmonics_detected,
                tolerance_hz=detection_cfg.harmonic_tolerance_hz,
                min_magnitude_ratio=detection_cfg.harmonic_min_magnitude_ratio,
            )
            if harmonics_detected:
                break

    signal_active, activity_info = detect_signal_activity(
        signals=signals,
        min_rms=detection_cfg.min_activity_rms,
        min_peak_to_mean=detection_cfg.min_activity_peak_to_mean,
    )

    if enable_debug:
        print(
            "[DEBUG] Signal activity check: "
            f"rms={activity_info['rms_energy']:.6f} "
            f"(threshold {activity_info['min_rms_threshold']:.6f}), "
            f"peak_to_mean={activity_info['peak_to_mean']:.2f} "
            f"(threshold {activity_info['min_peak_to_mean_threshold']:.2f})"
        )
        if detected_fundamental is not None:
            print(f"[DEBUG] Detected harmonic series with fundamental: {detected_fundamental:.1f} Hz")
        elif detection_cfg_targets:
            print(f"[DEBUG] No harmonic match (targets: {detection_cfg_targets})")

    info["harmonic_match"] = harmonics_detected
    info["detected_fundamental_hz"] = detected_fundamental
    info["signal_active"] = signal_active
    info["activity_info"] = activity_info

    if not harmonics_detected:
        if not signal_active:
            info["reason"] = "insufficient_activity"
            if enable_debug:
                print("No harmonic match and insufficient broadband activity. Skipping localization.")
            return False, info
        if detection_cfg.require_harmonic_match:
            info["reason"] = "harmonic_required"
            if enable_debug:
                print("Skipping localization because require_harmonic_match=True.")
            return False, info
        if enable_debug:
            print("[DEBUG] Proceeding with DOA estimation using broadband activity trigger.")
    else:
        if enable_debug:
            print("[DEBUG] Harmonic match satisfied - proceeding with DOA estimation.")

    info["reason"] = "ok"
    return True, info


def process_signals(
    signals: np.ndarray,
    cfg: AppConfig,
    mic_positions: np.ndarray,
    enable_debug: bool = False,
) -> Dict[str, Any]:
    """
    Full localization pipeline for a single microphone array.
    Returns a dictionary containing detection diagnostics and localization outputs.
    """
    detection_ok, info = _evaluate_detection(signals, cfg, enable_debug)
    if not detection_ok:
        info.setdefault("should_process", False)
        return info

    az_deg, el_deg, confidence = estimate_doa_az_el(
        signals=signals,
        fs=cfg.audio.samplerate,
        mic_positions_m=mic_positions,
        speed_of_sound=cfg.geometry.speed_of_sound,
        min_correlation_quality=cfg.detection.min_correlation_quality,
        min_rms_energy=cfg.detection.min_rms_energy,
        enable_debug=enable_debug,
    )

    if az_deg == 0.0 and el_deg == 0.0 and confidence == 0.0:
        info.update(
            {
                "should_process": False,
                "reason": "doa_failed",
                "azimuth_deg": az_deg,
                "elevation_deg": el_deg,
                "confidence": confidence,
            }
        )
        return info

    position_vector = estimate_position_from_tdoa(
        signals=signals,
        fs=cfg.audio.samplerate,
        mic_positions_m=mic_positions,
        speed_of_sound=cfg.geometry.speed_of_sound,
        ref_index=0,
        min_correlation_quality=cfg.detection.min_correlation_quality,
        enable_debug=enable_debug,
    )

    if position_vector is not None:
        distance_m = float(np.linalg.norm(position_vector))
        pairwise_baselines = [
            np.linalg.norm(mic_positions[i] - mic_positions[j])
            for i in range(mic_positions.shape[0])
            for j in range(i + 1, mic_positions.shape[0])
        ]
        max_baseline = max(pairwise_baselines) if pairwise_baselines else 0.0
        if max_baseline > 0:
            baseline_ratio = distance_m / max_baseline
            if baseline_ratio > 10:
                distance_confidence = max(0.1, 1.0 - (baseline_ratio - 10) * 0.1)
            elif baseline_ratio > 5:
                distance_confidence = 0.5 + 0.5 * (10 - baseline_ratio) / 5
            else:
                distance_confidence = 1.0

            if distance_m > max_baseline:
                tdoa_uncertainty_s = 3.9e-6
                distance_uncertainty = (
                    (distance_m ** 2) / (max_baseline * cfg.geometry.speed_of_sound) * tdoa_uncertainty_s
                )
                distance_uncertainty = max(distance_uncertainty, 0.01)
            else:
                distance_uncertainty = 0.01
        else:
            baseline_ratio = None
            distance_confidence = None
            distance_uncertainty = None

        per_mic_distances = [
            float(np.linalg.norm(position_vector - mic_positions[i])) for i in range(mic_positions.shape[0])
        ]
    else:
        distance_m = rough_distance_from_energy(signals) if cfg.plot.show_plots else None
        distance_confidence = None
        distance_uncertainty = None
        baseline_ratio = None
        per_mic_distances = None

    detected_flag = (
        confidence >= cfg.detection.min_confidence_threshold or cfg.detection.always_broadcast
    )

    direction_vector = np.array(
        [
            np.cos(np.deg2rad(el_deg)) * np.cos(np.deg2rad(az_deg)),
            np.cos(np.deg2rad(el_deg)) * np.sin(np.deg2rad(az_deg)),
            np.sin(np.deg2rad(el_deg)),
        ]
    )
    norm = np.linalg.norm(direction_vector)
    if norm > 0:
        direction_vector = direction_vector / norm

    info.update(
        {
            "should_process": True,
            "reason": "ok",
            "azimuth_deg": float(az_deg),
            "elevation_deg": float(el_deg),
            "confidence": float(confidence),
            "distance_m": float(distance_m) if distance_m is not None else None,
            "distance_confidence": float(distance_confidence) if distance_confidence is not None else None,
            "distance_uncertainty": float(distance_uncertainty) if distance_uncertainty is not None else None,
            "baseline_ratio": float(baseline_ratio) if baseline_ratio is not None else None,
            "per_mic_distances": per_mic_distances,
            "position_vector": position_vector,
            "detected": bool(detected_flag),
            "direction_vector": direction_vector,
        }
    )
    return info


def fuse_array_results(array_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Fuse localization outputs from multiple arrays into a single aggregate result.
    """
    if not array_results:
        return None

    best_result = max(array_results, key=lambda r: r.get("confidence", 0.0))

    vector_sum = np.zeros(3, dtype=np.float64)
    weight_sum = 0.0
    for res in array_results:
        vec = res.get("direction_vector")
        conf = max(res.get("confidence", 0.0), 0.0)
        if vec is not None and conf > 0:
            vector_sum += vec * conf
            weight_sum += conf

    if weight_sum <= 0:
        fused_vector = best_result.get("direction_vector", np.array([1.0, 0.0, 0.0]))
        weight_sum = best_result.get("confidence", 0.0)
        source = "best_only"
    else:
        fused_vector = vector_sum / weight_sum
        norm = np.linalg.norm(fused_vector)
        if norm > 0:
            fused_vector /= norm
        source = "weighted_average"

    azimuth_rad = np.arctan2(fused_vector[1], fused_vector[0])
    azimuth_deg = (np.degrees(azimuth_rad) + 360.0) % 360.0
    xy_norm = np.linalg.norm(fused_vector[:2])
    if xy_norm < 1e-10:
        elevation_deg = 90.0 if fused_vector[2] >= 0 else -90.0
    else:
        elevation_deg = np.degrees(np.arctan2(fused_vector[2], xy_norm))

    distance_values = [
        (res["distance_m"], max(res.get("confidence", 0.0), 1e-6))
        for res in array_results
        if res.get("distance_m") is not None
    ]
    if distance_values:
        total_weight = sum(w for _, w in distance_values)
        fused_distance = sum(d * w for d, w in distance_values) / total_weight
    else:
        fused_distance = None

    position_values = [
        (res["position_vector"], max(res.get("confidence", 0.0), 1e-6))
        for res in array_results
        if res.get("position_vector") is not None
    ]
    if position_values:
        total_weight = sum(w for _, w in position_values)
        fused_position = sum(vec * w for vec, w in position_values) / total_weight
    else:
        fused_position = None

    fused_confidence = min(1.0, max(weight_sum / len(array_results), best_result.get("confidence", 0.0)))

    return {
        "source": source,
        "azimuth_deg": float(azimuth_deg),
        "elevation_deg": float(elevation_deg),
        "confidence": float(fused_confidence),
        "distance_m": float(fused_distance) if fused_distance is not None else None,
        "position_vector": fused_position,
        "direction_vector": fused_vector,
        "reference_array": best_result,
    }


