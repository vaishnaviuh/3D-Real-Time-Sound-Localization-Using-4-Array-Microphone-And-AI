import os
import json
import numpy as np

from src.config import AppConfig, ArraySetup
from src.utils import countdown, ensure_dir, now_timestamp
from src.audio import record_multichannel
from src.doa import (
    build_mic_positions,
    apply_bandpass_filter,
)
from src.pipeline import process_signals, fuse_array_results
from src.plotting import plot_combined_2d_3d


def _vector_to_dict(vec: Optional[np.ndarray]) -> Optional[dict]:
    if vec is None:
        return None
    return {"x": float(vec[0]), "y": float(vec[1]), "z": float(vec[2])}


def _serialize_array_result(result: dict) -> dict:
    classification = result.get("classification")
    classification_json = None
    if classification:
        features = classification.get("features", {})
        features = {k: float(v) for k, v in features.items()}
        classification_json = {
            "predicted_class": classification.get("predicted_class"),
            "confidence": float(classification.get("confidence", 0.0)),
            "trigger_detected": bool(classification.get("trigger_detected", False)),
            "features": features,
        }

    return {
        "array_name": result.get("array_name"),
        "azimuth_deg": result.get("azimuth_deg"),
        "elevation_deg": result.get("elevation_deg"),
        "confidence": result.get("confidence"),
        "distance_m": result.get("distance_m"),
        "distance_confidence": result.get("distance_confidence"),
        "distance_uncertainty": result.get("distance_uncertainty"),
        "baseline_ratio": result.get("baseline_ratio"),
        "position": _vector_to_dict(result.get("position_vector")),
        "harmonic_match": result.get("harmonic_match"),
        "detected_fundamental_hz": result.get("detected_fundamental_hz"),
        "signal_active": result.get("signal_active"),
        "rms_energy": result.get("rms_energy"),
        "peak_to_mean": result.get("peak_to_mean"),
        "classification": classification_json,
        "detected": result.get("detected"),
    }


def _record_array_audio(cfg: AppConfig, array_cfg: ArraySetup) -> np.ndarray:
    return record_multichannel(
        samplerate=cfg.audio.samplerate,
        duration_s=cfg.audio.record_seconds,
        dtype=cfg.audio.dtype,
        channels_to_use=array_cfg.channels_to_use,
        device_query=array_cfg.device_query or cfg.audio.device_query,
        requested_channels=cfg.audio.requested_channels,
        blocksize=cfg.audio.blocksize,
        device_index=array_cfg.device_index,
    )


def run_once(cfg: AppConfig) -> None:
    arrays = cfg.resolved_arrays()
    if cfg.audio.countdown_seconds > 0:
        countdown(cfg.audio.countdown_seconds)

    array_geometries = [
        {
            "config": array_cfg,
            "mic_positions": build_mic_positions(
                radius_m=array_cfg.radius_m,
                angles_deg=array_cfg.mic_angles_deg,
                origin_xyz=array_cfg.origin_xyz,
            ),
        }
        for array_cfg in arrays
    ]

    mic_arrays_for_plot = [
        {"name": entry["config"].name, "color": entry["config"].color_hex, "positions": entry["mic_positions"]}
        for entry in array_geometries
    ]

    successful_results = []

    for entry in array_geometries:
        array_cfg: ArraySetup = entry["config"]
        mic_positions = entry["mic_positions"]
        print(f"[INFO] Capturing audio for {array_cfg.name}...")
        try:
            signals = _record_array_audio(cfg, array_cfg)
        except Exception as exc:
            print(f"[ERROR] Failed to record from {array_cfg.name}: {exc}")
            continue

        signals = apply_bandpass_filter(
            signals=signals,
            fs=cfg.audio.samplerate,
            low_freq=cfg.detection.min_freq_hz,
            high_freq=cfg.detection.max_freq_hz,
        )

        result = process_signals(
            signals=signals,
            cfg=cfg,
            mic_positions=mic_positions,
            enable_debug=cfg.detection.enable_debug,
        )

        if not result.get("should_process"):
            reason = result.get("reason", "unknown")
            print(f"[INFO] {array_cfg.name}: skipped ({reason})")
            continue

        result.update(
            {
                "array_name": array_cfg.name,
                "color": array_cfg.color_hex,
                "mic_positions": mic_positions,
                "signals": signals,
            }
        )
        successful_results.append(result)

    if not successful_results:
        print("No valid sound source detected across configured arrays.")
        return

    fused_result = fuse_array_results(successful_results) or {}
    reference_result = fused_result.get("reference_array", successful_results[0])

    fused_distance = fused_result.get("distance_m")
    if fused_distance is None:
        fused_distance = reference_result.get("distance_m")

    fused_position = fused_result.get("position_vector") or reference_result.get("position_vector")

    distance_print = f"{fused_distance:.2f} m" if fused_distance is not None else "unknown"
    fused_az = fused_result.get("azimuth_deg", reference_result.get("azimuth_deg", 0.0))
    fused_el = fused_result.get("elevation_deg", reference_result.get("elevation_deg", 0.0))
    print(
        f"FUSED → Azimuth: {fused_az:.1f}°, "
        f"Elevation: {fused_el:.1f}°, "
        f"Distance: {distance_print}"
    )
    print(f"Confidence: {fused_result.get('confidence', reference_result.get('confidence', 0.0)):.2f} "
          f"(source: {fused_result.get('source', 'single_array')})")

    for res in successful_results:
        dist_txt = f"{res['distance_m']:.2f} m" if res.get("distance_m") is not None else "unknown"
        print(
            f"  [{res['array_name']}] Az={res['azimuth_deg']:.1f}°, "
            f"El={res['elevation_deg']:.1f}°, Dist={dist_txt}, Conf={res['confidence']:.2f}"
        )

    if cfg.saving.enable_save_audio or cfg.saving.enable_save_results:
        ensure_dir(cfg.saving.output_dir)
        ts = now_timestamp()
        if cfg.saving.enable_save_audio:
            for res in successful_results:
                filename = f"audio_{res['array_name'].replace(' ', '_')}_{ts}.npy"
                np.save(os.path.join(cfg.saving.output_dir, filename), res["signals"])
        if cfg.saving.enable_save_results:
            payload = {
                "fused": {
                    "azimuth_deg": fused_result.get("azimuth_deg"),
                    "elevation_deg": fused_result.get("elevation_deg"),
                    "confidence": fused_result.get("confidence"),
                    "distance_m": fused_result.get("distance_m"),
                    "position": _vector_to_dict(fused_position),
                    "source": fused_result.get("source"),
                },
                "arrays": [_serialize_array_result(res) for res in successful_results],
            }
            with open(os.path.join(cfg.saving.output_dir, f"doa_{ts}.json"), "w") as f:
                json.dump(payload, f, indent=2)

    if cfg.plot.show_plots:
        plot_combined_2d_3d(
            azimuth_deg=fused_result.get("azimuth_deg", reference_result.get("azimuth_deg")),
            elevation_deg=fused_result.get("elevation_deg", reference_result.get("elevation_deg")),
            distance_m=fused_distance,
            mic_arrays=mic_arrays_for_plot,
        )


if __name__ == "__main__":
    config = AppConfig()
    run_once(config)


