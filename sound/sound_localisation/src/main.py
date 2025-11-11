import os
import json
import numpy as np

from src.config import AppConfig
from src.utils import countdown, ensure_dir, now_timestamp
from src.audio import record_multichannel
from src.doa import build_mic_positions, estimate_doa_az_el, rough_distance_from_energy, estimate_position_from_tdoa
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

    mic_positions = build_mic_positions(
        radius_m=cfg.geometry.radius_m, angles_deg=cfg.geometry.mic_angles_deg
    )
    az_deg, el_deg = estimate_doa_az_el(
        signals=signals,
        fs=cfg.audio.samplerate,
        mic_positions_m=mic_positions,
        speed_of_sound=cfg.geometry.speed_of_sound,
    )
    # Estimate distance via TDOA position solver; fallback to energy proxy if needed
    est_pos = estimate_position_from_tdoa(
        signals=signals,
        fs=cfg.audio.samplerate,
        mic_positions_m=mic_positions,
        speed_of_sound=cfg.geometry.speed_of_sound,
        ref_index=0,
    )
    if est_pos is not None:
        distance_m = float(np.linalg.norm(est_pos))
    else:
        distance_m = rough_distance_from_energy(signals) if cfg.plot.show_plots else None

    if distance_m is not None:
        print(f"Azimuth: {az_deg:.1f} deg, Elevation: {el_deg:.1f} deg, Distance~{distance_m:.2f} m")
    else:
        print(f"Azimuth: {az_deg:.1f} deg, Elevation: {el_deg:.1f} deg, Distance: unknown")

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


