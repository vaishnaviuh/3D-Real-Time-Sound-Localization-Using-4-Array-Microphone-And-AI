"""
FastAPI server for live sound localization dashboard.
Provides WebSocket endpoint for real-time DOA updates.
"""
import asyncio
import json
import numpy as np
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from src.config import AppConfig, ArraySetup
from src.audio import record_multichannel
from src.doa import (
    build_mic_positions,
    apply_bandpass_filter,
)
from src.classify import compute_spectrogram_all_channels
from src.pipeline import process_signals, fuse_array_results
from src.state import StateManager
from src.utils import (
    get_logger,
    validate_signals,
    validate_mic_positions,
    safe_validate,
    retry_with_backoff
)
import math

# Initialize logger and state manager
logger = get_logger("sound_localization.server")
state_manager = StateManager()


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    logger.info("Sound Localization Dashboard server starting...")
    logger.info("Open http://localhost:8000 in your browser")
    try:
        yield
    finally:
        logger.info("Shutting down server...")
        state_manager.processing.active = False
        if state_manager.processing.task:
            state_manager.processing.task.cancel()
            try:
                await state_manager.processing.task
            except asyncio.CancelledError:
                pass
        state_manager.cleanup_all()
        logger.info("Server shutdown complete.")


app = FastAPI(title="Sound Localization Dashboard", lifespan=lifespan)

# Global config and validation
config = AppConfig()
try:
    from src.utils.validators import validate_config
    validate_config(config)
except Exception as exc:
    logger.error(f"Configuration validation failed: {exc}")
    raise

# Array definitions and validation
array_definitions = [
    {
        "config": array_cfg,
        "mic_positions": build_mic_positions(
            radius_m=array_cfg.radius_m,
            angles_deg=array_cfg.mic_angles_deg,
            origin_xyz=array_cfg.origin_xyz,
        ),
    }
    for array_cfg in config.resolved_arrays()
]

for definition in array_definitions:
    is_valid, error = safe_validate(
        validate_mic_positions,
        definition["mic_positions"],
        expected_count=4,
    )
    if not is_valid:
        logger.warning(
            f"Invalid mic positions for {definition['config'].name}: {error}",
            event_type="mic_validation_warning",
        )

mic_arrays_payload = [
    {
        "name": definition["config"].name,
        "color": definition["config"].color_hex,
        "mic_positions": [
            {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
            for pos in definition["mic_positions"]
            if len(pos) >= 3
        ],
    }
    for definition in array_definitions
]


def _unit_vector(vec: np.ndarray) -> np.ndarray:
    vec = np.array(vec, dtype=np.float64)
    if len(vec) == 0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return vec
    return vec / norm


def _vector_to_angles(vec: np.ndarray) -> tuple[float, float]:
    vec = np.array(vec, dtype=np.float64)
    if len(vec) < 3:
        return 0.0, 0.0
    azimuth = (math.degrees(math.atan2(vec[1], vec[0])) + 360.0) % 360.0
    xy_norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
    elevation = math.degrees(math.atan2(vec[2], xy_norm))
    return float(azimuth), float(elevation)


def _smooth_direction_vector(new_vec: np.ndarray, confidence: float) -> tuple[np.ndarray, float]:
    """Smooth direction vector using state manager."""
    new_unit = _unit_vector(new_vec)

    alpha = np.clip(config.detection.smoothing_alpha, 0.0, 1.0)
    if state_manager.processing.smoothed_direction is None or alpha <= 0.0:
        state_manager.processing.smoothed_direction = new_unit
        state_manager.processing.smoothed_confidence = max(
            confidence, config.detection.min_fused_confidence
        )
        return (
            state_manager.processing.smoothed_direction,
            state_manager.processing.smoothed_confidence,
        )

    prev_unit = state_manager.processing.smoothed_direction
    prev_az, _ = _vector_to_angles(prev_unit)
    new_az, _ = _vector_to_angles(new_unit)
    az_diff = abs(new_az - prev_az)
    az_diff = min(az_diff, 360.0 - az_diff)

    if (
        az_diff > config.detection.max_azimuth_jump_deg
        and confidence < config.detection.min_confidence_for_jump
    ):
        return prev_unit, state_manager.processing.smoothed_confidence

    blended = _unit_vector(alpha * new_unit + (1 - alpha) * prev_unit)
    state_manager.processing.smoothed_direction = blended
    state_manager.processing.smoothed_confidence = max(
        confidence,
        state_manager.processing.smoothed_confidence * 0.8,
        config.detection.min_fused_confidence,
    )
    return (
        state_manager.processing.smoothed_direction,
        state_manager.processing.smoothed_confidence,
    )


def _smooth_position_vector(new_position: Optional[np.ndarray], confidence: float) -> Optional[np.ndarray]:
    """Smooth position vector using state manager."""
    if new_position is None:
        return state_manager.processing.smoothed_position

    smoothing = np.clip(config.detection.smoothing_alpha, 0.0, 1.0)
    new_pos = np.array(new_position, dtype=np.float64)
    if state_manager.processing.smoothed_position is None or smoothing <= 0.0:
        state_manager.processing.smoothed_position = new_pos
        return state_manager.processing.smoothed_position

    state_manager.processing.smoothed_position = (
        smoothing * new_pos + (1 - smoothing) * state_manager.processing.smoothed_position
    )
    return state_manager.processing.smoothed_position



async def broadcast_update(data: dict):
    """Broadcast DOA update to all connected clients."""
    message = json.dumps(data)
    disconnected = set()
    for connection in state_manager.connections.active_connections:
        try:
            await connection.send_text(message)
        except Exception as exc:
            logger.warning(f"Failed to send message to client: {exc}")
            disconnected.add(connection)
    for conn in disconnected:
        state_manager.connections.remove_connection(conn)


def _serialize_position(vec: Optional[np.ndarray]) -> Optional[dict]:
    if vec is None:
        return None
    if len(vec) < 3:
        return None
    return {"x": float(vec[0]), "y": float(vec[1]), "z": float(vec[2])}


def _serialize_classification(classification: Optional[dict]) -> Optional[dict]:
    if not classification:
        return None
    features = classification.get("features", {})
    features = {k: float(v) for k, v in features.items()}
    return {
        "predicted_class": classification.get("predicted_class"),
        "confidence": float(classification.get("confidence", 0.0)),
        "trigger_detected": bool(classification.get("trigger_detected", False)),
        "features": features,
    }


@retry_with_backoff(max_retries=2, initial_delay=0.5)
def _record_array_audio(array_cfg: ArraySetup, mic_positions: np.ndarray) -> np.ndarray:
    signals = record_multichannel(
        samplerate=config.audio.samplerate,
        duration_s=config.audio.record_seconds,
        dtype=config.audio.dtype,
        channels_to_use=array_cfg.channels_to_use,
        device_query=array_cfg.device_query or config.audio.device_query,
        requested_channels=config.audio.requested_channels,
        blocksize=config.audio.blocksize,
        device_index=array_cfg.device_index,
        use_simulation=config.simulation.enable_simulation,
        sim_config=config.simulation,
        mic_positions=mic_positions,
        speed_of_sound=config.geometry.speed_of_sound,
    )

    is_valid, error = safe_validate(validate_signals, signals, expected_channels=4)
    if not is_valid:
        raise ValueError(f"Signal validation failed for {array_cfg.name}: {error}")

    return signals


def process_audio_chunk():
    """Process a single audio chunk (runs in thread pool)."""
    try:
        successful_results = []
        best_result_overall = None
        spectrograms_data = None

        for definition in array_definitions:
            array_cfg: ArraySetup = definition["config"]
            mic_positions = definition["mic_positions"]
            try:
                signals = _record_array_audio(array_cfg, mic_positions)
                logger.log_audio_event("recorded", array_name=array_cfg.name)
            except Exception as exc:
                logger.error(
                    f"Failed to record from {array_cfg.name}: {exc}",
                    event_type="audio_recording_error",
                    array_name=array_cfg.name,
                )
                continue

            signals = apply_bandpass_filter(
                signals=signals,
                fs=config.audio.samplerate,
                low_freq=config.detection.min_freq_hz,
                high_freq=config.detection.max_freq_hz,
            )
            
            result = process_signals(
                signals=signals,
                cfg=config,
                mic_positions=mic_positions,
                enable_debug=config.detection.enable_debug,
            )

            if not result.get("should_process"):
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

            if best_result_overall is None or result["confidence"] > best_result_overall["confidence"]:
                best_result_overall = result

        if not successful_results:
            return None

        min_array_conf = config.detection.min_array_confidence
        filtered_results = [
            res for res in successful_results if res.get("confidence", 0.0) >= min_array_conf
        ]
        fusion_candidates = filtered_results if filtered_results else successful_results

        best_result = (
            max(fusion_candidates, key=lambda r: r.get("confidence", 0.0))
            if fusion_candidates
            else best_result_overall
        )

        fused_result = fuse_array_results(fusion_candidates)
        if fused_result is None:
            fused_result = {}
        reference_result = fused_result.get("reference_array")
        if reference_result is None:
            reference_result = best_result or successful_results[0]

        # Spectrograms from the best-performing array
        if config.classification.enable_spectrogram and best_result is not None:
            try:
                spectrograms, times, freqs = compute_spectrogram_all_channels(
                    signals=best_result["signals"],
                    fs=config.audio.samplerate,
                    n_fft=config.classification.spectrogram_n_fft,
                    hop_length=config.classification.spectrogram_hop_length,
                )
                downsample_factor = max(1, len(times) // 100)
                # spectrograms is list of arrays with shape (freq_bins, time_frames)
                # After downsampling: (freq_bins, downsampled_time_frames)
                spectrograms_data = {
                    "times": times[::downsample_factor].tolist(),
                    "freqs": freqs.tolist(),
                    "magnitudes": [spec[:, ::downsample_factor].tolist() for spec in spectrograms],
                }
                if config.detection.enable_debug:
                    logger.debug(
                        f"Computed spectrograms: {len(spectrograms)} channels, "
                        f"{len(spectrograms_data['times'])} time frames, "
                        f"{len(spectrograms_data['freqs'])} freq bins",
                        event_type="spectrogram_computed"
                    )
            except Exception as e:
                logger.warning(f"Failed to compute spectrograms: {e}", event_type="spectrogram_error")
                if config.detection.enable_debug:
                    logger.exception("Spectrogram computation error")
        
        distance_m = fused_result.get("distance_m")
        if distance_m is None:
            distance_m = reference_result.get("distance_m")

        fused_position = fused_result.get("position_vector")
        if fused_position is None:
            position_vector = reference_result.get("position_vector")
        else:
            position_vector = fused_position

        fused_direction = fused_result.get("direction_vector")
        if fused_direction is not None:
            smoothed_vec, smoothed_conf = _smooth_direction_vector(
                fused_direction, fused_result.get("confidence", 0.0)
            )
            fused_result["direction_vector"] = smoothed_vec
            az_deg, el_deg = _vector_to_angles(smoothed_vec)
            fused_result["azimuth_deg"] = az_deg
            fused_result["elevation_deg"] = el_deg
            fused_result["confidence"] = max(fused_result.get("confidence", 0.0), smoothed_conf)

        smoothed_position = _smooth_position_vector(position_vector, fused_result.get("confidence", 0.0))
        if smoothed_position is not None:
            fused_result["position_vector"] = smoothed_position
            position_3d_payload = _serialize_position(smoothed_position)
        else:
            position_3d_payload = _serialize_position(position_vector)

        array_payloads = []
        for res in successful_results:
            classification_json = _serialize_classification(res.get("classification"))
            res["_classification_json"] = classification_json
            array_payloads.append(
                {
                    "name": res["array_name"],
                    "color": res["color"],
                    "azimuth_deg": res.get("azimuth_deg"),
                    "elevation_deg": res.get("elevation_deg"),
                    "confidence": res.get("confidence"),
                    "distance_m": res.get("distance_m"),
                    "position_3d": _serialize_position(res.get("position_vector")),
                    "detected": res.get("detected"),
                    "harmonic_match": res.get("harmonic_match"),
                    "detected_fundamental_hz": res.get("detected_fundamental_hz"),
                    "signal_active": res.get("signal_active"),
                    "rms_energy": res.get("rms_energy"),
                    "peak_to_mean": res.get("peak_to_mean"),
                    "mic_positions": [
                        {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
                        for pos in res.get("mic_positions", [])
                        if len(pos) >= 3
                    ],
                    "classification": classification_json,
                }
            )

        detected_flag = any(res.get("detected") for res in successful_results)

        result_payload = {
            "azimuth_deg": fused_result.get("azimuth_deg", reference_result.get("azimuth_deg")),
            "elevation_deg": fused_result.get("elevation_deg", reference_result.get("elevation_deg")),
            "distance_m": distance_m,
            "confidence": fused_result.get("confidence", reference_result.get("confidence")),
            "position_3d": position_3d_payload,
            "detected": detected_flag,
            "harmonic_match": reference_result.get("harmonic_match"),
            "detected_fundamental_hz": reference_result.get("detected_fundamental_hz"),
            "signal_active": reference_result.get("signal_active"),
            "rms_energy": reference_result.get("rms_energy"),
            "peak_to_mean": reference_result.get("peak_to_mean"),
            "distance_confidence": reference_result.get("distance_confidence"),
            "distance_uncertainty_cm": (
                reference_result.get("distance_uncertainty") * 100.0 if reference_result.get("distance_uncertainty") else None
            ),
            "baseline_ratio": reference_result.get("baseline_ratio"),
            "per_mic_distances": reference_result.get("per_mic_distances"),
            "arrays": array_payloads,
            "mic_arrays": mic_arrays_payload,
        }

        classification_json = reference_result.get("_classification_json")
        if classification_json:
            result_payload["classification"] = classification_json

        if spectrograms_data is not None:
            result_payload["spectrograms"] = spectrograms_data

        # Override display with simulated ground truth if available
        if config.simulation.enable_simulation:
            sim_pos = np.array(config.simulation.last_position_xyz, dtype=np.float64)
            if np.linalg.norm(sim_pos) > 1e-6:
                sim_dist = float(np.linalg.norm(sim_pos))
                sim_az = (np.degrees(np.arctan2(sim_pos[1], sim_pos[0])) + 360.0) % 360.0
                sim_el = float(np.degrees(np.arctan2(sim_pos[2], np.linalg.norm(sim_pos[:2]))))
                result_payload["azimuth_deg"] = sim_az
                result_payload["elevation_deg"] = sim_el
                result_payload["distance_m"] = sim_dist
                result_payload["world_position"] = {
                    "x": float(sim_pos[0]),
                    "y": float(sim_pos[1]),
                    "z": float(sim_pos[2]),
                }
                result_payload["position_3d"] = result_payload["world_position"]
        
        return result_payload
    except Exception as e:
        logger.exception("Error processing audio chunk")
        return None


async def continuous_audio_processing():
    """Continuously process audio and broadcast DOA updates."""
    logger.info("Starting continuous audio processing...")
    state_manager.processing.active = True
    try:
        while state_manager.processing.active:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                state_manager.processing.executor, process_audio_chunk
            )

            if result is None:
                await asyncio.sleep(0.5)
                continue

            if result and (result.get("detected", False) or config.detection.always_broadcast):
                update_data = {
                    "type": "doa_update",
                    **result,
                    "timestamp": asyncio.get_event_loop().time(),
                }
                await broadcast_update(update_data)
                if config.detection.enable_debug:
                    logger.log_doa_result(
                        azimuth=result.get("azimuth_deg", 0.0),
                        elevation=result.get("elevation_deg", 0.0),
                        confidence=result.get("confidence", 0.0),
                    )

            await asyncio.sleep(0.1)

    except asyncio.CancelledError:
        logger.info("Audio processing cancelled.")
    except Exception:
        logger.exception("Error in audio processing loop")
    finally:
        state_manager.processing.active = False
        state_manager.processing.task = None
        logger.info("Audio processing stopped.")


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the dashboard HTML page."""
    import os
    html_path = os.path.join(os.path.dirname(__file__), "static", "dashboard.html")
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time DOA updates."""
    await websocket.accept()
    state_manager.connections.add_connection(websocket)
    logger.info(
        f"Client connected. Total connections: {state_manager.connections.get_count()}"
    )
    
    # Auto-start processing if not already running
    if not state_manager.processing.active and state_manager.processing.task is None:
        state_manager.processing.task = asyncio.create_task(continuous_audio_processing())
    
    # Send initial mic positions
    await websocket.send_json({
        "type": "init",
        "mic_arrays": mic_arrays_payload,
    })
    
    try:
        while True:
            # Wait for client messages (could be used for control commands)
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "start_processing" and not state_manager.processing.active:
                    state_manager.processing.task = asyncio.create_task(continuous_audio_processing())
                elif message.get("type") == "stop_processing":
                    state_manager.processing.active = False
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        state_manager.connections.remove_connection(websocket)
        logger.info(
            f"Client disconnected. Total connections: {state_manager.connections.get_count()}"
        )
        # Stop processing if no clients connected
        if not state_manager.connections.has_connections():
            state_manager.processing.active = False




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

