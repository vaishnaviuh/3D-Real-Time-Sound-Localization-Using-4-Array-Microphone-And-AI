"""
FastAPI server for live sound localization dashboard.
Provides WebSocket endpoint for real-time DOA updates.
"""
import asyncio
import json
import numpy as np
from typing import Set, Optional
from concurrent.futures import ThreadPoolExecutor
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


app = FastAPI(title="Sound Localization Dashboard")

# Store active WebSocket connections
active_connections: Set[WebSocket] = set()

# Global config
config = AppConfig()
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
mic_arrays_payload = [
    {
        "name": definition["config"].name,
        "color": definition["config"].color_hex,
        "mic_positions": [
            {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
            for pos in definition["mic_positions"]
        ],
    }
    for definition in array_definitions
]

# Processing state
processing_active = False
processing_task: Optional[asyncio.Task] = None
executor = ThreadPoolExecutor(max_workers=2)


async def broadcast_update(data: dict):
    """Broadcast DOA update to all connected clients."""
    message = json.dumps(data)
    disconnected = set()
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except Exception:
            disconnected.add(connection)
    active_connections.difference_update(disconnected)


def _serialize_position(vec: Optional[np.ndarray]) -> Optional[dict]:
    if vec is None:
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


def _record_array_audio(array_cfg: ArraySetup) -> np.ndarray:
    return record_multichannel(
        samplerate=config.audio.samplerate,
        duration_s=config.audio.record_seconds,
        dtype=config.audio.dtype,
        channels_to_use=array_cfg.channels_to_use,
        device_query=array_cfg.device_query or config.audio.device_query,
        requested_channels=config.audio.requested_channels,
        blocksize=config.audio.blocksize,
        device_index=array_cfg.device_index,
    )


def process_audio_chunk():
    """Process a single audio chunk (runs in thread pool)."""
    try:
        successful_results = []
        best_result = None
        spectrograms_data = None

        for definition in array_definitions:
            array_cfg: ArraySetup = definition["config"]
            mic_positions = definition["mic_positions"]
            try:
                signals = _record_array_audio(array_cfg)
            except Exception as exc:
                print(f"[ERROR] Failed to record from {array_cfg.name}: {exc}")
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

            if best_result is None or result["confidence"] > best_result["confidence"]:
                best_result = result

        if not successful_results:
            return None

        fused_result = fuse_array_results(successful_results)
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
                spectrograms_data = {
                    "times": times[::downsample_factor].tolist(),
                    "freqs": freqs.tolist(),
                    "magnitudes": [spec[:, ::downsample_factor].tolist() for spec in spectrograms],
                }
            except Exception as e:
                if config.detection.enable_debug:
                    print(f"[DEBUG] Failed to compute spectrograms: {e}")
        
        distance_m = fused_result.get("distance_m")
        if distance_m is None:
            distance_m = reference_result.get("distance_m")

        fused_position = fused_result.get("position_vector")
        if fused_position is None:
            position_vector = reference_result.get("position_vector")
        else:
            position_vector = fused_position

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
                        for pos in res["mic_positions"]
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
            "position_3d": _serialize_position(position_vector),
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
        
        return result_payload
    except Exception as e:
        print(f"Error processing audio: {e}")
        import traceback
        traceback.print_exc()
        return None


async def continuous_audio_processing():
    """Continuously process audio and broadcast DOA updates."""
    global processing_active
    
    print("Starting continuous audio processing...")
    processing_active = True
    
    try:
        while processing_active:
            # Run blocking audio processing in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(executor, process_audio_chunk)
            
            if result is None:
                await asyncio.sleep(0.5)
                continue
            
            # Always broadcast DOA updates (let dashboard filter by confidence)
            # This ensures markers are always shown when there's any signal
            if result and (result.get("detected", False) or config.detection.always_broadcast):
                update_data = {
                    "type": "doa_update",
                    **result,
                    "timestamp": asyncio.get_event_loop().time(),
                }
                
                # Broadcast to all connected clients
                await broadcast_update(update_data)
                if config.detection.enable_debug:
                    print(f"[DEBUG] Broadcasted DOA update: az={result.get('azimuth_deg')}, conf={result.get('confidence', 0):.3f}")
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.1)
            
    except asyncio.CancelledError:
        print("Audio processing cancelled.")
    except Exception as e:
        print(f"Error in audio processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        processing_active = False
        print("Audio processing stopped.")


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
    active_connections.add(websocket)
    print(f"Client connected. Total connections: {len(active_connections)}")
    
    # Auto-start processing if not already running
    global processing_task, processing_active
    if not processing_active and processing_task is None:
        processing_task = asyncio.create_task(continuous_audio_processing())
    
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
                if message.get("type") == "start_processing" and not processing_active:
                    processing_task = asyncio.create_task(continuous_audio_processing())
                elif message.get("type") == "stop_processing":
                    processing_active = False
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        active_connections.discard(websocket)
        print(f"Client disconnected. Total connections: {len(active_connections)}")
        # Stop processing if no clients connected
        if len(active_connections) == 0:
            processing_active = False


@app.on_event("startup")
async def startup_event():
    """Initialize on server startup."""
    print("Sound Localization Dashboard server starting...")
    print("Open http://localhost:8000 in your browser")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop audio processing on server shutdown."""
    global processing_active
    processing_active = False
    if processing_task:
        processing_task.cancel()
        try:
            await processing_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

