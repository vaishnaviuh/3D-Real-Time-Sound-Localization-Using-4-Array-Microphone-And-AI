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

from src.config import AppConfig
from src.audio import record_multichannel
from src.doa import (
    build_mic_positions,
    estimate_doa_az_el,
    estimate_position_from_tdoa,
    apply_bandpass_filter,
    detect_harmonics,
    auto_detect_candidate_fundamentals,
    detect_signal_activity,
)
from src.classify import classify_audio_signal, compute_spectrogram_all_channels


app = FastAPI(title="Sound Localization Dashboard")

# Store active WebSocket connections
active_connections: Set[WebSocket] = set()

# Global config
config = AppConfig()
mic_positions = build_mic_positions(
    radius_m=config.geometry.radius_m,
    angles_deg=config.geometry.mic_angles_deg
)

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


def process_audio_chunk():
    """Process a single audio chunk (runs in thread pool)."""
    try:
        # Record audio chunk (blocking call)
        signals = record_multichannel(
            samplerate=config.audio.samplerate,
            duration_s=config.audio.record_seconds,
            dtype=config.audio.dtype,
            channels_to_use=config.audio.channels_to_use,
            device_query=config.audio.device_query,
            requested_channels=config.audio.requested_channels,
            blocksize=config.audio.blocksize,
        )
        
        # Apply frequency filtering (300-4000 Hz)
        signals = apply_bandpass_filter(
            signals=signals,
            fs=config.audio.samplerate,
            low_freq=config.detection.min_freq_hz,
            high_freq=config.detection.max_freq_hz,
        )
        
        # CLASSIFICATION: Detect trigger sounds
        trigger_detected = False  # Default to FALSE - only trigger if explicitly detected
        predicted_class = "unknown"
        classification_confidence = 0.0
        classification_features = {}
        rule_matches = {}
        
        if config.classification.enable_classification:
            predicted_class, classification_confidence, classification_features, rule_matches = classify_audio_signal(
                signals=signals,
                fs=config.audio.samplerate,
                config=config.classification,
                channel_idx=None,
            )
            
            # Check if this is a trigger sound
            trigger_classes = []
            if config.classification.drone_enabled:
                trigger_classes.append('drone')
            if config.classification.mechanical_enabled:
                trigger_classes.append('mechanical')
            if config.classification.clap_enabled:
                trigger_classes.append('clap')
            
            trigger_detected = (
                predicted_class in trigger_classes and
                classification_confidence >= config.classification.min_trigger_confidence
            )
            
            if not trigger_detected:
                if config.detection.enable_debug:
                    print(f"[DEBUG] Sound classified as '{predicted_class}' (conf: {classification_confidence:.2f}) - IGNORED")
                    print(f"[DEBUG] Features: rms={classification_features.get('rms_energy', 0):.6f}, "
                          f"centroid={classification_features.get('spectral_centroid', 0):.1f}Hz")
                return None  # CRITICAL: Return None to skip ALL processing including DOA
            else:
                print(f"✓ TRIGGER SOUND DETECTED → Localization Active: {predicted_class.upper()} (conf: {classification_confidence:.2f})")
                if config.detection.enable_debug:
                    print(f"[DEBUG] Trigger features: rms={classification_features.get('rms_energy', 0):.6f}, "
                          f"centroid={classification_features.get('spectral_centroid', 0):.1f}Hz, "
                          f"bandwidth={classification_features.get('spectral_bandwidth', 0):.1f}Hz")
        else:
            # Classification disabled - this should not happen in production
            if config.detection.enable_debug:
                print("[WARNING] Classification is DISABLED - all sounds will be localized!")
        
        # Compute spectrograms for dashboard
        spectrograms_data = None
        if config.classification.enable_spectrogram:
            try:
                spectrograms, times, freqs = compute_spectrogram_all_channels(
                    signals=signals,
                    fs=config.audio.samplerate,
                    n_fft=config.classification.spectrogram_n_fft,
                    hop_length=config.classification.spectrogram_hop_length,
                )
                # Convert to list format for JSON (take every Nth sample to reduce data size)
                downsample_factor = max(1, len(times) // 100)  # Limit to ~100 time points
                spectrograms_data = {
                    'times': times[::downsample_factor].tolist(),
                    'freqs': freqs.tolist(),
                    'magnitudes': [
                        spec[:, ::downsample_factor].tolist() for spec in spectrograms
                    ],
                }
            except Exception as e:
                if config.detection.enable_debug:
                    print(f"[DEBUG] Failed to compute spectrograms: {e}")
        
        # IMPORTANT: When classification is enabled, skip old harmonic detection
        # Only run harmonic detection if classification is disabled
        harmonics_detected = False
        detected_fundamental = None
        
        if not config.classification.enable_classification:
            # OLD DETECTION: Only run when classification is disabled
            target_fundamentals = list(config.detection.target_harmonic_fundamentals_hz)

            if target_fundamentals:
                harmonics_detected, detected_fundamental = detect_harmonics(
                    signals=signals,
                    fs=config.audio.samplerate,
                    target_fundamentals_hz=target_fundamentals,
                    min_harmonics=config.detection.min_harmonics_detected,
                    tolerance_hz=config.detection.harmonic_tolerance_hz,
                    min_magnitude_ratio=config.detection.harmonic_min_magnitude_ratio,
                )
            elif config.detection.auto_detect_harmonics:
                candidate_fundamentals = auto_detect_candidate_fundamentals(
                    signals=signals,
                    fs=config.audio.samplerate,
                    min_freq_hz=config.detection.min_freq_hz,
                    max_freq_hz=config.detection.max_freq_hz,
                    min_peak_ratio=config.detection.auto_detect_min_peak_ratio,
                    max_candidates=config.detection.auto_detect_max_candidates,
                )
                if config.detection.enable_debug:
                    print(f"[DEBUG] Auto-detected harmonic candidates: {[f'{f:.1f}' for f in candidate_fundamentals]}")
                for candidate in candidate_fundamentals:
                    harmonics_detected, detected_fundamental = detect_harmonics(
                        signals=signals,
                        fs=config.audio.samplerate,
                        target_fundamentals_hz=[candidate],
                        min_harmonics=config.detection.min_harmonics_detected,
                        tolerance_hz=config.detection.harmonic_tolerance_hz,
                        min_magnitude_ratio=config.detection.harmonic_min_magnitude_ratio,
                    )
                    if harmonics_detected:
                        break
        else:
            # Classification enabled - skip harmonic detection, already validated trigger
            harmonics_detected = True  # Set to True since classification passed
            detected_fundamental = None
            if config.detection.enable_debug:
                print("[DEBUG] Classification validated - skipping harmonic detection")

        # When classification is enabled, we ONLY use classification - skip old detection logic
        if not config.classification.enable_classification:
            # OLD DETECTION LOGIC (only used when classification is disabled)
            signal_active, activity_info = detect_signal_activity(
                signals=signals,
                min_rms=config.detection.min_activity_rms,
                min_peak_to_mean=config.detection.min_activity_peak_to_mean,
            )
            if config.detection.enable_debug:
                print(
                    "[DEBUG] Signal activity check: "
                    f"rms={activity_info['rms_energy']:.6f} "
                    f"(threshold {activity_info['min_rms_threshold']:.6f}), "
                    f"peak_to_mean={activity_info['peak_to_mean']:.2f} "
                    f"(threshold {activity_info['min_peak_to_mean_threshold']:.2f})"
                )

            if not harmonics_detected:
                if config.detection.enable_debug:
                    print(f"[DEBUG] No harmonic match (targets: {config.detection.target_harmonic_fundamentals_hz})")
                if not signal_active:
                    if config.detection.enable_debug:
                        print("[DEBUG] No harmonic match and insufficient broadband activity. Skipping.")
                    return None
                if config.detection.require_harmonic_match:
                    if config.detection.enable_debug:
                        print("[DEBUG] require_harmonic_match=True => skipping.")
                    return None
                if config.detection.enable_debug:
                    print("[DEBUG] Proceeding with DOA estimation using broadband activity trigger.")
            elif config.detection.enable_debug and detected_fundamental is not None:
                print(f"[DEBUG] Detected harmonic series with fundamental: {detected_fundamental:.1f} Hz")
        else:
            # Classification is enabled - create activity_info for compatibility
            rms_energy = np.sqrt(np.mean(signals ** 2))
            peak_value = np.max(np.abs(signals))
            mean_value = np.mean(np.abs(signals))
            peak_to_mean = peak_value / max(mean_value, 1e-12)
            activity_info = {
                'rms_energy': float(rms_energy),
                'peak_to_mean': float(peak_to_mean),
            }
            signal_active = True  # Already validated by classification
            if config.detection.enable_debug:
                print("[DEBUG] Classification passed - proceeding with DOA estimation.")
        
        # Estimate DOA with confidence
        az_deg, el_deg, confidence = estimate_doa_az_el(
            signals=signals,
            fs=config.audio.samplerate,
            mic_positions_m=mic_positions,
            speed_of_sound=config.geometry.speed_of_sound,
            min_correlation_quality=config.detection.min_correlation_quality,
            min_rms_energy=config.detection.min_rms_energy,
            enable_debug=config.detection.enable_debug,
        )
        
        # Check if DOA calculation actually succeeded
        # If azimuth and elevation are both 0 with confidence 0, the solve failed
        doa_failed = (az_deg == 0.0 and el_deg == 0.0 and confidence == 0.0)
        
        if doa_failed:
            if config.detection.enable_debug:
                print(f"[WARNING] DOA calculation failed - angles are (0, 0) with confidence 0")
                print(f"[WARNING] This usually means insufficient microphone correlation or weak signals")
                print(f"[WARNING] Skipping this update - check microphone connections and signal levels")
            return None  # Don't broadcast failed DOA results
        
        # Always return DOA results, but mark detection status based on confidence
        # This allows dashboard to show markers even for low-confidence detections
        detected = confidence >= config.detection.min_confidence_threshold or config.detection.always_broadcast
        
        if not detected and config.detection.enable_debug:
            print(f"[DEBUG] Confidence {confidence:.3f} below threshold {config.detection.min_confidence_threshold:.3f}, but always_broadcast=True")
        
        # Estimate 3D position and distance
        position_3d = estimate_position_from_tdoa(
            signals=signals,
            fs=config.audio.samplerate,
            mic_positions_m=mic_positions,
            speed_of_sound=config.geometry.speed_of_sound,
            min_correlation_quality=config.detection.min_correlation_quality,
            enable_debug=config.detection.enable_debug,
        )
        
        if position_3d is not None:
            distance_m = np.linalg.norm(position_3d)
            x, y, z = position_3d
            
            # Calculate theoretical accuracy limits
            max_baseline = 2 * config.geometry.radius_m  # Array diameter
            baseline_ratio = distance_m / max_baseline if max_baseline > 0 else 0
            
            # Distance accuracy confidence (degrades with distance)
            if baseline_ratio > 10:
                distance_confidence = max(0.1, 1.0 - (baseline_ratio - 10) * 0.1)
            elif baseline_ratio > 5:
                distance_confidence = 0.5 + 0.5 * (10 - baseline_ratio) / 5
            else:
                distance_confidence = 1.0
            
            # Estimate distance uncertainty
            if distance_m > max_baseline:
                tdoa_uncertainty_s = 3.9e-6  # ~3.9 microseconds at 16kHz with 16x interpolation
                distance_uncertainty = (distance_m ** 2) / (max_baseline * config.geometry.speed_of_sound) * tdoa_uncertainty_s
                distance_uncertainty = max(distance_uncertainty, 0.01)
            else:
                distance_uncertainty = 0.01
            
            # Calculate per-microphone distances for debugging
            per_mic_distances = [
                float(np.linalg.norm(position_3d - mic_positions[i]))
                for i in range(4)
            ]
            if config.detection.enable_debug:
                print(f"[DEBUG] Per-mic distances: {[f'{d:.3f}' for d in per_mic_distances]} m")
                print(f"[DEBUG] Distance: {distance_m*100:.1f} cm, Baseline ratio: {baseline_ratio:.1f}x, Uncertainty: ±{distance_uncertainty*100:.1f} cm")
                if baseline_ratio > 10:
                    print(f"[WARNING] Distance >10x baseline ({baseline_ratio:.1f}x) - POOR accuracy expected")
                elif baseline_ratio > 5:
                    print(f"[WARNING] Distance >5x baseline ({baseline_ratio:.1f}x) - DEGRADED accuracy")
                # Specifically check mic 3 (index 2)
                if len(per_mic_distances) > 2:
                    mic3_dist = per_mic_distances[2]
                    mean_dist = np.mean(per_mic_distances)
                    if abs(mic3_dist - mean_dist) > 0.1:  # More than 10cm difference
                        print(f"[WARNING] Mic 3 distance ({mic3_dist:.3f} m) differs from mean ({mean_dist:.3f} m) by {abs(mic3_dist - mean_dist)*100:.1f} cm")
        else:
            distance_m = None
            x, y, z = None, None, None
            per_mic_distances = None
            distance_confidence = None
            distance_uncertainty = None
            baseline_ratio = None
        
        # Always return DOA values (azimuth/elevation are always computed)
        # The 'detected' flag indicates if confidence is above threshold
        result = {
            "azimuth_deg": float(az_deg) if not np.isnan(az_deg) else 0.0,
            "elevation_deg": float(el_deg) if not np.isnan(el_deg) else 0.0,
            "distance_m": float(distance_m) if distance_m is not None and not np.isnan(distance_m) else None,
            "distance_confidence": float(distance_confidence) if distance_confidence is not None else None,
            "distance_uncertainty_cm": float(distance_uncertainty * 100) if distance_uncertainty is not None else None,
            "baseline_ratio": float(baseline_ratio) if baseline_ratio is not None else None,
            "confidence": float(confidence),
            "position_3d": {
                "x": float(x) if x is not None and not np.isnan(x) else None,
                "y": float(y) if y is not None and not np.isnan(y) else None,
                "z": float(z) if z is not None and not np.isnan(z) else None,
            },
            "detected": detected,
            "harmonic_match": harmonics_detected,
            "detected_fundamental_hz": float(detected_fundamental) if detected_fundamental is not None else None,
            "signal_active": signal_active,
            "rms_energy": activity_info["rms_energy"],
            "peak_to_mean": activity_info["peak_to_mean"],
            "per_mic_distances": per_mic_distances if position_3d is not None else None,
        }
        
        # Add classification results
        if config.classification.enable_classification:
            result["classification"] = {
                "predicted_class": predicted_class,
                "confidence": float(classification_confidence),
                "trigger_detected": trigger_detected,
                "features": {k: float(v) for k, v in classification_features.items()},
            }
        
        # Add spectrograms
        if spectrograms_data is not None:
            result["spectrograms"] = spectrograms_data
        
        return result
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
                # Prepare update data
                update_data = {
                    "type": "doa_update",
                    **result,
                    "mic_positions": [
                        {"x": float(mic_positions[i, 0]), "y": float(mic_positions[i, 1]), "z": float(mic_positions[i, 2])}
                        for i in range(4)
                    ],
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
        "mic_positions": [
            {"x": float(mic_positions[i, 0]), "y": float(mic_positions[i, 1]), "z": float(mic_positions[i, 2])}
            for i in range(4)
        ],
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

