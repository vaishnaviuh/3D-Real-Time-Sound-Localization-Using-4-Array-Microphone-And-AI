import asyncio
import contextlib
import json
from typing import Set, cast, Tuple
from collections import deque

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from audio_capture import AudioCapture
from tdoa import TDOAEstimator
from angles import AngleEstimator
from triangulator import TriangulationLocalizer
from config import (SAMPLE_RATE, FREQ_MIN, FREQ_MAX, MIC_POSITIONS,
                    USE_FILTER, SOUND_SPEED)

from scipy.signal import butter, filtfilt


app = FastAPI(title="Sound Localization Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend (public/ directory)
app.mount("/static", StaticFiles(directory="public"), name="static")

@app.get("/favicon.ico")
async def favicon():
    return RedirectResponse(url="/static/favicon.ico", status_code=404)


@app.get("/")
async def root():
    """Redirect root to dashboard"""
    return RedirectResponse(url="/static/index.html")


class LocalizationStreamer:
    def __init__(self):
        self.audio = AudioCapture()
        self.tdoa = TDOAEstimator()
        self.angles = AngleEstimator(MIC_POSITIONS)
        self.sound_speed = float(SOUND_SPEED)
        self.tri = TriangulationLocalizer(MIC_POSITIONS, sound_speed=self.sound_speed)

        nyquist = SAMPLE_RATE / 2
        low = max(0.01, FREQ_MIN / nyquist)
        high = min(0.99, FREQ_MAX / nyquist)
        if high <= low:
            low = 0.02
            high = 0.95
        self.b, self.a = cast(Tuple[np.ndarray, np.ndarray], butter(4, [low, high], btype="band"))

        self.clients: Set[WebSocket] = set()
        self.running = False
        self.task: asyncio.Task | None = None

        self.smoothed_position = None
        self.smoothed_azimuth = 0.0
        self.smoothed_elevation = 0.0
        # Increased history for stability (reduces jitter)
        self.pos_hist: deque[np.ndarray] = deque(maxlen=5)  # 5 frames for stability
        self.tdoa_hist: deque[np.ndarray] = deque(maxlen=5)  # 5 frames for stability

        # No calibration - use channels as-is

    async def start(self):
        if self.running:
            print("Tracking already running - continuing...")
            return
        print("Starting continuous tracking...")
        self.audio.start_capture()
        self.running = True
        self.task = asyncio.create_task(self._loop())
        print("Tracking loop started - will run continuously until stopped")

    async def stop(self):
        self.running = False
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
            self.task = None
        self.audio.stop_capture()

    def set_temperature(self, temp_c: float | None):
        if temp_c is None:
            return
        try:
            t = float(temp_c)
            self.sound_speed = 331.0 + 0.6 * t
            self.tri = TriangulationLocalizer(MIC_POSITIONS, sound_speed=self.sound_speed)
        except Exception:
            pass

    def _process(self, audio_chunk: np.ndarray, frame_count: int = 0):
        signals = audio_chunk.T
        # Use channels directly - no calibration
        
        # Debug: print number of channels (only occasionally to avoid spam)
        if frame_count % 100 == 0:
            print(f"Number of channels: {len(signals)}")

        # Calculate signal energy BEFORE filtering (raw signal)
        raw_energies = [np.mean(sig**2) for sig in signals]
        raw_avg_energy = np.mean(raw_energies)
        raw_max_energy = np.max(raw_energies)
        raw_channel_energies = [float(e) for e in raw_energies]

        if USE_FILTER:
            signals = np.array([filtfilt(self.b, self.a, s) for s in signals])

        # Calculate signal energy AFTER filtering
        energies = [np.mean(sig**2) for sig in signals]
        avg_energy = np.mean(energies)
        max_energy = np.max(energies)
        channel_energies = [float(e) for e in energies]
        
        # Check if we have sufficient signal (higher threshold for drone detection)
        from config import ENERGY_THRESHOLD, MIN_SNR_DB
        
        # Calculate SNR
        min_energy = np.min(raw_energies)
        if min_energy > 1e-12:
            snr_linear = raw_avg_energy / min_energy
            snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else 0
        else:
            snr_db = 0
        
        # Debug: print signal levels every 50 frames to help diagnose
        if frame_count % 50 == 0:
            print(f"Signal check - Energy: {raw_avg_energy:.8f} (threshold: {ENERGY_THRESHOLD:.8f}), SNR: {snr_db:.1f}dB (min: {MIN_SNR_DB}dB), Max: {raw_max_energy:.8f}")
        
        # Reject if energy too low
        if raw_avg_energy < ENERGY_THRESHOLD:
            # Only return error status, don't block - let it through for debugging
            if frame_count % 50 == 0:
                print(f"  ⚠ Energy below threshold: {raw_avg_energy:.8f} < {ENERGY_THRESHOLD:.8f}")
            return {
                "status": "no_audio", 
                "message": f"Signal too weak (energy: {raw_avg_energy:.8f} < {ENERGY_THRESHOLD:.8f})",
                "energy_avg": float(raw_avg_energy),
                "energy_max": float(raw_max_energy),
                "raw_energy_avg": float(raw_avg_energy),
                "raw_energy_max": float(raw_max_energy),
                "snr_db": float(snr_db),
            }
        
        # Reject if SNR too low (noisy signal) - but be more lenient
        if snr_db < MIN_SNR_DB:
            if frame_count % 50 == 0:
                print(f"  ⚠ SNR too low: {snr_db:.1f}dB < {MIN_SNR_DB}dB")
            return {
                "status": "no_signal",
                "message": f"Signal too noisy (SNR: {snr_db:.1f}dB < {MIN_SNR_DB}dB)",
                "energy_avg": float(raw_avg_energy),
                "energy_max": float(raw_max_energy),
                "raw_energy_avg": float(raw_avg_energy),
                "raw_energy_max": float(raw_max_energy),
                "snr_db": float(snr_db),
            }
        
        # Simple TDOA estimation
        tdoa_matrix = self.tdoa.estimate_tdoa(signals, method="gcc_phat")
        
        # Check if we got any non-zero TDOA values
        max_tdoa = np.max(np.abs(tdoa_matrix))
        if max_tdoa < 1e-7:  # Essentially zero
            # TDOA estimation failed - signal may be too weak or too noisy
            return {
                "status": "no_signal", 
                "message": f"TDOA estimation failed (signal quality insufficient)",
                "energy_avg": float(avg_energy),
                "energy_max": float(max_energy),
                "raw_energy_avg": float(raw_avg_energy),
                "raw_energy_max": float(raw_max_energy),
                "snr_db": float(snr_db),
            }
        
        # Smoothing for stability - use median for robustness against outliers
        self.tdoa_hist.append(tdoa_matrix)
        # deque with maxlen auto-limits, no need to manually pop
        if len(self.tdoa_hist) >= 3:
            # Use median for better outlier rejection
            tdoa_avg = np.median(list(self.tdoa_hist), axis=0)
        else:
            # If not enough history, use mean
            tdoa_avg = np.mean(list(self.tdoa_hist), axis=0)
        
        # Check if source is directly above (all TDOA near zero)
        max_tdoa_abs = np.max(np.abs(tdoa_avg))
        source_directly_above = max_tdoa_abs < 5e-5  # Less than 50 microseconds difference

        # Special case: source directly above array
        if source_directly_above:
            # When source is directly above, TDOA is near zero
            # Estimate height from signal energy (closer = louder) or use previous position
            if self.smoothed_position is not None:
                # Use previous z-position if available (maintains stability)
                estimated_height = self.smoothed_position[2]
            else:
                # Estimate from signal energy: higher energy = closer source
                # Normalize energy to estimate distance (rough heuristic)
                # For a typical phone speaker at 50cm: energy ~0.000005-0.00001
                # Scale: energy 0.000001 -> ~1m, energy 0.00001 -> ~0.3m
                energy_val = float(raw_avg_energy)
                if energy_val > 1e-6:
                    # Inverse relationship: higher energy = closer
                    height_est = 0.5 / (energy_val * 1e5)
                    estimated_height = float(max(0.2, min(1.5, height_est)))
                else:
                    estimated_height = 0.5  # Default 50cm
            
            # Position is directly above center of array
            position = np.array([0.0, 0.0, estimated_height])
            azimuth = 0.0
            elevation = 90.0  # Straight up
            distance = estimated_height
            using_triangulation = True
        else:
            # Try 3D triangulation first (more accurate)
            position = self.tri.triangulate_3d(tdoa_avg)
            using_triangulation = position is not None
            
            # Get angles from TDOA for display
            azimuth, elevation = self.angles.estimate_azimuth_elevation(tdoa_avg)
            
            # If triangulation fails, fall back to angle-based estimation
            if position is None:
                # Estimate distance from TDOA
                distance = self.tri.estimate_distance_from_tdoa(tdoa_avg)
                
                # Simple position from angles and distance
                position = self.tri.estimate_from_angles(azimuth, elevation, distance)
            
            # Calculate distance from position (works for both methods)
            distance = np.linalg.norm(position)
            
            # Validate position - reject if it's too far from previous position (outlier rejection)
            if self.smoothed_position is not None:
                position_change = np.linalg.norm(position - self.smoothed_position)
                max_allowed_change = 0.3  # Stricter: reject if position jumps more than 30cm
                if position_change > max_allowed_change:
                    # Position jump too large - likely noise or false detection
                    # Use previous position with minimal update
                    position = 0.98 * self.smoothed_position + 0.02 * position
                    
                    # Also check if TDOA is consistent
                    if max_tdoa_abs < 1e-6:
                        # Very small TDOA but large position change - likely noise
                        # Reject this frame
                        return {
                            "status": "no_signal",
                            "message": "Position inconsistent - likely noise",
                            "energy_avg": float(avg_energy),
                            "energy_max": float(max_energy),
                            "raw_energy_avg": float(raw_avg_energy),
                            "raw_energy_max": float(raw_max_energy),
                            "snr_db": float(snr_db),
                        }
        
        # Add debug info every 100 frames
        if frame_count % 100 == 0:
            if source_directly_above:
                method = "directly-above"
            elif using_triangulation:
                method = "triangulation"
            else:
                method = "angle-based"
            print(f"  Method: {method}, Raw pos: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}), Distance: {distance:.3f}m, Max TDOA: {max_tdoa_abs*1e6:.2f}μs")
        
        # Adaptive smoothing based on position change
        if self.smoothed_position is None:
            self.smoothed_position = position.copy()
            self.smoothed_azimuth = azimuth
            self.smoothed_elevation = elevation
        else:
            position_change = np.linalg.norm(position - self.smoothed_position)
            
            # Use stronger smoothing for small changes (noise), lighter for large changes (real movement)
            if position_change < 0.05:  # Less than 5cm change - likely noise
                alpha = 0.9  # Strong smoothing to reduce jitter
            elif position_change < 0.2:  # 5-20cm change - moderate smoothing
                alpha = 0.7
            else:  # Large change - real movement, less smoothing
                alpha = 0.5
            
            # Apply exponential smoothing
            self.smoothed_position = alpha * self.smoothed_position + (1 - alpha) * position
            self.smoothed_azimuth = alpha * self.smoothed_azimuth + (1 - alpha) * azimuth
            self.smoothed_elevation = alpha * self.smoothed_elevation + (1 - alpha) * elevation

        # Build simple TDOA pairs table
        tdoa_pairs = []
        n = tdoa_avg.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                tdoa_pairs.append({"ch1": int(i), "ch2": int(j), "tdoa": float(tdoa_avg[i, j])})

        return {
            "x": float(self.smoothed_position[0]),
            "y": float(self.smoothed_position[1]),
            "z": float(self.smoothed_position[2]),
            "height": float(self.smoothed_position[2]),
            "azimuth": float(self.smoothed_azimuth),
            "elevation": float(self.smoothed_elevation),
            "distance": float(distance),
            "tdoa_pairs": tdoa_pairs,
            "status": "tracking",
            "message": "Source detected",
            "energy_avg": float(avg_energy),
            "energy_max": float(max_energy),
            "raw_energy_avg": float(raw_avg_energy),
            "raw_energy_max": float(raw_max_energy),
        }

    async def _loop(self):
        """Continuous tracking loop - runs indefinitely until stopped manually"""
        frame_count = 0
        try:
            while self.running:
                try:
                    chunk = self.audio.get_audio_chunk(timeout=0.05)  # Shorter timeout for faster response
                    if chunk is None:
                        await asyncio.sleep(0.001)  # Minimal delay
                        continue
                    
                    frame_count += 1
                    result = self._process(chunk, frame_count)
                    
                    # Always send result - never stop sending updates
                    if result is None:
                        # Send keepalive with last known position
                        result = {
                            "status": "waiting",
                            "message": "Waiting for signal...",
                            "x": float(self.smoothed_position[0]) if self.smoothed_position is not None else 0.0,
                            "y": float(self.smoothed_position[1]) if self.smoothed_position is not None else 0.0,
                            "z": float(self.smoothed_position[2]) if self.smoothed_position is not None else 0.0,
                            "azimuth": float(self.smoothed_azimuth),
                            "elevation": float(self.smoothed_elevation),
                        }
                    
                    # Add frame counter for debugging
                    result["frame"] = frame_count
                    
                    # Always send update - continuous tracking
                    payload = json.dumps(result)
                    if self.clients:
                        # Send immediately - never stop sending
                        disconnected = []
                        for ws in list(self.clients):
                            try:
                                await ws.send_text(payload)
                            except Exception as e:
                                print(f"Error sending to client: {e}")
                                disconnected.append(ws)
                        # Remove disconnected clients
                        for ws in disconnected:
                            self.clients.discard(ws)
                    
                    # Debug: print every 100 frames
                    if frame_count % 100 == 0:
                        print(f"Frame {frame_count}: Status={result.get('status')}, Pos=({result.get('x', 0):.3f}, {result.get('y', 0):.3f}, {result.get('z', 0):.3f})")
                    
                    # Continue immediately - no stopping
                except Exception as e:
                    # Log error but continue running
                    print(f"Error processing chunk: {e}")
                    import traceback
                    traceback.print_exc()
                    await asyncio.sleep(0.01)  # Small delay on error, then continue
                    continue
        except KeyboardInterrupt:
            print("Tracking stopped by user")
        except Exception as e:
            print(f"Fatal error in tracking loop: {e}")
        finally:
            if self.running:
                print("Tracking loop ended unexpectedly")
            self.audio.stop_capture()


streamer = LocalizationStreamer()


@app.on_event("shutdown")
async def _on_shutdown():
    await streamer.stop()


@app.get("/start")
async def start(temp_c: float | None = None):
    streamer.set_temperature(temp_c)
    await streamer.start()
    return {"status": "started"}


@app.get("/stop")
async def stop():
    await streamer.stop()
    return {"status": "stopped"}



@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    streamer.clients.add(websocket)
    # Ensure streaming is running
    await streamer.start()
    try:
        while True:
            # keep connection alive; client may send pings or ignore
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in streamer.clients:
            streamer.clients.remove(websocket)


