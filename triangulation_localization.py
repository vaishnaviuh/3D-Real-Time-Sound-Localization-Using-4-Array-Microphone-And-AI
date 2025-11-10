#!/usr/bin/env python3
"""
Sound Source Localization using Triangulation Method
ReSpeaker Microphone Array - 4 Channel Configuration
Uses channels 1-4 (ignores 0 and 5)
"""

import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt
from typing import cast, Tuple
import threading
import queue
import time
import os
from datetime import datetime

# Audio configuration
SAMPLE_RATE = 16000  # ReSpeaker typically uses 16kHz
CHUNK_SIZE = 2048  # Increased for better TDOA estimation
OVERLAP = 0.5  # 50% overlap for smoother processing
NUM_CHANNELS = 6  # ReSpeaker has 6 channels total
# Use channels 1-4, ignore channels 0 and 5
ACTIVE_CHANNELS = [1, 2, 3, 4]  # Ignore channel 0 and 5
NUM_ACTIVE_CHANNELS = len(ACTIVE_CHANNELS)
SOUND_SPEED = 343.0  # Speed of sound in m/s at 20°C

# Signal processing parameters
ENERGY_THRESHOLD = 0.000001  # Minimum energy to consider valid signal (more sensitive)
FREQ_MIN = 300  # Minimum frequency for bandpass filter (Hz)
FREQ_MAX = 7500  # Maximum frequency for bandpass filter (Hz) - must be < nyquist
SMOOTHING_ALPHA = 0.7  # Exponential smoothing factor (0-1, higher = more smoothing)
DEBUG_MODE = True  # Enable debug output
USE_FILTER = True  # Set to False to use raw audio if filter removes too much signal

# Output folder for saving plots
OUTPUT_FOLDER = "localization_results"  # Folder where plots will be saved

# Microphone positions in meters (square geometry, 4.5 cm spacing)
# Based on hardware: square pattern with 4.5 cm edge length
# Assuming square centered at origin
MIC_SPACING = 0.045  # 4.5 cm in meters (edge length of square)
HALF_EDGE = MIC_SPACING / 2.0  # Half edge length = 0.0225m
HALF_DIAGONAL = HALF_EDGE * np.sqrt(2)  # Half diagonal = 0.0318m

# Microphone positions (x, y, z) in meters
# Coordinate system: 
#   X: positive = right (DOA:0°)
#   Y: positive = top/forward (DOA:90°)
#   Z: positive = up (above array)
# MIC1: right, MIC2: top-left, MIC3: bottom-left, MIC4: bottom-right
MIC_POSITIONS = np.array([
    [HALF_EDGE, HALF_EDGE, 0.0],      # MIC1 (channel 1) - top-right: (+X, +Y)
    [-HALF_EDGE, HALF_EDGE, 0.0],    # MIC2 (channel 2) - top-left: (-X, +Y)
    [-HALF_EDGE, -HALF_EDGE, 0.0],   # MIC3 (channel 3) - bottom-left: (-X, -Y)
    [HALF_EDGE, -HALF_EDGE, 0.0],    # MIC4 (channel 4) - bottom-right: (+X, -Y)
])

# Derived physical limits based on array geometry
# Max time-difference-of-arrival is bounded by the maximum distance between any mic pair
_pair_dists = []
for _i in range(len(MIC_POSITIONS)):
    for _j in range(_i + 1, len(MIC_POSITIONS)):
        _pair_dists.append(float(np.linalg.norm(MIC_POSITIONS[_j] - MIC_POSITIONS[_i])))
MAX_MIC_PAIR_DISTANCE = max(_pair_dists) if _pair_dists else MIC_SPACING
MAX_TDOA_TIME = MAX_MIC_PAIR_DISTANCE / SOUND_SPEED
# Keep a small safety margin of +1 sample
MAX_TDOA_SAMPLES = int(np.ceil(SAMPLE_RATE * MAX_TDOA_TIME)) + 1

class AudioCapture:
    """Handles live audio capture from ReSpeaker microphone array"""
    
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio = None
        self.stream = None
        
    def find_respeaker_device(self):
        """Find ReSpeaker device index"""
        audio = pyaudio.PyAudio()
        device_index = None
        
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if 'respeaker' in info['name'].lower() or info['maxInputChannels'] >= 6:
                device_index = i
                print(f"Found audio device: {info['name']} (index {i})")
                print(f"  Channels: {info['maxInputChannels']}, Sample Rate: {info['defaultSampleRate']}")
                break
        
        audio.terminate()
        return device_index
    
    def start_capture(self, device_index=None):
        """Start audio capture"""
        self.audio = pyaudio.PyAudio()
        
        if device_index is None:
            device_index = self.find_respeaker_device()
            if device_index is None:
                print("ReSpeaker not found, using default device")
                device_index = None
        
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=NUM_CHANNELS,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            self.is_recording = True
            self.stream.start_stream()
            print("Audio capture started")
        except Exception as e:
            print(f"Error starting audio capture: {e}")
            raise
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Reshape to (chunk_size, num_channels)
        if len(audio_data) != self.chunk_size * NUM_CHANNELS:
            # Handle case where we get partial data
            expected_samples = self.chunk_size * NUM_CHANNELS
            if len(audio_data) < expected_samples:
                # Pad with zeros
                audio_data = np.pad(audio_data, (0, expected_samples - len(audio_data)), mode='constant')
            else:
                # Truncate
                audio_data = audio_data[:expected_samples]
        
        audio_data = audio_data.reshape(self.chunk_size, NUM_CHANNELS)
        
        # Extract only active channels (1-4)
        active_audio = audio_data[:, ACTIVE_CHANNELS]
        
        # Convert to float32 and normalize
        active_audio = active_audio.astype(np.float32) / 32768.0
        
        # Put in queue (non-blocking)
        try:
            self.audio_queue.put_nowait(active_audio)
        except queue.Full:
            pass  # Drop frame if queue is full
        
        return (None, pyaudio.paContinue)
    
    def get_audio_chunk(self, timeout=0.1):
        """Get next audio chunk from queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop_capture(self):
        """Stop audio capture"""
        self.is_recording = False
        try:
            if self.stream:
                try:
                    # Abort stream first to avoid ALSA drop errors on capture devices
                    if hasattr(self.stream, 'is_active') and self.stream.is_active():
                        if hasattr(self.stream, 'abort_stream'):
                            self.stream.abort_stream()  # type: ignore
                except Exception:
                    pass
                try:
                    if hasattr(self.stream, 'is_stopped') and not self.stream.is_stopped():
                        self.stream.stop_stream()
                except Exception:
                    pass
                try:
                    self.stream.close()
                except Exception:
                    pass
                finally:
                    self.stream = None
            if self.audio:
                try:
                    self.audio.terminate()
                except Exception:
                    pass
                finally:
                    self.audio = None
        finally:
            print("Audio capture stopped")


class TDOAEstimator:
    """Estimates Time Difference of Arrival (TDOA) between microphone pairs"""
    
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        # Limit search window to physically possible lag based on array geometry
        # This greatly reduces false peaks and improves accuracy
        self.max_tau = MAX_TDOA_SAMPLES
        self.search_range = self.max_tau * 2 + 1
        
    def cross_correlation(self, signal1, signal2):
        """Compute cross-correlation between two signals"""
        # Normalize signals
        signal1 = signal1 - np.mean(signal1)
        signal2 = signal2 - np.mean(signal2)
        
        # Compute cross-correlation
        correlation = np.correlate(signal1, signal2, mode='full')
        
        # Find the lag with maximum correlation
        lags = np.arange(-len(signal2) + 1, len(signal1))
        max_idx = np.argmax(np.abs(correlation))
        tau = lags[max_idx]
        
        return tau, correlation
    
    def gcc_phat(self, signal1, signal2):
        """Generalized Cross-Correlation with Phase Transform (GCC-PHAT)"""
        # Normalize signals
        signal1 = signal1 - np.mean(signal1)
        signal2 = signal2 - np.mean(signal2)
        
        # Remove DC and apply window
        signal1 = signal1 * np.hanning(len(signal1))
        signal2 = signal2 * np.hanning(len(signal2))
        
        # Compute FFT (use next power of 2 for efficiency)
        n = 2 ** int(np.ceil(np.log2(len(signal1) + len(signal2) - 1)))
        fft1 = np.fft.fft(signal1, n)
        fft2 = np.fft.fft(signal2, n)
        
        # Compute cross-power spectral density
        cross_power = fft1 * np.conj(fft2)
        
        # PHAT weighting
        magnitude = np.abs(cross_power)
        cross_power = cross_power / (magnitude + 1e-10)
        
        # Inverse FFT
        correlation = np.fft.ifft(cross_power)
        correlation = np.real(correlation)
        
        # Find delay in limited range around zero
        center = len(correlation) // 2
        search_start = max(0, center - self.max_tau)
        search_end = min(len(correlation), center + self.max_tau + 1)
        search_corr = correlation[search_start:search_end]
        
        if len(search_corr) == 0:
            return 0, correlation
        
        max_idx = np.argmax(np.abs(search_corr))
        tau = max_idx + search_start - center
        
        return tau, correlation
    
    def estimate_tdoa(self, signals, method='gcc_phat'):
        """Estimate TDOA for all microphone pairs"""
        num_mics = len(signals)
        tdoa_matrix = np.zeros((num_mics, num_mics))
        
        # Check signal energy
        energies = [np.mean(sig**2) for sig in signals]
        avg_energy = np.mean(energies)
        max_energy = np.max(energies)
        
        # Debug output (only print occasionally to avoid spam)
        if DEBUG_MODE and hasattr(self, '_debug_counter'):
            self._debug_counter += 1
            if self._debug_counter % 100 == 0:  # Print every 100 calls
                print(f"Signal energy - Avg: {avg_energy:.6f}, Max: {max_energy:.6f}, Threshold: {ENERGY_THRESHOLD:.6f}")
        elif DEBUG_MODE:
            self._debug_counter = 0
        
        # Only process if there's sufficient signal
        if avg_energy < ENERGY_THRESHOLD:
            return tdoa_matrix
        
        for i in range(num_mics):
            for j in range(i + 1, num_mics):
                # Skip if either signal is too weak (use lower threshold for individual channels)
                channel_threshold = ENERGY_THRESHOLD * 0.3  # More lenient for individual channels
                if energies[i] < channel_threshold or energies[j] < channel_threshold:
                    continue
                
                if method == 'gcc_phat':
                    tau, _ = self.gcc_phat(signals[i], signals[j])
                else:
                    tau, _ = self.cross_correlation(signals[i], signals[j])
                
                # Convert to time difference in seconds
                tdoa = tau / self.sample_rate
                # Clamp to physically possible TDOA for this microphone pair
                mic_vec = MIC_POSITIONS[j] - MIC_POSITIONS[i]
                mic_distance = float(np.linalg.norm(mic_vec))
                max_pair_tdoa = mic_distance / SOUND_SPEED
                if abs(tdoa) > max_pair_tdoa:
                    tdoa = float(np.sign(tdoa)) * max_pair_tdoa
                tdoa_matrix[i, j] = tdoa
                tdoa_matrix[j, i] = -tdoa
        
        return tdoa_matrix


class AngleEstimator:
    """Estimates azimuth and elevation angles from TDOA"""
    
    def __init__(self, mic_positions):
        self.mic_positions = mic_positions
        self.num_mics = len(mic_positions)
    
    def estimate_azimuth_elevation(self, tdoa_matrix):
        """Estimate azimuth and elevation from TDOA matrix using proper TDOA-to-angle conversion"""
        # For square array geometry
        # MIC1 (0): top-right, MIC2 (1): top-left
        # MIC3 (2): bottom-left, MIC4 (3): bottom-right
        
        # Use all microphone pairs to estimate direction
        # For a pair of mics, TDOA gives us the cosine of the angle between
        # the source direction and the mic pair vector
        
        x_components = []
        y_components = []
        z_components = []
        
        # Process all microphone pairs
        for i in range(self.num_mics):
            for j in range(i + 1, self.num_mics):
                tdoa = tdoa_matrix[i, j]
                if abs(tdoa) < 1e-7:
                    continue
                
                # Vector from mic i to mic j
                mic_vec = self.mic_positions[j] - self.mic_positions[i]
                mic_distance = np.linalg.norm(mic_vec)
                
                if mic_distance < 1e-6:
                    continue
                
                # Normalize mic vector
                mic_vec_norm = mic_vec / mic_distance
                
                # TDOA gives us: tdoa = (mic_distance / c) * cos(theta)
                # where theta is angle between source direction and mic pair vector
                cos_theta = np.clip(tdoa * SOUND_SPEED / mic_distance, -1, 1)
                
                # For far-field assumption, the source direction unit vector
                # projected onto mic_vec should equal cos_theta
                # We can't get full 3D direction from one pair, but we can get components
                
                # Estimate direction component along this mic pair
                direction_component = cos_theta * mic_vec_norm
                
                x_components.append(direction_component[0])
                y_components.append(direction_component[1])
                z_components.append(direction_component[2])
        
        # Average the direction components
        if x_components and y_components:
            avg_x = np.mean(x_components)
            avg_y = np.mean(y_components)
            avg_z = np.mean(z_components) if z_components else 0.0
            
            # Normalize to get unit direction vector
            direction_mag = np.sqrt(avg_x**2 + avg_y**2 + avg_z**2)
            if direction_mag > 1e-6:
                direction_vec = np.array([avg_x, avg_y, avg_z]) / direction_mag
                
                # Convert to azimuth and elevation
                # Azimuth: angle in xy plane (0 = +x, 90 = +y)
                azimuth_rad = np.arctan2(direction_vec[1], direction_vec[0])
                # Elevation: angle from xy plane
                elevation_rad = np.arcsin(np.clip(direction_vec[2], -1, 1))
            else:
                azimuth_rad = 0.0
                elevation_rad = 0.0
        else:
            azimuth_rad = 0.0
            elevation_rad = 0.0
        
        # Convert to degrees
        azimuth_deg = np.degrees(azimuth_rad)
        elevation_deg = np.degrees(elevation_rad)
        
        return azimuth_deg, elevation_deg


class TriangulationLocalizer:
    """Performs 3D triangulation to locate sound source"""
    
    def __init__(self, mic_positions):
        self.mic_positions = mic_positions
        self.num_mics = len(mic_positions)
    
    def triangulate_3d(self, tdoa_matrix):
        """Triangulate 3D position from TDOA matrix using multiple methods"""
        # Method 1: Use all microphone pairs with least squares
        # Collect all TDOA measurements
        tdoa_pairs = []
        mic_pairs = []
        
        for i in range(self.num_mics):
            for j in range(i + 1, self.num_mics):
                tdoa = tdoa_matrix[i, j]
                if abs(tdoa) > 1e-7:  # Only use significant TDOA values
                    tdoa_pairs.append(tdoa)
                    mic_pairs.append((i, j))
        
        if len(tdoa_pairs) < 3:  # Need at least 3 pairs for 3D
            return None
        
        # Use first microphone as reference for initial guess
        ref_mic = 0
        ref_pos = self.mic_positions[ref_mic]
        
        # Better initial guess: estimate from TDOA
        # For far-field, estimate direction first
        direction_estimate = np.zeros(3)
        for (i, j), tdoa in zip(mic_pairs, tdoa_pairs):
            mic_vec = self.mic_positions[j] - self.mic_positions[i]
            mic_dist = np.linalg.norm(mic_vec)
            if mic_dist > 1e-6:
                cos_theta = np.clip(tdoa * SOUND_SPEED / mic_dist, -1, 1)
                direction_estimate += cos_theta * (mic_vec / mic_dist)
        
        direction_estimate = direction_estimate / (len(mic_pairs) + 1e-10)
        dir_mag = np.linalg.norm(direction_estimate)
        if dir_mag > 1e-6:
            direction_estimate = direction_estimate / dir_mag
        
        # For near-field sources, TDOA-based distance estimation is unreliable
        # Instead, we'll try multiple close initial guesses and let optimization find the correct distance
        # Default to close range (10-15cm) which matches typical phone placement
        estimated_distance = 0.12  # Default 12cm - typical for phone above array
        
        def residual_function(pos):
            """Residual function for least squares - use all pairs"""
            residuals = []
            for (i, j), tdoa in zip(mic_pairs, tdoa_pairs):
                dist_i = np.linalg.norm(pos - self.mic_positions[i])
                dist_j = np.linalg.norm(pos - self.mic_positions[j])
                expected_tdoa = (dist_j - dist_i) / SOUND_SPEED
                residuals.append(expected_tdoa - tdoa)
            return np.array(residuals)
        
        # Try multiple initial guesses, prioritizing close sources first
        best_result = None
        best_cost = float('inf')
        
        # Prioritize close distances first (10-20cm range for phone above array)
        # Then try medium distances
        test_distances = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]
        test_z_heights = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50]
        
        # First pass: try close distances with various z-heights (most likely scenario)
        for init_dist in test_distances[:6]:  # First 6 distances (10-25cm)
            for z_height in test_z_heights[:5]:  # First 5 z-heights (10-20cm)
                try:
                    test_init = np.mean(self.mic_positions, axis=0) + init_dist * direction_estimate
                    test_init[2] = z_height
                    
                    # Bounds: allow very close sources (5cm to 2m), z from 0.05 to 0.5m above
                    bounds = ([-2, -2, 0.05], [2, 2, 0.5])
                    result = least_squares(residual_function, test_init, method='trf', 
                                         bounds=bounds, max_nfev=300, ftol=1e-7, 
                                         x_scale='jac', loss='linear')
                    
                    if result.success:
                        # Calculate distances from source to each mic
                        distances = [float(np.linalg.norm(result.x - mic)) for mic in self.mic_positions]
                        min_dist = min(distances)
                        max_dist = max(distances)
                        
                        # Validate: source should be 5cm to 2m from any mic
                        # Prefer solutions where all mics are at similar distances (indicates source is centered)
                        if 0.05 < min_dist < 2.0 and max_dist < 2.0:
                            # Weight cost by distance spread - prefer solutions where distances are similar
                            dist_spread = max_dist - min_dist
                            weighted_cost = result.cost * (1.0 + 0.1 * dist_spread)
                            
                            if weighted_cost < best_cost:
                                best_result = result
                                best_cost = weighted_cost
                except:
                    continue
        
        # Second pass: if no good solution found, try all distances
        if best_result is None or best_cost > 0.1:
            for init_dist in test_distances:
                for z_height in test_z_heights:
                    try:
                        test_init = np.mean(self.mic_positions, axis=0) + init_dist * direction_estimate
                        test_init[2] = z_height
                        
                        bounds = ([-2, -2, 0.05], [2, 2, 0.5])
                        result = least_squares(residual_function, test_init, method='trf', 
                                             bounds=bounds, max_nfev=300, ftol=1e-7,
                                             x_scale='jac', loss='linear')
                        
                        if result.success:
                            distances = [float(np.linalg.norm(result.x - mic)) for mic in self.mic_positions]
                            min_dist = min(distances)
                            max_dist = max(distances)
                            
                            if 0.05 < min_dist < 2.0 and max_dist < 2.0:
                                dist_spread = max_dist - min_dist
                                weighted_cost = result.cost * (1.0 + 0.1 * dist_spread)
                                
                                if weighted_cost < best_cost:
                                    best_result = result
                                    best_cost = weighted_cost
                    except:
                        continue
        
        # Accept solution if cost is reasonable (more lenient for near-field)
        if best_result is not None and best_cost < 0.05:
            return best_result.x
        
        return None
    
    def estimate_from_angles(self, azimuth_deg, elevation_deg, distance=0.12):
        """Estimate 3D position from angles and assumed distance"""
        azimuth = np.radians(azimuth_deg)
        elevation = np.radians(elevation_deg)
        
        x = distance * np.cos(elevation) * np.cos(azimuth)
        y = distance * np.cos(elevation) * np.sin(azimuth)
        z = distance * np.sin(elevation)
        
        # Ensure z is reasonable (5cm to 50cm for close sources)
        z = max(0.05, min(0.5, z))
        
        return np.array([x, y, z])


class SoundLocalization:
    """Main class for sound source localization"""
    
    def __init__(self):
        self.audio_capture = AudioCapture()
        self.tdoa_estimator = TDOAEstimator()
        self.angle_estimator = AngleEstimator(MIC_POSITIONS)
        self.triangulator = TriangulationLocalizer(MIC_POSITIONS)
        
        # Visualization data
        self.position_history = []
        self.angle_history = []
        self.tdoa_history = []
        self.max_history = 100
        
        # Smoothing filters
        self.smoothed_position = None
        self.smoothed_azimuth = 0.0
        self.smoothed_elevation = 0.0
        
        # Debug counters
        self.frame_count = 0
        self.detection_count = 0
        self.last_energy = 0.0
        self.energy_samples = []
        
        # Bandpass filter for signal preprocessing
        nyquist = SAMPLE_RATE / 2
        # Ensure frequencies are within valid range (0 < Wn < 1)
        low = max(0.01, FREQ_MIN / nyquist)  # Ensure > 0
        high = min(0.99, FREQ_MAX / nyquist)  # Ensure < 1
        if high <= low:
            # Fallback if frequencies are invalid
            low = 0.02
            high = 0.95
        self.b, self.a = cast(Tuple[np.ndarray, np.ndarray], butter(4, [low, high], btype='band'))
        
        # Plot setup
        self.fig = None
        self.ax_2d = None
        self.ax_3d = None
        self.animation = None
    
    def process_audio_chunk(self, audio_data):
        """Process a chunk of audio data"""
        if audio_data is None or audio_data.shape[1] != NUM_ACTIVE_CHANNELS:
            return None
        
        # Check raw audio energy before filtering
        raw_signals = audio_data.T
        raw_energies = [np.mean(sig**2) for sig in raw_signals]
        raw_avg_energy = np.mean(raw_energies)
        self.last_energy = raw_avg_energy
        self.energy_samples.append(raw_avg_energy)
        if len(self.energy_samples) > 100:
            self.energy_samples.pop(0)
        
        # Transpose to (num_channels, chunk_size)
        signals = raw_signals
        
        # Apply bandpass filter to each channel (or use raw if filter disabled)
        if USE_FILTER:
            filtered_signals = []
            for sig in signals:
                # Use filtfilt for zero-phase filtering
                filtered = filtfilt(self.b, self.a, sig)
                filtered_signals.append(filtered)
            signals = np.array(filtered_signals)
        else:
            # Use raw signals without filtering
            signals = np.array(raw_signals)
        
        # Estimate TDOA
        tdoa_matrix = self.tdoa_estimator.estimate_tdoa(signals, method='gcc_phat')
        
        # Check if we have valid TDOA measurements
        if np.allclose(tdoa_matrix, 0):
            self.frame_count += 1
            if self.frame_count % 50 == 0:  # Print every 50 frames
                avg_recent_energy = np.mean(self.energy_samples[-50:]) if len(self.energy_samples) > 0 else 0
                max_recent_energy = np.max(self.energy_samples[-50:]) if len(self.energy_samples) > 0 else 0
                print(f"No signal detected - Raw energy: avg={avg_recent_energy:.6f}, max={max_recent_energy:.6f}, threshold={ENERGY_THRESHOLD:.6f}")
                print(f"  If energy is very low, check: 1) Microphone is working, 2) Volume is up, 3) Channels are correct")
            return None
        
        self.detection_count += 1
        self.frame_count += 1
        
        # Debug: Print TDOA matrix occasionally
        if DEBUG_MODE and self.frame_count % 100 == 0:
            print(f"\nFrame {self.frame_count}, Detections: {self.detection_count}")
            print("TDOA matrix (microseconds):")
            for i in range(4):
                for j in range(4):
                    if i != j:
                        print(f"  MIC{i+1}-MIC{j+1}: {tdoa_matrix[i,j]*1e6:.2f} μs", end="  ")
                print()
        
        # Estimate angles
        azimuth, elevation = self.angle_estimator.estimate_azimuth_elevation(tdoa_matrix)
        
        # Triangulate 3D position
        position_3d = self.triangulator.triangulate_3d(tdoa_matrix)
        
        # Debug triangulation result
        if DEBUG_MODE and self.frame_count % 100 == 0:
            if position_3d is not None:
                print(f"Triangulation successful: ({position_3d[0]:.3f}, {position_3d[1]:.3f}, {position_3d[2]:.3f}) m")
            else:
                print("Triangulation failed, using angle-based estimation")
        
        # If triangulation fails, use angle-based estimation with close distance
        if position_3d is None:
            # Default to close range (10-15cm) which matches typical phone placement above array
            estimated_distance = 0.12  # 12cm - typical for phone above array
            position_3d = self.triangulator.estimate_from_angles(azimuth, elevation, estimated_distance)
        
        # Apply exponential smoothing
        if self.smoothed_position is None:
            self.smoothed_position = position_3d.copy()
            self.smoothed_azimuth = azimuth
            self.smoothed_elevation = elevation
        else:
            self.smoothed_position = (SMOOTHING_ALPHA * self.smoothed_position + 
                                      (1 - SMOOTHING_ALPHA) * position_3d)
            self.smoothed_azimuth = (SMOOTHING_ALPHA * self.smoothed_azimuth + 
                                    (1 - SMOOTHING_ALPHA) * azimuth)
            self.smoothed_elevation = (SMOOTHING_ALPHA * self.smoothed_elevation + 
                                      (1 - SMOOTHING_ALPHA) * elevation)
        
        return {
            'position': self.smoothed_position.copy(),
            'azimuth': self.smoothed_azimuth,
            'elevation': self.smoothed_elevation,
            'tdoa_matrix': tdoa_matrix,
            'raw_position': position_3d
        }
    
    def update_plots(self, frame):
        """Update plots with new data"""
        # Get latest audio chunk
        audio_data = self.audio_capture.get_audio_chunk()
        
        if audio_data is not None:
            result = self.process_audio_chunk(audio_data)
            
            if result:
                # Update history
                self.position_history.append(result['position'].copy())
                self.angle_history.append({
                    'azimuth': result['azimuth'],
                    'elevation': result['elevation']
                })
                self.tdoa_history.append(result['tdoa_matrix'].copy())
                
                # Limit history size
                if len(self.position_history) > self.max_history:
                    self.position_history.pop(0)
                    self.angle_history.pop(0)
                    self.tdoa_history.pop(0)
        
        # Clear and redraw plots
        if self.ax_2d is not None and self.ax_3d is not None:
            self.ax_2d.clear()
            self.ax_3d.clear()
        else:
            return
        
        # Plot 2D (top view)
        if self.position_history:
            positions = np.array(self.position_history)
            self.ax_2d.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.3, label='Trajectory')
            self.ax_2d.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='Current')
        
        # Plot microphone positions
        mic_2d = MIC_POSITIONS[:, :2]
        self.ax_2d.plot(mic_2d[:, 0], mic_2d[:, 1], 'ks', markersize=12, label='Microphones')
        for i, pos in enumerate(mic_2d):
            self.ax_2d.text(pos[0], pos[1], f'MIC{i+1}', fontsize=8, ha='center', va='bottom')
        
        self.ax_2d.set_xlabel('X (m)')
        self.ax_2d.set_ylabel('Y (m)')
        self.ax_2d.set_title('2D Sound Source Localization (Top View)')
        self.ax_2d.grid(True, alpha=0.3)
        self.ax_2d.legend()
        self.ax_2d.set_aspect('equal')
        self.ax_2d.set_xlim(-1, 1)
        self.ax_2d.set_ylim(-1, 1)
        
        # Plot 3D
        if self.position_history:
            positions = np.array(self.position_history)
            self.ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', alpha=0.3, label='Trajectory')
            self.ax_3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                             c='r', s=100, label='Current')
        
        # Plot microphone positions
        self.ax_3d.scatter(MIC_POSITIONS[:, 0], MIC_POSITIONS[:, 1], MIC_POSITIONS[:, 2],  # type: ignore
                          c='k', s=100, marker='s', label='Microphones')
        
        # Display current angles, position, and TDOA info
        if self.angle_history:
            current_angle = self.angle_history[-1]
            if self.position_history:
                current_pos = self.position_history[-1]
                if self.tdoa_history:
                    tdoa = self.tdoa_history[-1]
                    # Show max TDOA for debugging
                    max_tdoa = np.max(np.abs(tdoa))
                    info_text = (f'Azimuth: {current_angle["azimuth"]:.1f}°\n'
                                f'Elevation: {current_angle["elevation"]:.1f}°\n'
                                f'Position: ({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}) m\n'
                                f'Max TDOA: {max_tdoa*1e6:.1f} μs')
                else:
                    info_text = (f'Azimuth: {current_angle["azimuth"]:.1f}°\n'
                                f'Elevation: {current_angle["elevation"]:.1f}°\n'
                                f'Position: ({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}) m')
            else:
                info_text = (f'Azimuth: {current_angle["azimuth"]:.1f}°\n'
                            f'Elevation: {current_angle["elevation"]:.1f}°')
            self.ax_3d.text2D(0.02, 0.98, info_text, transform=self.ax_3d.transAxes,
                             fontsize=9, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title('3D Sound Source Localization')
        self.ax_3d.legend()
        self.ax_3d.set_xlim(-1, 1)
        self.ax_3d.set_ylim(-1, 1)
        self.ax_3d.set_zlim(-0.5, 0.5)
    
    def record_and_localize(self, duration=6.0):
        """Record audio for specified duration and localize source"""
        import time
        
        try:
            # Start audio capture
            self.audio_capture.start_capture()
            
            print("\n" + "=" * 60)
            print("Sound Source Localization - Recording Mode")
            print("=" * 60)
            
            # Countdown before recording
            print("\nStarting countdown...")
            for i in range(3, 0, -1):
                print(f"  {i}...")
                time.sleep(1)
            print("  RECORDING NOW!\n")
            
            # Record for specified duration
            print(f"Recording for {duration} seconds...")
            print("(Speak or make sound now)\n")
            
            start_time = time.time()
            recorded_chunks = []
            energy_samples = []
            
            print("Monitoring audio levels during recording...\n")
            
            while time.time() - start_time < duration:
                audio_data = self.audio_capture.get_audio_chunk(timeout=0.1)
                if audio_data is not None:
                    recorded_chunks.append(audio_data)
                    
                    # Check raw audio energy for diagnostics
                    raw_signals = audio_data.T
                    raw_energies = [np.mean(sig**2) for sig in raw_signals]
                    raw_avg_energy = np.mean(raw_energies)
                    raw_max_energy = np.max(raw_energies)
                    energy_samples.append(raw_avg_energy)
                    
                    elapsed = time.time() - start_time
                    if int(elapsed) != int(elapsed - 0.1):  # Print every second
                        remaining = duration - elapsed
                        # Show energy levels
                        recent_energy = np.mean(energy_samples[-10:]) if len(energy_samples) >= 10 else raw_avg_energy
                        status = "✓" if recent_energy > ENERGY_THRESHOLD else "✗"
                        print(f"  Recording... {remaining:.1f}s remaining | Energy: {recent_energy:.6f} {status}", end='\r')
            
            # Print final energy statistics
            if energy_samples:
                avg_energy = np.mean(energy_samples)
                max_energy = np.max(energy_samples)
                print(f"\n\nAudio Energy Statistics:")
                print(f"  Average: {avg_energy:.6f}")
                print(f"  Maximum: {max_energy:.6f}")
                print(f"  Threshold: {ENERGY_THRESHOLD:.6f}")
                if avg_energy < ENERGY_THRESHOLD:
                    print(f"  ⚠ Warning: Average energy below threshold!")
                    print(f"     Consider: 1) Increase phone volume, 2) Move phone closer, 3) Check microphone")
            
            print(f"\n\nRecording complete! Processing {len(recorded_chunks)} audio chunks...")
            
            # Process all recorded chunks
            all_positions = []
            all_angles = []
            all_tdoas = []
            
            for i, audio_data in enumerate(recorded_chunks):
                result = self.process_audio_chunk(audio_data)
                if result:
                    all_positions.append(result['position'].copy())
                    all_angles.append({
                        'azimuth': result['azimuth'],
                        'elevation': result['elevation']
                    })
                    all_tdoas.append(result['tdoa_matrix'].copy())
            
            if not all_positions:
                print("\n⚠ No valid source detected during recording!")
                print("\nTroubleshooting:")
                print("  1. Check microphone connection and volume")
                print("  2. Increase phone/audio source volume")
                print("  3. Move audio source closer to microphones")
                print("  4. Try speaking directly into microphones")
                print("  5. Check if channels 1-4 are correct (run --test-channels)")
                print(f"\n  Current energy threshold: {ENERGY_THRESHOLD:.6f}")
                print(f"  Try lowering ENERGY_THRESHOLD in code if signal is very quiet")
                if USE_FILTER:
                    print(f"  Try setting USE_FILTER = False to use raw audio without filtering")
                self.audio_capture.stop_capture()
                return
            
            # Calculate average position
            positions_array = np.array(all_positions)
            avg_position = np.mean(positions_array, axis=0)
            std_position = np.std(positions_array, axis=0)
            
            # Calculate average angles
            avg_azimuth = np.mean([a['azimuth'] for a in all_angles])
            avg_elevation = np.mean([a['elevation'] for a in all_angles])
            
            # Calculate average TDOA
            avg_tdoa = np.mean(all_tdoas, axis=0)
            
            print("\n" + "=" * 60)
            print("LOCALIZATION RESULTS")
            print("=" * 60)
            print(f"\nSource Position (average over {len(all_positions)} detections):")
            print(f"  X: {avg_position[0]:.4f} ± {std_position[0]:.4f} m (right=+, left=-)")
            print(f"  Y: {avg_position[1]:.4f} ± {std_position[1]:.4f} m (top/forward=+, bottom=-)")
            print(f"  Z: {avg_position[2]:.4f} ± {std_position[2]:.4f} m (up=+, down=-)")
            print(f"\nDistance from array center: {np.linalg.norm(avg_position):.4f} m")
            print(f"  Horizontal distance (XY plane): {np.linalg.norm(avg_position[:2]):.4f} m")
            print(f"  Height above array (Z): {avg_position[2]:.4f} m")
            
            # Debug: show expected vs actual
            print(f"\nExpected for phone 10-15cm above center:")
            print(f"  X should be ≈ 0.00 m (centered)")
            print(f"  Y should be ≈ 0.00 m (centered)")
            print(f"  Z should be ≈ 0.10-0.15 m (above array)")
            print(f"\nAngles:")
            print(f"  Azimuth: {avg_azimuth:.2f}°")
            print(f"  Elevation: {avg_elevation:.2f}°")
            print(f"\nMax TDOA: {np.max(np.abs(avg_tdoa))*1e6:.2f} microseconds")
            print("=" * 60)
            
            # Visualize results
            self.visualize_results(all_positions, avg_position, all_angles)
            
        except KeyboardInterrupt:
            print("\n\nRecording interrupted by user.")
        except Exception as e:
            print(f"\nError during recording: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.audio_capture.stop_capture()
            print("\nRecording stopped.")
    
    def visualize_results(self, positions, avg_position, angles):
        """Visualize the localization results with 2D and 3D plots and save to folder"""
        print("\n" + "=" * 60)
        print("DISPLAYING 2D AND 3D LOCALIZATION MAPS")
        print("=" * 60)
        
        # Create output folder if it doesn't exist
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
            print(f"Created output folder: {OUTPUT_FOLDER}")
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"localization_{timestamp}"
        
        fig = plt.figure(figsize=(18, 8))
        fig.suptitle('Sound Source Localization Results', fontsize=16, fontweight='bold')
        
        # ========== 2D PLOT (Top View) ==========
        ax_2d = fig.add_subplot(121)
        
        if len(positions) > 0:
            positions_array = np.array(positions)
            # Plot all individual detections
            ax_2d.scatter(positions_array[:, 0], positions_array[:, 1], 
                         c='blue', alpha=0.4, s=30, label=f'All detections ({len(positions)})', zorder=2)
            # Plot trajectory line
            ax_2d.plot(positions_array[:, 0], positions_array[:, 1], 
                      'b-', alpha=0.3, linewidth=2, zorder=1)
        
        # Average position (highlighted)
        ax_2d.plot(avg_position[0], avg_position[1], 'ro', 
                  markersize=18, markeredgecolor='darkred', markeredgewidth=2,
                  label='Average position', zorder=6)
        ax_2d.plot(avg_position[0], avg_position[1], 'r*', 
                  markersize=25, zorder=7)
        
        # Add text annotation for average position
        ax_2d.annotate(f'({avg_position[0]:.3f}, {avg_position[1]:.3f})',
                      xy=(avg_position[0], avg_position[1]),
                      xytext=(10, 10), textcoords='offset points',
                      fontsize=10, fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                      zorder=8)
        
        # Microphone positions
        mic_2d = MIC_POSITIONS[:, :2]
        ax_2d.plot(mic_2d[:, 0], mic_2d[:, 1], 'ks', 
                  markersize=15, markeredgecolor='white', markeredgewidth=1.5,
                  label='Microphones', zorder=5)
        for i, pos in enumerate(mic_2d):
            ax_2d.text(pos[0], pos[1], f'MIC{i+1}', 
                      fontsize=10, fontweight='bold', ha='center', va='bottom',
                      color='white', zorder=6)
        
        # Draw lines from array center to average position
        center = np.mean(MIC_POSITIONS[:, :2], axis=0)
        ax_2d.plot([center[0], avg_position[0]], [center[1], avg_position[1]],
                  'g--', linewidth=2, alpha=0.5, label='Distance', zorder=3)
        
        ax_2d.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
        ax_2d.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
        ax_2d.set_title('2D Sound Source Localization Map (Top View)', 
                       fontsize=14, fontweight='bold', pad=15)
        ax_2d.grid(True, alpha=0.4, linestyle='--')
        ax_2d.legend(loc='upper right', fontsize=10)
        ax_2d.set_aspect('equal')
        
        # Set limits with some margin
        margin = 0.2
        x_range = max(abs(avg_position[0]) + margin, 1.0)
        y_range = max(abs(avg_position[1]) + margin, 1.0)
        ax_2d.set_xlim(-x_range, x_range)
        ax_2d.set_ylim(-y_range, y_range)
        
        # ========== 3D PLOT ==========
        ax_3d = fig.add_subplot(122, projection='3d')
        
        if len(positions) > 0:
            positions_array = np.array(positions)
            # Plot all individual detections in 3D
            ax_3d.scatter(positions_array[:, 0], positions_array[:, 1], positions_array[:, 2],  # type: ignore
                         c='blue', alpha=0.4, s=30, label=f'All detections ({len(positions)})', zorder=2)
            # Plot trajectory line in 3D
            ax_3d.plot(positions_array[:, 0], positions_array[:, 1], positions_array[:, 2],
                      'b-', alpha=0.3, linewidth=2, zorder=1)
        
        # Average position (highlighted in 3D)
        ax_3d.scatter(avg_position[0], avg_position[1], avg_position[2],
                     c='r', s=300, marker='*', edgecolors='darkred', linewidths=2,
                     label='Average position', zorder=6)
        
        # Microphone positions in 3D
        ax_3d.scatter(MIC_POSITIONS[:, 0], MIC_POSITIONS[:, 1], MIC_POSITIONS[:, 2],  # type: ignore
                     c='k', s=150, marker='s', edgecolors='white', linewidths=1.5,
                     label='Microphones', zorder=5)
        
        # Draw lines from array center to average position in 3D
        center_3d = np.mean(MIC_POSITIONS, axis=0)
        ax_3d.plot([center_3d[0], avg_position[0]], 
                  [center_3d[1], avg_position[1]],
                  [center_3d[2], avg_position[2]],
                  'g--', linewidth=2, alpha=0.5, label='Distance', zorder=3)
        
        # Display detailed info box
        if angles:
            avg_azimuth = np.mean([a['azimuth'] for a in angles])
            avg_elevation = np.mean([a['elevation'] for a in angles])
            distance = np.linalg.norm(avg_position)
            info_text = (f'LOCALIZATION RESULTS\n'
                        f'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
                        f'Position (X, Y, Z):\n'
                        f'({avg_position[0]:.4f}, {avg_position[1]:.4f}, {avg_position[2]:.4f}) m\n\n'
                        f'Distance: {distance:.4f} m\n\n'
                        f'Azimuth: {avg_azimuth:.2f}°\n'
                        f'Elevation: {avg_elevation:.2f}°\n\n'
                        f'Detections: {len(positions)}')
        else:
            distance = np.linalg.norm(avg_position)
            info_text = (f'LOCALIZATION RESULTS\n'
                        f'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
                        f'Position (X, Y, Z):\n'
                        f'({avg_position[0]:.4f}, {avg_position[1]:.4f}, {avg_position[2]:.4f}) m\n\n'
                        f'Distance: {distance:.4f} m\n\n'
                        f'Detections: {len(positions)}')
        
        ax_3d.text2D(0.02, 0.98, info_text, transform=ax_3d.transAxes,
                     fontsize=10, fontweight='bold', verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', 
                              edgecolor='black', linewidth=2, alpha=0.9),
                     family='monospace', zorder=10)
        
        ax_3d.set_xlabel('X Position (m)', fontsize=12, fontweight='bold', labelpad=10)
        ax_3d.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold', labelpad=10)
        ax_3d.set_zlabel('Z Position (m)', fontsize=12, fontweight='bold', labelpad=10)
        ax_3d.set_title('3D Sound Source Localization Map', 
                       fontsize=14, fontweight='bold', pad=20)
        ax_3d.legend(loc='upper left', fontsize=10)
        
        # Set 3D limits with margin
        margin_3d = 0.2
        x_range_3d = max(abs(avg_position[0]) + margin_3d, 1.0)
        y_range_3d = max(abs(avg_position[1]) + margin_3d, 1.0)
        z_range_3d = max(abs(avg_position[2]) + margin_3d, 0.5)
        ax_3d.set_xlim(-x_range_3d, x_range_3d)
        ax_3d.set_ylim(-y_range_3d, y_range_3d)
        ax_3d.set_zlim(-0.3, z_range_3d)
        
        # Set 3D viewing angle for better visualization
        ax_3d.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        # Save the plot to file
        plot_path_png = os.path.join(OUTPUT_FOLDER, f"{filename}.png")
        plot_path_pdf = os.path.join(OUTPUT_FOLDER, f"{filename}.pdf")
        
        try:
            fig.savefig(plot_path_png, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\n✓ Plot saved as PNG: {plot_path_png}")
            
            fig.savefig(plot_path_pdf, bbox_inches='tight', facecolor='white')
            print(f"✓ Plot saved as PDF: {plot_path_pdf}")
        except Exception as e:
            print(f"\n⚠ Warning: Could not save plot: {e}")
        
        # Also save results data to text file
        results_path = os.path.join(OUTPUT_FOLDER, f"{filename}_results.txt")
        try:
            with open(results_path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("SOUND SOURCE LOCALIZATION RESULTS\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                if angles:
                    avg_azimuth = np.mean([a['azimuth'] for a in angles])
                    avg_elevation = np.mean([a['elevation'] for a in angles])
                    distance = np.linalg.norm(avg_position)
                    
                    f.write(f"Source Position (average over {len(positions)} detections):\n")
                    f.write(f"  X: {avg_position[0]:.6f} m\n")
                    f.write(f"  Y: {avg_position[1]:.6f} m\n")
                    f.write(f"  Z: {avg_position[2]:.6f} m\n\n")
                    f.write(f"Distance from array center: {distance:.6f} m\n\n")
                    f.write(f"Angles:\n")
                    f.write(f"  Azimuth: {avg_azimuth:.2f}°\n")
                    f.write(f"  Elevation: {avg_elevation:.2f}°\n\n")
                    f.write(f"Number of detections: {len(positions)}\n")
                else:
                    f.write(f"Position: ({avg_position[0]:.6f}, {avg_position[1]:.6f}, {avg_position[2]:.6f}) m\n")
                    f.write(f"Number of detections: {len(positions)}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("Microphone Positions:\n")
                for i, pos in enumerate(MIC_POSITIONS):
                    f.write(f"  MIC{i+1}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) m\n")
            
            print(f"✓ Results saved to: {results_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not save results file: {e}")
        
        print(f"\nAll files saved to folder: {OUTPUT_FOLDER}/")
        print("Plots displayed. Close the window to exit.")
        plt.show()
    
    def run(self):
        """Run the localization system (continuous mode)"""
        try:
            # Start audio capture
            self.audio_capture.start_capture()
            
            print("\n" + "=" * 60)
            print("Localization running... (Press Ctrl+C to stop)")
            print("The program will run continuously until you close the window.")
            print("=" * 60 + "\n")
            
            # Setup plots
            self.fig = plt.figure(figsize=(16, 6))
            self.ax_2d = self.fig.add_subplot(121)
            self.ax_3d = self.fig.add_subplot(122, projection='3d')
            
            # Start animation (update every 100ms for smoother performance)
            # This runs continuously - no time limit!
            self.animation = FuncAnimation(self.fig, self.update_plots, interval=100, blit=False, cache_frame_data=False)  # type: ignore
            
            plt.tight_layout()
            plt.show()  # This blocks until window is closed
            
        except KeyboardInterrupt:
            print("\nStopping...")
        except Exception as e:
            print(f"\nError during execution: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.audio_capture.stop_capture()
            plt.close()
            print("Localization stopped.")


def test_channels():
    """Test function to check which channels are receiving audio"""
    print("\n" + "=" * 60)
    print("Channel Test - Speak into microphone for 3 seconds")
    print("=" * 60)
    
    audio = pyaudio.PyAudio()
    device_index = None
    
    # Find ReSpeaker
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if 'respeaker' in info['name'].lower() or info['maxInputChannels'] >= 6:
            device_index = i
            print(f"Using device: {info['name']} (index {i})")
            print(f"  Max input channels: {info['maxInputChannels']}")
            print(f"  Default sample rate: {info['defaultSampleRate']}")
            break
    
    if device_index is None:
        print("ReSpeaker not found!")
        audio.terminate()
        return
    
    try:
        active_channels_found = []
        best_config = None
        
        # Try different channel configurations
        test_configs = [
            (6, "6 channels (default)"),
            (4, "4 channels"),
            (2, "2 channels (stereo)"),
        ]
        
        for num_test_channels, config_name in test_configs:
            print(f"\n--- Testing {config_name} ---")
            try:
                stream = audio.open(
                    format=pyaudio.paInt16,
                    channels=num_test_channels,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK_SIZE
                )
                
                print("Recording for 2 seconds... (speak now!)")
                channel_energies = [0.0] * num_test_channels
                num_frames = int(SAMPLE_RATE / CHUNK_SIZE * 2)
                
                for _ in range(num_frames):
                    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    audio_array = audio_array.reshape(CHUNK_SIZE, num_test_channels)
                    audio_float = audio_array.astype(np.float32) / 32768.0
                    
                    for ch in range(num_test_channels):
                        channel_energies[ch] += float(np.mean(audio_float[:, ch]**2))
                
                stream.stop_stream()
                stream.close()
                
                print("Channel Energy Levels:")
                current_active = []
                for ch in range(num_test_channels):
                    avg_energy = channel_energies[ch] / num_frames
                    status = "✓ ACTIVE" if avg_energy > 0.0001 else "✗ SILENT"
                    print(f"  Channel {ch}: {avg_energy:.6f} {status}")
                    if avg_energy > 0.0001:
                        current_active.append(ch)
                
                if len(current_active) >= 4:
                    print(f"\n✓ Found {len(current_active)} active channels!")
                    active_channels_found = current_active
                    best_config = (num_test_channels, current_active)
                    break
                elif len(current_active) > len(active_channels_found):
                    active_channels_found = current_active
                    best_config = (num_test_channels, current_active)
                    
            except Exception as e:
                print(f"  Error with {config_name}: {e}")
                continue
        
        # Final summary
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print("=" * 60)
        
        if len(active_channels_found) >= 4:
            print(f"✓ Good! Found 4+ active channels: {active_channels_found[:4]}")
            print(f"\nUpdate ACTIVE_CHANNELS in code to: {active_channels_found[:4]}")
            print(f"Update NUM_CHANNELS if needed (currently: {NUM_CHANNELS})")
        elif len(active_channels_found) > 0:
            print(f"⚠ Warning: Only found {len(active_channels_found)} active channel(s): {active_channels_found}")
            print(f"\nFor triangulation, you need at least 4 microphones.")
            print(f"Possible issues:")
            print(f"  1. ReSpeaker may need firmware/driver update")
            print(f"  2. Device may be in wrong mode (check ReSpeaker documentation)")
            print(f"  3. Some microphones may not be working")
            print(f"  4. Try using 'arecord' command to test: arecord -D hw:1,0 -f S16_LE -r 16000 -c 6 test.wav")
        else:
            print("✗ No active channels found! Check microphone connection.")
        
        print(f"\nCurrently configured to use channels: {ACTIVE_CHANNELS}")
        
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        audio.terminate()


def main():
    """Main function"""
    print("=" * 60)
    print("Sound Source Localization - Triangulation Method")
    print("ReSpeaker Microphone Array (Channels 1-4)")
    print("=" * 60)
    print(f"Microphone positions (m):")
    for i, pos in enumerate(MIC_POSITIONS):
        print(f"  MIC{i+1}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
    print(f"Microphone spacing: {MIC_SPACING*1000:.1f} mm")
    print("=" * 60)
    
    # Check for command line arguments
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test-channels':
            test_channels()
            return
        elif sys.argv[1] == '--continuous':
            print("\nTip: Run with '--test-channels' to verify which channels are active")
            print("Starting continuous localization...\n")
            localizer = SoundLocalization()
            localizer.run()
            return
        elif sys.argv[1] == '--duration':
            duration = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
        else:
            # Try to parse as duration number
            try:
                duration = float(sys.argv[1])
            except ValueError:
                duration = 6.0
    else:
        duration = 6.0  # Default 6 seconds
    
    print("\nTip: Run with '--test-channels' to verify which channels are active")
    print(f"Starting {duration}-second recording mode...\n")
    
    localizer = SoundLocalization()
    localizer.record_and_localize(duration=duration)


if __name__ == "__main__":
    main()

