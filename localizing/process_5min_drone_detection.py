#!/usr/bin/env python3
"""
Process full 5-minute audio recording:
1. Apply bandpass filter for drone detection
2. Check for harmonic signatures similar to drone sounds
3. Process as 5-minute chunks
4. Triangulate chunks with drone signatures
5. Plot comprehensive 2D/3D maps with sensor-to-drone markers
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import soundfile as sf
from scipy.signal import spectrogram, find_peaks, butter, filtfilt
import os
import time
import sys
from src.kml_parser import get_sensor_positions_xyz
from src.triangulation import TriangulationEngine

def apply_bandpass_filter(audio_data, sample_rate, low_freq=300.0, high_freq=4000.0):
    """Apply bandpass filter to enhance drone signal detection (300-4000 Hz)."""
    print(f"🔧 Applying bandpass filter ({low_freq}-{high_freq} Hz)...")
    sys.stdout.flush()
    
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Ensure frequencies are within valid range
    if low >= 1.0 or high >= 1.0:
        print(f"   ⚠️  Frequency too high for sample rate. Adjusting...")
        low = min(low, 0.9)
        high = min(high, 0.95)
    
    if low <= 0 or high <= low:
        print(f"   ⚠️  Invalid frequency range. Using default 100-3000 Hz...")
        low = 100.0 / nyquist
        high = 3000.0 / nyquist
    
    try:
        # Use 2nd order Butterworth filter
        b, a = butter(2, [low, high], btype='band')
        
        filtered_data = np.zeros_like(audio_data)
        
        for ch in range(audio_data.shape[0]):
            if ch % 5 == 0:  # Progress indicator
                print(f"      Filtering channel {ch+1}/{audio_data.shape[0]}...")
                sys.stdout.flush()
            
            # Apply filter with error handling
            try:
                filtered_data[ch] = filtfilt(b, a, audio_data[ch])
            except Exception as e:
                print(f"      ⚠️  Filter failed for channel {ch+1}, using original: {e}")
                filtered_data[ch] = audio_data[ch]
        
        print(f"   ✅ Filtering complete")
        sys.stdout.flush()
        return filtered_data
        
    except Exception as e:
        print(f"   ❌ Filter design failed: {e}")
        print(f"   📊 Using original audio data without filtering")
        return audio_data

def detect_drone_harmonics(signal, fs, min_fund_hz=80.0, max_fund_hz=800.0, min_harmonics=2):
    """
    Detect drone-like harmonic signatures in a signal.
    More lenient detection - looks for at least 2 harmonics.
    
    Returns:
        (has_drone, fundamental_freq, harmonic_score): Detection result
    """
    if signal.size < fs:  # Need at least 1 second
        return False, 0.0, 0.0
    
    # Compute spectrogram
    nperseg = min(4096, max(1024, fs // 4))
    noverlap = nperseg // 2
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # Average over time to get stable spectrum
    spectrum = np.mean(Sxx, axis=1)
    spectrum_db = 10.0 * np.log10(spectrum + 1e-12)
    
    # Focus on plausible drone fundamental band
    band_mask = (f >= min_fund_hz) & (f <= max_fund_hz)
    if not np.any(band_mask):
        return False, 0.0, 0.0
    
    f_band = f[band_mask]
    spec_band_db = spectrum_db[band_mask]
    
    # Find prominent peaks - more lenient threshold
    global_max = np.max(spec_band_db)
    min_height = global_max - 15.0  # More lenient: 15 dB instead of 12 dB
    peaks, peak_props = find_peaks(spec_band_db, height=min_height, distance=2)  # Smaller distance
    
    if peaks.size == 0:
        return False, 0.0, 0.0
    
    candidate_freqs = f_band[peaks]
    candidate_mags = spec_band_db[peaks]
    
    # For each candidate fundamental, check for harmonics
    best_fund = 0.0
    best_count = 0
    best_score = 0.0
    
    for fund_idx, fund in enumerate(candidate_freqs):
        harmonic_count = 0
        harmonic_energy = candidate_mags[fund_idx]
        
        # Check for harmonics at 2x, 3x, 4x, 5x
        for h in range(2, 6):
            harmonic_freq = fund * h
            if harmonic_freq > f[-1]:
                break
            
            # Find closest frequency bin
            idx = np.argmin(np.abs(f - harmonic_freq))
            tolerance = fund * 0.08  # More lenient tolerance: 8% instead of 5%
            if abs(f[idx] - harmonic_freq) <= tolerance:
                peak_energy = spectrum_db[idx]
                # More lenient threshold: 25 dB instead of 20 dB
                if peak_energy > (global_max - 25.0):
                    harmonic_count += 1
                    harmonic_energy += peak_energy
        
        # Score: number of harmonics * total harmonic energy
        score = harmonic_count * (harmonic_energy + 10.0)
        
        # More lenient: require at least 2 harmonics (instead of 3)
        if harmonic_count >= min_harmonics - 1:  # -1 because we count from 2nd harmonic
            if score > best_score:
                best_score = score
                best_count = harmonic_count
                best_fund = fund
    
    # More lenient: require at least 2 harmonics
    if best_count >= min_harmonics - 1 and best_fund > 0.0:
        return True, best_fund, best_score
    
    return False, 0.0, 0.0

def combine_cluster_channels(audio_chunk):
    """Combine channels 16-20 into single cluster signal for triangulation."""
    if audio_chunk.shape[0] < 20:
        return audio_chunk
    
    # Create 16-channel output (15 individual + 1 cluster)
    result = np.zeros((16, audio_chunk.shape[1]))
    result[:15] = audio_chunk[:15]
    cluster_signal = np.mean(audio_chunk[15:20], axis=0)
    result[15] = cluster_signal
    return result

def process_5min_chunks_with_drone_detection(wav_path, kml_path, chunk_duration=300.0):
    """
    Process audio in 5-minute chunks, detect drone harmonics, and triangulate.
    
    Returns:
        (sensor_names, sensor_positions, detections)
    """
    print("🎵 Full 5-Minute Audio - Drone Harmonic Detection & Triangulation")
    print("=" * 70)
    
    # Load audio file
    print(f"📂 Loading: {wav_path}")
    sys.stdout.flush()
    audio_data, sample_rate = sf.read(wav_path, always_2d=True)
    audio_data = audio_data.T  # Shape: (channels, samples)
    
    duration_s = audio_data.shape[1] / sample_rate
    print(f"   ✅ Loaded: {audio_data.shape[0]} channels, {duration_s:.1f}s ({duration_s/60:.1f} min), {sample_rate}Hz")
    sys.stdout.flush()
    
    # Apply bandpass filter
    filtered_audio = apply_bandpass_filter(audio_data, sample_rate, low_freq=300.0, high_freq=4000.0)
    
    # Load sensor positions
    print(f"📍 Loading sensors: {kml_path}")
    sys.stdout.flush()
    sensor_names, sensor_positions_array = get_sensor_positions_xyz(
        kml_path, add_opposite_sensors=True
    )
    print(f"   ✅ Loaded: {len(sensor_names)} sensors (15 original + 1 cluster)")
    sys.stdout.flush()
    
    # Initialize triangulation engine
    print("🔧 Initializing triangulation engine...")
    sys.stdout.flush()
    triangulation_engine = TriangulationEngine(
        sensor_positions=sensor_positions_array,
        speed_of_sound=343.0
    )
    
    # Process as 5-minute chunks
    chunk_samples = int(chunk_duration * sample_rate)
    total_samples = filtered_audio.shape[1]
    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples  # Ceiling division
    
    print(f"🔍 Processing: {num_chunks} chunk(s) of {chunk_duration/60:.1f} minutes each")
    sys.stdout.flush()
    
    # Analysis parameters
    confidence_threshold = 0.15
    
    print(f"   Thresholds: confidence>{confidence_threshold}")
    print(f"   Processing: All chunks with signal energy > 0.0005 (harmonic detection optional)")
    sys.stdout.flush()
    
    # Process chunks
    detections = []
    start_time = time.time()
    
    for chunk_idx in range(num_chunks):
        start_sample = chunk_idx * chunk_samples
        end_sample = min(start_sample + chunk_samples, total_samples)
        chunk = filtered_audio[:, start_sample:end_sample]
        
        chunk_duration_actual = (end_sample - start_sample) / sample_rate
        timestamp = start_sample / sample_rate
        
        print(f"\n📊 Processing chunk {chunk_idx+1}/{num_chunks} ({chunk_duration_actual:.1f}s, t={timestamp:.1f}s)...")
        sys.stdout.flush()
        
        # Combine cluster channels
        processed_chunk = combine_cluster_channels(chunk)
        
        # Check signal energy - process all chunks with sufficient signal
        rms_levels = np.sqrt(np.mean(processed_chunk**2, axis=1))
        max_energy = np.max(rms_levels)
        active_channels = np.sum(rms_levels > 0.0005)  # Very low threshold
        
        print(f"   📊 Signal energy: {max_energy:.6f}, active channels: {active_channels}")
        sys.stdout.flush()
        
        if max_energy < 0.0005 or active_channels < 3:
            print(f"   🔇 Low signal energy - skipping")
            sys.stdout.flush()
            continue
        
        # Try to detect harmonics (optional - for info only)
        cluster_signal = processed_chunk[15] if processed_chunk.shape[0] >= 16 else np.mean(processed_chunk, axis=0)
        has_drone, fundamental, harmonic_score = detect_drone_harmonics(cluster_signal, sample_rate)
        
        if has_drone:
            print(f"   ✅ Drone harmonics detected! (fundamental={fundamental:.1f} Hz, score={harmonic_score:.1f})")
        else:
            print(f"   ⚠️  No clear harmonics, but processing anyway (signal energy: {max_energy:.6f})")
            fundamental = 0.0
            harmonic_score = 0.0
        
        sys.stdout.flush()
        
        # Split large chunk into smaller sub-chunks for triangulation
        sub_chunk_duration = 10.0  # 10-second sub-chunks
        sub_chunk_samples = int(sub_chunk_duration * sample_rate)
        num_sub_chunks = (end_sample - start_sample) // sub_chunk_samples
        
        print(f"   🔍 Triangulating {num_sub_chunks} sub-chunks...")
        sys.stdout.flush()
        
        for sub_idx in range(num_sub_chunks):
            sub_start = start_sample + sub_idx * sub_chunk_samples
            sub_end = min(sub_start + sub_chunk_samples, end_sample)
            sub_chunk = filtered_audio[:, sub_start:sub_end]
            sub_timestamp = sub_start / sample_rate
            
            processed_sub = combine_cluster_channels(sub_chunk)
            
            try:
                # Perform triangulation
                result = triangulation_engine.triangulate_audio_chunk(
                    processed_sub.T,  # Shape: (samples, channels)
                    sample_rate,
                    tdoa_method='gcc_phat',
                    triangulation_method='robust'
                )
                
                if not result:
                    continue
                
                # Validate result
                sensor_centroid = np.mean(sensor_positions_array, axis=0)
                distance_from_centroid = np.linalg.norm(result.position - sensor_centroid)
                
                # More lenient validation - accept more results
                is_valid = (
                    result.confidence >= 0.05 and  # Lower confidence threshold
                    result.residual_error < 200.0 and  # Allow higher error
                    distance_from_centroid < 1000.0 and  # Allow farther distances
                    -50.0 <= result.position[2] <= 300.0 and  # Wider height range
                    not np.any(np.isnan(result.position)) and
                    not np.any(np.isinf(result.position))
                )
                
                if is_valid:
                    # Find nearest sensor
                    distances_to_sensors = np.linalg.norm(
                        sensor_positions_array - result.position, axis=1
                    )
                    nearest_sensor_idx = np.argmin(distances_to_sensors)
                    nearest_sensor_distance = distances_to_sensors[nearest_sensor_idx]
                    
                    detection = {
                        'timestamp': sub_timestamp,
                        'position': result.position.copy(),
                        'confidence': result.confidence,
                        'residual_error': result.residual_error,
                        'method': result.method,
                        'num_sensors_used': result.num_sensors_used,
                        'fundamental_freq': fundamental,
                        'harmonic_score': harmonic_score,
                        'nearest_sensor_idx': nearest_sensor_idx,
                        'nearest_sensor_name': sensor_names[nearest_sensor_idx],
                        'nearest_sensor_distance': nearest_sensor_distance
                    }
                    detections.append(detection)
                    
                    if len(detections) % 10 == 0 or result.confidence > 0.3:
                        print(f"      ✅ t={sub_timestamp:6.1f}s: ({result.position[0]:6.1f}, {result.position[1]:6.1f}, {result.position[2]:6.1f}) "
                              f"conf={result.confidence:.3f} nearest={sensor_names[nearest_sensor_idx]} "
                              f"({nearest_sensor_distance:.1f}m)")
                        sys.stdout.flush()
            
            except Exception as e:
                if sub_idx % 10 == 0:
                    print(f"      ❌ Sub-chunk {sub_idx+1}: {str(e)[:40]}...")
                    sys.stdout.flush()
    
    # Final results
    elapsed_total = time.time() - start_time
    print(f"\n🎯 Analysis Complete!")
    print(f"   ⏱️  Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"   📊 Processed: {num_chunks} chunk(s)")
    print(f"   🎯 Drone detections: {len(detections)}")
    sys.stdout.flush()
    
    if detections:
        confidences = [d['confidence'] for d in detections]
        print(f"   📈 Confidence: avg={np.mean(confidences):.3f}, range={np.min(confidences):.3f}-{np.max(confidences):.3f}")
        print(f"   📍 Detections span: {min([d['timestamp'] for d in detections]):.1f}s - {max([d['timestamp'] for d in detections]):.1f}s")
        sys.stdout.flush()
    
    return sensor_names, sensor_positions_array.tolist(), detections

def create_comprehensive_maps(sensor_names, sensor_positions, detections, output_dir="plots"):
    """
    Create comprehensive 2D and 3D maps showing sensors, detections, and markers.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🗺️  Creating Comprehensive Maps (2D + 3D)...")
    sys.stdout.flush()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(24, 12))
    plt.style.use('dark_background')
    
    # 2D plot (left side)
    ax1 = plt.subplot(121)
    
    # 3D plot (right side)
    ax2 = plt.subplot(122, projection='3d')
    
    # Separate sensor types
    original_indices = [i for i, name in enumerate(sensor_names) if "CLUSTER" not in name]
    cluster_indices = [i for i, name in enumerate(sensor_names) if "CLUSTER" in name]
    
    positions_arr = np.array(sensor_positions, dtype=float)
    
    # Extract coordinates
    orig_x = [sensor_positions[i][0] for i in original_indices]
    orig_y = [sensor_positions[i][1] for i in original_indices]
    orig_z = [sensor_positions[i][2] for i in original_indices]
    
    cluster_pos = sensor_positions[cluster_indices[0]] if cluster_indices else None
    
    # === 2D PLOT ===
    plt.sca(ax1)
    
    # Plot sensors in 2D
    if original_indices:
        ax1.scatter(orig_x, orig_y, c='cyan', s=150, marker='o', 
                   label=f'Sensors 1-15 ({len(original_indices)})', 
                   alpha=0.9, edgecolors='white', linewidth=2, zorder=5)
        
        # Add sensor labels
        for i, idx in enumerate(original_indices):
            pos = sensor_positions[idx]
            name = sensor_names[idx]
            try:
                num = int(name.split()[-1])
                label = f"S{num:02d}"
            except (ValueError, IndexError):
                label = name
            
            ax1.annotate(label, (pos[0], pos[1]), xytext=(5, 5),
                        textcoords='offset points', fontsize=9, color='white',
                        weight='bold', ha='left', va='bottom', zorder=6)
    
    # Plot cluster in 2D
    if cluster_pos is not None:
        ax1.scatter([cluster_pos[0]], [cluster_pos[1]], c='orange', s=400, 
                   marker='*', label='Cluster (Ch 16-20)', alpha=0.9,
                   edgecolors='white', linewidth=2, zorder=5)
        ax1.annotate('CLUSTER', (cluster_pos[0], cluster_pos[1]),
                    xytext=(15, 15), textcoords='offset points',
                    fontsize=12, color='orange', weight='bold', ha='center', zorder=6)
    
    # Plot detections and markers
    if detections:
        det_x = [d['position'][0] for d in detections]
        det_y = [d['position'][1] for d in detections]
        det_z = [d['position'][2] for d in detections]
        confidences = [d['confidence'] for d in detections]
        timestamps = [d['timestamp'] for d in detections]
        
        # Sort by timestamp
        sorted_indices = np.argsort(timestamps)
        
        # Draw lines from nearest sensors to detections in 2D
        for det_idx in sorted_indices:
            detection = detections[det_idx]
            det_pos = detection['position']
            nearest_idx = detection['nearest_sensor_idx']
            sensor_pos = positions_arr[nearest_idx]
            
            # Draw line from sensor to detection
            ax1.plot([sensor_pos[0], det_pos[0]], [sensor_pos[1], det_pos[1]],
                    'yellow', linestyle='--', alpha=0.4, linewidth=1.5, zorder=2)
            
            # Mark nearest sensor with a highlight
            if nearest_idx in original_indices:
                ax1.scatter([sensor_pos[0]], [sensor_pos[1]], c='lime', s=200,
                           marker='s', alpha=0.6, edgecolors='white', linewidth=1.5, zorder=4)
        
        # Plot detections
        scatter2d = ax1.scatter(det_x, det_y, c=confidences, s=[60 + c * 100 for c in confidences],
                              cmap='viridis', alpha=0.8, edgecolors='white', linewidth=1.5,
                              label=f'Drone Detections ({len(detections)})', zorder=3)
        
        # Annotate detections with nearest sensor info
        for i, (x, y, det) in enumerate(zip(det_x, det_y, detections)):
            nearest_name = det['nearest_sensor_name']
            try:
                num = int(nearest_name.split()[-1]) if "CLUSTER" not in nearest_name else "C"
                label = f"→S{num:02d}" if num != "C" else "→CL"
            except:
                label = "→?"
            ax1.annotate(label, (x, y), xytext=(8, 8), textcoords='offset points',
                        fontsize=7, color='yellow', weight='bold', ha='left', va='bottom', zorder=6)
        
        # Show flight path
        if len(detections) > 1:
            sorted_detections = [detections[i] for i in sorted_indices]
            path_x = [d['position'][0] for d in sorted_detections]
            path_y = [d['position'][1] for d in sorted_detections]
            ax1.plot(path_x, path_y, 'yellow', alpha=0.6, linewidth=2.5,
                    label='Flight Path', zorder=1)
            
            # Mark start and end
            ax1.scatter(path_x[0], path_y[0], c='green', s=200, marker='^',
                       label='First Detection', edgecolors='white', linewidth=2, zorder=4)
            ax1.scatter(path_x[-1], path_y[-1], c='red', s=200, marker='v',
                       label='Last Detection', edgecolors='white', linewidth=2, zorder=4)
    else:
        scatter2d = None
        print("   ⚠️  No detections to plot - showing sensors only")
        sys.stdout.flush()
    
    # 2D formatting
    ax1.set_xlabel('X (meters)', fontsize=13, weight='bold')
    ax1.set_ylabel('Y (meters)', fontsize=13, weight='bold')
    ax1.set_title('2D Map - Drone Detections with Nearest Sensor Markers', 
                  fontsize=15, pad=20, weight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Set reasonable limits
    all_x = orig_x + (det_x if detections else [])
    all_y = orig_y + (det_y if detections else [])
    if all_x and all_y:
        margin = max(100, (max(all_x) - min(all_x)) * 0.2, (max(all_y) - min(all_y)) * 0.2)
        ax1.set_xlim([min(all_x) - margin, max(all_x) + margin])
        ax1.set_ylim([min(all_y) - margin, max(all_y) + margin])
    
    # === 3D PLOT ===
    
    # Plot sensors in 3D
    if original_indices:
        ax2.scatter(orig_x, orig_y, orig_z, c='cyan', s=100, marker='o',
                   label=f'Sensors 1-15 ({len(original_indices)})',
                   alpha=0.9, edgecolors='white', linewidth=1.5)
        
        # Add sensor labels in 3D
        for i, idx in enumerate(original_indices):
            pos = sensor_positions[idx]
            name = sensor_names[idx]
            try:
                num = int(name.split()[-1])
                label = f"S{num:02d}"
            except (ValueError, IndexError):
                label = name
            
            ax2.text(pos[0], pos[1], pos[2] + 1, label, fontsize=8,
                    color='white', weight='bold')
    
    # Plot cluster in 3D
    if cluster_pos is not None:
        ax2.scatter([cluster_pos[0]], [cluster_pos[1]], [cluster_pos[2]],
                   c='orange', s=300, marker='*', label='Cluster (Ch 16-20)',
                   alpha=0.9, edgecolors='white', linewidth=2)
        ax2.text(cluster_pos[0], cluster_pos[1], cluster_pos[2] + 2, 'CLUSTER',
                fontsize=11, color='orange', weight='bold', ha='center')
    
    # Draw lines from sensors to detections in 3D
    if detections:
        for det_idx in sorted_indices:
            detection = detections[det_idx]
            det_pos = detection['position']
            nearest_idx = detection['nearest_sensor_idx']
            sensor_pos = positions_arr[nearest_idx]
            
            # Draw line from sensor to detection
            ax2.plot([sensor_pos[0], det_pos[0]], 
                    [sensor_pos[1], det_pos[1]],
                    [sensor_pos[2], det_pos[2]],
                    'yellow', linestyle='--', alpha=0.5, linewidth=2)
            
            # Mark nearest sensor
            if nearest_idx in original_indices:
                ax2.scatter([sensor_pos[0]], [sensor_pos[1]], [sensor_pos[2]],
                           c='lime', s=150, marker='s', alpha=0.7,
                           edgecolors='white', linewidth=1.5)
        
        # Plot detections in 3D
        scatter3d = ax2.scatter(det_x, det_y, det_z, c=confidences,
                               s=[80 + c * 120 for c in confidences],
                               cmap='viridis', alpha=0.9, edgecolors='white',
                               linewidth=1.5, label=f'Drone Detections ({len(detections)})')
        
        # Show 3D flight path
        if len(detections) > 1:
            sorted_detections = [detections[i] for i in sorted_indices]
            path_x = [d['position'][0] for d in sorted_detections]
            path_y = [d['position'][1] for d in sorted_detections]
            path_z = [d['position'][2] for d in sorted_detections]
            ax2.plot(path_x, path_y, path_z, 'yellow', alpha=0.7,
                    linewidth=3, label='3D Flight Path')
            
            # Mark start and end in 3D
            ax2.scatter(path_x[0], path_y[0], path_z[0], c='green', s=250,
                       marker='^', label='First Detection', edgecolors='white', linewidth=2)
            ax2.scatter(path_x[-1], path_y[-1], path_z[-1], c='red', s=250,
                       marker='v', label='Last Detection', edgecolors='white', linewidth=2)
    else:
        scatter3d = None
    
    # 3D formatting
    ax2.set_xlabel('X (meters)', fontsize=12, weight='bold')
    ax2.set_ylabel('Y (meters)', fontsize=12, weight='bold')
    ax2.set_zlabel('Height (meters)', fontsize=12, weight='bold')
    ax2.set_title('3D Map - Sensor-to-Drone Markers', fontsize=15, pad=20, weight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.view_init(elev=20, azim=45)
    ax2.grid(True, alpha=0.3)
    
    # Set reasonable 3D limits
    if all_x and all_y and (det_z if detections else []):
        margin_3d = max(100, (max(all_x) - min(all_x)) * 0.2)
        height_margin = max(50, (max(det_z) - min(det_z)) * 0.3) if detections else 100
        ax2.set_xlim([min(all_x) - margin_3d, max(all_x) + margin_3d])
        ax2.set_ylim([min(all_y) - margin_3d, max(all_y) + margin_3d])
        ax2.set_zlim([min(0, min(det_z) - 20), max(det_z) + height_margin])
    
    # Add colorbar
    if scatter2d is not None:
        plt.colorbar(scatter2d, ax=ax1, label='Confidence', shrink=0.8)
    if scatter3d is not None:
        plt.colorbar(scatter3d, ax=ax2, label='Confidence', shrink=0.8)
    
    # Add info box
    info_lines = [
        f"Full 5-Minute Audio Analysis - Drone Harmonic Detection:",
        f"• Total detections: {len(detections)}",
    ]
    
    if detections:
        info_lines.append(f"• Time span: {min(timestamps):.1f}s - {max(timestamps):.1f}s ({max(timestamps)-min(timestamps):.1f}s)")
        info_lines.append(f"• Avg confidence: {np.mean(confidences):.3f}")
        info_lines.append(f"• Best confidence: {max(confidences):.3f}")
        heights = [d['position'][2] for d in detections]
        info_lines.append(f"• Height range: {min(heights):.1f}m - {max(heights):.1f}m")
        
        # Count detections per nearest sensor
        sensor_counts = {}
        for d in detections:
            name = d['nearest_sensor_name']
            sensor_counts[name] = sensor_counts.get(name, 0) + 1
        
        info_lines.append(f"• Most active sensor: {max(sensor_counts.items(), key=lambda x: x[1])[0]} ({max(sensor_counts.values())} detections)")
    else:
        info_lines.append("• No drone detections found")
        info_lines.append("• Check filter settings and harmonic detection parameters")
    
    info_text = "\n".join(info_lines)
    plt.figtext(0.02, 0.02, info_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, "5min_drone_detection_map.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"   💾 Saved: {output_file}")
    sys.stdout.flush()
    
    plt.show()
    
    return output_file

def main():
    """Main function."""
    wav_path = "multi-20251122-141610-627897594.wav"
    kml_path = "Sensor-Locations-BOP-Dharma.kml"
    
    if not os.path.exists(wav_path):
        print(f"❌ WAV file not found: {wav_path}")
        return
    
    if not os.path.exists(kml_path):
        print(f"❌ KML file not found: {kml_path}")
        return
    
    try:
        # Process full 5-minute audio
        sensor_names, sensor_positions, detections = process_5min_chunks_with_drone_detection(
            wav_path, kml_path, chunk_duration=300.0  # 5 minutes = 300 seconds
        )
        
        # Create comprehensive maps
        output_file = create_comprehensive_maps(sensor_names, sensor_positions, detections)
        
        print(f"\n🎉 Full Analysis Complete!")
        print(f"   📊 Detections: {len(detections)} drone sound detections")
        print(f"   🗺️  Map: {output_file}")
        sys.stdout.flush()
        
        if detections:
            print(f"\n📈 Detection Summary:")
            sensor_counts = {}
            for d in detections:
                name = d['nearest_sensor_name']
                sensor_counts[name] = sensor_counts.get(name, 0) + 1
            
            print(f"   Detections per sensor:")
            for name, count in sorted(sensor_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"      {name}: {count} detections")
            sys.stdout.flush()
        else:
            print(f"   ⚠️  No valid detections found")
            print(f"   💡 Try adjusting harmonic detection parameters or filter settings")
            sys.stdout.flush()
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()

if __name__ == "__main__":
    main()

