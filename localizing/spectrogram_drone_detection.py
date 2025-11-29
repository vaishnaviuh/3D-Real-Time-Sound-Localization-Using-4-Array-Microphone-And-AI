#!/usr/bin/env python3
"""
Analyze spectrograms to detect drone sounds, then triangulate and create maps.
Uses spectrogram pattern analysis to identify drone signatures.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import soundfile as sf
from scipy.signal import spectrogram, find_peaks
from scipy.signal import butter, filtfilt
import os
import time
import sys
from src.kml_parser import get_sensor_positions_xyz
from src.triangulation import TriangulationEngine

def apply_bandpass_filter(audio_data, sample_rate, low_freq=300.0, high_freq=4000.0):
    """Apply bandpass filter to enhance drone signal detection."""
    print(f"🔧 Applying bandpass filter ({low_freq}-{high_freq} Hz)...")
    sys.stdout.flush()
    
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    if low >= 1.0 or high >= 1.0:
        low = min(low, 0.9)
        high = min(high, 0.95)
    
    if low <= 0 or high <= low:
        low = 100.0 / nyquist
        high = 3000.0 / nyquist
    
    try:
        b, a = butter(2, [low, high], btype='band')
        filtered_data = np.zeros_like(audio_data)
        
        for ch in range(audio_data.shape[0]):
            if ch % 5 == 0:
                print(f"      Filtering channel {ch+1}/{audio_data.shape[0]}...")
                sys.stdout.flush()
            try:
                filtered_data[ch] = filtfilt(b, a, audio_data[ch])
            except:
                filtered_data[ch] = audio_data[ch]
        
        print(f"   ✅ Filtering complete")
        sys.stdout.flush()
        return filtered_data
    except Exception as e:
        print(f"   ❌ Filter failed: {e}, using original")
        return audio_data

def analyze_spectrogram_for_drone(signal, fs, min_fund_hz=80.0, max_fund_hz=800.0):
    """
    Analyze spectrogram to detect drone-like patterns.
    
    Returns:
        (has_drone, fundamental_freq, confidence_score, spectrogram_data)
    """
    if signal.size < fs:  # Need at least 1 second
        return False, 0.0, 0.0, None
    
    # Compute spectrogram with good resolution
    nperseg = min(4096, max(2048, fs // 2))
    noverlap = nperseg // 2
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, 
                            window='hann', scaling='density')
    
    # Convert to dB
    Sxx_db = 10.0 * np.log10(Sxx + 1e-12)
    
    # Focus on drone frequency band
    band_mask = (f >= min_fund_hz) & (f <= max_fund_hz)
    if not np.any(band_mask):
        return False, 0.0, 0.0, None
    
    f_band = f[band_mask]
    Sxx_band = Sxx_db[band_mask, :]
    
    # Average over time to get stable spectrum
    spectrum_avg = np.mean(Sxx_band, axis=1)
    
    # Find peaks in the spectrum
    global_max = np.max(spectrum_avg)
    min_height = global_max - 18.0  # More lenient: 18 dB below peak
    peaks, peak_props = find_peaks(spectrum_avg, height=min_height, distance=2)
    
    if peaks.size == 0:
        return False, 0.0, 0.0, None
    
    candidate_freqs = f_band[peaks]
    candidate_mags = spectrum_avg[peaks]
    
    # Analyze each candidate for harmonic structure
    best_fund = 0.0
    best_score = 0.0
    best_harmonic_count = 0
    
    for fund_idx, fund in enumerate(candidate_freqs):
        harmonic_count = 0
        harmonic_energy = candidate_mags[fund_idx]
        harmonic_freqs = []
        
        # Check for harmonics at 2x, 3x, 4x, 5x, 6x
        for h in range(2, 7):
            harmonic_freq = fund * h
            if harmonic_freq > f[-1]:
                break
            
            # Find closest frequency bin
            idx = np.argmin(np.abs(f - harmonic_freq))
            tolerance = fund * 0.1  # 10% tolerance
            if abs(f[idx] - harmonic_freq) <= tolerance:
                # Check if this frequency has significant energy
                peak_energy = spectrum_avg[np.argmin(np.abs(f_band - harmonic_freq))]
                if peak_energy > (global_max - 25.0):  # 25 dB below peak
                    harmonic_count += 1
                    harmonic_energy += peak_energy
                    harmonic_freqs.append(harmonic_freq)
        
        # Score based on number of harmonics and their strength
        # Also check temporal consistency
        temporal_consistency = 0.0
        if len(harmonic_freqs) > 0:
            # Check if harmonics are consistent over time
            for hf in harmonic_freqs[:3]:  # Check first 3 harmonics
                hf_idx = np.argmin(np.abs(f - hf))
                if hf_idx < Sxx_db.shape[0]:
                    # Calculate coefficient of variation (lower = more consistent)
                    time_series = Sxx_db[hf_idx, :]
                    if np.std(time_series) > 0:
                        cv = np.std(time_series) / (np.mean(time_series) + 1e-12)
                        temporal_consistency += 1.0 / (1.0 + cv)  # Higher is better
        
        # Combined score
        score = harmonic_count * 10.0 + harmonic_energy * 0.1 + temporal_consistency * 5.0
        
        if harmonic_count >= 2:  # At least 2 harmonics
            if score > best_score:
                best_score = score
                best_fund = fund
                best_harmonic_count = harmonic_count
    
    # Determine if drone is detected
    has_drone = (best_harmonic_count >= 2 and best_fund > 0.0 and best_score > 5.0)
    
    return has_drone, best_fund, best_score, (f, t, Sxx_db)

def combine_cluster_channels(audio_chunk):
    """Combine channels 16-20 into single cluster signal."""
    if audio_chunk.shape[0] < 20:
        return audio_chunk
    result = np.zeros((16, audio_chunk.shape[1]))
    result[:15] = audio_chunk[:15]
    cluster_signal = np.mean(audio_chunk[15:20], axis=0)
    result[15] = cluster_signal
    return result

def process_with_spectrogram_analysis(wav_path, kml_path, chunk_duration=10.0):
    """
    Process audio using spectrogram analysis to detect drone sounds.
    """
    print("🎵 Spectrogram-Based Drone Detection & Triangulation")
    print("=" * 70)
    
    # Load audio
    print(f"📂 Loading: {wav_path}")
    sys.stdout.flush()
    audio_data, sample_rate = sf.read(wav_path, always_2d=True)
    audio_data = audio_data.T  # Shape: (channels, samples)
    
    duration_s = audio_data.shape[1] / sample_rate
    print(f"   ✅ Loaded: {audio_data.shape[0]} channels, {duration_s:.1f}s ({duration_s/60:.1f} min), {sample_rate}Hz")
    sys.stdout.flush()
    
    # Apply filter
    filtered_audio = apply_bandpass_filter(audio_data, sample_rate, low_freq=300.0, high_freq=4000.0)
    
    # Load sensors
    print(f"📍 Loading sensors: {kml_path}")
    sys.stdout.flush()
    sensor_names, sensor_positions_array = get_sensor_positions_xyz(
        kml_path, add_opposite_sensors=True
    )
    print(f"   ✅ Loaded: {len(sensor_names)} sensors")
    sys.stdout.flush()
    
    # Initialize triangulation
    print("🔧 Initializing triangulation engine...")
    sys.stdout.flush()
    triangulation_engine = TriangulationEngine(
        sensor_positions=sensor_positions_array,
        speed_of_sound=343.0
    )
    
    # Process in chunks
    chunk_samples = int(chunk_duration * sample_rate)
    total_samples = filtered_audio.shape[1]
    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
    
    print(f"🔍 Processing: {num_chunks} chunks of {chunk_duration}s each")
    print(f"   Analyzing spectrograms to detect drone signatures...")
    sys.stdout.flush()
    
    detections = []
    start_time = time.time()
    
    for chunk_idx in range(num_chunks):
        start_sample = chunk_idx * chunk_samples
        end_sample = min(start_sample + chunk_samples, total_samples)
        chunk = filtered_audio[:, start_sample:end_sample]
        timestamp = start_sample / sample_rate
        
        if (chunk_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            progress = (chunk_idx + 1) / num_chunks * 100
            print(f"   📊 {progress:5.1f}% ({chunk_idx+1}/{num_chunks}) | {elapsed:.1f}s | Detections: {len(detections)}")
            sys.stdout.flush()
        
        # Combine cluster channels
        processed_chunk = combine_cluster_channels(chunk)
        
        # Analyze spectrogram on cluster channel
        cluster_signal = processed_chunk[15] if processed_chunk.shape[0] >= 16 else np.mean(processed_chunk, axis=0)
        
        # Try cluster channel first
        has_drone, fundamental, score, spec_data = analyze_spectrogram_for_drone(cluster_signal, sample_rate)
        
        # If not detected, try average of first 15 channels
        if not has_drone:
            avg_signal = np.mean(processed_chunk[:15], axis=0) if processed_chunk.shape[0] >= 15 else cluster_signal
            has_drone, fundamental, score, spec_data = analyze_spectrogram_for_drone(avg_signal, sample_rate)
        
        # If still not detected, try individual channels
        if not has_drone:
            for ch_idx in range(min(10, processed_chunk.shape[0])):
                ch_signal = processed_chunk[ch_idx]
                has_drone, fundamental, score, spec_data = analyze_spectrogram_for_drone(ch_signal, sample_rate)
                if has_drone:
                    break
        
        if not has_drone:
            continue  # Skip this chunk
        
        print(f"   ✅ Chunk {chunk_idx+1} (t={timestamp:.1f}s): Drone detected! (fund={fundamental:.1f} Hz, score={score:.1f})")
        sys.stdout.flush()
        
        # Triangulate this chunk
        try:
            result = triangulation_engine.triangulate_audio_chunk(
                processed_chunk.T,
                sample_rate,
                tdoa_method='gcc_phat',
                triangulation_method='robust'
            )
            
            if not result:
                continue
            
            # Lenient validation
            sensor_centroid = np.mean(sensor_positions_array, axis=0)
            distance_from_centroid = np.linalg.norm(result.position - sensor_centroid)
            
            is_valid = (
                result.confidence >= 0.05 and
                result.residual_error < 300.0 and
                distance_from_centroid < 1500.0 and
                -100.0 <= result.position[2] <= 400.0 and
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
                    'timestamp': timestamp,
                    'position': result.position.copy(),
                    'confidence': result.confidence,
                    'residual_error': result.residual_error,
                    'method': result.method,
                    'num_sensors_used': result.num_sensors_used,
                    'fundamental_freq': fundamental,
                    'spectrogram_score': score,
                    'nearest_sensor_idx': nearest_sensor_idx,
                    'nearest_sensor_name': sensor_names[nearest_sensor_idx],
                    'nearest_sensor_distance': nearest_sensor_distance
                }
                detections.append(detection)
                
                if len(detections) % 5 == 0:
                    print(f"      ✅ t={timestamp:6.1f}s: ({result.position[0]:6.1f}, {result.position[1]:6.1f}, {result.position[2]:6.1f}) "
                          f"conf={result.confidence:.3f} nearest={sensor_names[nearest_sensor_idx]}")
                    sys.stdout.flush()
        
        except Exception as e:
            if chunk_idx % 20 == 0:
                print(f"      ❌ Chunk {chunk_idx+1}: {str(e)[:40]}...")
                sys.stdout.flush()
    
    elapsed_total = time.time() - start_time
    print(f"\n🎯 Analysis Complete!")
    print(f"   ⏱️  Total time: {elapsed_total:.1f}s")
    print(f"   📊 Processed: {num_chunks} chunks")
    print(f"   🎯 Detections: {len(detections)}")
    sys.stdout.flush()
    
    if detections:
        confidences = [d['confidence'] for d in detections]
        print(f"   📈 Confidence: avg={np.mean(confidences):.3f}, range={np.min(confidences):.3f}-{np.max(confidences):.3f}")
        sys.stdout.flush()
    
    return sensor_names, sensor_positions_array.tolist(), detections

def create_maps(sensor_names, sensor_positions, detections, output_dir="plots"):
    """Create 2D and 3D maps with sensor-to-drone markers."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🗺️  Creating Maps...")
    sys.stdout.flush()
    
    fig = plt.figure(figsize=(24, 12))
    plt.style.use('dark_background')
    
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122, projection='3d')
    
    # Separate sensor types
    original_indices = [i for i, name in enumerate(sensor_names) if "CLUSTER" not in name]
    cluster_indices = [i for i, name in enumerate(sensor_names) if "CLUSTER" in name]
    
    positions_arr = np.array(sensor_positions, dtype=float)
    
    orig_x = [sensor_positions[i][0] for i in original_indices]
    orig_y = [sensor_positions[i][1] for i in original_indices]
    orig_z = [sensor_positions[i][2] for i in original_indices]
    
    cluster_pos = sensor_positions[cluster_indices[0]] if cluster_indices else None
    
    # === 2D PLOT ===
    plt.sca(ax1)
    
    # Plot sensors
    if original_indices:
        ax1.scatter(orig_x, orig_y, c='cyan', s=150, marker='o', 
                   label=f'Sensors 1-15 ({len(original_indices)})', 
                   alpha=0.9, edgecolors='white', linewidth=2, zorder=5)
        
        for i, idx in enumerate(original_indices):
            pos = sensor_positions[idx]
            name = sensor_names[idx]
            try:
                num = int(name.split()[-1])
                label = f"S{num:02d}"
            except:
                label = name
            ax1.annotate(label, (pos[0], pos[1]), xytext=(5, 5),
                        textcoords='offset points', fontsize=9, color='white',
                        weight='bold', ha='left', va='bottom', zorder=6)
    
    if cluster_pos is not None:
        ax1.scatter([cluster_pos[0]], [cluster_pos[1]], c='orange', s=400, 
                   marker='*', label='Cluster', alpha=0.9,
                   edgecolors='white', linewidth=2, zorder=5)
        ax1.annotate('CLUSTER', (cluster_pos[0], cluster_pos[1]),
                    xytext=(15, 15), textcoords='offset points',
                    fontsize=12, color='orange', weight='bold', ha='center', zorder=6)
    
    # Plot detections
    if detections:
        det_x = [d['position'][0] for d in detections]
        det_y = [d['position'][1] for d in detections]
        det_z = [d['position'][2] for d in detections]
        confidences = [d['confidence'] for d in detections]
        timestamps = [d['timestamp'] for d in detections]
        
        sorted_indices = np.argsort(timestamps)
        
        # Draw lines from sensors to detections
        for det_idx in sorted_indices:
            detection = detections[det_idx]
            det_pos = detection['position']
            nearest_idx = detection['nearest_sensor_idx']
            sensor_pos = positions_arr[nearest_idx]
            
            ax1.plot([sensor_pos[0], det_pos[0]], [sensor_pos[1], det_pos[1]],
                    'yellow', linestyle='--', alpha=0.4, linewidth=1.5, zorder=2)
            
            if nearest_idx in original_indices:
                ax1.scatter([sensor_pos[0]], [sensor_pos[1]], c='lime', s=200,
                           marker='s', alpha=0.6, edgecolors='white', linewidth=1.5, zorder=4)
        
        scatter2d = ax1.scatter(det_x, det_y, c=confidences, s=[60 + c * 100 for c in confidences],
                              cmap='viridis', alpha=0.8, edgecolors='white', linewidth=1.5,
                              label=f'Detections ({len(detections)})', zorder=3)
        
        for i, (x, y, det) in enumerate(zip(det_x, det_y, detections)):
            nearest_name = det['nearest_sensor_name']
            try:
                num = int(nearest_name.split()[-1]) if "CLUSTER" not in nearest_name else "C"
                label = f"→S{num:02d}" if num != "C" else "→CL"
            except:
                label = "→?"
            ax1.annotate(label, (x, y), xytext=(8, 8), textcoords='offset points',
                        fontsize=7, color='yellow', weight='bold', ha='left', va='bottom', zorder=6)
        
        if len(detections) > 1:
            sorted_detections = [detections[i] for i in sorted_indices]
            path_x = [d['position'][0] for d in sorted_detections]
            path_y = [d['position'][1] for d in sorted_detections]
            ax1.plot(path_x, path_y, 'yellow', alpha=0.6, linewidth=2.5,
                    label='Flight Path', zorder=1)
            ax1.scatter(path_x[0], path_y[0], c='green', s=200, marker='^',
                       label='First', edgecolors='white', linewidth=2, zorder=4)
            ax1.scatter(path_x[-1], path_y[-1], c='red', s=200, marker='v',
                       label='Last', edgecolors='white', linewidth=2, zorder=4)
    else:
        scatter2d = None
    
    ax1.set_xlabel('X (meters)', fontsize=13, weight='bold')
    ax1.set_ylabel('Y (meters)', fontsize=13, weight='bold')
    ax1.set_title('2D Map - Spectrogram-Based Detection', fontsize=15, pad=20, weight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    if detections:
        all_x = orig_x + det_x
        all_y = orig_y + det_y
        margin = max(100, (max(all_x) - min(all_x)) * 0.2)
        ax1.set_xlim([min(all_x) - margin, max(all_x) + margin])
        ax1.set_ylim([min(all_y) - margin, max(all_y) + margin])
    
    # === 3D PLOT ===
    if original_indices:
        ax2.scatter(orig_x, orig_y, orig_z, c='cyan', s=100, marker='o',
                   label=f'Sensors ({len(original_indices)})',
                   alpha=0.9, edgecolors='white', linewidth=1.5)
    
    if cluster_pos is not None:
        ax2.scatter([cluster_pos[0]], [cluster_pos[1]], [cluster_pos[2]],
                   c='orange', s=300, marker='*', label='Cluster',
                   alpha=0.9, edgecolors='white', linewidth=2)
    
    if detections:
        for det_idx in sorted_indices:
            detection = detections[det_idx]
            det_pos = detection['position']
            nearest_idx = detection['nearest_sensor_idx']
            sensor_pos = positions_arr[nearest_idx]
            
            ax2.plot([sensor_pos[0], det_pos[0]], 
                    [sensor_pos[1], det_pos[1]],
                    [sensor_pos[2], det_pos[2]],
                    'yellow', linestyle='--', alpha=0.5, linewidth=2)
            
            if nearest_idx in original_indices:
                ax2.scatter([sensor_pos[0]], [sensor_pos[1]], [sensor_pos[2]],
                           c='lime', s=150, marker='s', alpha=0.7,
                           edgecolors='white', linewidth=1.5)
        
        scatter3d = ax2.scatter(det_x, det_y, det_z, c=confidences,
                               s=[80 + c * 120 for c in confidences],
                               cmap='viridis', alpha=0.9, edgecolors='white',
                               linewidth=1.5, label=f'Detections ({len(detections)})')
        
        if len(detections) > 1:
            sorted_detections = [detections[i] for i in sorted_indices]
            path_x = [d['position'][0] for d in sorted_detections]
            path_y = [d['position'][1] for d in sorted_detections]
            path_z = [d['position'][2] for d in sorted_detections]
            ax2.plot(path_x, path_y, path_z, 'yellow', alpha=0.7,
                    linewidth=3, label='3D Path')
    else:
        scatter3d = None
    
    ax2.set_xlabel('X (meters)', fontsize=12, weight='bold')
    ax2.set_ylabel('Y (meters)', fontsize=12, weight='bold')
    ax2.set_zlabel('Height (meters)', fontsize=12, weight='bold')
    ax2.set_title('3D Map - Spectrogram-Based Detection', fontsize=15, pad=20, weight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.view_init(elev=20, azim=45)
    ax2.grid(True, alpha=0.3)
    
    if scatter2d is not None:
        plt.colorbar(scatter2d, ax=ax1, label='Confidence', shrink=0.8)
    if scatter3d is not None:
        plt.colorbar(scatter3d, ax=ax2, label='Confidence', shrink=0.8)
    
    info_lines = [f"Spectrogram-Based Detection Results:", f"• Total detections: {len(detections)}"]
    if detections:
        info_lines.append(f"• Time span: {min(timestamps):.1f}s - {max(timestamps):.1f}s")
        info_lines.append(f"• Avg confidence: {np.mean(confidences):.3f}")
    info_text = "\n".join(info_lines)
    plt.figtext(0.02, 0.02, info_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8))
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "spectrogram_drone_detection_map.png")
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
        sensor_names, sensor_positions, detections = process_with_spectrogram_analysis(
            wav_path, kml_path, chunk_duration=10.0
        )
        
        output_file = create_maps(sensor_names, sensor_positions, detections)
        
        print(f"\n🎉 Analysis Complete!")
        print(f"   📊 Detections: {len(detections)}")
        print(f"   🗺️  Map: {output_file}")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()

if __name__ == "__main__":
    main()

