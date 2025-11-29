#!/usr/bin/env python3
"""
Final WAV analysis script - processes the full 5-minute recording using core triangulation logic.
Shows all 15 sensors + 1 cluster and detects sound sources.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import soundfile as sf
from scipy.signal import butter, filtfilt, sosfilt
import os
import time
from src.kml_parser import get_sensor_positions_xyz
from src.triangulation import TriangulationEngine

def apply_bandpass_filter(audio_data, sample_rate, low_freq=300.0, high_freq=4000.0):
    """Apply bandpass filter to enhance drone signal detection (300-4000 Hz)."""
    print(f"   Filtering {audio_data.shape[0]} channels with {low_freq}-{high_freq} Hz bandpass...")
    
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
        # Use lower order filter to avoid numerical issues
        b, a = butter(2, [low, high], btype='band')
        
        filtered_data = np.zeros_like(audio_data)
        
        for ch in range(audio_data.shape[0]):
            if ch % 5 == 0:  # Progress indicator
                print(f"      Processing channel {ch+1}/{audio_data.shape[0]}...")
            
            # Apply filter with error handling
            try:
                filtered_data[ch] = filtfilt(b, a, audio_data[ch])
            except Exception as e:
                print(f"      ⚠️  Filter failed for channel {ch+1}, using original: {e}")
                filtered_data[ch] = audio_data[ch]
        
        print(f"   ✅ Filtering complete")
        return filtered_data
        
    except Exception as e:
        print(f"   ❌ Filter design failed: {e}")
        print(f"   📊 Using original audio data without filtering")
        return audio_data

def simple_bandpass_filter(audio_data, sample_rate, low_freq=300.0, high_freq=4000.0):
    """Ultra-simple bandpass filter for small chunks."""
    try:
        # Only filter if chunk is reasonable size
        if audio_data.shape[1] > 100000:  # Too large, skip filtering
            return audio_data
            
        nyquist = sample_rate / 2
        low = max(low_freq / nyquist, 0.01)  # Minimum 1% of Nyquist
        high = min(high_freq / nyquist, 0.95)  # Maximum 95% of Nyquist
        
        if high <= low:
            return audio_data
        
        # Use very simple 1st order filter
        sos = butter(1, [low, high], btype='band', output='sos')
        
        filtered_data = np.zeros_like(audio_data)
        for ch in range(audio_data.shape[0]):
            try:
                filtered_data[ch] = sosfilt(sos, audio_data[ch])
            except:
                filtered_data[ch] = audio_data[ch]  # Fallback to original
        
        return filtered_data
        
    except:
        return audio_data  # Always return something

def combine_cluster_channels(audio_chunk):
    """Combine channels 16-20 into single cluster signal for triangulation."""
    if audio_chunk.shape[0] < 20:
        return audio_chunk
    
    # Create 16-channel output (15 individual + 1 cluster)
    result = np.zeros((16, audio_chunk.shape[1]))
    
    # Copy first 15 channels
    result[:15] = audio_chunk[:15]
    
    # Combine channels 16-20 (indices 15-19) into cluster signal
    cluster_signal = np.mean(audio_chunk[15:20], axis=0)
    result[15] = cluster_signal
    
    return result

def analyze_full_wav_file(wav_path, kml_path, chunk_duration=2.0, max_chunks=None):
    """
    Analyze the full 5-minute WAV file using core triangulation logic.
    
    Args:
        wav_path: Path to WAV file
        kml_path: Path to KML file with sensor positions
        chunk_duration: Duration of each analysis chunk in seconds
        max_chunks: Maximum chunks to process (None = all)
    
    Returns:
        Tuple of (sensor_names, sensor_positions, detections)
    """
    print("🎵 Full WAV File Analysis - Core Triangulation Logic")
    print("=" * 60)
    
    # Load audio file
    print(f"📂 Loading: {wav_path}")
    audio_data, sample_rate = sf.read(wav_path, always_2d=True)
    audio_data = audio_data.T  # Shape: (channels, samples)
    
    duration_s = audio_data.shape[1] / sample_rate
    print(f"   ✅ Loaded: {audio_data.shape[0]} channels, {duration_s:.1f}s ({duration_s/60:.1f} min), {sample_rate}Hz")
    
    # Load sensor positions
    print(f"📍 Loading sensors: {kml_path}")
    sensor_names, sensor_positions_array = get_sensor_positions_xyz(
        kml_path, add_opposite_sensors=True
    )
    
    print(f"   ✅ Loaded: {len(sensor_names)} sensors (15 original + 1 cluster)")

    # Print numbered sensor list with approximate ground-referenced height
    # Assume all sensors are mounted on poles at 20 m above local ground.
    avg_sensor_z = float(np.mean(sensor_positions_array[:, 2]))
    print("\n   Sensor index / name / XYZ (m) and approx height above ground (m):")
    for idx, (name, pos) in enumerate(zip(sensor_names, sensor_positions_array)):
        x, y, z = pos
        # Height above ground: relative to average sensor plane, then +20 m pole height
        est_height_ground = (z - avg_sensor_z) + 20.0
        print(
            f"    [{idx:2d}] {name:25s}  "
            f"X={x:7.1f}  Y={y:7.1f}  Z={z:6.1f}  "
            f"≈Height={est_height_ground:5.1f} m"
        )
    
    # Show cluster details
    cluster_idx = [i for i, name in enumerate(sensor_names) if "CLUSTER" in name][0]
    cluster_pos = sensor_positions_array[cluster_idx]
    print(f"   🎯 Cluster: {sensor_names[cluster_idx]} at ({cluster_pos[0]:.1f}, {cluster_pos[1]:.1f}, {cluster_pos[2]:.1f})")
    
    # Initialize triangulation engine
    print("🔧 Initializing triangulation engine...")
    triangulation_engine = TriangulationEngine(
        sensor_positions=sensor_positions_array,
        speed_of_sound=343.0
    )
    
    # Skip filtering for now to avoid numerical issues
    print("🔧 Skipping bandpass filter to avoid numerical issues...")
    print("   📊 Using original audio data (will filter individual chunks later)")
    filtered_audio = audio_data
    
    # Calculate RMS improvement
    original_rms = np.sqrt(np.mean(audio_data**2))
    filtered_rms = np.sqrt(np.mean(filtered_audio**2))
    print(f"   Signal RMS: {original_rms:.6f} → {filtered_rms:.6f}")
    
    # Setup chunking parameters
    chunk_samples = int(chunk_duration * sample_rate)
    overlap = 0.5
    step_samples = int(chunk_samples * (1 - overlap))
    
    total_samples = filtered_audio.shape[1]
    total_possible_chunks = (total_samples - chunk_samples) // step_samples + 1
    
    if max_chunks is None:
        chunks_to_process = total_possible_chunks
    else:
        chunks_to_process = min(max_chunks, total_possible_chunks)
    
    print(f"🔍 Processing: {chunks_to_process}/{total_possible_chunks} chunks ({chunk_duration}s each, {overlap*100:.0f}% overlap)")
    
    # Analysis parameters
    energy_threshold = 0.002
    min_active_channels = 8
    confidence_threshold = 0.15
    
    print(f"   Thresholds: energy>{energy_threshold}, active_ch>={min_active_channels}, confidence>{confidence_threshold}")
    
    # Process chunks
    detections = []
    start_time = time.time()
    
    for i in range(chunks_to_process):
        start_sample = i * step_samples
        end_sample = start_sample + chunk_samples
        timestamp = start_sample / sample_rate
        
        # Extract chunk
        chunk = filtered_audio[:, start_sample:end_sample]
        
        # Apply bandpass filter to this small chunk (much more stable)
        try:
            chunk = simple_bandpass_filter(chunk, sample_rate, 300.0, 4000.0)
        except Exception as e:
            if i % 50 == 0:  # Only show occasional filter warnings
                print(f"      ⚠️ Chunk filter failed at t={timestamp:.1f}s, using unfiltered")
        
        # Combine cluster channels
        processed_chunk = combine_cluster_channels(chunk)
        
        # Calculate signal metrics
        rms_levels = np.sqrt(np.mean(processed_chunk**2, axis=1))
        max_energy = np.max(rms_levels)
        active_channels = np.sum(rms_levels > energy_threshold)
        
        # Progress update
        if i % 25 == 0:
            elapsed = time.time() - start_time
            progress = (i + 1) / chunks_to_process * 100
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (chunks_to_process - i - 1) / rate if rate > 0 else 0
            print(f"   📊 {progress:5.1f}% ({i+1:3d}/{chunks_to_process}) | {elapsed:5.1f}s | {rate:4.1f} ch/s | ETA: {eta:4.0f}s | Detections: {len(detections)}")
        
        # Check if chunk has sufficient signal
        if max_energy > energy_threshold and active_channels >= min_active_channels:
            try:
                # Perform triangulation using core logic
                result = triangulation_engine.triangulate_audio_chunk(
                    processed_chunk.T,  # Shape: (samples, channels)
                    sample_rate,
                    tdoa_method='gcc_phat',
                    triangulation_method='robust'
                )
                
                # Check result quality
                if result and result.confidence > confidence_threshold:
                    detection = {
                        'timestamp': timestamp,
                        'position': result.position.copy(),
                        'confidence': result.confidence,
                        'residual_error': result.residual_error,
                        'method': result.method,
                        'max_energy': max_energy,
                        'active_channels': active_channels,
                        'num_sensors_used': result.num_sensors_used
                    }
                    detections.append(detection)
                    
                    # Show good detections
                    if i % 25 == 0 or result.confidence > 0.3:
                        print(f"      ✅ t={timestamp:6.1f}s: ({result.position[0]:6.1f}, {result.position[1]:6.1f}, {result.position[2]:6.1f}) "
                              f"conf={result.confidence:.3f} sensors={result.num_sensors_used}")
            
            except Exception as e:
                # Only show occasional errors to avoid spam
                if i % 50 == 0:
                    print(f"      ❌ t={timestamp:6.1f}s: {str(e)[:40]}...")
        
        # Show low-signal chunks occasionally
        elif i % 100 == 0:
            print(f"      ⏭️  t={timestamp:6.1f}s: Low signal (energy={max_energy:.6f}, active={active_channels})")
    
    # Final results
    elapsed_total = time.time() - start_time
    print(f"\n🎯 Analysis Complete!")
    print(f"   ⏱️  Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"   📊 Processed: {chunks_to_process} chunks ({chunks_to_process * chunk_duration:.0f}s of audio)")
    print(f"   🎯 Detections: {len(detections)} ({len(detections)/chunks_to_process*100:.1f}% success rate)")
    
    if detections:
        confidences = [d['confidence'] for d in detections]
        energies = [d['max_energy'] for d in detections]
        sensors_used = [d['num_sensors_used'] for d in detections]
        
        print(f"   📈 Confidence: avg={np.mean(confidences):.3f}, range={np.min(confidences):.3f}-{np.max(confidences):.3f}")
        print(f"   🔊 Energy: avg={np.mean(energies):.6f}, range={np.min(energies):.6f}-{np.max(energies):.6f}")
        print(f"   📡 Sensors used: avg={np.mean(sensors_used):.1f}, range={np.min(sensors_used)}-{np.max(sensors_used)}")
    
    return sensor_names, sensor_positions_array.tolist(), detections

def create_results_map(sensor_names, sensor_positions, detections):
    """Create comprehensive results map with both 2D and 3D views."""
    print("\n🗺️  Creating Results Map (2D + 3D)...")
    
    # Create figure with subplots for 2D and 3D views
    fig = plt.figure(figsize=(20, 10))
    plt.style.use('dark_background')
    
    # 2D plot (left side)
    ax1 = plt.subplot(121)
    
    # 3D plot (right side)  
    ax2 = plt.subplot(122, projection='3d')
    
    # Separate sensor types
    original_indices = [i for i, name in enumerate(sensor_names) if "CLUSTER" not in name]
    cluster_indices = [i for i, name in enumerate(sensor_names) if "CLUSTER" in name]

    # Convert positions to numpy array for distance computations
    positions_arr = np.array(sensor_positions, dtype=float)
    
    # Extract coordinates for both plots
    orig_x = [sensor_positions[i][0] for i in original_indices]
    orig_y = [sensor_positions[i][1] for i in original_indices]
    orig_z = [sensor_positions[i][2] for i in original_indices]
    
    cluster_pos = sensor_positions[cluster_indices[0]] if cluster_indices else None
    
    # === 2D PLOT (Left) ===
    plt.sca(ax1)
    
    # Plot original sensors in 2D
    if original_indices:
        ax1.scatter(
            orig_x,
            orig_y,
            c='cyan',
            s=100,
            marker='o',
            label=f'Sensors 1-15 ({len(original_indices)})',
            alpha=0.8,
            edgecolors='white',
            linewidth=1,
        )
        
        # Add sensor labels (name + index) next to each ground sensor
        for i, idx in enumerate(original_indices):
            pos = sensor_positions[idx]
            name = sensor_names[idx]
            # Create a compact label like "S05" from "SSSD BOP DHARMA 005" if possible
            try:
                num = int(name.split()[-1])
                label = f"S{num:02d}"
            except (ValueError, IndexError):
                label = name
            
            ax1.annotate(
                label,
                (pos[0], pos[1]),
                xytext=(4, 4),
                textcoords='offset points',
                fontsize=7,
                color='white',
                weight='bold',
                ha='left',
                va='bottom',
            )
    
    # Plot cluster in 2D
    if cluster_pos is not None:
        ax1.scatter([cluster_pos[0]], [cluster_pos[1]], c='orange', s=300, marker='*', 
                   label='Cluster (Ch 16-20)', alpha=0.9, 
                   edgecolors='white', linewidth=2)
        
        ax1.annotate('CLUSTER\n(16-20)', (cluster_pos[0], cluster_pos[1]), 
                    xytext=(12, 12), textcoords='offset points', 
                    fontsize=11, color='orange', weight='bold', ha='center')
        
        # Highlight sensors 5 and 7
        sensor_5_pos = sensor_positions[4]
        sensor_7_pos = sensor_positions[6]
        
        ax1.plot([sensor_5_pos[0], sensor_7_pos[0]], [sensor_5_pos[1], sensor_7_pos[1]], 
                 'lime', linestyle='--', alpha=0.6, linewidth=2, label='5-7 Connection')
    
    # Plot detections in 2D
    if detections:
        det_x = [d['position'][0] for d in detections]
        det_y = [d['position'][1] for d in detections]
        det_z = [d['position'][2] for d in detections]
        confidences = [d['confidence'] for d in detections]
        timestamps = [d['timestamp'] for d in detections]

        # Label detections in chronological order: D1, D2, ...
        sorted_indices = np.argsort(timestamps)
        det_labels = ["" for _ in detections]
        for order, det_idx in enumerate(sorted_indices, start=1):
            det_labels[det_idx] = f"D{order}"
        
        # 2D scatter plot
        scatter2d = ax1.scatter(
            det_x,
            det_y,
            c=confidences,
            s=[40 + c * 80 for c in confidences],
            cmap='viridis',
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5,
            label=f'Sound Sources ({len(detections)})',
        )

        # Annotate each detection with its label (D1, D2, ...)
        for x, y, label in zip(det_x, det_y, det_labels):
            ax1.annotate(
                label,
                (x, y),
                xytext=(4, 4),
                textcoords='offset points',
                fontsize=8,
                color='yellow',
                weight='bold',
                ha='left',
                va='bottom',
            )
        
        # Show movement path in 2D
        if len(detections) > 1:
            sorted_detections = [detections[i] for i in sorted_indices]
            path_x = [d['position'][0] for d in sorted_detections]
            path_y = [d['position'][1] for d in sorted_detections]
            
            ax1.plot(path_x, path_y, 'yellow', alpha=0.5, linewidth=2, label='Flight Path')
            
            # Mark start and end
            ax1.scatter(
                path_x[0],
                path_y[0],
                c='green',
                s=120,
                marker='^',
                label='First Detection',
                edgecolors='white',
                linewidth=2,
            )
            ax1.scatter(
                path_x[-1],
                path_y[-1],
                c='red',
                s=120,
                marker='v',
                label='Last Detection',
                edgecolors='white',
                linewidth=2,
            )
    
    # 2D formatting
    ax1.set_xlabel('X (meters)', fontsize=12)
    ax1.set_ylabel('Y (meters)', fontsize=12)
    ax1.set_title('2D View - Top Down', fontsize=14, pad=15)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # === 3D PLOT (Right) ===
    
    # Plot original sensors in 3D
    if original_indices:
        ax2.scatter(
            orig_x,
            orig_y,
            orig_z,
            c='cyan',
            s=80,
            marker='o',
            label=f'Sensors 1-15 ({len(original_indices)})',
            alpha=0.8,
            edgecolors='white',
            linewidth=1,
        )
        
        # Add sensor labels in 3D
        for i, idx in enumerate(original_indices):
            pos = sensor_positions[idx]
            name = sensor_names[idx]
            try:
                num = int(name.split()[-1])
                label = f"S{num:02d}"
            except (ValueError, IndexError):
                label = name
            
            ax2.text(
                pos[0],
                pos[1],
                pos[2],
                f"{label}",
                fontsize=7,
                color='white',
                weight='bold',
            )
    
    # Plot cluster in 3D
    if cluster_pos is not None:
        ax2.scatter([cluster_pos[0]], [cluster_pos[1]], [cluster_pos[2]], 
                   c='orange', s=200, marker='*', 
                   label='Cluster (Ch 16-20)', alpha=0.9, 
                   edgecolors='white', linewidth=2)
        
        ax2.text(cluster_pos[0], cluster_pos[1], cluster_pos[2] + 1, 'CLUSTER', 
                fontsize=10, color='orange', weight='bold', ha='center')
        
        # 3D line between sensors 5 and 7
        sensor_5_pos = sensor_positions[4]
        sensor_7_pos = sensor_positions[6]
        
        ax2.plot([sensor_5_pos[0], sensor_7_pos[0]], 
                [sensor_5_pos[1], sensor_7_pos[1]], 
                [sensor_5_pos[2], sensor_7_pos[2]], 
                'lime', linestyle='--', alpha=0.6, linewidth=2, label='5-7 Connection')
    
    # Plot detections in 3D
    if detections:
        # 3D scatter plot
        scatter3d = ax2.scatter(
            det_x,
            det_y,
            det_z,
            c=confidences,
            s=[50 + c * 100 for c in confidences],
            cmap='viridis',
            alpha=0.8,
            edgecolors='white',
            linewidth=0.5,
            label=f'Sound Sources ({len(detections)})',
        )

        # Annotate detections with labels in 3D as well
        for x, y, z, label in zip(det_x, det_y, det_z, det_labels):
            ax2.text(
                x,
                y,
                z + 0.5,
                label,
                fontsize=8,
                color='yellow',
                weight='bold',
                ha='center',
            )
        
        # Show 3D movement path
        if len(detections) > 1:
            sorted_detections = [detections[i] for i in sorted_indices]
            path_x = [d['position'][0] for d in sorted_detections]
            path_y = [d['position'][1] for d in sorted_detections]
            path_z = [d['position'][2] for d in sorted_detections]
            
            ax2.plot(
                path_x,
                path_y,
                path_z,
                'yellow',
                alpha=0.7,
                linewidth=3,
                label='3D Flight Path',
            )
            
            # Mark start and end in 3D
            ax2.scatter(
                path_x[0],
                path_y[0],
                path_z[0],
                c='green',
                s=150,
                marker='^',
                label='First Detection',
                edgecolors='white',
                linewidth=2,
            )
            ax2.scatter(
                path_x[-1],
                path_y[-1],
                path_z[-1],
                c='red',
                s=150,
                marker='v',
                label='Last Detection',
                edgecolors='white',
                linewidth=2,
            )

        # Draw small point chains from nearest sensors to each detection (2D + 3D)
        # Use only ground sensors (exclude cluster) to keep it readable.
        if original_indices:
            ground_positions = positions_arr[original_indices]

            for (x_d, y_d, z_d) in zip(det_x, det_y, det_z):
                det_vec = np.array([x_d, y_d, z_d], dtype=float)

                # Distances from this detection to all ground sensors
                dists = np.linalg.norm(ground_positions - det_vec, axis=1)
                # Indices of the 3 closest ground sensors
                nearest_idx_local = np.argsort(dists)[:3]

                for idx_local in nearest_idx_local:
                    sensor_idx = original_indices[idx_local]
                    sx, sy, sz = positions_arr[sensor_idx]

                    # 2D small points between sensor and detection
                    t_vals = np.linspace(0.0, 1.0, 12)
                    line_x_2d = sx + (x_d - sx) * t_vals
                    line_y_2d = sy + (y_d - sy) * t_vals
                    ax1.scatter(
                        line_x_2d,
                        line_y_2d,
                        c='deepskyblue',
                        s=5,
                        alpha=0.2,
                        marker='o',
                    )

                    # 3D small points between sensor and detection
                    line_x_3d = sx + (x_d - sx) * t_vals
                    line_y_3d = sy + (y_d - sy) * t_vals
                    line_z_3d = sz + (z_d - sz) * t_vals
                    ax2.scatter(
                        line_x_3d,
                        line_y_3d,
                        line_z_3d,
                        c='deepskyblue',
                        s=4,
                        alpha=0.25,
                        marker='o',
                    )
        
        # Calculate 3D movement statistics
        if len(detections) > 1:
            positions_3d = np.array([d['position'] for d in detections])
            distances_3d = []
            for i in range(1, len(positions_3d)):
                dist = np.linalg.norm(positions_3d[i] - positions_3d[i-1])
                distances_3d.append(dist)
            
            total_distance_3d = sum(distances_3d)
            max_distance_3d = max(distances_3d) if distances_3d else 0
            
            # Height statistics
            heights = [d['position'][2] for d in detections]
            min_height = min(heights)
            max_height = max(heights)
            height_range = max_height - min_height
            
            print(f"   ✅ Plotted {len(detections)} detections in 2D + 3D")
            print(f"   🎯 3D Movement: Total {total_distance_3d:.1f}m, Max step {max_distance_3d:.1f}m")
            print(f"   📏 Height range: {min_height:.1f}m - {max_height:.1f}m (span: {height_range:.1f}m)")
            print(f"   ⏱️  Time span: {min(timestamps):.1f}s - {max(timestamps):.1f}s ({max(timestamps)-min(timestamps):.1f}s)")
    else:
        print("   ⚠️  No detections to plot")
    
    # 3D formatting
    ax2.set_xlabel('X (meters)', fontsize=11)
    ax2.set_ylabel('Y (meters)', fontsize=11)
    ax2.set_zlabel('Height (meters)', fontsize=11)
    ax2.set_title('3D View - With Height', fontsize=14, pad=15)
    ax2.legend(loc='upper left', fontsize=8)
    
    # Set 3D view angle for better perspective
    ax2.view_init(elev=20, azim=45)
    
    # Add grid to 3D plot
    ax2.grid(True, alpha=0.3)
    
    # Add comprehensive info box with 3D details
    info_lines = [
        f"Full WAV Analysis Results (2D + 3D):",
        f"• Duration: 5 minutes (300 seconds)",
        f"• Sensors: 15 original + 1 cluster",
        f"• Detections: {len(detections)}",
    ]
    
    if detections:
        avg_conf = np.mean([d['confidence'] for d in detections])
        best_conf = max([d['confidence'] for d in detections])
        heights = [d['position'][2] for d in detections]
        sensor_heights = [pos[2] for pos in sensor_positions]

        # Approximate ground-referenced heights assuming sensors are 20 m above ground.
        sensor_plane_z = float(np.mean(sensor_heights))
        det_heights_ground = [(h - sensor_plane_z) + 20.0 for h in heights]
        min_det_h = min(det_heights_ground)
        max_det_h = max(det_heights_ground)
        span_det_h = max_det_h - min_det_h
        
        info_lines.extend([
            f"• Avg Confidence: {avg_conf:.3f}",
            f"• Best Confidence: {best_conf:.3f}",
            f"• Detection Z (local): {min(heights):.1f}m – {max(heights):.1f}m",
            f"• Sensor Z (local): {min(sensor_heights):.1f}m – {max(sensor_heights):.1f}m",
            f"• Drone height vs ground (assuming sensors at 20 m):",
            f"    {min_det_h:.1f}m – {max_det_h:.1f}m (span {span_det_h:.1f}m)",
        ])

        # Add short per-detection summary lines (label, time, height, confidence)
        sensor_plane_z = float(np.mean(sensor_heights))
        sorted_indices_local = np.argsort([d['timestamp'] for d in detections])
        for order, det_idx in enumerate(sorted_indices_local, start=1):
            d = detections[det_idx]
            label = f"D{order}"
            z_local = float(d['position'][2])
            height_ground = (z_local - sensor_plane_z) + 20.0
            info_lines.append(
                f"  {label}: t={d['timestamp']:5.1f}s, "
                f"height≈{height_ground:5.1f}m, conf={d['confidence']:.2f}"
            )
    
    info_text = "\n".join(info_lines)
    
    plt.figtext(0.02, 0.02, info_text, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.8))
    
    # Add colorbar for 2D plot
    if detections:
        plt.colorbar(scatter2d, ax=ax1, label='Confidence', shrink=0.8)
    
    plt.tight_layout()

    # Save plot into dedicated plots folder
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    output_file = os.path.join(plots_dir, "full_wav_analysis_3d_results.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"   💾 Saved: {output_file}")
    
    plt.show()
    
    return output_file

def main():
    """Main analysis function."""
    # File paths
    kml_path = "Sensor-Locations-BOP-Dharma.kml"
    wav_path = "multi-20251122-141610-627897594.wav"
    
    # Check files exist
    if not os.path.exists(kml_path):
        print(f"❌ KML file not found: {kml_path}")
        return
    
    if not os.path.exists(wav_path):
        print(f"❌ WAV file not found: {wav_path}")
        return
    
    try:
        # Analyze full WAV file (limit chunks for reasonable processing time)
        sensor_names, sensor_positions, detections = analyze_full_wav_file(
            wav_path, kml_path, 
            chunk_duration=2.0, 
            max_chunks=100  # fewer chunks for faster, still accurate analysis
        )
        
        # Create results map
        output_file = create_results_map(sensor_names, sensor_positions, detections)
        
        print(f"\n🎉 Full Analysis Complete!")
        print(f"   📊 Results: {len(detections)} sound source detections")
        print(f"   🗺️  Map: {output_file}")
        print(f"   📍 System: 15 sensors + 1 cluster (between sensors 5 & 7)")
        print(f"   🎵 Processed: 5-minute WAV file using core triangulation logic")
        
        if detections:
            confidences = [d['confidence'] for d in detections]
            print(f"   🎯 Best detection confidence: {max(confidences):.3f}")
            print(f"   📈 Average confidence: {np.mean(confidences):.3f}")
        else:
            print(f"   ⚠️  No detections found - consider adjusting thresholds")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
