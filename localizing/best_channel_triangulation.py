#!/usr/bin/env python3
"""
Find the channel with best harmonics, select best 120-second (2-minute) chunk, and create triangulation plots.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import soundfile as sf
from scipy.signal import spectrogram, find_peaks
import os
from src.kml_parser import get_sensor_positions_xyz
from src.triangulation import TriangulationEngine

def compute_harmonic_score(signal, fs, min_fund_hz=80.0, max_fund_hz=800.0):
    """
    Compute a harmonic score for a signal by detecting harmonic series.
    
    Returns:
        harmonic_score: Higher values indicate more prominent harmonics
        fundamental_freq: Detected fundamental frequency (if any)
        num_harmonics: Number of harmonics detected
    """
    if signal.size < fs:  # Need at least 1 second
        return 0.0, None, 0
    
    # Compute spectrogram
    nperseg = min(4096, max(1024, fs // 4))
    noverlap = nperseg // 2
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # Average over time to get stable spectrum
    spectrum = np.mean(Sxx, axis=1)
    spectrum_db = 10.0 * np.log10(spectrum + 1e-12)
    
    # Focus on plausible fundamental band
    band_mask = (f >= min_fund_hz) & (f <= max_fund_hz)
    if not np.any(band_mask):
        return 0.0, None, 0
    
    f_band = f[band_mask]
    spec_band_db = spectrum_db[band_mask]
    
    # Find prominent peaks
    global_max = np.max(spec_band_db)
    min_height = global_max - 12.0  # 12 dB below peak
    peaks, peak_props = find_peaks(spec_band_db, height=min_height, distance=3)
    
    if peaks.size == 0:
        return 0.0, None, 0
    
    candidate_freqs = f_band[peaks]
    candidate_mags = spec_band_db[peaks]
    
    # For each candidate fundamental, check for harmonics
    best_score = 0.0
    best_fund = None
    best_num_harmonics = 0
    
    for fund_idx, fund in enumerate(candidate_freqs):
        # Check for harmonics at 2x, 3x, 4x, 5x the fundamental
        harmonic_freqs = [fund * n for n in range(2, 6)]  # 2nd, 3rd, 4th, 5th harmonics
        tolerance_hz = fund * 0.05  # 5% tolerance
        
        num_harmonics_found = 0
        harmonic_energy = candidate_mags[fund_idx]
        
        for harmonic_freq in harmonic_freqs:
            if harmonic_freq > f[-1]:  # Beyond Nyquist
                break
            
            # Find closest frequency bin
            idx = np.argmin(np.abs(f - harmonic_freq))
            if abs(f[idx] - harmonic_freq) <= tolerance_hz:
                # Check if there's a peak near this frequency
                peak_energy = spectrum_db[idx]
                if peak_energy > (global_max - 20.0):  # At least 20 dB below peak
                    num_harmonics_found += 1
                    harmonic_energy += peak_energy
        
        # Score: number of harmonics * total harmonic energy
        score = num_harmonics_found * (harmonic_energy + 10.0)  # Add offset to avoid zero
        
        if score > best_score:
            best_score = score
            best_fund = fund
            best_num_harmonics = num_harmonics_found + 1  # +1 for fundamental
    
    return best_score, best_fund, best_num_harmonics

def find_best_channel_and_chunk(wav_path, chunk_duration=120.0):
    """
    Find the channel with best harmonics and the best 120-second (2-minute) chunk within that channel.
    
    Returns:
        best_channel_idx: Index of best channel
        best_start_sample: Start sample of best chunk
        best_score: Harmonic score
        best_fundamental: Fundamental frequency
        sample_rate: Sample rate
        audio_data: Full audio data (all channels)
    """
    import sys
    print(f"📂 Loading WAV file: {wav_path}")
    sys.stdout.flush()
    audio_data, sample_rate = sf.read(wav_path, always_2d=True)
    audio_data = audio_data.T  # Shape: (channels, samples)
    
    duration_s = audio_data.shape[1] / sample_rate
    num_channels = audio_data.shape[0]
    print(f"   ✅ Loaded: {num_channels} channels, {duration_s:.1f}s, {sample_rate}Hz")
    
    # Analyze each channel to find the one with best harmonics
    print(f"\n🔍 Analyzing all {num_channels} channels for harmonic content...")
    import sys
    sys.stdout.flush()
    
    chunk_samples = int(chunk_duration * sample_rate)
    step_samples = chunk_samples // 2  # 50% overlap
    
    best_overall_score = 0.0
    best_channel_idx = 0
    best_start_sample = 0
    best_fundamental = None
    best_num_harmonics = 0
    
    for ch_idx in range(num_channels):
        print(f"   Processing channel {ch_idx+1}/{num_channels}...", end='', flush=True)
        signal = audio_data[ch_idx]
        total_samples = len(signal)
        num_chunks = (total_samples - chunk_samples) // step_samples + 1
        
        # Limit chunks to process for speed (sample every Nth chunk)
        # For 120-second chunks with 50% overlap, we get ~2.5 chunks per minute
        # Process every 10th chunk to cover the file efficiently
        chunk_step = max(1, num_chunks // 20)  # Process up to 20 chunks per channel for speed
        
        channel_best_score = 0.0
        channel_best_start = 0
        channel_best_fund = None
        
        # Find best chunk in this channel
        for i in range(0, num_chunks, chunk_step):
            start_sample = i * step_samples
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk = signal[start_sample:end_sample]
            
            if len(chunk) < chunk_samples * 0.9:  # Skip incomplete chunks
                continue
            
            score, fundamental, num_harmonics = compute_harmonic_score(chunk, sample_rate)
            
            if score > channel_best_score:
                channel_best_score = score
                channel_best_start = start_sample
                channel_best_fund = fundamental
        
        import sys
        if channel_best_fund:
            print(f"   Channel {ch_idx+1:2d}: best score = {channel_best_score:8.1f}, fundamental = {channel_best_fund:.1f} Hz")
        else:
            print(f"   Channel {ch_idx+1:2d}: best score = {channel_best_score:8.1f}, fundamental = None")
        sys.stdout.flush()
        
        # Update overall best
        if channel_best_score > best_overall_score:
            best_overall_score = channel_best_score
            best_channel_idx = ch_idx
            best_start_sample = channel_best_start
            best_fundamental = channel_best_fund
    
    best_start_time = best_start_sample / sample_rate
    best_end_time = (best_start_sample + chunk_samples) / sample_rate
    
    print(f"\n🎯 Best channel and chunk found!")
    print(f"   Best channel: {best_channel_idx+1} (index {best_channel_idx})")
    print(f"   Best chunk: {best_start_time:.1f}s - {best_end_time:.1f}s")
    print(f"   Harmonic score: {best_overall_score:.1f}")
    print(f"   Fundamental frequency: {best_fundamental:.1f} Hz" if best_fundamental else "   Fundamental frequency: None")
    
    return best_channel_idx, best_start_sample, best_overall_score, best_fundamental, sample_rate, audio_data

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

def create_triangulation_plots(audio_chunk, start_time, kml_path, sample_rate, output_dir="plots"):
    """
    Create triangulation plots for the selected chunk.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📊 Creating triangulation plots...")
    
    # Load sensor positions
    print(f"📍 Loading sensors: {kml_path}")
    sensor_names, sensor_positions_array = get_sensor_positions_xyz(
        kml_path, add_opposite_sensors=True
    )
    print(f"   ✅ Loaded: {len(sensor_names)} sensors (15 original + 1 cluster)")
    
    # Combine cluster channels
    processed_chunk = combine_cluster_channels(audio_chunk)
    
    # Initialize triangulation engine
    print("🔧 Initializing triangulation engine...")
    triangulation_engine = TriangulationEngine(
        sensor_positions=sensor_positions_array,
        speed_of_sound=343.0
    )
    
    # Perform triangulation - split 120s chunk into smaller sub-chunks for better results
    print("🎯 Performing triangulation on 120-second (2-minute) chunk...")
    print("   Splitting into sub-chunks and validating each result...")
    
    # Split into sub-chunks of 10 seconds each for more attempts
    # For 120-second chunk, we can have up to 12 sub-chunks
    sub_chunk_duration = 10.0
    sub_chunk_samples = int(sub_chunk_duration * sample_rate)
    num_sub_chunks = min(12, processed_chunk.shape[1] // sub_chunk_samples)
    
    best_result = None
    best_score = -float('inf')
    
    for sub_idx in range(num_sub_chunks):
        start_sub = sub_idx * sub_chunk_samples
        end_sub = min(start_sub + sub_chunk_samples, processed_chunk.shape[1])
        sub_chunk = processed_chunk[:, start_sub:end_sub]
        
        try:
            result = triangulation_engine.triangulate_audio_chunk(
                sub_chunk.T,  # Shape: (samples, channels)
                sample_rate,
                tdoa_method='gcc_phat',
                triangulation_method='robust'
            )
            
            if not result:
                continue
            
            # Validate result - check if position is reasonable
            sensor_centroid = np.mean(sensor_positions_array, axis=0)
            sensor_std = np.std(sensor_positions_array, axis=0)
            
            # Calculate distance from sensor centroid
            distance_from_centroid = np.linalg.norm(result.position - sensor_centroid)
            
            # Very strict validation criteria - reject poor quality results
            # 1. Confidence must be at least 0.15 (minimum threshold)
            confidence_valid = result.confidence >= 0.15
            
            # 2. Residual error must be reasonable (less than 100m for acceptable results)
            residual_valid = result.residual_error < 100.0
            
            # 3. Position must be within reasonable distance from sensors (max 500m)
            max_reasonable_distance = 500.0
            distance_valid = distance_from_centroid < max_reasonable_distance
            
            # 4. Height must be reasonable (between 0m and 200m for drones)
            height_valid = 0.0 <= result.position[2] <= 200.0
            
            # 5. Check for NaN/Inf
            numeric_valid = (not np.any(np.isnan(result.position)) and 
                            not np.any(np.isinf(result.position)))
            
            # 6. Additional check: residual error should be less than distance from centroid
            # This ensures the solution is well-constrained
            error_ratio_valid = result.residual_error < distance_from_centroid * 0.5
            
            # 7. Critical check: if confidence is essentially zero, reject immediately
            if result.confidence < 0.01:
                confidence_valid = False
            
            # All checks must pass
            is_valid = (confidence_valid and residual_valid and distance_valid and 
                       height_valid and numeric_valid and error_ratio_valid)
            
            if not is_valid:
                reasons = []
                if not confidence_valid:
                    reasons.append(f"low confidence ({result.confidence:.3f} < 0.15)")
                if not residual_valid:
                    reasons.append(f"high error ({result.residual_error:.1f}m >= 100m)")
                if not distance_valid:
                    reasons.append(f"too far ({distance_from_centroid:.1f}m >= 500m)")
                if not height_valid:
                    reasons.append(f"bad height ({result.position[2]:.1f}m)")
                if not numeric_valid:
                    reasons.append("NaN/Inf values")
                if not error_ratio_valid:
                    reasons.append(f"poor error ratio (err={result.residual_error:.1f}m vs dist={distance_from_centroid:.1f}m)")
                print(f"   ⚠️  Sub-chunk {sub_idx+1}/{num_sub_chunks}: Invalid - {', '.join(reasons)}")
                continue  # Skip this invalid result
            
            # Double-check validation before accepting result
            # This is a safety check to ensure we never accept invalid results
            if not (confidence_valid and residual_valid and distance_valid and 
                   height_valid and numeric_valid and error_ratio_valid):
                print(f"   ⚠️  Sub-chunk {sub_idx+1}/{num_sub_chunks}: Validation check failed - skipping")
                continue
            
            # Additional safety: reject if confidence is essentially zero
            if result.confidence < 0.01:
                print(f"   ⚠️  Sub-chunk {sub_idx+1}/{num_sub_chunks}: Confidence too low ({result.confidence:.3f}) - skipping")
                continue
            
            # Score: weighted combination of confidence and residual error
            # Higher confidence and lower error = better score
            score = result.confidence * 10.0 - (result.residual_error / 10.0)
            if score > best_score:
                best_score = score
                best_result = result
                print(f"   ✅ Sub-chunk {sub_idx+1}/{num_sub_chunks}: Valid result "
                      f"(conf={result.confidence:.3f}, err={result.residual_error:.1f}m, "
                      f"dist={distance_from_centroid:.1f}m)")
        
        except Exception as e:
            print(f"   ❌ Sub-chunk {sub_idx+1}/{num_sub_chunks}: Error - {str(e)[:50]}")
            continue
    
    if not best_result:
        print("   ❌ No valid triangulation results found")
        print("   💡 All sub-chunks produced invalid results (low confidence, high error, or out of bounds)")
        print("   💡 Try a different time chunk or check audio quality")
        return None
    
    # Final validation check before plotting - be VERY strict
    sensor_centroid = np.mean(sensor_positions_array, axis=0)
    final_distance = np.linalg.norm(best_result.position - sensor_centroid)
    
    # Re-validate with even stricter criteria before plotting
    # Reject if ANY of these conditions are true
    confidence_too_low = best_result.confidence < 0.15
    error_too_high = best_result.residual_error >= 100.0
    distance_too_far = final_distance >= 500.0
    height_invalid = best_result.position[2] < 0 or best_result.position[2] > 200.0
    has_nan_inf = np.any(np.isnan(best_result.position)) or np.any(np.isinf(best_result.position))
    error_ratio_bad = best_result.residual_error >= final_distance * 0.5
    
    if (confidence_too_low or error_too_high or distance_too_far or 
        height_invalid or has_nan_inf or error_ratio_bad):
        print(f"\n   ❌ Final validation FAILED - result quality too poor to plot")
        print(f"      Confidence: {best_result.confidence:.3f} {'❌' if confidence_too_low else '✅'} (need >= 0.15)")
        print(f"      Residual error: {best_result.residual_error:.1f}m {'❌' if error_too_high else '✅'} (need < 100m)")
        print(f"      Distance from sensors: {final_distance:.1f}m {'❌' if distance_too_far else '✅'} (need < 500m)")
        print(f"      Height: {best_result.position[2]:.1f}m {'❌' if height_invalid else '✅'} (need 0-200m)")
        if has_nan_inf:
            print(f"      NaN/Inf values: ❌")
        if error_ratio_bad:
            print(f"      Error ratio: ❌ (err={best_result.residual_error:.1f}m >= {final_distance*0.5:.1f}m)")
        print(f"\n   💡 Cannot create plot - all triangulation results were invalid")
        print(f"   💡 This chunk may not have sufficient signal quality for accurate triangulation")
        return None
    
    # One final absolute check - never plot if confidence is essentially zero
    # This is a hard stop - no exceptions
    if best_result is None:
        print(f"\n   ❌ CRITICAL: best_result is None - cannot plot")
        return None
        
    if best_result.confidence < 0.01:
        print(f"\n   ❌ CRITICAL: Result has confidence {best_result.confidence:.3f} - rejecting")
        print(f"   💡 This indicates the triangulation failed completely")
        return None
    
    # Final hard check - if any critical metric fails, reject
    if (best_result.residual_error >= 100.0 or 
        best_result.confidence < 0.15 or
        np.any(np.isnan(best_result.position)) or 
        np.any(np.isinf(best_result.position))):
        print(f"\n   ❌ CRITICAL: Final hard check failed - rejecting result")
        print(f"      Confidence: {best_result.confidence:.3f}")
        print(f"      Residual error: {best_result.residual_error:.1f}m")
        print(f"      Position: {best_result.position}")
        return None
    
    result = best_result
    
    # ABSOLUTE FINAL CHECK - right before plotting
    # If this fails, we should never reach the plotting code
    assert result is not None, "result cannot be None"
    assert result.confidence >= 0.15, f"confidence {result.confidence:.3f} must be >= 0.15"
    assert result.residual_error < 100.0, f"residual_error {result.residual_error:.1f}m must be < 100m"
    assert not np.any(np.isnan(result.position)), "position cannot contain NaN"
    assert not np.any(np.isinf(result.position)), "position cannot contain Inf"
    
    print(f"\n   ✅ Best triangulation result selected and validated!")
    print(f"      Position: ({result.position[0]:.1f}, {result.position[1]:.1f}, {result.position[2]:.1f})")
    print(f"      Confidence: {result.confidence:.3f}")
    print(f"      Residual error: {result.residual_error:.3f} m")
    print(f"      Sensors used: {result.num_sensors_used}")
    print(f"      Method: {result.method}")
    print(f"   🎨 Creating plots with validated result...")
    
    # Create plots
    fig = plt.figure(figsize=(20, 10))
    plt.style.use('dark_background')
    
    # 2D plot (left side)
    ax1 = plt.subplot(121)
    
    # 3D plot (right side)  
    ax2 = plt.subplot(122, projection='3d')
    
    # Separate sensor types
    original_indices = [i for i, name in enumerate(sensor_names) if "CLUSTER" not in name]
    cluster_indices = [i for i, name in enumerate(sensor_names) if "CLUSTER" in name]
    
    # Convert positions to numpy array
    positions_arr = np.array(sensor_positions_array, dtype=float)
    
    # Extract coordinates
    orig_x = [sensor_positions_array[i][0] for i in original_indices]
    orig_y = [sensor_positions_array[i][1] for i in original_indices]
    orig_z = [sensor_positions_array[i][2] for i in original_indices]
    
    cluster_pos = sensor_positions_array[cluster_indices[0]] if cluster_indices else None
    
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
        
        # Add sensor labels
        for i, idx in enumerate(original_indices):
            pos = sensor_positions_array[idx]
            name = sensor_names[idx]
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
    
    # Plot detected source in 2D
    det_x = result.position[0]
    det_y = result.position[1]
    det_z = result.position[2]
    
    ax1.scatter(
        [det_x],
        [det_y],
        c='yellow',
        s=200 + result.confidence * 200,
        marker='D',
        alpha=0.9,
        edgecolors='white',
        linewidth=2,
        label=f'Sound Source (conf={result.confidence:.3f})',
    )
    
    ax1.annotate(
        'SOURCE',
        (det_x, det_y),
        xytext=(8, 8),
        textcoords='offset points',
        fontsize=12,
        color='yellow',
        weight='bold',
        ha='left',
        va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7)
    )
    
    # Draw lines from nearest sensors to detection
    if original_indices:
        ground_positions = positions_arr[original_indices]
        det_vec = np.array([det_x, det_y, det_z], dtype=float)
        dists = np.linalg.norm(ground_positions - det_vec, axis=1)
        nearest_idx_local = np.argsort(dists)[:3]
        
        for idx_local in nearest_idx_local:
            sensor_idx = original_indices[idx_local]
            sx, sy, sz = positions_arr[sensor_idx]
            
            t_vals = np.linspace(0.0, 1.0, 12)
            line_x_2d = sx + (det_x - sx) * t_vals
            line_y_2d = sy + (det_y - sy) * t_vals
            ax1.scatter(
                line_x_2d,
                line_y_2d,
                c='deepskyblue',
                s=5,
                alpha=0.2,
                marker='o',
            )
    
    # Set reasonable axis limits for 2D plot
    sensor_x_range = max(orig_x) - min(orig_x) if orig_x else 1000
    sensor_y_range = max(orig_y) - min(orig_y) if orig_y else 1000
    sensor_x_center = np.mean(orig_x) if orig_x else 0
    sensor_y_center = np.mean(orig_y) if orig_y else 0
    
    # Expand view to show detection with some margin
    margin = max(200, sensor_x_range * 0.3, sensor_y_range * 0.3)
    ax1.set_xlim([sensor_x_center - margin, sensor_x_center + margin])
    ax1.set_ylim([sensor_y_center - margin, sensor_y_center + margin])
    
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
            pos = sensor_positions_array[idx]
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
    
    # Plot detected source in 3D
    ax2.scatter(
        [det_x],
        [det_y],
        [det_z],
        c='yellow',
        s=200 + result.confidence * 200,
        marker='D',
        alpha=0.9,
        edgecolors='white',
        linewidth=2,
        label=f'Sound Source (conf={result.confidence:.3f})',
    )
    
    ax2.text(
        det_x,
        det_y,
        det_z + 2,
        'SOURCE',
        fontsize=12,
        color='yellow',
        weight='bold',
        ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
    )
    
    # Draw lines from nearest sensors to detection in 3D
    if original_indices:
        for idx_local in nearest_idx_local:
            sensor_idx = original_indices[idx_local]
            sx, sy, sz = positions_arr[sensor_idx]
            
            t_vals = np.linspace(0.0, 1.0, 12)
            line_x_3d = sx + (det_x - sx) * t_vals
            line_y_3d = sy + (det_y - sy) * t_vals
            line_z_3d = sz + (det_z - sz) * t_vals
            ax2.scatter(
                line_x_3d,
                line_y_3d,
                line_z_3d,
                c='deepskyblue',
                s=4,
                alpha=0.25,
                marker='o',
            )
    
    # Set reasonable axis limits for 3D plot
    sensor_z_range = max(orig_z) - min(orig_z) if orig_z else 10
    sensor_z_center = np.mean(orig_z) if orig_z else 0
    
    # Expand view with reasonable height range
    margin_3d = max(200, sensor_x_range * 0.3, sensor_y_range * 0.3)
    height_margin = max(100, sensor_z_range + 200)  # Allow up to 200m above/below sensors
    
    ax2.set_xlim([sensor_x_center - margin_3d, sensor_x_center + margin_3d])
    ax2.set_ylim([sensor_y_center - margin_3d, sensor_y_center + margin_3d])
    ax2.set_zlim([sensor_z_center - 50, sensor_z_center + height_margin])
    
    # 3D formatting
    ax2.set_xlabel('X (meters)', fontsize=11)
    ax2.set_ylabel('Y (meters)', fontsize=11)
    ax2.set_zlabel('Height (meters)', fontsize=11)
    ax2.set_title('3D View - With Height', fontsize=14, pad=15)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.view_init(elev=20, azim=45)
    ax2.grid(True, alpha=0.3)
    
    # Add info box
    info_lines = [
        f"Triangulation Results (Best Channel & 120s Chunk):",
        f"• Time: {start_time:.1f}s - {start_time+120:.1f}s",
        f"• Position: ({det_x:.1f}, {det_y:.1f}, {det_z:.1f}) m",
        f"• Confidence: {result.confidence:.3f}",
        f"• Residual error: {result.residual_error:.3f} m",
        f"• Sensors used: {result.num_sensors_used}",
        f"• Method: {result.method}",
    ]
    
    info_text = "\n".join(info_lines)
    plt.figtext(0.02, 0.02, info_text, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, "best_channel_triangulation.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"   💾 Saved: {output_file}")
    
    plt.show()
    
    return output_file

def main():
    """Main function."""
    import sys
    import io
    
    # Fix encoding issues on Windows
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    sys.stdout.flush()
    
    wav_path = "multi-20251122-141610-627897594.wav"
    kml_path = "Sensor-Locations-BOP-Dharma.kml"
    
    if not os.path.exists(wav_path):
        print(f"❌ WAV file not found: {wav_path}")
        return
    
    if not os.path.exists(kml_path):
        print(f"❌ KML file not found: {kml_path}")
        return
    
    try:
        # Find best channel and chunk
        best_channel_idx, best_start_sample, score, fundamental, sample_rate, audio_data = \
            find_best_channel_and_chunk(wav_path, chunk_duration=120.0)
        
        # Extract the 120-second (2-minute) chunk from ALL channels
        chunk_samples = int(120.0 * sample_rate)
        best_end_sample = min(best_start_sample + chunk_samples, audio_data.shape[1])
        best_chunk = audio_data[:, best_start_sample:best_end_sample]
        
        best_start_time = best_start_sample / sample_rate
        
        print(f"\n📊 Using 120-second (2-minute) chunk from all channels for triangulation...")
        print(f"   Channel {best_channel_idx+1} had best harmonics")
        print(f"   Chunk: {best_start_time:.1f}s - {best_start_time+120:.1f}s")
        
        # Create triangulation plots
        output_file = create_triangulation_plots(
            best_chunk, best_start_time, kml_path, sample_rate
        )
        
        print(f"\n🎉 Analysis Complete!")
        print(f"   📊 Best channel: {best_channel_idx+1}")
        print(f"   ⏱️  Best chunk: {best_start_time:.1f}s - {best_start_time+120:.1f}s")
        print(f"   🎯 Harmonic score: {score:.1f}")
        print(f"   💾 Plot saved: {output_file}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

