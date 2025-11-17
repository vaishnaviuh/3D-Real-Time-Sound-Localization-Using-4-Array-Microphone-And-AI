from typing import Tuple, List, Optional
import numpy as np
from scipy.signal import fftconvolve, butter, filtfilt
from scipy.optimize import least_squares
from math import atan2, degrees


def _gcc_phat(sig: np.ndarray, ref: np.ndarray, fs: int, max_tau: float, interp: int = 16) -> Tuple[float, float]:
    """
    GCC-PHAT between two signals to estimate time delay (seconds).
    Limits search to +/- max_tau.
    Returns (tau, peak_quality) where peak_quality is the normalized correlation peak value.
    """
    n = sig.shape[0] + ref.shape[0]
    nfft = 1
    while nfft < n:
        nfft <<= 1
    SIG = np.fft.rfft(sig, n=nfft)
    REF = np.fft.rfft(ref, n=nfft)
    R = SIG * np.conj(REF)
    R /= np.maximum(np.abs(R), 1e-12)
    cc = np.fft.irfft(R, n=nfft * interp)
    max_shift = int(interp * fs * max_tau)
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    cc_abs = np.abs(cc)
    peak_idx = np.argmax(cc_abs)
    peak_value = cc_abs[peak_idx]
    # Normalize peak quality: peak / (mean + std) to measure how distinct the peak is
    mean_cc = np.mean(cc_abs)
    std_cc = np.std(cc_abs)
    peak_quality = peak_value / max(mean_cc + std_cc, 1e-12)
    shift = peak_idx - max_shift
    tau = shift / float(interp * fs)
    return tau, peak_quality


def build_mic_positions(radius_m: float, angles_deg: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Returns mic positions (4, 3) in meters on XY plane (z=0).
    """
    positions = []
    for ang in angles_deg:
        theta = np.deg2rad(ang)
        x = radius_m * np.cos(theta)
        y = radius_m * np.sin(theta)
        positions.append([x, y, 0.0])
    return np.array(positions, dtype=np.float64)


def compute_signal_quality(signals: np.ndarray) -> Tuple[float, float]:
    """
    Compute signal quality metrics.
    Returns (rms_energy, peak_to_mean_ratio)
    - rms_energy: RMS energy across all channels (normalized)
    - peak_to_mean_ratio: Ratio of peak signal to mean signal
    """
    # Compute RMS energy per channel
    rms_per_channel = np.sqrt(np.mean(np.square(signals), axis=0))
    rms_energy = float(np.mean(rms_per_channel))
    
    # Compute peak-to-mean ratio as a quality indicator
    abs_signals = np.abs(signals)
    peak_value = np.max(abs_signals)
    mean_value = np.mean(abs_signals)
    peak_to_mean = peak_value / max(mean_value, 1e-12)
    
    return rms_energy, float(peak_to_mean)


def estimate_doa_az_el(
    signals: np.ndarray,
    fs: int,
    mic_positions_m: np.ndarray,
    speed_of_sound: float = 343.0,
    min_correlation_quality: float = 0.8,
    min_rms_energy: float = 0.0001,
    enable_debug: bool = False,
) -> Tuple[float, float, float]:
    """
    Estimate azimuth and elevation using pairwise TDOA + LS direction fit.
    signals: (N, 4)
    mic_positions_m: (4, 3)
    min_correlation_quality: Minimum peak quality threshold (default 1.5)
    min_rms_energy: Minimum RMS energy threshold (default 0.001)
    Returns (azimuth_deg, elevation_deg, confidence) where confidence is 0.0-1.0
    """
    assert signals.shape[1] == 4, "Expected 4 channels"
    assert mic_positions_m.shape == (4, 3)

    # Check signal quality first
    rms_energy, peak_to_mean = compute_signal_quality(signals)
    
    # Debug output
    if enable_debug:
        print(f"[DEBUG] RMS Energy: {rms_energy:.6f} (threshold: {min_rms_energy:.6f}), Peak-to-Mean: {peak_to_mean:.2f}")
    
    # If signal is too weak, return low confidence but still compute DOA
    # This allows tracking even weak signals
    signal_too_weak = rms_energy < min_rms_energy
    if signal_too_weak and enable_debug:
        print(f"[DEBUG] Signal weak (RMS: {rms_energy:.6f} < {min_rms_energy:.6f}), but continuing...")

    # High-level: for all unique pairs, GCC-PHAT TDOA, then solve A u = b (LS), u=unit vector.
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    max_mic_spacing = np.max(
        [np.linalg.norm(mic_positions_m[j] - mic_positions_m[i]) for i, j in pairs]
    )
    max_tau = max_mic_spacing / speed_of_sound

    tdoas = []
    A_rows = []
    correlation_qualities = []
    for i, j in pairs:
        tau_ij, peak_quality = _gcc_phat(signals[:, i], signals[:, j], fs, max_tau=max_tau)
        tdoas.append(tau_ij)
        correlation_qualities.append(peak_quality)
        diff = (mic_positions_m[j] - mic_positions_m[i]) / speed_of_sound  # shape (3,)
        A_rows.append(diff)
    
    # Check correlation quality - if peaks are too weak, likely just noise
    avg_correlation_quality = float(np.mean(correlation_qualities))
    if enable_debug:
        print(f"[DEBUG] Avg Correlation Quality: {avg_correlation_quality:.3f} (threshold: {min_correlation_quality:.3f})")
        print(f"[DEBUG] Per-pair correlation qualities: {[f'{q:.3f}' for q in correlation_qualities]}")
        print(f"[DEBUG] TDOAs (µs): {[f'{t*1e6:.2f}' for t in tdoas]}")
    
    # Check if we have any valid correlations at all
    if avg_correlation_quality < 0.1:  # Very low correlation - likely no signal
        if enable_debug:
            print(f"[DEBUG] Very low correlation quality ({avg_correlation_quality:.3f}) - likely no coherent signal")
        return 0.0, 0.0, 0.0
    
    # Don't reject based on correlation quality - compute DOA anyway and use confidence
    correlation_too_low = avg_correlation_quality < min_correlation_quality
    if correlation_too_low and enable_debug:
        print(f"[DEBUG] Correlation quality low ({avg_correlation_quality:.3f} < {min_correlation_quality:.3f}), but continuing...")
    
    A = np.vstack(A_rows)  # (num_pairs, 3)
    b = np.array(tdoas)  # (num_pairs,)
    
    is_planar_array = np.ptp(mic_positions_m[:, 2]) < 1e-4
    
    if enable_debug:
        print(f"[DEBUG] A matrix shape: {A.shape}, b vector shape: {b.shape}")
        cond_A = np.linalg.cond(A)
        print(f"[DEBUG] A condition number: {cond_A:.2e}")
        if is_planar_array:
            print("[DEBUG] Mic array is planar (z-baseline ≈ 0) - elevation may be unreliable")
    
    # Least squares for direction vector
    u, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)  # shape (3,)
    
    # Check if solution is valid
    if rank < 3:
        if is_planar_array:
            if enable_debug:
                print(f"[DEBUG] 3D solve rank={rank} < 3, falling back to 2D azimuth-only solve")
            A_xy = A[:, :2]
            uxuy, res2, rank2, s2 = np.linalg.lstsq(A_xy, b, rcond=None)
            if rank2 < 2:
                if enable_debug:
                    print(f"[DEBUG] 2D solve also failed: rank={rank2} < 2")
                return 0.0, 0.0, 0.0
            u = np.array([uxuy[0], uxuy[1], 0.0])
        else:
            if enable_debug:
                print(f"[DEBUG] DOA solve failed: rank={rank} < 3 (insufficient mic correlation)")
            return 0.0, 0.0, 0.0
    
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-10 or np.allclose(u, 0):
        if enable_debug:
            print(f"[DEBUG] DOA solve failed: direction vector too small (norm={u_norm:.2e})")
        return 0.0, 0.0, 0.0
    
    u = u / u_norm

    # Convert to spherical: azimuth in XY, elevation from XY plane
    # Azimuth: angle in XY plane (0° = +X axis, 90° = +Y axis)
    azimuth_rad = atan2(u[1], u[0])
    azimuth_deg = (degrees(azimuth_rad) + 360.0) % 360.0
    
    # Elevation: angle above XY plane (positive = above, negative = below)
    xy_norm = np.linalg.norm(u[:2])
    if xy_norm < 1e-10:
        # Source is directly above/below (u[2] dominates)
        elevation_deg = 90.0 if u[2] > 0 else -90.0
    else:
        elevation_rad = atan2(u[2], xy_norm)
        elevation_deg = degrees(elevation_rad)
    
    if enable_debug:
        print(f"[DEBUG] DOA result: azimuth={azimuth_deg:.1f}°, elevation={elevation_deg:.1f}°, direction_vector=[{u[0]:.3f}, {u[1]:.3f}, {u[2]:.3f}]")
    
    # Compute confidence based on correlation quality and signal energy
    # Always give some confidence, even if below thresholds (just lower)
    # Normalize correlation quality - use min_correlation_quality as baseline
    if avg_correlation_quality > min_correlation_quality:
        corr_confidence = min(1.0, (avg_correlation_quality - min_correlation_quality) / 2.0)
    else:
        # Below threshold, but still give some confidence (scaled down)
        corr_confidence = max(0.0, avg_correlation_quality / min_correlation_quality) * 0.3
    
    # Normalize RMS energy
    if rms_energy > min_rms_energy:
        energy_confidence = min(1.0, rms_energy / 0.01)
    else:
        # Below threshold, but still give some confidence (scaled down)
        energy_confidence = max(0.0, rms_energy / min_rms_energy) * 0.3
    
    # Combined confidence - give more weight to correlation (it's more reliable)
    confidence = (corr_confidence * 0.6 + energy_confidence * 0.4)
    # Always give at least a small confidence value so we can track
    confidence = max(confidence, 0.05)
    
    if enable_debug:
        print(f"[DEBUG] Confidence: {confidence:.3f} (corr: {corr_confidence:.3f}, energy: {energy_confidence:.3f})")
    
    return azimuth_deg, elevation_deg, float(confidence)


def apply_bandpass_filter(signals: np.ndarray, fs: int, low_freq: float, high_freq: float) -> np.ndarray:
    """
    Apply bandpass filter to signals to keep only frequencies in [low_freq, high_freq] range.
    signals: (N, M) array
    Returns filtered signals with same shape.
    """
    nyquist = fs / 2.0
    low_norm = max(0.01, min(low_freq / nyquist, 0.99))
    high_norm = max(0.01, min(high_freq / nyquist, 0.99))
    
    if low_norm >= high_norm:
        # Invalid range, return original
        return signals
    
    # Design Butterworth bandpass filter
    b, a = butter(N=4, Wn=[low_norm, high_norm], btype='band')
    
    # Apply filter to each channel
    filtered = np.zeros_like(signals)
    for ch in range(signals.shape[1]):
        filtered[:, ch] = filtfilt(b, a, signals[:, ch])
    
    return filtered


def detect_harmonics(
    signals: np.ndarray,
    fs: int,
    target_fundamentals_hz: List[float],
    min_harmonics: int = 2,
    tolerance_hz: float = 50.0,
    min_magnitude_ratio: float = 0.1,
) -> Tuple[bool, Optional[float]]:
    """
    Detect if specific harmonic series are present in the signal.
    
    Args:
        signals: (N, M) array of audio signals
        fs: Sampling rate
        target_fundamentals_hz: List of fundamental frequencies to look for
        min_harmonics: Minimum number of harmonics (including fundamental) that must be present
        tolerance_hz: Frequency tolerance for matching harmonics
        min_magnitude_ratio: Minimum peak magnitude relative to max peak
    
    Returns:
        (detected, detected_fundamental): True if harmonics detected, and the fundamental freq found
    """
    if not target_fundamentals_hz:
        # Harmonic detection disabled
        return True, None
    
    # Use first channel for harmonic detection (or average across channels)
    signal = np.mean(signals, axis=1)
    
    # Compute FFT
    n = len(signal)
    nfft = 1
    while nfft < n:
        nfft <<= 1
    
    fft_signal = np.fft.rfft(signal, n=nfft)
    magnitude = np.abs(fft_signal)
    
    # Frequency axis
    freqs = np.fft.rfftfreq(nfft, 1.0/fs)
    
    # Normalize magnitude
    max_mag = np.max(magnitude)
    if max_mag < 1e-12:
        return False, None
    
    magnitude_norm = magnitude / max_mag
    
    # Find peaks above threshold
    peak_threshold = min_magnitude_ratio
    peak_indices = np.where(magnitude_norm > peak_threshold)[0]
    peak_freqs = freqs[peak_indices]
    peak_mags = magnitude_norm[peak_indices]
    
    if len(peak_freqs) < min_harmonics:
        return False, None
    
    # Check each target fundamental
    for fundamental_hz in target_fundamentals_hz:
        harmonics_detected = 0
        detected_harmonic_freqs = []
        
        # Check for fundamental and its harmonics (up to 10th harmonic)
        for harmonic_num in range(1, 11):
            target_freq = fundamental_hz * harmonic_num
            
            # Skip if outside frequency range
            if target_freq > fs / 2:
                break
            
            # Find peaks near this harmonic frequency
            matches = np.abs(peak_freqs - target_freq) <= tolerance_hz
            if np.any(matches):
                # Find the best match (highest magnitude)
                match_indices = np.where(matches)[0]
                best_match_idx = match_indices[np.argmax(peak_mags[matches])]
                detected_freq = peak_freqs[best_match_idx]
                detected_mag = peak_mags[best_match_idx]
                
                harmonics_detected += 1
                detected_harmonic_freqs.append((detected_freq, detected_mag, harmonic_num))
        
        # Check if we found enough harmonics
        if harmonics_detected >= min_harmonics:
            return True, fundamental_hz
    
    return False, None


def auto_detect_candidate_fundamentals(
    signals: np.ndarray,
    fs: int,
    min_freq_hz: float,
    max_freq_hz: float,
    min_peak_ratio: float = 0.3,
    max_candidates: int = 5,
) -> List[float]:
    """
    Automatically identify likely fundamental frequencies by finding strong tonal peaks.
    Returns a list of candidate fundamentals sorted by strength (Hz).
    """
    signal = np.mean(signals, axis=1)
    n = len(signal)
    nfft = 1
    while nfft < n:
        nfft <<= 1

    fft_signal = np.fft.rfft(signal, n=nfft)
    magnitude = np.abs(fft_signal)
    if magnitude.size == 0:
        return []

    max_mag = np.max(magnitude)
    if max_mag < 1e-12:
        return []

    magnitude_norm = magnitude / max_mag
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)

    # Filter by frequency range and minimum peak ratio
    valid_mask = (
        (freqs >= min_freq_hz)
        & (freqs <= max_freq_hz)
        & (magnitude_norm >= min_peak_ratio)
    )
    candidate_freqs = freqs[valid_mask]
    candidate_mags = magnitude_norm[valid_mask]
    if candidate_freqs.size == 0:
        return []

    # Sort by magnitude descending and take top candidates, enforcing uniqueness
    sort_indices = np.argsort(candidate_mags)[::-1]
    unique_candidates: List[float] = []
    for idx in sort_indices:
        freq = float(candidate_freqs[idx])
        if all(abs(freq - existing) > 5.0 for existing in unique_candidates):
            unique_candidates.append(freq)
        if len(unique_candidates) >= max_candidates:
            break
    return unique_candidates


def detect_signal_activity(
    signals: np.ndarray,
    min_rms: float,
    min_peak_to_mean: float,
) -> Tuple[bool, dict]:
    """
    Detect general signal activity using RMS energy and peak-to-mean ratio.
    Returns (activity_detected, diagnostics_dict).
    """
    rms_energy, peak_to_mean = compute_signal_quality(signals)
    activity_detected = (rms_energy >= min_rms) and (peak_to_mean >= min_peak_to_mean)
    diagnostics = {
        "rms_energy": float(rms_energy),
        "peak_to_mean": float(peak_to_mean),
        "min_rms_threshold": float(min_rms),
        "min_peak_to_mean_threshold": float(min_peak_to_mean),
    }
    return activity_detected, diagnostics


def rough_distance_from_energy(signals: np.ndarray, ref_db_at_1m: float = -20.0) -> float:
    """
    Extremely rough distance proxy from RMS level (dBFS-like).
    Requires calibration to be meaningful. Returns meters.
    """
    eps = 1e-12
    rms = np.sqrt(np.mean(np.square(signals), axis=0)).mean()
    db = 20.0 * np.log10(max(rms, eps))
    # Simple inverse square model: level drop 6 dB per doubling distance
    # distance = 1m * 10^((ref_db - db)/20)
    distance_m = 10 ** ((ref_db_at_1m - db) / 20.0)
    return float(max(distance_m, 0.1))


def _compute_tdoas_vs_ref(
    signals: np.ndarray, fs: int, ref_index: int, speed_of_sound: float, mic_positions_m: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    Compute GCC-PHAT TDOAs relative to reference mic.
    Returns:
      - tdoas: array of shape (num_mics,) where tdoas[ref_index] == 0
      - max_tau: bound based on maximum baseline
      - avg_correlation_quality: average correlation peak quality
    """
    num_mics = signals.shape[1]
    # Determine max_tau from geometry (largest mic spacing)
    pair_indices: List[Tuple[int, int]] = [(i, j) for i in range(num_mics) for j in range(i + 1, num_mics)]
    max_mic_spacing = np.max(
        [np.linalg.norm(mic_positions_m[j] - mic_positions_m[i]) for i, j in pair_indices]
    )
    max_tau = max_mic_spacing / speed_of_sound

    tdoas = np.zeros(num_mics, dtype=np.float64)
    correlation_qualities = []
    per_pair_qualities = {}  # Store quality for each pair for debugging
    
    for i in range(num_mics):
        if i == ref_index:
            tdoas[i] = 0.0
        else:
            tau, peak_quality = _gcc_phat(signals[:, i], signals[:, ref_index], fs, max_tau=max_tau)
            tdoas[i] = tau
            correlation_qualities.append(peak_quality)
            per_pair_qualities[(ref_index, i)] = peak_quality
    
    avg_correlation_quality = float(np.mean(correlation_qualities)) if correlation_qualities else 0.0
    
    # Debug: Check if mic 3 (index 2) has poor correlation
    if len(correlation_qualities) > 2:
        mic3_quality = per_pair_qualities.get((ref_index, 2), 0.0)
        if mic3_quality < avg_correlation_quality * 0.5:
            # This will be printed in estimate_position_from_tdoa if enable_debug
            pass
    
    return tdoas, max_tau, avg_correlation_quality


def estimate_position_from_tdoa(
    signals: np.ndarray,
    fs: int,
    mic_positions_m: np.ndarray,
    speed_of_sound: float = 343.0,
    ref_index: int = 0,
    initial_guess: Optional[np.ndarray] = None,
    min_correlation_quality: float = 1.5,
    enable_debug: bool = False,
) -> Optional[np.ndarray]:
    """
    Estimate 3D source position using TDOAs relative to a reference mic via nonlinear least squares.
    signals: (N, M) with M=4
    mic_positions_m: (M, 3)
    min_correlation_quality: Minimum correlation quality threshold
    enable_debug: Print debug information about TDOAs and per-mic distances
    Returns position np.ndarray shape (3,) or None if solve fails or quality is too low.
    """
    assert signals.shape[1] == mic_positions_m.shape[0], "Signal/mic count mismatch"
    num_mics = mic_positions_m.shape[0]
    assert num_mics >= 4, "Need at least 4 microphones for Position from TDOA"

    tdoas_meas, _, avg_correlation_quality = _compute_tdoas_vs_ref(
        signals=signals, fs=fs, ref_index=ref_index, speed_of_sound=speed_of_sound, mic_positions_m=mic_positions_m
    )
    
    # Debug: Print per-microphone TDOAs and signal levels
    if enable_debug:
        print(f"[DEBUG] TDOAs vs ref mic {ref_index}:")
        for i in range(num_mics):
            rms_i = np.sqrt(np.mean(signals[:, i] ** 2))
            mic_angle = np.degrees(np.arctan2(mic_positions_m[i][1], mic_positions_m[i][0]))
            print(f"  Mic {i} (angle={mic_angle:.1f}°): TDOA={tdoas_meas[i]*1e6:.2f} µs, RMS={rms_i:.6f}")
        
        # Check for anomalous TDOAs (especially mic 3)
        tdoas_us = [t * 1e6 for t in tdoas_meas]
        mean_tdoa = np.mean([abs(t) for t in tdoas_us if t != 0])
        for i in range(num_mics):
            if i != ref_index and abs(tdoas_us[i]) > 3 * mean_tdoa and mean_tdoa > 0:
                print(f"[WARNING] Mic {i} TDOA ({tdoas_us[i]:.2f} µs) is anomalously large compared to mean ({mean_tdoa:.2f} µs)")
    
    # Check correlation quality before attempting position estimation
    if avg_correlation_quality < min_correlation_quality:
        if enable_debug:
            print(f"[DEBUG] Correlation quality {avg_correlation_quality:.3f} < {min_correlation_quality:.3f}, skipping position estimation")
        return None

    # Initial guess: small height above array center or along estimated direction if available
    if initial_guess is None:
        initial_guess = np.array([0.0, 0.0, 0.1], dtype=np.float64)

    def residuals(src_pos: np.ndarray) -> np.ndarray:
        # Predicted delays vs ref: (|s - m_i| - |s - m_ref|)/c
        distances = np.linalg.norm(src_pos[None, :] - mic_positions_m, axis=1)
        tau_pred = (distances - distances[ref_index]) / speed_of_sound
        res = tdoas_meas - tau_pred
        res[ref_index] = 0.0
        return res

    try:
        result = least_squares(residuals, initial_guess, method="trf", max_nfev=200, ftol=1e-10, xtol=1e-10)
        if not result.success:
            if enable_debug:
                print(f"[DEBUG] Position estimation failed: {result.message}")
            return None
        
        estimated_pos = result.x.astype(np.float64)
        
        # Calculate distance from array center
        distance_from_center = np.linalg.norm(estimated_pos)
        
        # Calculate theoretical accuracy limits based on array geometry
        # Maximum baseline (diameter of array)
        max_baseline = 2 * np.max([np.linalg.norm(mic_positions_m[i]) for i in range(num_mics)])
        # Theoretical angular resolution: ~wavelength / baseline for far field
        # For 1kHz: wavelength = 343/1000 = 0.343m, baseline = 0.086m
        # Angular resolution ≈ 0.343/0.086 ≈ 4 radians ≈ 230 degrees (very poor!)
        # Distance accuracy degrades as distance increases relative to baseline
        
        # Estimate distance uncertainty based on TDOA measurement accuracy
        # TDOA accuracy is limited by: sampling rate (1/fs), interpolation factor, and correlation quality
        # At 16kHz with 16x interpolation: ~1/(16000*16) ≈ 3.9 microseconds
        # This corresponds to distance uncertainty: c * dt ≈ 343 * 3.9e-6 ≈ 1.3mm per mic pair
        # But as distance increases, small TDOA errors cause large distance errors
        
        # Far-field approximation: distance >> baseline
        # Distance uncertainty scales with distance^2 / baseline
        if distance_from_center > max_baseline:
            # Far field: distance uncertainty grows quadratically
            tdoa_uncertainty_us = 3.9  # Approximate TDOA uncertainty in microseconds
            tdoa_uncertainty_s = tdoa_uncertainty_us * 1e-6
            distance_uncertainty = (distance_from_center ** 2) / (max_baseline * speed_of_sound) * tdoa_uncertainty_s
            distance_uncertainty = max(distance_uncertainty, 0.01)  # At least 1cm
        else:
            # Near field: better accuracy
            distance_uncertainty = 0.01  # ~1cm in near field
        
        # Calculate confidence based on distance and array geometry
        # Accuracy degrades significantly beyond ~10x baseline (for 4.3cm radius, ~43cm)
        baseline_ratio = distance_from_center / max_baseline
        if baseline_ratio > 10:
            distance_confidence = max(0.1, 1.0 - (baseline_ratio - 10) * 0.1)  # Degrades beyond 10x
        elif baseline_ratio > 5:
            distance_confidence = 0.5 + 0.5 * (10 - baseline_ratio) / 5  # 0.5 to 1.0
        else:
            distance_confidence = 1.0  # Good accuracy within 5x baseline
        
        # Debug: Calculate and print per-microphone distances
        if enable_debug:
            print(f"[DEBUG] Estimated position: [{estimated_pos[0]:.3f}, {estimated_pos[1]:.3f}, {estimated_pos[2]:.3f}] m")
            print(f"[DEBUG] Distance from center: {distance_from_center:.3f} m ({distance_from_center*100:.1f} cm)")
            print(f"[DEBUG] Array baseline: {max_baseline:.3f} m ({max_baseline*100:.1f} cm)")
            print(f"[DEBUG] Baseline ratio: {baseline_ratio:.1f}x (accuracy degrades >10x)")
            print(f"[DEBUG] Estimated distance uncertainty: ±{distance_uncertainty*100:.1f} cm")
            print(f"[DEBUG] Distance confidence: {distance_confidence:.2f}")
            if baseline_ratio > 10:
                print(f"[WARNING] Distance ({distance_from_center*100:.1f} cm) is >10x baseline - accuracy is POOR")
            elif baseline_ratio > 5:
                print(f"[WARNING] Distance ({distance_from_center*100:.1f} cm) is >5x baseline - accuracy is DEGRADED")
            print(f"[DEBUG] Per-microphone distances from source:")
            for i in range(num_mics):
                dist_i = np.linalg.norm(estimated_pos - mic_positions_m[i])
                print(f"  Mic {i} (angle={np.degrees(np.arctan2(mic_positions_m[i][1], mic_positions_m[i][0])):.1f}°): {dist_i:.3f} m")
            
            # Check for anomalies (mic with very different distance)
            distances = [np.linalg.norm(estimated_pos - mic_positions_m[i]) for i in range(num_mics)]
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            for i in range(num_mics):
                diff = abs(distances[i] - mean_dist)
                if diff > max(0.1, 2 * std_dist):  # More than 10cm or 2 std devs
                    mic_angle = np.degrees(np.arctan2(mic_positions_m[i][1], mic_positions_m[i][0]))
                    print(f"[WARNING] Mic {i} (angle={mic_angle:.1f}°) distance ({distances[i]:.3f} m) differs significantly from mean ({mean_dist:.3f} m, std={std_dist:.3f} m)")
                    print(f"[WARNING]   Difference: {diff:.3f} m ({diff*100:.1f} cm)")
                    # Check if this mic's TDOA is also anomalous
                    if i < len(tdoas_meas):
                        print(f"[WARNING]   Mic {i} TDOA: {tdoas_meas[i]*1e6:.2f} µs")
                        # Check signal quality
                        rms_i = np.sqrt(np.mean(signals[:, i] ** 2))
                        rms_ref = np.sqrt(np.mean(signals[:, ref_index] ** 2))
                        print(f"[WARNING]   Mic {i} RMS: {rms_i:.6f}, Ref mic RMS: {rms_ref:.6f}, Ratio: {rms_i/max(rms_ref, 1e-12):.2f}")
        
        return estimated_pos
    except Exception as e:
        if enable_debug:
            print(f"[DEBUG] Position estimation exception: {e}")
        return None


