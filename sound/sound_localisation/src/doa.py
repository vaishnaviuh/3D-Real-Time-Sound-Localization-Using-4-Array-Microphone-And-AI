from typing import Tuple, List, Optional
import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import least_squares
from math import atan2, degrees


def _gcc_phat(sig: np.ndarray, ref: np.ndarray, fs: int, max_tau: float, interp: int = 16) -> float:
    """
    GCC-PHAT between two signals to estimate time delay (seconds).
    Limits search to +/- max_tau.
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
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau


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


def estimate_doa_az_el(
    signals: np.ndarray,
    fs: int,
    mic_positions_m: np.ndarray,
    speed_of_sound: float = 343.0,
) -> Tuple[float, float]:
    """
    Estimate azimuth and elevation using pairwise TDOA + LS direction fit.
    signals: (N, 4)
    mic_positions_m: (4, 3)
    Returns (azimuth_deg, elevation_deg)
    """
    assert signals.shape[1] == 4, "Expected 4 channels"
    assert mic_positions_m.shape == (4, 3)

    # High-level: for all unique pairs, GCC-PHAT TDOA, then solve A u = b (LS), u=unit vector.
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    max_mic_spacing = np.max(
        [np.linalg.norm(mic_positions_m[j] - mic_positions_m[i]) for i, j in pairs]
    )
    max_tau = max_mic_spacing / speed_of_sound

    tdoas = []
    A_rows = []
    for i, j in pairs:
        tau_ij = _gcc_phat(signals[:, i], signals[:, j], fs, max_tau=max_tau)
        tdoas.append(tau_ij)
        diff = (mic_positions_m[j] - mic_positions_m[i]) / speed_of_sound  # shape (3,)
        A_rows.append(diff)
    A = np.vstack(A_rows)  # (num_pairs, 3)
    b = np.array(tdoas)  # (num_pairs,)

    # Least squares for direction vector
    u, *_ = np.linalg.lstsq(A, b, rcond=None)  # shape (3,)
    if np.allclose(u, 0):
        return 0.0, 0.0
    u = u / np.maximum(np.linalg.norm(u), 1e-12)

    # Convert to spherical: azimuth in XY, elevation from XY plane
    azimuth_rad = atan2(u[1], u[0])
    azimuth_deg = (degrees(azimuth_rad) + 360.0) % 360.0
    # Elevation: angle above XY plane
    elevation_rad = atan2(u[2], np.linalg.norm(u[:2]))
    elevation_deg = degrees(elevation_rad)
    return azimuth_deg, elevation_deg


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
) -> Tuple[np.ndarray, float]:
    """
    Compute GCC-PHAT TDOAs relative to reference mic.
    Returns:
      - tdoas: array of shape (num_mics,) where tdoas[ref_index] == 0
      - max_tau: bound based on maximum baseline
    """
    num_mics = signals.shape[1]
    # Determine max_tau from geometry (largest mic spacing)
    pair_indices: List[Tuple[int, int]] = [(i, j) for i in range(num_mics) for j in range(i + 1, num_mics)]
    max_mic_spacing = np.max(
        [np.linalg.norm(mic_positions_m[j] - mic_positions_m[i]) for i, j in pair_indices]
    )
    max_tau = max_mic_spacing / speed_of_sound

    tdoas = np.zeros(num_mics, dtype=np.float64)
    for i in range(num_mics):
        if i == ref_index:
            tdoas[i] = 0.0
        else:
            tdoas[i] = _gcc_phat(signals[:, i], signals[:, ref_index], fs, max_tau=max_tau)
    return tdoas, max_tau


def estimate_position_from_tdoa(
    signals: np.ndarray,
    fs: int,
    mic_positions_m: np.ndarray,
    speed_of_sound: float = 343.0,
    ref_index: int = 0,
    initial_guess: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Estimate 3D source position using TDOAs relative to a reference mic via nonlinear least squares.
    signals: (N, M) with M=4
    mic_positions_m: (M, 3)
    Returns position np.ndarray shape (3,) or None if solve fails.
    """
    assert signals.shape[1] == mic_positions_m.shape[0], "Signal/mic count mismatch"
    num_mics = mic_positions_m.shape[0]
    assert num_mics >= 4, "Need at least 4 microphones for Position from TDOA"

    tdoas_meas, _ = _compute_tdoas_vs_ref(
        signals=signals, fs=fs, ref_index=ref_index, speed_of_sound=speed_of_sound, mic_positions_m=mic_positions_m
    )

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
            return None
        return result.x.astype(np.float64)
    except Exception:
        return None


