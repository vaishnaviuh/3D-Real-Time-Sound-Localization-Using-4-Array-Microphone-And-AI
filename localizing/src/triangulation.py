"""
Advanced triangulation engine for multi-sensor sound localization.
Implements various triangulation algorithms for 20+ sensor poles.
"""
import os
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.optimize import minimize, least_squares
from scipy.signal import correlate, find_peaks
import warnings
from src.utils import get_logger

logger = get_logger("sound_localization.triangulation")

@dataclass
class TriangulationResult:
    """Result of triangulation computation."""
    position: np.ndarray  # Estimated 3D position (x, y, z)
    confidence: float  # Confidence score 0-1
    residual_error: float  # RMS error of the solution
    num_sensors_used: int  # Number of sensors that contributed
    tdoa_matrix: np.ndarray  # Time difference of arrival matrix
    distance_matrix: np.ndarray  # Distance difference matrix
    method: str  # Triangulation method used
    convergence_info: Dict  # Additional convergence information


class TriangulationEngine:
    """
    Multi-sensor triangulation engine using Time Difference of Arrival (TDOA).
    """
    
    def __init__(self, sensor_positions: np.ndarray, speed_of_sound: float = 343.0):
        """
        Initialize triangulation engine.
        
        Args:
            sensor_positions: Array of shape (N, 3) with sensor positions
            speed_of_sound: Speed of sound in m/s
        """
        self.sensor_positions = np.array(sensor_positions, dtype=np.float64)
        self.num_sensors = self.sensor_positions.shape[0]
        self.speed_of_sound = speed_of_sound
        
        if self.num_sensors < 4:
            logger.warning(f"Only {self.num_sensors} sensors available. Need at least 4 for 3D triangulation.")
        
        logger.info(f"Initialized triangulation engine with {self.num_sensors} sensors")
    
    def compute_tdoa_matrix(self, sensor_audio: np.ndarray, sample_rate: int, 
                           method: str = 'gcc_phat') -> np.ndarray:
        """
        Compute Time Difference of Arrival (TDOA) matrix between all sensor pairs.
        
        Args:
            sensor_audio: Audio data of shape (num_samples, num_sensors)
            sample_rate: Audio sample rate in Hz
            method: TDOA estimation method ('gcc_phat', 'cross_correlation', 'phase_transform')
            
        Returns:
            TDOA matrix of shape (num_sensors, num_sensors) in seconds
            tdoa[i,j] = time for sound to reach sensor j relative to sensor i
        """
        num_samples, num_sensors = sensor_audio.shape
        tdoa_matrix = np.zeros((num_sensors, num_sensors))
        
        for i in range(num_sensors):
            for j in range(i + 1, num_sensors):
                if method == 'gcc_phat':
                    delay = self._gcc_phat_delay(sensor_audio[:, i], sensor_audio[:, j], sample_rate)
                elif method == 'cross_correlation':
                    delay = self._cross_correlation_delay(sensor_audio[:, i], sensor_audio[:, j], sample_rate)
                elif method == 'phase_transform':
                    delay = self._phase_transform_delay(sensor_audio[:, i], sensor_audio[:, j], sample_rate)
                else:
                    raise ValueError(f"Unknown TDOA method: {method}")
                
                tdoa_matrix[i, j] = delay
                tdoa_matrix[j, i] = -delay  # Symmetric
        
        return tdoa_matrix
    
    def _gcc_phat_delay(self, signal1: np.ndarray, signal2: np.ndarray, sample_rate: int) -> float:
        """
        Generalized Cross-Correlation with Phase Transform (GCC-PHAT) delay estimation.
        More robust to noise and reverberation than simple cross-correlation.
        """
        # Zero-pad signals to avoid circular correlation artifacts
        n = len(signal1) + len(signal2) - 1
        n_fft = 2 ** int(np.ceil(np.log2(n)))
        
        # FFT of both signals
        fft1 = np.fft.fft(signal1, n_fft)
        fft2 = np.fft.fft(signal2, n_fft)
        
        # Cross-power spectral density
        cross_psd = fft1 * np.conj(fft2)
        
        # Phase transform (PHAT weighting)
        magnitude = np.abs(cross_psd)
        magnitude[magnitude < 1e-10] = 1e-10  # Avoid division by zero
        gcc_phat = cross_psd / magnitude
        
        # IFFT to get correlation
        correlation = np.fft.ifft(gcc_phat)
        correlation = np.real(correlation)
        
        # Find peak
        peak_idx = np.argmax(np.abs(correlation))
        
        # Convert to delay in seconds
        if peak_idx > n_fft // 2:
            peak_idx -= n_fft
        
        delay_seconds = peak_idx / sample_rate
        return delay_seconds
    
    def _cross_correlation_delay(self, signal1: np.ndarray, signal2: np.ndarray, sample_rate: int) -> float:
        """Simple cross-correlation delay estimation."""
        correlation = correlate(signal1, signal2, mode='full')
        peak_idx = np.argmax(np.abs(correlation))
        delay_samples = peak_idx - (len(correlation) // 2)
        return delay_samples / sample_rate
    
    def _phase_transform_delay(self, signal1: np.ndarray, signal2: np.ndarray, sample_rate: int) -> float:
        """Phase transform delay estimation."""
        # This is a simplified version - could be enhanced
        return self._gcc_phat_delay(signal1, signal2, sample_rate)
    
    def triangulate_least_squares(self, tdoa_matrix: np.ndarray, 
                                 initial_guess: Optional[np.ndarray] = None) -> TriangulationResult:
        """
        Triangulate source position using least squares optimization.
        
        Args:
            tdoa_matrix: TDOA matrix of shape (num_sensors, num_sensors)
            initial_guess: Initial position guess (x, y, z). If None, uses centroid.
            
        Returns:
            TriangulationResult with estimated position and metadata
        """
        if initial_guess is None:
            # Use sensor centroid as initial guess, elevated by 50m
            initial_guess = np.mean(self.sensor_positions, axis=0)
            initial_guess[2] += 50.0  # Assume source is elevated
        
        # Convert TDOA to distance differences
        distance_diff_matrix = tdoa_matrix * self.speed_of_sound
        
        def residual_function(source_pos):
            """Compute residuals for least squares optimization."""
            residuals = []
            source_pos = np.array(source_pos)
            
            for i in range(self.num_sensors):
                for j in range(i + 1, self.num_sensors):
                    # Predicted distance difference
                    dist_i = np.linalg.norm(source_pos - self.sensor_positions[i])
                    dist_j = np.linalg.norm(source_pos - self.sensor_positions[j])
                    predicted_diff = dist_i - dist_j
                    
                    # Observed distance difference
                    observed_diff = distance_diff_matrix[i, j]
                    
                    # Residual
                    residuals.append(predicted_diff - observed_diff)
            
            return np.array(residuals)
        
        # Solve using least squares
        try:
            result = least_squares(residual_function, initial_guess, method='lm')
            
            # Compute confidence based on residual error and convergence
            residual_error = np.sqrt(np.mean(result.fun**2))
            confidence = np.exp(-residual_error / 10.0)  # Exponential decay with error
            confidence = max(0.0, min(1.0, confidence))
            
            # Count how many sensor pairs contributed
            num_pairs = self.num_sensors * (self.num_sensors - 1) // 2
            
            return TriangulationResult(
                position=result.x,
                confidence=confidence,
                residual_error=residual_error,
                num_sensors_used=self.num_sensors,
                tdoa_matrix=tdoa_matrix,
                distance_matrix=distance_diff_matrix,
                method='least_squares',
                convergence_info={
                    'success': result.success,
                    'nfev': result.nfev,
                    'cost': result.cost,
                    'optimality': result.optimality
                }
            )
            
        except Exception as e:
            logger.error(f"Least squares triangulation failed: {e}")
            # Return fallback result
            return TriangulationResult(
                position=initial_guess,
                confidence=0.0,
                residual_error=float('inf'),
                num_sensors_used=0,
                tdoa_matrix=tdoa_matrix,
                distance_matrix=distance_diff_matrix,
                method='least_squares_failed',
                convergence_info={'error': str(e)}
            )
    
    def triangulate_weighted_least_squares(self, tdoa_matrix: np.ndarray, 
                                          signal_strengths: np.ndarray,
                                          initial_guess: Optional[np.ndarray] = None) -> TriangulationResult:
        """
        Weighted least squares triangulation using signal strength as weights.
        
        Args:
            tdoa_matrix: TDOA matrix
            signal_strengths: Signal strength at each sensor (for weighting)
            initial_guess: Initial position guess
            
        Returns:
            TriangulationResult
        """
        if initial_guess is None:
            initial_guess = np.mean(self.sensor_positions, axis=0)
            initial_guess[2] += 50.0
        
        # Normalize signal strengths to use as weights
        weights = signal_strengths / np.max(signal_strengths)
        distance_diff_matrix = tdoa_matrix * self.speed_of_sound
        
        def weighted_residual_function(source_pos):
            """Weighted residual function."""
            residuals = []
            residual_weights = []
            source_pos = np.array(source_pos)
            
            for i in range(self.num_sensors):
                for j in range(i + 1, self.num_sensors):
                    dist_i = np.linalg.norm(source_pos - self.sensor_positions[i])
                    dist_j = np.linalg.norm(source_pos - self.sensor_positions[j])
                    predicted_diff = dist_i - dist_j
                    observed_diff = distance_diff_matrix[i, j]
                    
                    residual = predicted_diff - observed_diff
                    weight = np.sqrt(weights[i] * weights[j])  # Geometric mean of weights
                    
                    residuals.append(residual * weight)
                    residual_weights.append(weight)
            
            return np.array(residuals)
        
        try:
            result = least_squares(weighted_residual_function, initial_guess, method='lm')
            
            residual_error = np.sqrt(np.mean(result.fun**2))
            confidence = np.exp(-residual_error / 8.0)  # Slightly more optimistic for weighted
            confidence = max(0.0, min(1.0, confidence))
            
            return TriangulationResult(
                position=result.x,
                confidence=confidence,
                residual_error=residual_error,
                num_sensors_used=self.num_sensors,
                tdoa_matrix=tdoa_matrix,
                distance_matrix=distance_diff_matrix,
                method='weighted_least_squares',
                convergence_info={
                    'success': result.success,
                    'nfev': result.nfev,
                    'cost': result.cost
                }
            )
            
        except Exception as e:
            logger.error(f"Weighted least squares triangulation failed: {e}")
            return TriangulationResult(
                position=initial_guess,
                confidence=0.0,
                residual_error=float('inf'),
                num_sensors_used=0,
                tdoa_matrix=tdoa_matrix,
                distance_matrix=distance_diff_matrix,
                method='weighted_least_squares_failed',
                convergence_info={'error': str(e)}
            )
    
    def triangulate_robust(self, tdoa_matrix: np.ndarray, 
                          signal_strengths: Optional[np.ndarray] = None) -> TriangulationResult:
        """
        Robust triangulation using multiple methods and outlier rejection.
        
        Args:
            tdoa_matrix: TDOA matrix
            signal_strengths: Optional signal strengths for weighting
            
        Returns:
            Best TriangulationResult from multiple methods
        """
        results = []
        
        # Try multiple initial guesses
        sensor_centroid = np.mean(self.sensor_positions, axis=0)
        initial_guesses = [
            sensor_centroid + np.array([0, 0, 50]),    # Above centroid
            sensor_centroid + np.array([50, 0, 30]),   # East of centroid
            sensor_centroid + np.array([-50, 0, 30]),  # West of centroid
            sensor_centroid + np.array([0, 50, 30]),   # North of centroid
            sensor_centroid + np.array([0, -50, 30]),  # South of centroid
        ]
        
        # Try least squares with different initial guesses
        for i, guess in enumerate(initial_guesses):
            try:
                result = self.triangulate_least_squares(tdoa_matrix, guess)
                result.method = f'least_squares_init_{i}'
                results.append(result)
            except Exception as e:
                logger.debug(f"Initial guess {i} failed: {e}")
        
        # Try weighted least squares if signal strengths available
        if signal_strengths is not None:
            for i, guess in enumerate(initial_guesses[:3]):  # Try fewer for weighted
                try:
                    result = self.triangulate_weighted_least_squares(tdoa_matrix, signal_strengths, guess)
                    result.method = f'weighted_ls_init_{i}'
                    results.append(result)
                except Exception as e:
                    logger.debug(f"Weighted LS with guess {i} failed: {e}")
        
        if not results:
            logger.error("All triangulation methods failed")
            return TriangulationResult(
                position=sensor_centroid,
                confidence=0.0,
                residual_error=float('inf'),
                num_sensors_used=0,
                tdoa_matrix=tdoa_matrix,
                distance_matrix=np.zeros_like(tdoa_matrix),
                method='all_failed',
                convergence_info={}
            )
        
        # Select best result based on combination of confidence and residual error
        def score_result(result):
            if result.residual_error == float('inf'):
                return -float('inf')
            return result.confidence - result.residual_error / 100.0
        
        best_result = max(results, key=score_result)
        best_result.method = f'robust_{best_result.method}'
        
        logger.info(f"Robust triangulation selected method: {best_result.method}")
        logger.info(f"Position: ({best_result.position[0]:.1f}, {best_result.position[1]:.1f}, {best_result.position[2]:.1f})")
        logger.info(f"Confidence: {best_result.confidence:.3f}, Residual: {best_result.residual_error:.3f}")
        
        return best_result
    
    def triangulate_audio_chunk(self, sensor_audio: np.ndarray, sample_rate: int,
                               tdoa_method: str = 'gcc_phat',
                               triangulation_method: str = 'robust') -> TriangulationResult:
        """
        Complete triangulation pipeline for an audio chunk.
        
        Args:
            sensor_audio: Audio data of shape (num_samples, num_sensors)
            sample_rate: Audio sample rate
            tdoa_method: TDOA estimation method
            triangulation_method: Triangulation method ('least_squares', 'weighted', 'robust')
            
        Returns:
            TriangulationResult
        """
        # Compute TDOA matrix
        tdoa_matrix = self.compute_tdoa_matrix(sensor_audio, sample_rate, tdoa_method)
        
        # Compute signal strengths (RMS energy per sensor)
        signal_strengths = np.sqrt(np.mean(sensor_audio**2, axis=0))
        
        # Triangulate based on method
        if triangulation_method == 'least_squares':
            result = self.triangulate_least_squares(tdoa_matrix)
        elif triangulation_method == 'weighted':
            result = self.triangulate_weighted_least_squares(tdoa_matrix, signal_strengths)
        elif triangulation_method == 'robust':
            result = self.triangulate_robust(tdoa_matrix, signal_strengths)
        else:
            raise ValueError(f"Unknown triangulation method: {triangulation_method}")
        
        return result


def validate_triangulation_result(result: TriangulationResult, 
                                sensor_positions: np.ndarray,
                                max_distance: float = 5000.0) -> bool:
    """
    Validate triangulation result for reasonableness.
    
    Args:
        result: TriangulationResult to validate
        sensor_positions: Sensor positions for bounds checking
        max_distance: Maximum reasonable distance from sensors
        
    Returns:
        True if result seems reasonable
    """
    if result.confidence < 0.1:
        return False
    
    if result.residual_error > 50.0:  # More than 50m error
        return False
    
    # Check if position is within reasonable bounds
    sensor_center = np.mean(sensor_positions, axis=0)
    distance_from_center = np.linalg.norm(result.position - sensor_center)
    
    if distance_from_center > max_distance:
        return False
    
    # Check if height is reasonable (not underground, not too high)
    if result.position[2] < -100 or result.position[2] > 1000:
        return False
    
    return True


if __name__ == "__main__":
    # Test triangulation with synthetic data
    from src.kml_parser import get_sensor_positions_xyz
    
    kml_path = "Sensor-Locations-BOP-Dharma.kml"
    if not os.path.exists(kml_path):
        print(f"KML file not found: {kml_path}")
        exit(1)
    
    try:
        # Load sensor positions
        pole_names, pole_positions = get_sensor_positions_xyz(kml_path)
        print(f"Loaded {len(pole_names)} sensor positions")
        
        # Create triangulation engine
        engine = TriangulationEngine(pole_positions)
        
        # Create synthetic audio data for testing
        sample_rate = 16000
        duration = 0.25  # seconds
        num_samples = int(sample_rate * duration)
        
        # Simulate a source at known position
        true_source_pos = np.array([100.0, 150.0, 75.0])  # x, y, z in meters
        print(f"True source position: ({true_source_pos[0]:.1f}, {true_source_pos[1]:.1f}, {true_source_pos[2]:.1f})")
        
        # Generate synthetic sensor audio (random noise plus delayed sinusoid)
        synthetic_audio = np.random.randn(num_samples, len(pole_positions)) * 0.1
        
        # Add a simple signal with realistic delays
        signal_freq = 1000  # Hz
        t = np.linspace(0, duration, num_samples)
        base_signal = np.sin(2 * np.pi * signal_freq * t)
        
        for i, pole_pos in enumerate(pole_positions):
            distance = np.linalg.norm(true_source_pos - pole_pos)
            delay_samples = int(distance / 343.0 * sample_rate)
            attenuation = 1.0 / (distance ** 1.2)
            
            if delay_samples < num_samples:
                delayed_signal = np.zeros_like(base_signal)
                delayed_signal[delay_samples:] = base_signal[:-delay_samples] if delay_samples > 0 else base_signal
                synthetic_audio[:, i] += delayed_signal * attenuation
        
        # Perform triangulation
        print("\nPerforming triangulation...")
        result = engine.triangulate_audio_chunk(synthetic_audio, sample_rate)
        
        print(f"Estimated position: ({result.position[0]:.1f}, {result.position[1]:.1f}, {result.position[2]:.1f})")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Residual error: {result.residual_error:.3f} m")
        print(f"Method: {result.method}")
        
        # Calculate error
        error = np.linalg.norm(result.position - true_source_pos)
        print(f"Triangulation error: {error:.1f} m")
        
        # Validate result
        is_valid = validate_triangulation_result(result, pole_positions)
        print(f"Result validation: {'PASS' if is_valid else 'FAIL'}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
