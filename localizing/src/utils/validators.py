"""
Input validation utilities for audio processing pipeline.
"""
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_signals(
    signals: np.ndarray,
    expected_channels: int = 4,
    min_samples: int = 100,
    check_finite: bool = True
) -> bool:
    """
    Validate audio signals array.
    
    Args:
        signals: Audio signals array
        expected_channels: Expected number of channels
        min_samples: Minimum number of samples required
        check_finite: Whether to check for NaN/Inf values
    
    Returns:
        True if valid
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(signals, np.ndarray):
        raise ValidationError(f"Signals must be numpy array, got {type(signals)}")
    
    if signals.ndim != 2:
        raise ValidationError(f"Signals must be 2D array (samples, channels), got shape {signals.shape}")
    
    if signals.shape[1] != expected_channels:
        raise ValidationError(
            f"Expected {expected_channels} channels, got {signals.shape[1]}"
        )
    
    if signals.shape[0] < min_samples:
        raise ValidationError(
            f"Expected at least {min_samples} samples, got {signals.shape[0]}"
        )
    
    if check_finite:
        if np.any(np.isnan(signals)):
            raise ValidationError("Signals contain NaN values")
        if np.any(np.isinf(signals)):
            raise ValidationError("Signals contain Inf values")
    
    return True


def validate_mic_positions(
    mic_positions: np.ndarray,
    expected_count: int = 4,
    check_dimensions: bool = True
) -> bool:
    """
    Validate microphone positions array.
    
    Args:
        mic_positions: Microphone positions array
        expected_count: Expected number of microphones
        check_dimensions: Whether to check 3D coordinates
    
    Returns:
        True if valid
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(mic_positions, np.ndarray):
        raise ValidationError(f"Mic positions must be numpy array, got {type(mic_positions)}")
    
    if mic_positions.ndim != 2:
        raise ValidationError(f"Mic positions must be 2D array, got shape {mic_positions.shape}")
    
    if mic_positions.shape[0] != expected_count:
        raise ValidationError(
            f"Expected {expected_count} microphones, got {mic_positions.shape[0]}"
        )
    
    if check_dimensions and mic_positions.shape[1] != 3:
        raise ValidationError(
            f"Expected 3D coordinates, got {mic_positions.shape[1]} dimensions"
        )
    
    if np.any(np.isnan(mic_positions)):
        raise ValidationError("Mic positions contain NaN values")
    
    if np.any(np.isinf(mic_positions)):
        raise ValidationError("Mic positions contain Inf values")
    
    return True


def validate_config(config) -> bool:
    """
    Validate application configuration.
    
    Args:
        config: Application configuration object
    
    Returns:
        True if valid
    
    Raises:
        ValidationError: If validation fails
    """
    # Validate audio config
    if hasattr(config, 'audio'):
        audio = config.audio
        if audio.samplerate <= 0:
            raise ValidationError(f"Invalid sample rate: {audio.samplerate}")
        if audio.record_seconds <= 0:
            raise ValidationError(f"Invalid record duration: {audio.record_seconds}")
    
    # Validate geometry config
    if hasattr(config, 'geometry'):
        geom = config.geometry
        if geom.radius_m <= 0:
            raise ValidationError(f"Invalid microphone radius: {geom.radius_m}")
        if geom.speed_of_sound <= 0:
            raise ValidationError(f"Invalid speed of sound: {geom.speed_of_sound}")
    
    # Validate detection config
    if hasattr(config, 'detection'):
        det = config.detection
        if det.min_freq_hz >= det.max_freq_hz:
            raise ValidationError(
                f"Invalid frequency range: {det.min_freq_hz} >= {det.max_freq_hz}"
            )
    
    return True


def validate_doa_result(result: dict) -> bool:
    """
    Validate DOA estimation result.
    
    Args:
        result: DOA result dictionary
    
    Returns:
        True if valid
    
    Raises:
        ValidationError: If validation fails
    """
    required_keys = ['azimuth_deg', 'elevation_deg', 'confidence']
    for key in required_keys:
        if key not in result:
            raise ValidationError(f"Missing required key in DOA result: {key}")
    
    azimuth = result['azimuth_deg']
    elevation = result['elevation_deg']
    confidence = result['confidence']
    
    if not (0 <= azimuth < 360):
        raise ValidationError(f"Azimuth out of range: {azimuth}")
    
    if not (-90 <= elevation <= 90):
        raise ValidationError(f"Elevation out of range: {elevation}")
    
    if not (0 <= confidence <= 1):
        raise ValidationError(f"Confidence out of range: {confidence}")
    
    return True


def safe_validate(func, *args, **kwargs):
    """
    Safely validate with error handling.
    
    Args:
        func: Validation function to call
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        func(*args, **kwargs)
        return True, None
    except ValidationError as e:
        return False, str(e)
    except Exception as e:
        logger.exception(f"Unexpected error in validation: {e}")
        return False, f"Validation error: {str(e)}"

