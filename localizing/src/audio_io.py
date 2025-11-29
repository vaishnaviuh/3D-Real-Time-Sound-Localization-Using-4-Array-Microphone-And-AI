import numpy as np
import librosa
from pathlib import Path
import soundfile as sf
from typing import Tuple, Optional, Union


def load_wav_file(file_path: Union[str, Path],
                  target_sr: Optional[int] = None,
                  mono: bool = True,
                  logging: bool = False) -> Tuple[np.ndarray, int]:
    """
    Load a WAV file.

    Parameters:
    -----------
    file_path : str or Path
        Path to the WAV file (absolute or relative to current working directory).
    target_sr : int, optional
        Target sample rate for resampling. If None, keeps original sample rate.
    mono : bool, default=True
        Whether to convert to mono if stereo.
    logging : bool, default=False
        Whether to print loading information.

    Returns:
    --------
    audio_data : np.ndarray
        Audio data as numpy array
    sample_rate : int
        Sample rate of the audio

    Example:
    --------
    >>> audio, sr = load_wav_file("recording.wav")
    >>> print(f"Audio shape: {audio.shape}, Sample rate: {sr}")
    """
    # Convert to Path object
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"WAV file not found: {file_path}")

    try:
        # Load audio using librosa (handles various formats well)
        audio_data, sample_rate = librosa.load(
            str(file_path),
            sr=target_sr,
            mono=mono
        )

        if logging:
            print(f"Loaded: {file_path.name}")
            print(f"  Duration: {len(audio_data)/sample_rate:.2f} seconds")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Shape: {audio_data.shape}")

        return audio_data, sample_rate

    except Exception as e:
        # Fallback to soundfile if librosa fails
        try:
            audio_data, sample_rate = sf.read(str(file_path))

            # Convert to mono if stereo and mono=True
            if mono and len(audio_data.shape) > 1:
                if logging:
                    print("Converting to mono by selecting the first channel")
                audio_data = audio_data[:, 0]

            # Resample if target_sr is specified
            if target_sr is not None and target_sr != sample_rate:
                if logging:
                    print(f"Resampling to target sample rate {target_sr}")
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
                sample_rate = target_sr

            if logging:
                print(f"Loaded (soundfile): {file_path.name}")
                print(f"  Duration: {len(audio_data)/sample_rate:.2f} seconds")
                print(f"  Sample rate: {sample_rate} Hz")
                print(f"  Shape: {audio_data.shape}")

            return audio_data, sample_rate

        except Exception as e2:
            raise RuntimeError(f"Failed to load WAV file {file_path}: {e2}")


def get_audio_segment(audio_data: np.ndarray,
                      sample_rate: int,
                      start_time: Optional[float],
                      end_time: Optional[float]) -> np.ndarray:
    """
    Return a slice of audio_data between start_time and end_time (seconds).
    
    Parameters:
    -----------
    audio_data : np.ndarray
        Audio data array
    sample_rate : int
        Sample rate in Hz
    start_time : float, optional
        Start time in seconds (None = beginning)
    end_time : float, optional
        End time in seconds (None = end)
    
    Returns:
    --------
    audio_segment : np.ndarray
        Extracted audio segment
    """
    if start_time is not None:
        start_sample = int(start_time * sample_rate)
    else:
        start_sample = 0

    if end_time is not None:
        end_sample = int(end_time * sample_rate)
    else:
        end_sample = len(audio_data)

    start_sample = max(0, min(start_sample, len(audio_data)))
    end_sample = max(start_sample, min(end_sample, len(audio_data)))

    audio_segment = audio_data[start_sample:end_sample]
    if len(audio_segment) == 0:
        raise ValueError("No audio data in the specified time range")

    return audio_segment


def save_audio_to_wav(
    audio_data: np.ndarray,
    sample_rate: int,
    filename: Union[str, Path],
    normalize: bool = True,
    target_dbfs: float = -20.0
) -> bool:
    """
    Save audio data to a WAV file with optional normalization.

    Parameters
    ----------
    audio_data : np.ndarray
        Audio data array (can be 1D for mono or 2D for multichannel).
        Shape should be (num_samples,) for mono or (num_samples, num_channels) for multichannel.
    sample_rate : int
        Sampling rate in Hz.
    filename : str or Path
        Output filename (should end with .wav).
    normalize : bool, optional
        Whether to normalize the audio to prevent clipping. Default is True.
    target_dbfs : float, optional
        Target dBFS level for normalization (default -20 dBFS).

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    # Ensure audio_data is a numpy array
    audio_data = np.array(audio_data)

    # Handle stereo vs mono
    if audio_data.ndim == 1:
        # Mono - keep as is
        pass
    elif audio_data.ndim == 2:
        # Multichannel - ensure it's in the right format (samples, channels)
        if audio_data.shape[0] == 2 and audio_data.shape[1] > 2:
            # Likely (channels, samples) format, transpose to (samples, channels)
            audio_data = audio_data.T
    else:
        print("Error: Audio data must be 1D (mono) or 2D (multichannel)")
        return False

    # Normalize if requested
    if normalize:
        # Calculate RMS
        rms = np.sqrt(np.mean(audio_data**2))

        if rms > 0:
            # Convert target_dbfs to linear scale
            target_rms = 10**(target_dbfs / 20)

            # Calculate scaling factor
            scale_factor = target_rms / rms

            # Apply scaling, but prevent clipping
            max_scale = 0.95 / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 0 else 1.0
            scale_factor = min(scale_factor, max_scale)

            audio_data = audio_data * scale_factor

    # Ensure audio data is float32 for soundfile
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # Save to WAV file
    try:
        sf.write(str(filename), audio_data, sample_rate)
        print(f"Successfully saved audio to: {filename}")
        print(f"Duration: {len(audio_data) / sample_rate:.2f} seconds")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Channels: {audio_data.shape[-1] if audio_data.ndim > 1 else 1}")
        return True
    except Exception as e:
        print(f"Error saving WAV file: {e}")
        return False

