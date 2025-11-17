"""
Sound classification module for detecting trigger sounds (drone, mechanical noise, claps)
while ignoring speech and background noise.
"""
from typing import Dict, Tuple, Optional, List
import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq


def compute_mel_spectrogram(
    audio: np.ndarray,
    fs: int,
    n_mels: int = 128,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mel-spectrogram from audio signal.
    
    Args:
        audio: 1D audio signal
        fs: Sample rate
        n_mels: Number of mel filter banks
        hop_length: Hop length for STFT
        n_fft: FFT window size
    
    Returns:
        (mel_spec, times, freqs): Mel spectrogram, time axis, mel frequency axis
    """
    # Simple mel filter bank approximation (linear in mel scale)
    # For production, use librosa.mel_filters, but we avoid extra dependency
    mel_max = 2595 * np.log10(1 + (fs / 2) / 700)
    mel_points = np.linspace(0, mel_max, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    
    # Compute STFT
    f, t, stft = signal.stft(
        audio,
        fs=fs,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        window='hann',
    )
    magnitude = np.abs(stft)
    
    # Apply mel filter bank (simplified)
    mel_filters = np.zeros((n_mels, len(f)))
    for i in range(n_mels):
        lower = hz_points[i]
        center = hz_points[i + 1]
        upper = hz_points[i + 2]
        for j, freq in enumerate(f):
            if lower <= freq <= upper:
                if freq <= center:
                    mel_filters[i, j] = (freq - lower) / (center - lower)
                else:
                    mel_filters[i, j] = (upper - freq) / (upper - center)
    
    mel_spec = np.dot(mel_filters, magnitude)
    mel_spec = np.log10(mel_spec + 1e-10)  # Log scale
    
    return mel_spec, t, hz_points[1:-1]  # Return mel frequencies


def extract_audio_features(
    audio: np.ndarray,
    fs: int,
) -> Dict[str, float]:
    """
    Extract audio features for classification.
    
    Features:
    - spectral_centroid: Center of mass of spectrum
    - spectral_bandwidth: Spread of spectrum
    - spectral_rolloff: Frequency below which 85% of energy is contained
    - zero_crossing_rate: Rate of sign changes
    - harmonic_ratio: Ratio of harmonic to total energy
    
    Args:
        audio: 1D audio signal
        fs: Sample rate
    
    Returns:
        Dictionary of feature values
    """
    # Compute FFT
    n = len(audio)
    nfft = 1
    while nfft < n:
        nfft <<= 1
    
    fft_signal = rfft(audio, n=nfft)
    magnitude = np.abs(fft_signal)
    freqs = rfftfreq(nfft, 1.0/fs)
    
    # Normalize magnitude
    magnitude_norm = magnitude / (np.max(magnitude) + 1e-12)
    
    # Spectral centroid (weighted mean frequency)
    spectral_centroid = np.sum(freqs * magnitude_norm) / (np.sum(magnitude_norm) + 1e-12)
    
    # Spectral bandwidth (standard deviation around centroid)
    spectral_bandwidth = np.sqrt(
        np.sum(((freqs - spectral_centroid) ** 2) * magnitude_norm) / (np.sum(magnitude_norm) + 1e-12)
    )
    
    # Spectral rolloff (85% energy cutoff)
    cumsum_mag = np.cumsum(magnitude_norm)
    rolloff_threshold = 0.85 * cumsum_mag[-1]
    rolloff_idx = np.where(cumsum_mag >= rolloff_threshold)[0]
    spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else fs / 2
    
    # Zero crossing rate
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / (2.0 * len(audio))
    zero_crossing_rate = zero_crossings * fs
    
    # Harmonic ratio (simplified: ratio of peaks to total energy)
    # Find peaks in spectrum
    peaks, _ = signal.find_peaks(magnitude_norm, height=0.1, distance=10)
    if len(peaks) > 0:
        peak_energy = np.sum(magnitude_norm[peaks])
        total_energy = np.sum(magnitude_norm)
        harmonic_ratio = peak_energy / (total_energy + 1e-12)
    else:
        harmonic_ratio = 0.0
    
    # RMS energy
    rms_energy = np.sqrt(np.mean(audio ** 2))
    
    # Peak to RMS ratio
    peak_value = np.max(np.abs(audio))
    peak_to_rms = peak_value / (rms_energy + 1e-12)
    
    return {
        'spectral_centroid': float(spectral_centroid),
        'spectral_bandwidth': float(spectral_bandwidth),
        'spectral_rolloff': float(spectral_rolloff),
        'zero_crossing_rate': float(zero_crossing_rate),
        'harmonic_ratio': float(harmonic_ratio),
        'rms_energy': float(rms_energy),
        'peak_to_rms': float(peak_to_rms),
    }


def classify_sound_rule_based(
    features: Dict[str, float],
    config,  # ClassificationConfig - using Any to avoid circular import
) -> Tuple[str, float, Dict[str, bool]]:
    """
    Rule-based classifier for trigger sounds.
    
    Classes:
    - 'drone': Low-frequency harmonics, high energy, moderate bandwidth
    - 'mechanical': High energy, wide bandwidth, moderate harmonics
    - 'clap': Very short, high peak-to-RMS, wide bandwidth
    - 'speech': Lower energy, moderate bandwidth, low harmonics
    - 'background': Low energy, low activity
    - 'unknown': Doesn't match any pattern
    
    Args:
        features: Extracted audio features
        config: Classification configuration
    
    Returns:
        (predicted_class, confidence, rule_matches): Class name, confidence (0-1), rule match details
    """
    sc = features['spectral_centroid']
    sbw = features['spectral_bandwidth']
    sro = features['spectral_rolloff']
    zcr = features['zero_crossing_rate']
    hr = features['harmonic_ratio']
    rms = features['rms_energy']
    ptr = features['peak_to_rms']
    
    rule_matches = {}
    scores = {}
    
    # DRONE detection rules
    drone_score = 0.0
    drone_rules = []
    if config.drone_enabled:
        if sc < config.drone_max_centroid:
            drone_score += 0.3
            drone_rules.append('low_centroid')
        if hr > config.drone_min_harmonic_ratio:
            drone_score += 0.3
            drone_rules.append('harmonic')
        if rms > config.drone_min_rms:
            drone_score += 0.2
            drone_rules.append('high_energy')
        if sbw < config.drone_max_bandwidth:
            drone_score += 0.2
            drone_rules.append('narrow_bandwidth')
    scores['drone'] = drone_score
    rule_matches['drone'] = drone_rules
    
    # MECHANICAL noise detection rules
    mech_score = 0.0
    mech_rules = []
    if config.mechanical_enabled:
        if rms > config.mechanical_min_rms:
            mech_score += 0.3
            mech_rules.append('high_energy')
        if sbw > config.mechanical_min_bandwidth:
            mech_score += 0.3
            mech_rules.append('wide_bandwidth')
        if sc > config.mechanical_min_centroid:
            mech_score += 0.2
            mech_rules.append('mid_high_centroid')
        if hr > config.mechanical_min_harmonic_ratio:
            mech_score += 0.2
            mech_rules.append('some_harmonics')
    scores['mechanical'] = mech_score
    rule_matches['mechanical'] = mech_rules
    
    # CLAP detection rules
    clap_score = 0.0
    clap_rules = []
    if config.clap_enabled:
        if ptr > config.clap_min_peak_to_rms:
            clap_score += 0.4
            clap_rules.append('high_peak')
        if sbw > config.clap_min_bandwidth:
            clap_score += 0.3
            clap_rules.append('wide_bandwidth')
        if rms > config.clap_min_rms:
            clap_score += 0.3
            clap_rules.append('high_energy')
    scores['clap'] = clap_score
    rule_matches['clap'] = clap_rules
    
    # SPEECH detection rules (to filter out)
    speech_score = 0.0
    speech_rules = []
    if rms < config.speech_max_rms:
        speech_score += 0.3
        speech_rules.append('low_energy')
    if sbw < config.speech_max_bandwidth:
        speech_score += 0.3
        speech_rules.append('moderate_bandwidth')
    if hr < config.speech_max_harmonic_ratio:
        speech_score += 0.2
        speech_rules.append('low_harmonics')
    if zcr > config.speech_min_zcr:
        speech_score += 0.2
        speech_rules.append('high_zcr')
    scores['speech'] = speech_score
    rule_matches['speech'] = speech_rules
    
    # BACKGROUND noise (low activity)
    background_score = 0.0
    background_rules = []
    if rms < config.background_max_rms:
        background_score += 0.5
        background_rules.append('low_energy')
    if ptr < config.background_max_peak_to_rms:
        background_score += 0.5
        background_rules.append('low_peak')
    scores['background'] = background_score
    rule_matches['background'] = background_rules
    
    # Find best match (excluding speech and background if they're not targets)
    target_classes = []
    if config.drone_enabled:
        target_classes.append('drone')
    if config.mechanical_enabled:
        target_classes.append('mechanical')
    if config.clap_enabled:
        target_classes.append('clap')
    
    if not target_classes:
        # No targets enabled, accept all
        best_class = max(scores.items(), key=lambda x: x[1])[0]
        confidence = scores[best_class]
    else:
        # Only consider target classes
        target_scores = {k: v for k, v in scores.items() if k in target_classes}
        if not target_scores or max(target_scores.values()) < config.min_trigger_confidence:
            # No target class meets threshold
            best_class = 'unknown'
            confidence = 0.0
        else:
            best_class = max(target_scores.items(), key=lambda x: x[1])[0]
            confidence = scores[best_class]
    
    # Reject if speech or background score is too high (unless they're targets)
    if best_class in target_classes:
        # Check rejection thresholds FIRST - be more aggressive
        speech_score = scores.get('speech', 0)
        background_score = scores.get('background', 0)
        
        if speech_score > config.speech_rejection_threshold:
            best_class = 'speech_rejected'
            confidence = 0.0
        elif background_score > config.background_rejection_threshold:
            best_class = 'background_rejected'
            confidence = 0.0
        # Also reject if target score is not significantly higher than speech/background
        elif speech_score > 0.3 and confidence < speech_score + 0.2:
            # Target detected but speech score is close - likely speech
            best_class = 'speech_rejected'
            confidence = 0.0
        elif background_score > 0.4 and confidence < background_score + 0.2:
            # Target detected but background score is close - likely background
            best_class = 'background_rejected'
            confidence = 0.0
    
    # Final check: if best_class is unknown or rejected, ensure confidence is 0
    if best_class not in target_classes:
        confidence = 0.0
    
    return best_class, confidence, rule_matches


def classify_audio_signal(
    signals: np.ndarray,
    fs: int,
    config,  # ClassificationConfig - using Any to avoid circular import
    channel_idx: Optional[int] = None,
) -> Tuple[str, float, Dict[str, float], Dict[str, bool]]:
    """
    Classify audio signal from multichannel input.
    
    Args:
        signals: (N, M) array of audio signals
        fs: Sample rate
        config: Classification configuration
        channel_idx: Which channel to use (None = average all)
    
    Returns:
        (predicted_class, confidence, features, rule_matches)
    """
    # Use specified channel or average
    if channel_idx is not None:
        audio = signals[:, channel_idx]
    else:
        audio = np.mean(signals, axis=1)
    
    # Check minimum energy threshold first (reject pure noise/quiet signals)
    rms_energy = np.sqrt(np.mean(audio ** 2))
    min_energy = getattr(config, 'min_energy_for_classification', 0.0003)
    
    if rms_energy < min_energy:
        # Signal too weak, classify as background
        features = extract_audio_features(audio, fs)
        return 'background', 0.0, features, {}
    
    # Extract features
    features = extract_audio_features(audio, fs)
    
    # Classify
    predicted_class, confidence, rule_matches = classify_sound_rule_based(features, config)
    
    # Additional strictness: if confidence is very low, force to unknown
    if confidence < 0.3:
        predicted_class = 'unknown'
        confidence = 0.0
    
    return predicted_class, confidence, features, rule_matches


def compute_spectrogram_all_channels(
    signals: np.ndarray,
    fs: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Compute spectrograms for all channels.
    
    Args:
        signals: (N, M) array of audio signals
        fs: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
    
    Returns:
        (spectrograms, times, freqs): List of spectrograms per channel, time axis, frequency axis
    """
    num_channels = signals.shape[1]
    spectrograms = []
    
    for ch in range(num_channels):
        f, t, stft = signal.stft(
            signals[:, ch],
            fs=fs,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            window='hann',
        )
        magnitude = np.abs(stft)
        spectrograms.append(magnitude)
    
    # Use first channel's time/freq axes (all should be same)
    f, t, _ = signal.stft(
        signals[:, 0],
        fs=fs,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        window='hann',
    )
    
    return spectrograms, t, f

