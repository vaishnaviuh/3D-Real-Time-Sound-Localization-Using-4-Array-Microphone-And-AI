#!/usr/bin/env python3
"""
Create spectrograms for original and filtered audio files.
This script works with already processed files to avoid memory issues.
"""

import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os

def create_spectrogram_from_file(input_file, title, filename, max_duration=30):
    """Create and save spectrogram from audio file."""
    print(f"Creating spectrogram for {title}...")
    
    # Load audio file
    audio, sample_rate = librosa.load(input_file, sr=None, duration=max_duration)
    
    # Create spectrogram with smaller parameters to save memory
    plt.figure(figsize=(12, 8))
    
    # Use smaller hop_length and n_fft to reduce memory usage
    hop_length = 1024
    n_fft = 2048
    
    # Use librosa for better spectrogram with noise reduction
    stft = librosa.stft(audio, hop_length=hop_length, n_fft=n_fft)
    
    # Apply noise reduction by smoothing the magnitude spectrum
    magnitude = np.abs(stft)
    
    # Apply median filtering to reduce noise
    from scipy.ndimage import median_filter
    magnitude_smoothed = median_filter(magnitude, size=(3, 3))
    
    # Convert to dB with noise floor
    D = librosa.amplitude_to_db(magnitude_smoothed, ref=np.max, top_db=80)
    
    # Plot spectrogram with viridis colormap and noise reduction
    librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='hz', 
                            hop_length=hop_length, cmap='viridis', vmin=-80, vmax=0)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram - {title}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, 8000)  # Full frequency range
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Spectrogram saved to: {filename}")

def main():
    """Create spectrograms for all files."""
    files = [
        ('drone_data1.wav', 'drone_data1_filtered.wav'),
        ('drone_data2.WAV', 'drone_data2_filtered.wav')
    ]
    
    for original_file, filtered_file in files:
        if os.path.exists(original_file):
            # Original spectrogram
            base_name = os.path.splitext(original_file)[0]
            create_spectrogram_from_file(
                original_file,
                f"Original - {original_file}",
                f"{base_name}_original_spectrogram.png"
            )
        
        if os.path.exists(filtered_file):
            # Filtered spectrogram
            base_name = os.path.splitext(original_file)[0]
            create_spectrogram_from_file(
                filtered_file,
                f"High-pass Filtered (1000 Hz) - {original_file}",
                f"{base_name}_filtered_spectrogram.png"
            )
    
    print("All spectrograms created!")

if __name__ == "__main__":
    main()
