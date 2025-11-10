#!/usr/bin/env python3
"""
High-pass filter audio files with 1000 Hz cutoff and generate spectrograms.
Processes files in chunks to avoid memory issues.
"""

import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, sosfilt
import os
from tqdm import tqdm

def design_highpass_filter(cutoff_freq, sample_rate, order=5):
    """Design a high-pass Butterworth filter using SOS (Second-Order Sections)."""
    from scipy.signal import butter, sosfilt
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    sos = butter(order, normalized_cutoff, btype='high', analog=False, output='sos')
    return sos

def apply_highpass_filter(data, cutoff_freq, sample_rate, order=5):
    """Apply high-pass filter to audio data using SOS filter."""
    sos = design_highpass_filter(cutoff_freq, sample_rate, order)
    filtered_data = sosfilt(sos, data)
    return filtered_data

def process_audio_chunked(input_file, output_file, cutoff_freq=1000, chunk_size=44100*30):
    """
    Process audio file in chunks to avoid memory issues.
    chunk_size: number of samples per chunk (default: 30 seconds at 44.1kHz)
    """
    print(f"Processing {input_file}...")
    
    # Get file info
    info = sf.info(input_file)
    sample_rate = info.samplerate
    total_samples = info.frames
    
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Total samples: {total_samples}")
    print(f"Duration: {total_samples/sample_rate:.2f} seconds")
    print(f"Chunk size: {chunk_size} samples ({chunk_size/sample_rate:.2f} seconds)")
    
    # Design filter once
    sos = design_highpass_filter(cutoff_freq, sample_rate)
    
    # Process in chunks
    filtered_chunks = []
    original_chunks = []
    
    with sf.SoundFile(input_file, 'r') as f:
        num_chunks = (total_samples + chunk_size - 1) // chunk_size
        
        for i in tqdm(range(num_chunks), desc="Processing chunks"):
            start_sample = i * chunk_size
            end_sample = min((i + 1) * chunk_size, total_samples)
            
            # Read chunk
            f.seek(start_sample)
            chunk = f.read(end_sample - start_sample)
            
            # Store original chunk for spectrogram (only first few chunks to save memory)
            if i < 3:  # Only keep first 3 chunks for spectrogram
                original_chunks.append(chunk.copy())
            
            # Apply high-pass filter
            if len(chunk) > 50:  # Ensure chunk is large enough for filter
                filtered_chunk = sosfilt(sos, chunk)
            else:
                filtered_chunk = chunk  # Skip filtering for very small chunks
            filtered_chunks.append(filtered_chunk)
    
    # Combine all chunks
    filtered_audio = np.concatenate(filtered_chunks)
    
    # For spectrogram, use first few chunks of original audio
    if original_chunks:
        original_audio = np.concatenate(original_chunks)
    else:
        # If no original chunks saved, read a small portion for spectrogram
        with sf.SoundFile(input_file, 'r') as f:
            original_audio = f.read(sample_rate * 60)  # First 60 seconds
    
    # Save filtered audio
    sf.write(output_file, filtered_audio, sample_rate)
    print(f"Filtered audio saved to: {output_file}")
    
    return original_audio, filtered_audio, sample_rate

def create_spectrogram(audio, sample_rate, title, filename, max_duration=30):
    """Create and save spectrogram of audio data."""
    print(f"Creating spectrogram for {title}...")
    
    # Limit duration for spectrogram (first max_duration seconds)
    max_samples = int(sample_rate * max_duration)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        print(f"Truncated to {max_duration} seconds for spectrogram")
    
    # Create spectrogram with smaller parameters to save memory
    plt.figure(figsize=(12, 8))
    
    # Use smaller hop_length and n_fft to reduce memory usage
    hop_length = 1024  # Increased from 512
    n_fft = 2048       # Keep reasonable
    
    # Use librosa for better spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, hop_length=hop_length, n_fft=n_fft)), ref=np.max)
    
    # Plot spectrogram
    librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='hz', 
                            hop_length=hop_length, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram - {title}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, 8000)  # Focus on relevant frequency range
    
    # Add cutoff line
    plt.axhline(y=1000, color='red', linestyle='--', linewidth=2, 
                label='High-pass cutoff (1000 Hz)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Spectrogram saved to: {filename}")

def main():
    """Main processing function."""
    # File paths
    files = [
        ('drone_data1.wav', 'drone_data1_filtered.wav'),
        ('drone_data2.WAV', 'drone_data2_filtered.wav')
    ]
    
    cutoff_freq = 1000  # Hz
    
    for input_file, output_file in files:
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping...")
            continue
            
        try:
            # Process audio
            original_audio, filtered_audio, sample_rate = process_audio_chunked(
                input_file, output_file, cutoff_freq
            )
            
            # Create spectrograms
            base_name = os.path.splitext(input_file)[0]
            
            # Original spectrogram
            create_spectrogram(
                original_audio, sample_rate, 
                f"Original - {input_file}",
                f"{base_name}_original_spectrogram.png"
            )
            
            # Filtered spectrogram
            create_spectrogram(
                filtered_audio, sample_rate,
                f"High-pass Filtered (1000 Hz) - {input_file}",
                f"{base_name}_filtered_spectrogram.png"
            )
            
            print(f"Completed processing {input_file}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            continue
    
    print("All processing completed!")

if __name__ == "__main__":
    main()
