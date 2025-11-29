#!/usr/bin/env python3
"""
Find the 20-second chunk with highest harmonics in the WAV file and create plots.
"""
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import spectrogram, find_peaks
import os

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

def find_best_harmonic_chunk(wav_path, chunk_duration=20.0):
    """
    Find the 20-second chunk with highest harmonics.
    
    Returns:
        best_chunk_data: Audio data for best chunk
        best_start_time: Start time of best chunk in seconds
        best_score: Harmonic score
        best_fundamental: Fundamental frequency
        sample_rate: Sample rate
    """
    print(f"📂 Loading WAV file: {wav_path}")
    audio_data, sample_rate = sf.read(wav_path, always_2d=True)
    audio_data = audio_data.T  # Shape: (channels, samples)
    
    duration_s = audio_data.shape[1] / sample_rate
    print(f"   ✅ Loaded: {audio_data.shape[0]} channels, {duration_s:.1f}s, {sample_rate}Hz")
    
    # Use the first channel (or average across channels) for harmonic analysis
    if audio_data.shape[0] > 1:
        print("   📊 Using average of all channels for harmonic analysis")
        signal = np.mean(audio_data, axis=0)
    else:
        signal = audio_data[0]
    
    # Split into 20-second chunks with 50% overlap
    chunk_samples = int(chunk_duration * sample_rate)
    step_samples = chunk_samples // 2  # 50% overlap
    
    total_samples = len(signal)
    num_chunks = (total_samples - chunk_samples) // step_samples + 1
    
    print(f"🔍 Analyzing {num_chunks} chunks ({chunk_duration}s each, 50% overlap)...")
    
    best_score = 0.0
    best_start_sample = 0
    best_fundamental = None
    best_num_harmonics = 0
    
    for i in range(num_chunks):
        start_sample = i * step_samples
        end_sample = min(start_sample + chunk_samples, total_samples)
        chunk = signal[start_sample:end_sample]
        
        if len(chunk) < chunk_samples * 0.9:  # Skip incomplete chunks
            continue
        
        score, fundamental, num_harmonics = compute_harmonic_score(chunk, sample_rate)
        start_time = start_sample / sample_rate
        
        if score > best_score:
            best_score = score
            best_start_sample = start_sample
            best_fundamental = fundamental
            best_num_harmonics = num_harmonics
        
        if (i + 1) % 10 == 0:
            print(f"   Processed {i+1}/{num_chunks} chunks... (best so far: t={best_start_sample/sample_rate:.1f}s, score={best_score:.1f})")
    
    best_start_time = best_start_sample / sample_rate
    best_end_sample = min(best_start_sample + chunk_samples, total_samples)
    best_chunk = signal[best_start_sample:best_end_sample]
    
    print(f"\n🎯 Best chunk found!")
    print(f"   Start time: {best_start_time:.1f}s")
    print(f"   End time: {best_end_sample/sample_rate:.1f}s")
    print(f"   Harmonic score: {best_score:.1f}")
    print(f"   Fundamental frequency: {best_fundamental:.1f} Hz" if best_fundamental else "   Fundamental frequency: None")
    print(f"   Number of harmonics: {best_num_harmonics}")
    
    return best_chunk, best_start_time, best_score, best_fundamental, sample_rate, audio_data

def create_spectrogram_plots(chunk_data, start_time, fundamental, sample_rate, output_dir="plots"):
    """
    Create comprehensive spectrogram plots for the best chunk.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📊 Creating spectrogram plots...")
    
    # Compute spectrogram
    nperseg = min(4096, max(1024, sample_rate // 4))
    noverlap = nperseg // 2
    f, t, Sxx = spectrogram(chunk_data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    
    # Convert to dB
    Sxx_db = 10.0 * np.log10(Sxx + 1e-12)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Main spectrogram (full frequency range)
    ax1 = plt.subplot(3, 1, 1)
    im1 = ax1.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
    ax1.set_ylabel('Frequency (Hz)', fontsize=12)
    ax1.set_title(f'Spectrogram - Best Harmonic Chunk (t={start_time:.1f}s to {start_time+20:.1f}s)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlim([0, t[-1]])
    ax1.set_ylim([0, min(8000, f[-1])])  # Show up to 8 kHz
    plt.colorbar(im1, ax=ax1, label='Power (dB)')
    
    # Mark fundamental and harmonics if detected
    if fundamental:
        for n in range(1, 6):  # Fundamental + 4 harmonics
            freq = fundamental * n
            if freq < f[-1]:
                ax1.axhline(y=freq, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
                ax1.text(t[-1]*0.98, freq, f'{n}f₀', color='red', fontsize=9, 
                        verticalalignment='center', horizontalalignment='right', fontweight='bold')
    
    # Focused view (0-2000 Hz) for better harmonic visibility
    ax2 = plt.subplot(3, 1, 2)
    im2 = ax2.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
    ax2.set_ylabel('Frequency (Hz)', fontsize=12)
    ax2.set_title('Focused View (0-2000 Hz) - Harmonic Structure', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, t[-1]])
    ax2.set_ylim([0, 2000])
    plt.colorbar(im2, ax=ax2, label='Power (dB)')
    
    # Mark harmonics more clearly in focused view
    if fundamental:
        colors = ['red', 'orange', 'yellow', 'lime', 'cyan']
        for n in range(1, 6):
            freq = fundamental * n
            if freq < 2000:
                color = colors[(n-1) % len(colors)]
                ax2.axhline(y=freq, color=color, linestyle='--', alpha=0.8, linewidth=2)
                ax2.text(t[-1]*0.98, freq, f'{n}f₀ = {freq:.1f} Hz', color=color, 
                        fontsize=10, verticalalignment='center', horizontalalignment='right', 
                        fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
    
    # Average spectrum (time-averaged)
    ax3 = plt.subplot(3, 1, 3)
    avg_spectrum = np.mean(Sxx_db, axis=1)
    ax3.plot(f, avg_spectrum, 'b-', linewidth=1.5, label='Average Spectrum')
    ax3.set_xlabel('Frequency (Hz)', fontsize=12)
    ax3.set_ylabel('Power (dB)', fontsize=12)
    ax3.set_title('Time-Averaged Power Spectrum', fontsize=14, fontweight='bold')
    ax3.set_xlim([0, 2000])
    ax3.grid(True, alpha=0.3)
    
    # Mark peaks in spectrum
    peaks, peak_props = find_peaks(avg_spectrum, height=np.max(avg_spectrum)-20, distance=10)
    if len(peaks) > 0:
        peak_freqs = f[peaks]
        peak_mags = avg_spectrum[peaks]
        ax3.scatter(peak_freqs, peak_mags, color='red', s=50, zorder=5, label='Peaks')
        
        # Annotate prominent peaks
        for freq, mag in zip(peak_freqs[:10], peak_mags[:10]):  # Top 10 peaks
            if freq < 2000:
                ax3.annotate(f'{freq:.1f} Hz', (freq, mag), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, color='red', fontweight='bold')
    
    # Mark fundamental and harmonics on spectrum
    if fundamental:
        for n in range(1, 6):
            freq = fundamental * n
            if freq < 2000:
                idx = np.argmin(np.abs(f - freq))
                ax3.axvline(x=freq, color='green', linestyle=':', alpha=0.6, linewidth=1)
                ax3.text(freq, avg_spectrum[idx] + 2, f'{n}f₀', 
                        color='green', fontsize=9, ha='center', fontweight='bold')
    
    ax3.legend()
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, "best_harmonic_chunk_spectrogram.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   💾 Saved: {output_file}")
    
    plt.show()
    
    return output_file

def main():
    """Main function."""
    import sys
    sys.stdout.flush()  # Ensure output is flushed
    
    wav_path = "multi-20251122-141610-627897594.wav"
    
    if not os.path.exists(wav_path):
        print(f"❌ WAV file not found: {wav_path}")
        sys.stdout.flush()
        return
    
    try:
        # Find best harmonic chunk
        best_chunk, start_time, score, fundamental, sample_rate, full_audio = find_best_harmonic_chunk(
            wav_path, chunk_duration=20.0
        )
        
        # Create plots
        output_file = create_spectrogram_plots(
            best_chunk, start_time, fundamental, sample_rate
        )
        
        print(f"\n🎉 Analysis Complete!")
        print(f"   📊 Best chunk: {start_time:.1f}s - {start_time+20:.1f}s")
        print(f"   🎯 Harmonic score: {score:.1f}")
        print(f"   📈 Fundamental: {fundamental:.1f} Hz" if fundamental else "   📈 Fundamental: Not detected")
        print(f"   💾 Plot saved: {output_file}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

