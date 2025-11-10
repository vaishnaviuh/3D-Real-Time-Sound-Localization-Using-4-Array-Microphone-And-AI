import numpy as np
from config import SAMPLE_RATE, ENERGY_THRESHOLD, DEBUG_MODE, MIN_PEAK_RATIO


class TDOAEstimator:
    """Estimates Time Difference of Arrival (TDOA) between microphone pairs"""
    
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        max_delay_time = 0.01  # 10ms
        self.max_tau = int(sample_rate * max_delay_time)
        self.search_range = self.max_tau * 2 + 1
        self._debug_counter = 0
        
    def cross_correlation(self, signal1, signal2):
        signal1 = signal1 - np.mean(signal1)
        signal2 = signal2 - np.mean(signal2)
        correlation = np.correlate(signal1, signal2, mode='full')
        lags = np.arange(-len(signal2) + 1, len(signal1))
        max_idx = np.argmax(np.abs(correlation))
        tau = lags[max_idx]
        return tau, correlation
    
    def gcc_phat(self, signal1, signal2):
        signal1 = signal1 - np.mean(signal1)
        signal2 = signal2 - np.mean(signal2)
        
        # Normalize signals to help with peak detection
        norm1 = np.linalg.norm(signal1)
        norm2 = np.linalg.norm(signal2)
        if norm1 > 1e-10:
            signal1 = signal1 / norm1
        if norm2 > 1e-10:
            signal2 = signal2 / norm2
        
        signal1 = signal1 * np.hanning(len(signal1))
        signal2 = signal2 * np.hanning(len(signal2))
        n = 2 ** int(np.ceil(np.log2(len(signal1) + len(signal2) - 1)))
        fft1 = np.fft.fft(signal1, n)
        fft2 = np.fft.fft(signal2, n)
        cross_power = fft1 * np.conj(fft2)
        magnitude = np.abs(cross_power)
        # Use smaller epsilon to avoid division issues
        cross_power = cross_power / (magnitude + 1e-12)
        correlation = np.fft.ifft(cross_power)
        correlation = np.real(correlation)
        center = len(correlation) // 2
        search_start = max(0, center - self.max_tau)
        search_end = min(len(correlation), center + self.max_tau + 1)
        search_corr = correlation[search_start:search_end]
        if len(search_corr) == 0:
            return 0, correlation
        
        # Find peak with better detection
        abs_corr = np.abs(search_corr)
        max_idx = np.argmax(abs_corr)
        peak_value = abs_corr[max_idx]
        
        # Calculate statistics for quality check
        mean_corr = np.mean(abs_corr)
        std_corr = np.std(abs_corr)
        median_corr = np.median(abs_corr)
        
        # Improved peak detection: peak must be significantly above noise floor
        # Use both mean and median to be more robust
        noise_floor = float(max(float(mean_corr), float(median_corr)))
        peak_ratio = float(peak_value) / (noise_floor + 1e-12)
        
        # Require peak to be MIN_PEAK_RATIO times above noise floor
        if peak_ratio >= MIN_PEAK_RATIO and peak_value > std_corr * 2.0:
            tau = max_idx + search_start - center
            return tau, correlation
        else:
            # Peak not significant enough - reject as noise
            return 0, correlation
    
    def estimate_tdoa(self, signals, method='gcc_phat'):
        num_mics = len(signals)
        tdoa_matrix = np.zeros((num_mics, num_mics))
        energies = [np.mean(sig**2) for sig in signals]
        avg_energy = np.mean(energies)
        max_energy = np.max(energies)
        min_energy = np.min(energies)
        
        # Calculate SNR (signal-to-noise ratio)
        # Use min energy as noise floor estimate
        if min_energy > 1e-12:
            snr_linear = avg_energy / min_energy
            snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else 0
        else:
            snr_db = 0
        
        if DEBUG_MODE:
            self._debug_counter += 1
            if self._debug_counter % 100 == 0:
                print(f"Signal energy - Avg: {avg_energy:.6f}, Max: {max_energy:.6f}, Min: {min_energy:.6f}, SNR: {snr_db:.1f}dB, Threshold: {ENERGY_THRESHOLD:.6f}")
        
        # Require both minimum energy AND minimum SNR
        if avg_energy < ENERGY_THRESHOLD:
            return tdoa_matrix
        
        # Require minimum SNR to filter out noisy signals (but more lenient)
        from config import MIN_SNR_DB
        if snr_db < MIN_SNR_DB:
            if DEBUG_MODE and self._debug_counter % 50 == 0:
                print(f"  Rejected: SNR too low ({snr_db:.1f}dB < {MIN_SNR_DB}dB)")
            return tdoa_matrix
        
        # Require at least 2 channels with significant energy (more lenient threshold)
        channel_threshold = ENERGY_THRESHOLD * 0.3  # Lowered from 0.5 to 0.3 - more lenient
        active_channels = sum(1 for e in energies if e >= channel_threshold)
        if active_channels < 2:
            if DEBUG_MODE and self._debug_counter % 50 == 0:
                print(f"  Rejected: Only {active_channels} active channel(s) (need at least 2, threshold: {channel_threshold:.8f})")
            return tdoa_matrix
        
        for i in range(num_mics):
            for j in range(i + 1, num_mics):
                # Require both channels to have sufficient energy
                if energies[i] < channel_threshold or energies[j] < channel_threshold:
                    continue
                
                if method == 'gcc_phat':
                    tau, corr = self.gcc_phat(signals[i], signals[j])
                    # If GCC-PHAT failed (tau=0), try cross-correlation as fallback
                    if tau == 0:
                        tau, _ = self.cross_correlation(signals[i], signals[j])
                    
                    # Check if we found a valid peak (not just noise)
                    if abs(tau) < self.max_tau and tau != 0:  # Within expected range and non-zero
                        tdoa = tau / self.sample_rate
                        tdoa_matrix[i, j] = tdoa
                        tdoa_matrix[j, i] = -tdoa
                else:
                    tau, _ = self.cross_correlation(signals[i], signals[j])
                    if abs(tau) < self.max_tau and tau != 0:
                        tdoa = tau / self.sample_rate
                        tdoa_matrix[i, j] = tdoa
                        tdoa_matrix[j, i] = -tdoa
        return tdoa_matrix


