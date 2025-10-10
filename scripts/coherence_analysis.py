from scipy.signal import butter, filtfilt, coherence
import librosa
import numpy as np

def bandpass_filter(signal, sr, lowcut=300, highcut=3000, order=4):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

file = "/content/DroneSound_noMuff.wav"
y, sr = librosa.load(file, sr=None, mono=False)
if y.ndim == 1:
    y = np.expand_dims(y, axis=0)

num_channels = min(5, y.shape[0])
print(f"Processing {num_channels} channels.")

n_fft = 2048
hop_length = 512

channel_specs = []
for i in range(num_channels):
    S = librosa.stft(y[i], n_fft=n_fft, hop_length=hop_length)
    channel_specs.append(S)
channel_specs = np.stack(channel_specs, axis=-1)

window_size = 2048
step_size = 512
nperseg = 256

all_coh_specs = []
for i in range(num_channels):
    sig1 = bandpass_filter(y[i], sr)
    for j in range(i+1, num_channels):
        sig2 = bandpass_filter(y[j], sr)
        n_windows = (len(sig1) - window_size) // step_size + 1
        coh_spec = []
        for w in range(n_windows):
            start = w * step_size
            end = start + window_size
            f, Cxy = coherence(sig1[start:end], sig2[start:end], fs=sr, nperseg=nperseg)
            coh_spec.append(Cxy)
        coh_spec = np.array(coh_spec).T
        all_coh_specs.append(coh_spec)

min_time_bins = min(spec.shape[1] for spec in all_coh_specs)
all_coh_specs_trunc = [spec[:, :min_time_bins] for spec in all_coh_specs]
median_coh_spec = np.median(np.stack(all_coh_specs_trunc, axis=-1), axis=-1)
