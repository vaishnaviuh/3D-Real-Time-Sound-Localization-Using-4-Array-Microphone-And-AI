import numpy as np
import pyaudio
import queue
from config import SAMPLE_RATE, CHUNK_SIZE, NUM_CHANNELS, ACTIVE_CHANNELS


class AudioCapture:
    """Handles live audio capture from ReSpeaker microphone array"""
    
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio = None
        self.stream = None
        
    def find_respeaker_device(self):
        """Find ReSpeaker device index"""
        audio = pyaudio.PyAudio()
        device_index = None
        
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if 'respeaker' in info['name'].lower() or info['maxInputChannels'] >= 6:
                device_index = i
                print(f"Found audio device: {info['name']} (index {i})")
                print(f"  Channels: {info['maxInputChannels']}, Sample Rate: {info['defaultSampleRate']}")
                break
        
        audio.terminate()
        return device_index
    
    def start_capture(self, device_index=None):
        """Start audio capture"""
        self.audio = pyaudio.PyAudio()
        
        if device_index is None:
            device_index = self.find_respeaker_device()
            if device_index is None:
                print("ReSpeaker not found, using default device")
                device_index = None
        
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=NUM_CHANNELS,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            self.is_recording = True
            self.stream.start_stream()
            print("Audio capture started")
        except Exception as e:
            print(f"Error starting audio capture: {e}")
            raise
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Reshape to (chunk_size, num_channels)
        if len(audio_data) != self.chunk_size * NUM_CHANNELS:
            expected_samples = self.chunk_size * NUM_CHANNELS
            if len(audio_data) < expected_samples:
                audio_data = np.pad(audio_data, (0, expected_samples - len(audio_data)), mode='constant')
            else:
                audio_data = audio_data[:expected_samples]
        audio_data = audio_data.reshape(self.chunk_size, NUM_CHANNELS)
        
        # Extract only active channels (1-4)
        active_audio = audio_data[:, ACTIVE_CHANNELS]
        
        # Convert to float32 and normalize
        active_audio = active_audio.astype(np.float32) / 32768.0
        
        # Put in queue (non-blocking)
        try:
            self.audio_queue.put_nowait(active_audio)
        except queue.Full:
            pass  # Drop frame if queue is full
        
        return (None, pyaudio.paContinue)
    
    def get_audio_chunk(self, timeout=0.1):
        """Get next audio chunk from queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop_capture(self):
        """Stop audio capture"""
        self.is_recording = False
        try:
            if self.stream:
                try:
                    if hasattr(self.stream, 'is_active') and self.stream.is_active():
                        self.stream.abort_stream()
                except Exception:
                    pass
                try:
                    if hasattr(self.stream, 'is_stopped') and not self.stream.is_stopped():
                        self.stream.stop_stream()
                except Exception:
                    pass
                try:
                    self.stream.close()
                except Exception:
                    pass
                finally:
                    self.stream = None
            if self.audio:
                try:
                    self.audio.terminate()
                except Exception:
                    pass
                finally:
                    self.audio = None
        finally:
            print("Audio capture stopped")


