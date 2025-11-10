import numpy as np

# Audio configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024  # Smaller chunks for lower latency
OVERLAP = 0.5
NUM_CHANNELS = 6
# Use channels 1-4, ignore channels 0 and 5
ACTIVE_CHANNELS = [1, 2, 3, 4]
NUM_ACTIVE_CHANNELS = len(ACTIVE_CHANNELS)
SOUND_SPEED = 343.0

# Signal processing parameters
# Balanced thresholds: filter noise but still detect drones and loud sources
ENERGY_THRESHOLD = 0.000001  # Lowered to 0.000001 (was 0.00001) - detect drones but filter quiet noise
MIN_SNR_DB = 6.0  # Lowered to 6dB (was 10dB) - still filters noisy signals but more lenient
MIN_PEAK_RATIO = 2.0  # Lowered to 2.0x (was 2.5x) - still requires good peak but more lenient
FREQ_MIN = 300
FREQ_MAX = 7500
SMOOTHING_ALPHA = 0.7
DEBUG_MODE = True
USE_FILTER = False  # Disable filter to preserve TDOA information

# Output folder
OUTPUT_FOLDER = "localization_results"

# Microphone geometry
# ReSpeaker 4-Mic circular board has mics on a cross (DOA labels on silk):
# DOA:0° at right (MIC1), DOA:90° at top (MIC2), DOA:180° at left (MIC3), DOA:270° at bottom (MIC4)
# Use a radius from PCB center to each mic. Typical radius is ~0.045–0.048 m. Adjust if needed.
MIC_RADIUS = 0.046  # meters (measure center-to-mic distance for best accuracy)

# Microphone positions (x, y, z) in meters, origin at array center
# Coordinate system: X:+right (DOA:0), Y:+forward/top (DOA:90), Z:+up
MIC_POSITIONS = np.array([
    [ MIC_RADIUS,  0.0,        0.0],   # MIC1 (ch1) - right  (DOA 0°)
    [ 0.0,         MIC_RADIUS, 0.0],   # MIC2 (ch2) - top    (DOA 90°)
    [-MIC_RADIUS,  0.0,        0.0],   # MIC3 (ch3) - left   (DOA 180°)
    [ 0.0,        -MIC_RADIUS, 0.0],   # MIC4 (ch4) - bottom (DOA 270°)
])

# Explicit channel-to-physical mic mapping (ALSA channel index -> MIC number)
# We use channels 1..4 as active; adjust if your device exposes a different order.
CHANNEL_TO_MIC = {
    1: "MIC1_DOA0",
    2: "MIC2_DOA90",
    3: "MIC3_DOA180",
    4: "MIC4_DOA270",
}


