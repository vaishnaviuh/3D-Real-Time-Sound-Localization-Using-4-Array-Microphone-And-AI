# Sound Source Localization using Triangulation Method

This project implements real-time sound source localization using a ReSpeaker microphone array with 4 channels (channels 1-4, ignoring channels 0 and 5).

## Features

- **Live Audio Capture**: Real-time audio input from ReSpeaker microphone array
- **TDOA Estimation**: Time Difference of Arrival calculation using GCC-PHAT (Generalized Cross-Correlation with Phase Transform)
- **Angle Estimation**: Azimuth and elevation angle calculation
- **3D Triangulation**: 3D position estimation using least squares optimization
- **Real-time Visualization**: 2D and 3D plots showing sound source position and trajectory

## Hardware Configuration

- **Microphone Array**: ReSpeaker with 6 channels (using channels 1-4)
- **Geometry**: Square pattern with 4.5 cm spacing between microphones
- **Microphone Positions**:
  - MIC1 (Channel 1): Top-right
  - MIC2 (Channel 2): Top-left
  - MIC3 (Channel 3): Bottom-left
  - MIC4 (Channel 4): Bottom-right

## Installation

1. Install system dependencies (for PyAudio):
   ```bash
   sudo apt-get update
   sudo apt-get install portaudio19-dev python3-pyaudio
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the localization script:
```bash
python3 triangulation_localization.py
```

The script will:
1. Automatically detect the ReSpeaker device
2. Start capturing live audio from channels 1-4
3. Display real-time 2D and 3D visualizations
4. Show current azimuth and elevation angles

Press `Ctrl+C` to stop the program.

## Algorithm Details

### TDOA Estimation
- Uses GCC-PHAT method for robust time delay estimation
- Computes time differences between all microphone pairs
- Maximum delay limited to 100ms (configurable)

### Angle Estimation
- Calculates azimuth and elevation from TDOA measurements
- Uses multiple microphone pairs for improved accuracy
- Averages estimates from different pairs

### Triangulation
- Uses least squares optimization to solve for 3D position
- Based on hyperbolic positioning equations
- Falls back to angle-based estimation if triangulation fails

## Parameters

Key parameters in the code (can be adjusted):
- `SAMPLE_RATE`: 16000 Hz (default for ReSpeaker)
- `CHUNK_SIZE`: 1024 samples
- `MIC_SPACING`: 0.045 m (4.5 cm)
- `SOUND_SPEED`: 343.0 m/s (at 20°C)

## Troubleshooting

- **No audio device found**: Make sure ReSpeaker is connected and recognized by the system
- **Poor localization accuracy**: 
  - Check microphone positions match the configuration
  - Ensure adequate signal-to-noise ratio
  - Verify microphone spacing is correct
- **Performance issues**: Reduce `CHUNK_SIZE` or increase `interval` in animation

## License

This project is provided as-is for research and educational purposes.

