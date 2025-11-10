# 3D Real-Time Sound Localization Using 4-Array Microphone and AI

This project implements real-time sound source localization using a ReSpeaker microphone array with 4 channels (channels 1-4, ignoring channels 0 and 5).

## Features

- **Live Audio Capture**: Real-time audio input from ReSpeaker microphone array
- **TDOA Estimation**: Time Difference of Arrival calculation using GCC-PHAT (Generalized Cross-Correlation with Phase Transform)
- **Angle Estimation**: Azimuth and elevation angle calculation
- **3D Triangulation**: 3D position estimation using least squares optimization
- **Real-time Visualization**: 2D and 3D plots showing sound source position and trajectory
- **Web Dashboard**: Real-time web interface for monitoring localization results

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

### Standalone Localization Script

Run the localization script:
```bash
python3 triangulation_localization.py
```

Options:
- `python3 triangulation_localization.py --continuous` - Continuous real-time mode
- `python3 triangulation_localization.py --duration 10` - Record for 10 seconds
- `python3 triangulation_localization.py --test-channels` - Test which channels are active

### Web Server Mode

Run the web server for real-time dashboard:
```bash
./run_server.sh
# or
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in your browser.

## Algorithm Details

### TDOA Estimation
- Uses GCC-PHAT method for robust time delay estimation
- Computes time differences between all microphone pairs
- Maximum delay limited based on microphone geometry

### Angle Estimation
- Calculates azimuth and elevation from TDOA measurements
- Uses multiple microphone pairs for improved accuracy
- Averages estimates from different pairs

### Triangulation
- Uses least squares optimization to solve for 3D position
- Based on hyperbolic positioning equations
- Optimized for near-field sources (10-50cm range)
- Falls back to angle-based estimation if triangulation fails

## Parameters

Key parameters in the code (can be adjusted):
- `SAMPLE_RATE`: 16000 Hz (default for ReSpeaker)
- `CHUNK_SIZE`: 2048 samples
- `MIC_SPACING`: 0.045 m (4.5 cm)
- `SOUND_SPEED`: 343.0 m/s (at 20°C)
- `ENERGY_THRESHOLD`: Signal detection threshold

## Project Structure

- `triangulation_localization.py` - Main standalone localization script
- `server.py` - FastAPI web server for real-time dashboard
- `tdoa.py` - TDOA estimation module
- `angles.py` - Angle estimation module
- `triangulator.py` - 3D triangulation module
- `audio_capture.py` - Audio capture interface
- `config.py` - Configuration parameters
- `public/` - Web dashboard frontend files

## Troubleshooting

- **No audio device found**: Make sure ReSpeaker is connected and recognized by the system
- **Poor localization accuracy**: 
  - Check microphone positions match the configuration
  - Ensure adequate signal-to-noise ratio
  - Verify microphone spacing is correct
  - Test channels with `--test-channels` option
- **Performance issues**: Reduce `CHUNK_SIZE` or increase `interval` in animation
- **Source detected at wrong distance**: Check TDOA values in debug output, verify microphone geometry

## Author

Vaishnavi Hiremath

## License

This project is provided as-is for research and educational purposes.
