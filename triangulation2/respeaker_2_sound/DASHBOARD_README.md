# Live Sound Localization Dashboard

A modern, real-time web dashboard for live sound localization using FastAPI and WebSockets.

## Features

-    **Real-time Audio Processing**: Continuous sound localization with live updates
- 📊 **Interactive Visualizations**: 
  - 2D localization map with microphone positions
  - 3D localization map with interactive camera controls
  - Trajectory history showing sound source movement
- 📈 **Live Statistics**: 
  - Azimuth, Elevation, Distance
  - Update rate (FPS)
-    **Modern UI**: Beautiful gradient design with responsive layout
-    **WebSocket Communication**: Low-latency real-time updates

## Installation

1. Install dependencies:
```bash
cd /home/head-node-5/sound_localisation
. .venv/bin/activate
pip install -r requirements.txt
```

## Running the Dashboard

1. Start the FastAPI server:
```bash
python -m src.server
```

Or using uvicorn directly:
```bash
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

2. Open your web browser and navigate to:
```
http://localhost:8000
```

3. The dashboard will automatically connect via WebSocket and start processing audio.

## Usage

- **Start Processing**: Click the "Start Processing" button to begin continuous audio capture and localization
- **Stop Processing**: Click "Stop Processing" to pause audio processing
- **Interactive 3D View**: Rotate, zoom, and pan the 3D plot by clicking and dragging
- **Real-time Updates**: Watch the sound source position update in real-time as you move around the microphone array

## Architecture

- **Backend**: FastAPI server with WebSocket support
- **Audio Processing**: Runs in thread pool to avoid blocking the event loop
- **Frontend**: Vanilla JavaScript with Plotly.js for 3D visualization
- **Communication**: WebSocket for bidirectional real-time communication

## Configuration

The dashboard uses the same configuration as the main application (`src/config.py`). You can modify:
- Audio device selection
- Sample rate
- Recording duration
- Microphone geometry
- Speed of sound

## Troubleshooting

- **No audio detected**: Check that your ReSpeaker device is connected and recognized
- **WebSocket connection fails**: Ensure the server is running and accessible
- **Slow updates**: Reduce `record_seconds` in config for faster processing
- **Distance inaccurate**: The TDOA-based distance estimation works best for near-field sources (within 1-2 meters)

## Network Access

To access the dashboard from other devices on your network:
1. Start the server with `--host 0.0.0.0`
2. Find your machine's IP address: `hostname -I`
3. Access from another device: `http://<your-ip>:8000`

