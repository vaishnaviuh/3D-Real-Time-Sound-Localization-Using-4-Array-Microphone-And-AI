#!/bin/bash
# Startup script for Sound Localization Dashboard

cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Start the FastAPI server
echo "Starting Sound Localization Dashboard..."
echo "Open http://localhost:8000 in your browser"
python -m src.server


