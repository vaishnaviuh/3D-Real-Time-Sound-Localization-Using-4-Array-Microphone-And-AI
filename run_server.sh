#!/bin/bash
# Activate virtual environment and run the server
cd "$(dirname "$0")"
source venv/bin/activate
uvicorn server:app --reload --host 0.0.0.0 --port 8000

