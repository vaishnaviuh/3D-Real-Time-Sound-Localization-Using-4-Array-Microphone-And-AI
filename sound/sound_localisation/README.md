## Live Sound Localization (ReSpeaker, channels 1–4)

This project records from a ReSpeaker microphone array, using channels 1–4, performs DOA estimation via GCC-PHAT, and visualizes the result in 2D and 3D with matplotlib. It includes a 3-second countdown before capture and optional saving of raw audio and results.

### Setup
1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Run
```bash
python -m src.main
```

If your ReSpeaker device name differs, update `device_query` in `src/config.py`.

### Notes
- Only channels 1–4 are used; channels 0 and 5 are ignored.
- Mic geometry is approximated as 4 mics on a circle. Tune `radius_m` and `mic_angles_deg` in `src/config.py` to match your hardware orientation.
- Distance is a rough proxy derived from signal energy and is not calibrated.


