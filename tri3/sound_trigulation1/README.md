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

### Using Two ReSpeaker Arrays
- The default config already declares two `ArraySetup` entries (`Array A` on device index 0 at the origin, `Array B` on device index 1 offset by 35 cm on +X). Update `device_index` (or `device_query`) and `origin_xyz` to match where your two USB ReSpeakers actually sit.
- Each entry lets you set its `name`, `channels_to_use`, `mic_angles_deg`, `radius_m`, and a dashboard `color_hex`. These values control how the microphones are drawn in the 2D/3D plots and how per-array cards are labeled.
- If you only have one array, simply drop the second `ArraySetup` from `AppConfig.arrays`. If you have more, append additional entries.
- The processing pipeline localizes each array independently, then fuses the direction vectors for a combined “best guess”; the dashboard still shows every individual array result so you can compare confidence.

### Harmonic Detection & Frequency Filtering

The system now includes intelligent sound detection that only triggers localization for specific sounds:

1. **Frequency Filtering**: Only processes sounds in the 300-4000 Hz range (configurable via `min_freq_hz` and `max_freq_hz` in `DetectionConfig`).

2. **Harmonic Detection**: Lets you gate DOA estimation on specific harmonic series so random sounds are ignored. When the list is empty and `auto_detect_harmonics=True`, the app will automatically look for strong tonal peaks (e.g. drone hums) and use them as triggers.

   To configure harmonic detection, set `target_harmonic_fundamentals_hz` in `src/config.py`:
   ```python
   target_harmonic_fundamentals_hz: List[float] = [440.0, 880.0]  # Example: A4 note and its octave
   ```

   - If the list is empty (`[]`), harmonic detection is disabled and all sounds in the frequency range will be localized.
   - The system checks for at least `min_harmonics_detected` harmonics (default: 2) in the harmonic series.
   - Adjust `harmonic_tolerance_hz` (default: 50 Hz) for frequency matching tolerance.
   - Adjust `harmonic_min_magnitude_ratio` (default: 0.1) for minimum peak strength relative to maximum.
   - Set `require_harmonic_match=True` if you want to *skip* localization whenever the harmonic check fails. Leave it `False` (default) to keep computing azimuth/elevation but mark results that lacked a harmonic match.
   - Use `auto_detect_harmonics=True` (default) to let the system discover tonal sources on the fly when no explicit fundamentals are provided.
   - Broadband activity (voice, loud noise) is still tracked via RMS/peak checks even when harmonics are missing.

   Example: To detect a 1000 Hz tone and its harmonics:
   ```python
   target_harmonic_fundamentals_hz: List[float] = [1000.0]
   min_harmonics_detected: int = 3  # Require fundamental + 2 harmonics
   ```


