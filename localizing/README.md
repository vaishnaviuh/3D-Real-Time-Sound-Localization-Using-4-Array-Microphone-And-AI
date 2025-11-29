## Multi‑Sensor Drone Sound Triangulation (20‑Channel WAV + KML)

This project localizes airborne sound sources (e.g. drones) using a **20‑channel WAV recording** and **15 fixed sensors + 1 cluster** defined in a KML file. It computes **TDOA (Time Difference of Arrival)** using GCC‑PHAT and performs **3D triangulation** to estimate the sound source position, then visualizes results in clear **2D and 3D maps**.

The core triangulation logic lives in `src/triangulation.py` and is not modified by the plotting/analysis scripts.

### Setup
1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure the following files are present in the project root:
   - `multi-20251122-141610-627897594.wav` (20‑channel, 5‑minute recording)
   - `Sensor-Locations-BOP-Dharma.kml` (15 sensor pole locations)

All generated plots are saved into the `plots/` folder.

### Sensor Geometry (KML → Local XYZ)
- `Sensor-Locations-BOP-Dharma.kml` defines **15 sensor poles** in GPS.
- `src/kml_parser.py` converts GPS to a **local XYZ coordinate system** (meters) and adds:
  - **Sensors 1–15**: real poles, mounted ~1.5 m above ground.
  - **Cluster sensor**: one virtual sensor between poles 5 and 7, representing channels **16–20** combined.

### Main Scripts

- **1. Real WAV analysis without filtering (safe pipeline)**
  - **File**: `no_filter_analysis.py`
  - **What it does**:
    - Reads the 20‑channel WAV in **2‑second chunks** (chunk‑by‑chunk, low memory).
    - Combines channels **16–20** into a single **cluster channel** (sensor 16).
    - **No bandpass filter** – uses raw audio to avoid `filtfilt` numerical issues.
    - For chunks with enough energy and active sensors, calls `TriangulationEngine.triangulate_audio_chunk(...)` with:
      - `tdoa_method='gcc_phat'`
      - `triangulation_method='robust'`
    - Collects detections (timestamp, 3D position, confidence, residual error).
    - Plots **2D and 3D maps** of the detections.
  - **Run**:
    ```bash
    python -u no_filter_analysis.py
    ```
  - **Outputs** (in `plots/`):
    - `no_filter_results_2d.png` – 2D top‑view map (sensors + detections).
    - `no_filter_results_3d.png` – 3D height map (sensors on ground, detections at height).

- **2. Final clean 2D + 3D maps (simple command to run, simulated detections)**
  - **Files**: `final_detection_maps.py`, `height_triang.py`
  - **What it does**:
    - Loads sensors from the KML and simulates representative drone detections (no WAV needed).
    - Creates **two separate maps**:
      - `plots/final_2d_map.png`: 2D top view (sensors + cluster + detections with confidence labels).
      - `plots/final_3d_map.png`: 3D view (ground sensors, drone heights, path, and an idealized TDOA summary between nearest sensors and each detection).
    - Pure visualization; does **not** change any core triangulation logic.
  - **Run (single command)**:
    ```bash
    cd C:\Users\vaish\OneDrive\Desktop\triangulation
    python -u height_triang.py
    ```

- **3. Accurate height triangulation demo (sensors at 1.5 m)**
  - **File**: `accurate_height_triangulation.py`
  - **What it does**:
    - Loads sensors and then shifts all sensor Z‑coordinates by **+1.5 m** to model pole‑mounted sensors.
    - Plots:
      - 2D top view (sensors + drones).
      - 3D view showing sensor plane (1.5 m) vs ground and drone heights.
      - Bar/line charts comparing **height accuracy** for ground vs elevated sensors and showing **vertical angle** improvements.
    - Demonstrates how raising sensors improves **height accuracy** (better geometry, larger elevation angles).
  - **Run**:
    ```bash
    python accurate_height_triangulation.py
    ```
  - **Output**: `plots/accurate_height_triangulation.png`

- **4. Full WAV analysis with bandpass filter (optional)**
  - **File**: `final_wav_analysis.py`
  - **What it does**:
    - End‑to‑end analysis of the 20‑channel WAV with **300–4000 Hz bandpass filtering** (applied safely, chunk‑wise, with fallbacks).
    - Uses the same triangulation engine (`src/triangulation.py`) to compute 3D positions over time.
    - Can be used when filter stability is acceptable on your machine.

### Core Logic (for reference)

- **TDOA & Triangulation** (`src/triangulation.py`)
  - **TDOA estimation**: GCC‑PHAT cross‑correlation between sensor pairs to estimate sample delays.
  - **Triangulation**: least‑squares and robust solvers that find the 3D point best matching all TDOAs.
  - The plotting scripts **do not modify** this logic; they only visualize results and geometry.

- **Plotting Utilities** (`src/plotting.py`)
  - Helper functions for 2D/3D sound maps and multi‑array visualizations (for generic DOA use‑cases).

### Outputs
- All key plots are saved in the **`plots/`** directory:
  - `no_filter_results.png` – real detection map from your 20‑channel WAV (no filter).
  - `improved_triangulation_plot.png` – clean 2D + 3D triangulation view with TDOA summary.
  - `accurate_height_triangulation.png` – height‑accuracy and geometry demonstration.

This README describes the current triangulation workflow (20‑channel WAV + KML sensors + 2D/3D plots) without changing the underlying triangulation algorithms. 
