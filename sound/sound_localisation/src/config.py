import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class AudioConfig:
    device_query: Optional[str] = "ReSpeaker"  # substring to match input device
    samplerate: int = 16000
    record_seconds: float = 2.0
    dtype: str = "float32"
    # If None, will use device's max input channels then slice channels_to_use
    requested_channels: Optional[int] = None
    # Use 1–4 only (exclude 0 and 5). Zero-based channel indices on device.
    channels_to_use: Tuple[int, int, int, int] = (1, 2, 3, 4)
    countdown_seconds: int = 3
    blocksize: int = 0  # 0 lets PortAudio choose optimal


@dataclass
class GeometryConfig:
    # Planar circular 4-mic subset; angles in degrees for channels 1–4
    # Radius in meters (tune to your hardware; ReSpeaker 6-mic circle ~4.3 cm)
    radius_m: float = 0.043
    mic_angles_deg: Tuple[float, float, float, float] = (0.0, 90.0, 180.0, 270.0)
    speed_of_sound: float = 343.0  # m/s at ~20°C


@dataclass
class SaveConfig:
    enable_save_audio: bool = True
    enable_save_results: bool = True
    output_dir: str = os.path.join(os.getcwd(), "outputs")


@dataclass
class PlotConfig:
    show_plots: bool = True
    interactive_3d: bool = False  # if True, use plotly; else use mpl 3d


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    saving: SaveConfig = field(default_factory=SaveConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)


