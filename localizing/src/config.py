import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class ArraySetup:
    """
    Configuration for an individual microphone array instance.
    Allows running multiple physical arrays (e.g., two ReSpeaker boards) simultaneously.
    """
    name: str = "Array A"
    device_query: Optional[str] = None  # Fallback to AudioConfig.device_query if None
    device_index: Optional[int] = None  # Optional explicit sounddevice index override
    channels_to_use: Tuple[int, int, int, int] = (1, 2, 3, 4)
    radius_m: float = 0.043
    mic_angles_deg: Tuple[float, float, float, float] = (0.0, 90.0, 180.0, 270.0)
    origin_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Center position in meters
    color_hex: str = "#22c55e"  # Dashboard accent color


@dataclass
class AudioConfig:
    device_query: Optional[str] = "ReSpeaker"  # substring to match input device
    samplerate: int = 16000
    record_seconds: float = 0.25  # ~2 Hz update rate for localization
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
class ClassificationConfig:
    """Configuration for sound classification and trigger system."""
    # Enable classification (if False, all sounds pass through)
    # Disabled to allow localization to run on all detected sounds during tuning
    enable_classification: bool = False
    
    # Which sound classes to detect as triggers
    drone_enabled: bool = True
    mechanical_enabled: bool = True
    clap_enabled: bool = True
    
    # Minimum confidence to trigger localization (0.0-1.0)
    # Higher = more strict, only very confident detections trigger
    min_trigger_confidence: float = 0.6
    
    # Rejection thresholds (if speech/background score > threshold, reject even if target detected)
    # Lower = more aggressive rejection of unwanted sounds
    speech_rejection_threshold: float = 0.5  # Reject if speech score > 0.5
    background_rejection_threshold: float = 0.6  # Reject if background score > 0.6
    
    # Minimum RMS energy to even consider classification (filters out pure noise)
    min_energy_for_classification: float = 0.0003
    
    # DRONE detection rules (stricter to avoid false positives)
    drone_max_centroid: float = 1800.0  # Hz (lowered from 2000)
    drone_min_harmonic_ratio: float = 0.35  # Increased from 0.3
    drone_min_rms: float = 0.0015  # Increased from 0.001
    drone_max_bandwidth: float = 1400.0  # Hz (lowered from 1500)
    
    # MECHANICAL noise detection rules (stricter)
    mechanical_min_rms: float = 0.003  # Increased from 0.002
    mechanical_min_bandwidth: float = 2200.0  # Hz (increased from 2000)
    mechanical_min_centroid: float = 600.0  # Hz (increased from 500)
    mechanical_min_harmonic_ratio: float = 0.25  # Increased from 0.2
    
    # CLAP detection rules (stricter)
    clap_min_peak_to_rms: float = 6.0  # Increased from 5.0
    clap_min_bandwidth: float = 3200.0  # Hz (increased from 3000)
    clap_min_rms: float = 0.0015  # Increased from 0.001
    
    # SPEECH rejection rules
    speech_max_rms: float = 0.0015
    speech_max_bandwidth: float = 2500.0  # Hz
    speech_max_harmonic_ratio: float = 0.25
    speech_min_zcr: float = 1000.0  # zero crossings per second
    
    # BACKGROUND rejection rules
    background_max_rms: float = 0.0005
    background_max_peak_to_rms: float = 2.0
    
    # Spectrogram computation
    spectrogram_n_fft: int = 2048
    spectrogram_hop_length: int = 512
    enable_spectrogram: bool = True  # Send spectrograms to dashboard


@dataclass
class SimulationConfig:
    """Configuration for simulated audio data (when no microphones available)."""
    enable_simulation: bool = True  # Enable simulation mode
    # Base source position (meters) relative to Array A (ground level)
    source_position_xyz: Tuple[float, float, float] = (-50.0, 0.0, 95.0)
    # Drone sound characteristics
    fundamental_freq_hz: float = 350.0  # Typical drone fundamental frequency
    num_harmonics: int = 5  # Number of harmonics to include
    harmonic_amplitudes: List[float] = field(default_factory=lambda: [1.0, 0.6, 0.4, 0.3, 0.2])  # Relative amplitudes
    fm_depth_hz: float = 15.0  # Frequency modulation depth
    fm_rate_hz: float = 2.0  # Frequency modulation rate
    rotor_speed_variation_hz: float = 8.0  # Slow rotor speed swings
    rotor_speed_variation_rate_hz: float = 0.6
    rotor_speed_jitter_hz: float = 1.5  # Random jitter per sample
    signal_amplitude: float = 0.15  # Overall signal amplitude
    noise_level: float = 0.02  # Background noise level
    amplitude_mod_depth: float = 0.15
    amplitude_mod_rate_hz: float = 0.4
    distance_attenuation_power: float = 1.2
    # Movement simulation parameters
    enable_movement: bool = True
    movement_path_type: str = "waypoints"  # Drone flies between arrays
    movement_speed_mps: float = 100.0 / 55.0  # Traverse 100 m in ~55 s
    movement_direction: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    circle_radius_m: float = 1.5
    circle_height_m: float = 95.0
    circle_period_s: float = 55.0
    waypoint_positions: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [
            (0.0, -50.0, 95.0),
            (0.0, 0.0, 95.0),
            (0.0, 50.0, 95.0),
            (0.0, 0.0, 95.0),
        ]
    )
    waypoint_speed_mps: float = 100.0 / 55.0
    waypoint_loop: bool = True
    vertical_oscillation_amp_m: float = 1.0
    vertical_oscillation_freq_hz: float = 0.05
    hover_jitter_std_m: float = 0.05
    current_time_s: float = 0.0  # internal simulation clock
    last_position_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    last_velocity_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class DetectionConfig:
    # Minimum correlation peak quality (peak / (mean + std))
    # Lower values = more sensitive, higher values = less false positives
    # Set very low to allow almost all detections - can filter later by confidence
    min_correlation_quality: float = 0.8
    
    # Minimum RMS energy threshold (normalized)
    # Lower values = more sensitive to quiet sounds
    # Set very low to allow almost all detections
    min_rms_energy: float = 0.0005
    
    # Minimum confidence threshold (0.0-1.0) to report a detection
    # Set to 0.0 to always report detections (dashboard can filter by confidence visually)
    # Lower values = more detections (including weak ones)
    # Higher values = only strong, clear detections
    min_confidence_threshold: float = 0.0
    
    # Always broadcast DOA updates even if confidence is low (let dashboard decide)
    always_broadcast: bool = True
    
    # Enable debug output to see signal quality metrics
    enable_debug: bool = True
    
    # Frequency filtering: only process sounds in this range (Hz)
    min_freq_hz: float = 300.0
    max_freq_hz: float = 4000.0
    
    # Harmonic detection: only trigger DOA when specific harmonics are present
    # List of fundamental frequencies (Hz) to detect - if any harmonic series is found, trigger
    # Example: [440, 880] would detect A4 note or its octave
    # Empty list means harmonic detection is disabled
    target_harmonic_fundamentals_hz: List[float] = field(default_factory=lambda: [])
    
    # Harmonic detection parameters
    # Minimum number of harmonics that must be present (including fundamental)
    min_harmonics_detected: int = 2
    # Tolerance for harmonic frequency matching (Hz)
    harmonic_tolerance_hz: float = 50.0
    # Minimum magnitude threshold for harmonic peaks (relative to max)
    harmonic_min_magnitude_ratio: float = 0.1
    # Require harmonic match before running DOA. If False, DOA still runs but is flagged.
    require_harmonic_match: bool = False
    # Automatically look for dominant tonal fundamentals when explicit list is empty
    auto_detect_harmonics: bool = True
    # Minimum normalized peak magnitude to consider a frequency a candidate fundamental
    auto_detect_min_peak_ratio: float = 0.2
    # Maximum number of candidate fundamentals to test when auto detecting
    auto_detect_max_candidates: int = 5
    # Activity detection thresholds (used when harmonics missing)
    min_activity_rms: float = 0.0005
    min_activity_peak_to_mean: float = 1.2
    # Array-level gating and smoothing
    min_array_confidence: float = 0.35
    min_fused_confidence: float = 0.3
    smoothing_alpha: float = 0.7
    max_azimuth_jump_deg: float = 30.0
    min_confidence_for_jump: float = 0.45


def _default_arrays() -> List[ArraySetup]:
    """
    Provide two array definitions by default so the dashboard shows
    separate geometries out of the box. Adjust device_index/origin per setup.
    """
    return [
        ArraySetup(
            name="Array A",
            device_query="ReSpeaker",
            device_index=5,
            channels_to_use=(1, 2, 3, 4),
            radius_m=0.043,
            mic_angles_deg=(0.0, 90.0, 180.0, 270.0),
            origin_xyz=(-50.0, 0.0, 0.0),
            color_hex="#22c55e",
        ),
        ArraySetup(
            name="Array B",
            device_query="ReSpeaker",
            device_index=6,
            channels_to_use=(1, 2, 3, 4),
            radius_m=0.043,
            mic_angles_deg=(0.0, 90.0, 180.0, 270.0),
            origin_xyz=(50.0, 0.0, 0.0),  # 100 m separation relative to Array A
            color_hex="#f97316",
        ),
    ]


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    saving: SaveConfig = field(default_factory=SaveConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    arrays: List[ArraySetup] = field(default_factory=_default_arrays)

    def resolved_arrays(self) -> List[ArraySetup]:
        """
        Return the configured array list, or fall back to a single-array setup
        derived from the legacy Audio/Geometry config for backwards compatibility.
        """
        if self.arrays:
            return self.arrays
        return [
            ArraySetup(
                name="Array A",
                device_query=self.audio.device_query,
                channels_to_use=self.audio.channels_to_use,
                radius_m=self.geometry.radius_m,
                mic_angles_deg=self.geometry.mic_angles_deg,
                origin_xyz=(0.0, 0.0, 0.0),
                color_hex="#22c55e",
            )
        ]

