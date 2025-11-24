from typing import Optional, Tuple, List
import numpy as np
import sounddevice as sd
from src.config import SimulationConfig


def _find_input_device(query_substring: Optional[str]) -> Optional[int]:
    if query_substring is None:
        return None
    query_lower = query_substring.lower()
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        name = dev.get("name", "")
        max_in = dev.get("max_input_channels", 0)
        if max_in > 0 and query_lower in str(name).lower():
            return idx
    return None


def record_multichannel(
    samplerate: int,
    duration_s: float,
    dtype: str,
    channels_to_use: Tuple[int, ...],
    device_query: Optional[str] = None,
    requested_channels: Optional[int] = None,
    blocksize: int = 0,
    device_index: Optional[int] = None,
    use_simulation: bool = False,
    sim_config: Optional[SimulationConfig] = None,
    mic_positions: Optional[np.ndarray] = None,
    speed_of_sound: float = 343.0,
) -> np.ndarray:
    """
    Record multichannel audio and return only channels indexed by channels_to_use.
    Shape: (num_samples, len(channels_to_use))
    
    If use_simulation is True, generates simulated drone sound instead of recording.
    """
    # Use simulation if enabled
    if use_simulation and sim_config is not None and mic_positions is not None:
        return simulate_drone_audio(
            samplerate=samplerate,
            duration_s=duration_s,
            channels_to_use=channels_to_use,
            mic_positions=mic_positions,
            sim_config=sim_config,
            speed_of_sound=speed_of_sound,
        )
    
    # Otherwise, try to record from real device
    if device_index is None:
        device_index = _find_input_device(device_query)
        if device_index is not None:
            dev_info = sd.query_devices(device_index)
        else:
            dev_info = sd.query_devices(kind="input")
            device_index = sd.default.device[0]
            if device_index is None:
                # fallback to default input device index from dev_info
                device_index = sd.query_devices().index(dev_info)
    else:
        dev_info = sd.query_devices(device_index)

    max_in = dev_info["max_input_channels"]
    if max_in < len(channels_to_use):
        raise RuntimeError(
            f"Input device has only {max_in} input channels; need at least {len(channels_to_use)}."
        )

    # Determine how many to record; we will slice later to channels_to_use
    channels = requested_channels if requested_channels is not None else max_in
    channels = max(channels, max(channels_to_use) + 1)

    frames = int(round(duration_s * samplerate))
    print(
        f"Recording from device '{dev_info['name']}' (index {device_index}) at "
        f"{samplerate} Hz, {channels} ch, {duration_s:.2f} s"
    )
    audio = sd.rec(
        frames=frames,
        samplerate=samplerate,
        channels=channels,
        dtype=dtype,
        device=device_index,
        blocking=True,
    )
    # Slice to the 4 desired channels
    selected = audio[:, list(channels_to_use)]
    return selected.astype(np.float32, copy=False)


def simulate_drone_audio(
    samplerate: int,
    duration_s: float,
    channels_to_use: Tuple[int, ...],
    mic_positions: np.ndarray,
    sim_config: SimulationConfig,
    speed_of_sound: float = 343.0,
) -> np.ndarray:
    """
    Simulate realistic drone audio with time-varying source motion and rotor dynamics.
    """
    num_samples = int(round(duration_s * samplerate))
    t_local = np.linspace(0.0, duration_s, num_samples, dtype=np.float32)
    t_global = t_local + sim_config.current_time_s

    # Generate moving source path and rotor sound
    source_positions = _generate_source_trajectory(t_global, sim_config)
    sim_config.last_velocity_xyz = tuple(((source_positions[-1] - source_positions[0]) /
                                          max(duration_s, 1e-6)).tolist())
    sim_config.last_position_xyz = tuple(source_positions[-1].tolist())

    base_signal = _generate_harmonic_signal(t_global, samplerate, sim_config)

    # Apply amplitude modulation (minor prop-wash pulsing)
    if sim_config.amplitude_mod_depth > 0:
        amp_env = 1.0 + sim_config.amplitude_mod_depth * np.sin(
            2 * np.pi * sim_config.amplitude_mod_rate_hz * t_global
        )
        base_signal *= amp_env.astype(np.float32)

    mic_signals = _render_multichannel_audio(
        base_signal=base_signal,
        source_positions=source_positions,
        mic_positions=mic_positions,
        samplerate=samplerate,
        speed_of_sound=speed_of_sound,
        sim_config=sim_config,
    )

    # Map to requested channels (ensure proper ordering)
    output = np.zeros((num_samples, len(channels_to_use)), dtype=np.float32)
    for idx, ch in enumerate(channels_to_use):
        if idx < mic_signals.shape[1]:
            output[:, idx] = mic_signals[:, idx]

    print(
        f"[SIMULATION] Generated moving drone audio ({sim_config.movement_path_type}) "
        f"for {len(channels_to_use)} channels, {samplerate} Hz, {duration_s:.2f} s"
    )

    sim_config.current_time_s += duration_s
    return output


def _generate_source_trajectory(times: np.ndarray, sim_config: SimulationConfig) -> np.ndarray:
    """Produce time-varying source positions for the simulated drone."""
    base = np.ones((times.size, 3), dtype=np.float32) * np.array(sim_config.source_position_xyz, dtype=np.float32)

    if not sim_config.enable_movement:
        return base

    path_type = sim_config.movement_path_type.lower()
    if path_type == "circle":
        period = max(sim_config.circle_period_s, 1.0)
        angle = 2 * np.pi * times / period
        base[:, 0] += sim_config.circle_radius_m * np.cos(angle)
        base[:, 1] += sim_config.circle_radius_m * np.sin(angle)
        base[:, 2] = sim_config.circle_height_m
    elif path_type == "waypoints" and sim_config.waypoint_positions:
        base = _interpolate_waypoints(times, sim_config)
    else:  # default linear path
        direction = np.array(sim_config.movement_direction, dtype=np.float32)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        displacement = direction * sim_config.movement_speed_mps * (times[:, None])
        base += displacement

    if sim_config.vertical_oscillation_amp_m > 0:
        base[:, 2] += sim_config.vertical_oscillation_amp_m * np.sin(
            2 * np.pi * sim_config.vertical_oscillation_freq_hz * times
        )

    if sim_config.hover_jitter_std_m > 0:
        base += np.random.normal(0.0, sim_config.hover_jitter_std_m, base.shape).astype(np.float32)

    return base


def _interpolate_waypoints(times: np.ndarray, sim_config: SimulationConfig) -> np.ndarray:
    waypoints = np.array(sim_config.waypoint_positions, dtype=np.float32)
    if sim_config.waypoint_loop:
        waypoints = np.vstack([waypoints, waypoints[0]])

    # Precompute cumulative distances
    diffs = np.diff(waypoints, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    segment_lengths[segment_lengths < 1e-6] = 1e-6
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]
    if total_length <= 0:
        return np.ones((t.size, 3), dtype=np.float32) * waypoints[0]

    travel_dist = sim_config.waypoint_speed_mps * times
    if sim_config.waypoint_loop:
        travel_dist = np.mod(travel_dist, total_length)
    else:
        travel_dist = np.clip(travel_dist, 0.0, total_length - 1e-6)

    positions = np.zeros((times.size, 3), dtype=np.float32)
    for idx, dist in enumerate(travel_dist):
        seg_idx = np.searchsorted(cumulative, dist, side="right") - 1
        seg_idx = max(0, min(seg_idx, len(segment_lengths) - 1))
        seg_start = waypoints[seg_idx]
        seg_end = waypoints[seg_idx + 1]
        seg_frac = (dist - cumulative[seg_idx]) / segment_lengths[seg_idx]
        positions[idx] = seg_start + seg_frac * (seg_end - seg_start)
    return positions


def _generate_harmonic_signal(times: np.ndarray, samplerate: int, sim_config: SimulationConfig) -> np.ndarray:
    """Create base rotor harmonic signal with modulation and jitter."""
    num_samples = times.size
    signal = np.zeros(num_samples, dtype=np.float32)

    random_jitter = (
        np.random.randn(num_samples).astype(np.float32) * sim_config.rotor_speed_jitter_hz
        if sim_config.rotor_speed_jitter_hz > 0
        else np.zeros(num_samples, dtype=np.float32)
    )

    variation = (
        sim_config.rotor_speed_variation_hz
        * np.sin(2 * np.pi * sim_config.rotor_speed_variation_rate_hz * times)
    )

    for harmonic_idx in range(sim_config.num_harmonics):
        base_freq = sim_config.fundamental_freq_hz * (harmonic_idx + 1)
        freq = base_freq + variation + random_jitter
        freq = np.maximum(freq, 1.0)

        amplitude = (
            sim_config.signal_amplitude
            * sim_config.harmonic_amplitudes[min(harmonic_idx, len(sim_config.harmonic_amplitudes) - 1)]
        )

        if sim_config.fm_depth_hz > 0:
            phase_mod = (
                sim_config.fm_depth_hz
                * np.sin(2 * np.pi * sim_config.fm_rate_hz * times + harmonic_idx * 0.3)
            )
        else:
            phase_mod = 0.0

        phase = 2 * np.pi * np.cumsum(freq) / samplerate + phase_mod
        signal += amplitude * np.sin(phase).astype(np.float32)

    signal += np.random.randn(num_samples).astype(np.float32) * sim_config.noise_level
    return signal


def _render_multichannel_audio(
    base_signal: np.ndarray,
    source_positions: np.ndarray,
    mic_positions: np.ndarray,
    samplerate: int,
    speed_of_sound: float,
    sim_config: SimulationConfig,
) -> np.ndarray:
    """Render per-microphone signals with time-varying TDOA and attenuation."""
    num_samples = base_signal.shape[0]
    num_mics = mic_positions.shape[0]

    distances = np.linalg.norm(
        source_positions[:, None, :] - mic_positions[None, :, :],
        axis=2,
    )  # shape (samples, mics)

    ref_idx = np.argmin(distances.mean(axis=0))
    ref_distance = distances[:, ref_idx][:, None]
    time_delays = (distances - ref_distance) / speed_of_sound
    delay_samples = time_delays * samplerate

    base_indices = np.arange(num_samples, dtype=np.float32)
    mic_output = np.zeros((num_samples, num_mics), dtype=np.float32)

    for mic_idx in range(num_mics):
        sample_positions = base_indices - delay_samples[:, mic_idx].astype(np.float32)
        mic_signal = np.interp(
            sample_positions,
            base_indices,
            base_signal,
            left=0.0,
            right=0.0,
        )

        attenuation = 1.0 / np.maximum(distances[:, mic_idx], 0.2) ** sim_config.distance_attenuation_power
        mic_signal *= attenuation.astype(np.float32)

        mic_signal += np.random.randn(num_samples).astype(np.float32) * sim_config.noise_level
        mic_output[:, mic_idx] = mic_signal

    return mic_output


