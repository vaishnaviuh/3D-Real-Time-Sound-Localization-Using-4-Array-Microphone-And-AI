from typing import Optional, Tuple, List
import numpy as np
import sounddevice as sd


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
    channels_to_use: Tuple[int, int, int, int],
    device_query: Optional[str] = None,
    requested_channels: Optional[int] = None,
    blocksize: int = 0,
) -> np.ndarray:
    """
    Record multichannel audio and return only channels indexed by channels_to_use.
    Shape: (num_samples, 4)
    """
    device_index = _find_input_device(device_query)
    if device_index is not None:
        dev_info = sd.query_devices(device_index)
    else:
        dev_info = sd.query_devices(kind="input")
        device_index = sd.default.device[0]
        if device_index is None:
            # fallback to default input device index from dev_info
            device_index = sd.query_devices().index(dev_info)

    max_in = dev_info["max_input_channels"]
    if max_in < 4:
        raise RuntimeError(
            f"Input device has only {max_in} input channels; need at least 4."
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


