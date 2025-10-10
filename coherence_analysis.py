import numpy as np
from typing import Tuple

def process_multi_channel_coherence(
        tf_maps: np.ndarray,
        calc_coherence: callable,
        coherence_threshold: float = 0.5,
        logging: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process multi-channel time-frequency data to calculate coherence between all channel pairs
    and return the average of the most coherent pair with a coherence mask.

    Parameters
    ----------
    tf_maps : np.ndarray
        Time-frequency maps with shape (K, L, M) where:
        - K: number of channels
        - L: number of frequency bins
        - M: number of time samples
    calc_coherence : callable
        Function that calculates coherence between two time-frequency maps.
        Should accept two arrays of shape (L, M) and return coherence of shape (L, M)
    coherence_threshold : float, default=0.5
        Threshold for coherence mask. Pixels with coherence below this will be masked.
    logging : bool, default=False
        If True, print debug information about the coherence analysis.

    Returns
    -------
    tuple
        (average_tf_map, coherence_mask) where:
        - average_tf_map: np.ndarray of shape (L, M) - average of the most coherent channel pair
        - coherence_mask: np.ndarray of shape (L, M) - boolean mask where True indicates
          coherence >= threshold, False indicates coherence < threshold
    """
    K, L, M = tf_maps.shape

    if K < 2:
        raise ValueError("At least 2 channels are required for coherence analysis")

    # Calculate number of channel pairs: K choose 2
    num_pairs = K * (K - 1) // 2

    if logging:
        print(f"[COHERENCE] Processing {K} channels, {num_pairs} channel pairs")
        print(f"[COHERENCE] Time-frequency map shape: {L} freq bins × {M} time samples")

    # Initialize arrays to store coherence values and track best pair
    coherence_streams = np.zeros((num_pairs, L, M))
    max_coherence = np.zeros((L, M))
    best_pair_indices = np.zeros((L, M), dtype=int)

    # Calculate coherence for all channel pairs
    pair_idx = 0
    for i in range(K):
        for j in range(i + 1, K):
            if logging:
                print(f"[COHERENCE] Calculating coherence for channel pair ({i}, {j})")

            # Calculate coherence between channels i and j
            coherence = calc_coherence(tf_maps[i], tf_maps[j])
            coherence_streams[pair_idx] = coherence

            # Update best coherence and corresponding pair indices
            better_mask = coherence > max_coherence
            max_coherence[better_mask] = coherence[better_mask]
            best_pair_indices[better_mask] = pair_idx

            pair_idx += 1

    # Create the average time-frequency map of the most coherent pair
    average_tf_map = np.zeros((L, M))

    # For each pixel, find the most coherent pair and average those two channels
    for l in range(L):
        for m in range(M):
            best_pair = best_pair_indices[l, m]

            # Convert pair index back to channel indices
            # This is the inverse of the pair calculation above
            i, j = _pair_index_to_channels(best_pair, K)

            # Average the two most coherent channels for this pixel
            average_tf_map[l, m] = (tf_maps[i, l, m] + tf_maps[j, l, m]) / 2.0

    # Create coherence mask
    coherence_mask = max_coherence >= coherence_threshold

    if logging:
        coherence_stats = {
            'mean': np.mean(max_coherence),
            'std': np.std(max_coherence),
            'min': np.min(max_coherence),
            'max': np.max(max_coherence),
            'above_threshold': np.sum(coherence_mask) / (L * M) * 100
        }
        print(f"[COHERENCE] Coherence statistics:")
        print(f"  Mean: {coherence_stats['mean']:.3f}")
        print(f"  Std: {coherence_stats['std']:.3f}")
        print(f"  Range: [{coherence_stats['min']:.3f}, {coherence_stats['max']:.3f}]")
        print(f"  Above threshold ({coherence_threshold}): {coherence_stats['above_threshold']:.1f}%")

    return average_tf_map, coherence_mask


def _pair_index_to_channels(pair_idx: int, K: int) -> Tuple[int, int]:
    """
    Convert a pair index back to the corresponding channel indices.

    This is the inverse of the pair enumeration used in process_multi_channel_coherence.

    Parameters
    ----------
    pair_idx : int
        Index of the channel pair (0-based)
    K : int
        Total number of channels

    Returns
    -------
    tuple
        (i, j) where i < j are the channel indices for this pair
    """
    # This implements the inverse of the enumeration:
    # for i in range(K):
    #     for j in range(i + 1, K):
    #         pair_idx += 1

    i = 0
    remaining = pair_idx

    while remaining >= (K - 1 - i):
        remaining -= (K - 1 - i)
        i += 1

    j = i + 1 + remaining
    return i, j


