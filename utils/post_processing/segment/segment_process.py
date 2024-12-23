import numpy as np


def compute_segment_mask(segment_map: np.ndarray, threshold: float) -> np.ndarray:
    segment_map = segment_map.squeeze()
    mask: np.ndarray = np.zeros_like(segment_map).astype(np.uint8)
    mask[segment_map > threshold] = 1

    mask *= 255

    return mask
