import cv2
import numpy as np


def anomaly_map_to_color_map(anomaly_map: np.ndarray, normalize: bool = True) -> np.ndarray:
    if normalize:
        anomaly_map = (anomaly_map - anomaly_map.min()) / np.ptp(anomaly_map)
    anomaly_map = anomaly_map * 255
    anomaly_map = anomaly_map.astype(np.uint8)

    anomaly_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    anomaly_map = cv2.cvtColor(anomaly_map, cv2.COLOR_BGR2RGB)
    return anomaly_map


def superimpose_anomaly_map(
    anomaly_map: np.ndarray, image: np.ndarray, alpha: float = 0.4, gamma: int = 0, normalize: bool = False
) -> np.ndarray:
    anomaly_map = anomaly_map_to_color_map(
        anomaly_map.squeeze(), normalize=normalize)
    superimposed_map = cv2.addWeighted(
        anomaly_map, alpha, image, (1 - alpha), gamma)
    return superimposed_map


def compute_anomaly_mask(anomaly_map: np.ndarray, threshold: float) -> np.ndarray:
    anomaly_map = anomaly_map.squeeze()
    mask: np.ndarray = np.zeros_like(anomaly_map).astype(np.uint8)
    mask[anomaly_map > threshold] = 1

    mask *= 255

    return mask


def compute_anomaly_predict(anomaly_map: np.ndarray, threshold: float):
    anomaly_map = anomaly_map.squeeze()
    prediction = np.any(anomaly_map.numpy() > threshold)

    return prediction
