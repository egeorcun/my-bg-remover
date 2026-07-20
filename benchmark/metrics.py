"""Matting quality metrics.

Contract: pred and gt are float32, (H, W), [0, 1]. Following the literature
convention, SAD/Grad/Conn are divided by 1000 (for small, readable numbers).
"""
import numpy as np
from scipy import ndimage


def _check(pred: np.ndarray, gt: np.ndarray) -> None:
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")


def sad(pred: np.ndarray, gt: np.ndarray) -> float:
    _check(pred, gt)
    return float(np.abs(pred - gt).sum()) / 1000.0


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    _check(pred, gt)
    return float(np.abs(pred - gt).mean())


def mse(pred: np.ndarray, gt: np.ndarray) -> float:
    _check(pred, gt)
    return float(((pred - gt) ** 2).mean())


def _gauss_gradient(img: np.ndarray, sigma: float) -> np.ndarray:
    gx = ndimage.gaussian_filter(img, sigma, order=[0, 1])
    gy = ndimage.gaussian_filter(img, sigma, order=[1, 0])
    return np.sqrt(gx**2 + gy**2)


def grad_error(pred: np.ndarray, gt: np.ndarray, sigma: float = 1.4) -> float:
    """Uses scipy's gaussian derivative; values are internally consistent but not
    directly comparable to MATLAB-based published numbers."""
    _check(pred, gt)
    pred_g = _gauss_gradient(pred.astype(np.float64), sigma)
    gt_g = _gauss_gradient(gt.astype(np.float64), sigma)
    return float(((pred_g - gt_g) ** 2).sum()) / 1000.0


def conn_error(pred: np.ndarray, gt: np.ndarray, step: float = 0.1) -> float:
    _check(pred, gt)
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    thresh_steps = np.arange(0, 1 + step, step)
    round_down_map = -np.ones_like(gt)
    for i in range(1, len(thresh_steps)):
        gt_thresh = gt >= thresh_steps[i]
        pred_thresh = pred >= thresh_steps[i]
        intersection = (gt_thresh & pred_thresh).astype(np.uint8)
        labels, num = ndimage.label(intersection)
        if num == 0:
            omega = np.zeros_like(gt)
        else:
            sizes = ndimage.sum(intersection, labels, range(1, num + 1))
            omega = (labels == (np.argmax(sizes) + 1)).astype(np.float64)
        flag = (round_down_map == -1) & (omega == 0)
        round_down_map[flag] = thresh_steps[i - 1]
    round_down_map[round_down_map == -1] = 1
    gt_diff = gt - round_down_map
    pred_diff = pred - round_down_map
    phi_gt = 1 - gt_diff * (gt_diff >= 0.15)
    phi_pred = 1 - pred_diff * (pred_diff >= 0.15)
    return float(np.abs(phi_pred - phi_gt).sum()) / 1000.0


def all_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    return {
        "sad": sad(pred, gt),
        "mae": mae(pred, gt),
        "mse": mse(pred, gt),
        "grad": grad_error(pred, gt),
        "conn": conn_error(pred, gt),
    }
