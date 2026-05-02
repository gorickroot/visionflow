"""
VisionFlow Heatmap Engine
Accumulates detection centroids into a smooth activity heatmap overlay.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import cv2


class HeatmapEngine:
    """
    Generates a persistent Gaussian heatmap from detection centroids.
    The heatmap decays over time so recent activity is brighter.

    Usage:
        hm = HeatmapEngine((1280, 720))
        hm.update(detections)          # call each frame
        annotated = hm.overlay(frame)  # blend into frame
    """

    def __init__(
        self,
        resolution: Tuple[int, int],
        decay: float = 0.97,
        blur_radius: int = 61,
        alpha: float = 0.45,
    ):
        w, h = resolution
        self.shape = (h, w)
        self.decay = decay
        self.blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        self.alpha = alpha

        self._accumulator = np.zeros((h, w), dtype=np.float32)
        self._colormap = cv2.COLORMAP_TURBO  # vivid: blue→green→yellow→red

    def update(self, detections: List[Dict[str, Any]]):
        """Add Gaussian blobs at detection centroids and apply decay."""
        # Decay
        self._accumulator *= self.decay

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            h, w = self.shape

            # Clamp to frame
            cx = max(0, min(cx, w - 1))
            cy = max(0, min(cy, h - 1))

            # Gaussian splash
            spread = max(int((x2 - x1 + y2 - y1) / 4), 20)
            self._splash_gaussian(cx, cy, spread)

    def _splash_gaussian(self, cx: int, cy: int, sigma: int):
        """Add a 2D Gaussian centred at (cx, cy)."""
        h, w = self.shape
        y_lo = max(0, cy - sigma * 3)
        y_hi = min(h, cy + sigma * 3)
        x_lo = max(0, cx - sigma * 3)
        x_hi = min(w, cx + sigma * 3)

        xs = np.arange(x_lo, x_hi) - cx
        ys = np.arange(y_lo, y_hi) - cy
        xg, yg = np.meshgrid(xs, ys)
        gauss = np.exp(-(xg ** 2 + yg ** 2) / (2 * sigma ** 2))
        self._accumulator[y_lo:y_hi, x_lo:x_hi] += gauss * 255

    def overlay(self, frame: np.ndarray) -> np.ndarray:
        """Blend heatmap onto frame and return result."""
        # Normalise to [0, 255]
        acc_norm = self._accumulator.copy()
        max_val = acc_norm.max()
        if max_val > 0:
            acc_norm = (acc_norm / max_val * 255).astype(np.uint8)
        else:
            return frame

        # Smooth
        blurred = cv2.GaussianBlur(acc_norm, (self.blur_radius, self.blur_radius), 0)

        # Colorise
        colored = cv2.applyColorMap(blurred, self._colormap)

        # Mask out low-activity areas
        mask = blurred > 15
        colored[~mask] = 0

        # Blend
        result = frame.copy()
        result[mask] = cv2.addWeighted(
            frame, 1 - self.alpha, colored, self.alpha, 0
        )[mask]

        return result

    def reset(self):
        self._accumulator[:] = 0

    def save_snapshot(self, path: str):
        """Save current heatmap as PNG."""
        acc_norm = self._accumulator.copy()
        max_val = acc_norm.max()
        if max_val > 0:
            acc_norm = (acc_norm / max_val * 255).astype(np.uint8)
        colored = cv2.applyColorMap(acc_norm, self._colormap)
        cv2.imwrite(path, colored)
