"""
VisionFlow Detector
YOLOv8 wrapper with industry-specific class filtering and color mapping.
Falls back gracefully if ultralytics not installed (shows install prompt).
"""

import sys
from typing import List, Dict, Any, Optional
import numpy as np

# Industry-specific class filters (COCO class names)
INDUSTRY_FILTERS = {
    "general": None,  # All 80 COCO classes
    "retail": [
        "person", "bottle", "cup", "bowl", "backpack",
        "handbag", "suitcase", "cell phone", "laptop",
        "chair", "couch", "dining table",
    ],
    "security": [
        "person", "car", "truck", "motorcycle", "bicycle",
        "backpack", "handbag", "suitcase", "knife", "scissors",
        "cell phone",
    ],
    "industrial": [
        "person", "truck", "car", "forklift",
        "bottle", "cup", "bowl",  # placeholder for custom PPE classes
    ],
    "automotive": [
        "car", "truck", "bus", "motorcycle", "bicycle",
        "person", "traffic light", "stop sign",
    ],
}

# Neon color palette per class category
CLASS_COLORS = {
    # People
    "person":        (0, 255, 200),    # cyan-green
    # Vehicles
    "car":           (255, 180, 0),    # amber
    "truck":         (255, 140, 0),
    "bus":           (255, 100, 0),
    "motorcycle":    (255, 200, 50),
    "bicycle":       (200, 255, 50),
    # Animals
    "dog":           (150, 100, 255),
    "cat":           (180, 80, 255),
    # Objects/bags
    "backpack":      (255, 80, 150),
    "handbag":       (255, 60, 120),
    "suitcase":      (255, 40, 100),
    # Electronics
    "laptop":        (80, 200, 255),
    "cell phone":    (60, 180, 255),
    "tv":            (40, 160, 255),
    # Default
    "__default__":   (0, 200, 255),
}

# Alert classes (shown with red accent in security mode)
ALERT_CLASSES = {"knife", "scissors", "gun", "rifle"}


class YOLODetector:
    """
    Wraps Ultralytics YOLOv8 for real-time detection.

    Usage:
        detector = YOLODetector(model_name="yolov8n", conf=0.35)
        detections = detector.detect(frame)  # frame: np.ndarray BGR
    """

    def __init__(
        self,
        model_name: str = "yolov8n",
        conf: float = 0.35,
        iou: float = 0.45,
        device: str = "auto",
        industry: str = "general",
    ):
        self.conf = conf
        self.iou = iou
        self.industry = industry
        self.class_filter = INDUSTRY_FILTERS.get(industry)
        self._model = None
        self._class_names = {}

        # Resolve device
        self.device = self._resolve_device(device)

        # Load model
        self._load_model(model_name)

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _load_model(self, model_name: str):
        try:
            from ultralytics import YOLO
            # Auto-downloads weights on first run (~6MB for nano)
            self._model = YOLO(f"{model_name}.pt")
            self._model.to(self.device)
            # Cache class names
            self._class_names = self._model.names  # {0: 'person', 1: 'bicycle', ...}
            print(f"\033[32m  ✓ Model loaded: {model_name} on {self.device}\033[0m")
        except ImportError:
            print("\033[33m  ⚠ ultralytics not found — running in DEMO mode\033[0m")
            print("  Install: pip install ultralytics")
            self._model = None
            self._class_names = {0: "person", 1: "car", 2: "truck"}

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection on a BGR frame.
        Returns list of dicts with: bbox, label, conf, class_id, color
        """
        if self._model is None:
            return self._demo_detections(frame)

        results = self._model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                label = self._class_names.get(class_id, f"cls_{class_id}")

                # Industry filter
                if self.class_filter and label not in self.class_filter:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                color = CLASS_COLORS.get(label, CLASS_COLORS["__default__"])
                # Alert class override
                if label in ALERT_CLASSES:
                    color = (0, 60, 255)

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "label": label,
                    "conf": conf,
                    "class_id": class_id,
                    "color": color,
                    "is_alert": label in ALERT_CLASSES,
                })

        return detections

    def _demo_detections(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Synthetic detections for demo/testing without model weights.
        Oscillates to simulate live detection.
        """
        import time
        t = time.time()
        h, w = frame.shape[:2]
        detections = []

        # Simulate 2-4 detections
        templates = [
            {"label": "person", "bbox_rel": (0.1, 0.15, 0.3, 0.85), "conf": 0.91},
            {"label": "person", "bbox_rel": (0.55, 0.2, 0.75, 0.9), "conf": 0.87},
            {"label": "car",    "bbox_rel": (0.3, 0.5, 0.7, 0.95), "conf": 0.82},
        ]

        for i, tmpl in enumerate(templates):
            # Add slight oscillation
            jitter = 0.01 * np.sin(t * 2 + i)
            rx1, ry1, rx2, ry2 = tmpl["bbox_rel"]
            bbox = (
                (rx1 + jitter) * w, (ry1 + jitter) * h,
                (rx2 + jitter) * w, (ry2 + jitter) * h,
            )
            label = tmpl["label"]
            detections.append({
                "bbox": bbox,
                "label": label,
                "conf": tmpl["conf"] + 0.01 * np.sin(t + i),
                "class_id": i,
                "color": CLASS_COLORS.get(label, CLASS_COLORS["__default__"]),
                "is_alert": False,
            })

        return detections
