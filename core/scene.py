"""
VisionFlow Scene Analyzer
Classifies scene context and computes crowd density metrics.
"""

from typing import List, Dict, Any
import numpy as np


# Scene classification rules per industry
SCENE_RULES = {
    "general": {
        "thresholds": {"empty": 0, "sparse": 2, "moderate": 5, "crowded": 10},
    },
    "retail": {
        "thresholds": {"empty": 0, "light traffic": 2, "busy": 6, "peak": 12},
        "kpis": ["customer_count", "dwell_estimate"],
    },
    "security": {
        "thresholds": {"clear": 0, "low risk": 2, "elevated": 5, "high alert": 8},
        "watch_classes": ["person", "backpack", "suitcase"],
    },
    "industrial": {
        "thresholds": {"safe": 0, "active": 3, "busy": 7, "overcrowded": 12},
    },
    "automotive": {
        "thresholds": {"free flow": 0, "light": 3, "moderate": 8, "congested": 15},
        "vehicle_classes": ["car", "truck", "bus", "motorcycle"],
    },
}


class SceneAnalyzer:
    """
    Analyses a frame + its detections to produce scene-level metadata:
      - Scene density label
      - Dominant object class
      - Industry-specific KPIs
      - Anomaly flags
    """

    def __init__(self, industry: str = "general"):
        self.industry = industry
        self.rules = SCENE_RULES.get(industry, SCENE_RULES["general"])
        self._history: List[int] = []  # rolling object counts

    def analyse(
        self, frame: np.ndarray, detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        count = len(detections)
        self._history.append(count)
        if len(self._history) > 30:
            self._history.pop(0)

        avg_count = sum(self._history) / len(self._history)
        density = self._classify_density(count)
        dominant = self._dominant_class(detections)
        anomaly = self._detect_anomaly(count, avg_count)

        meta = {
            "scene": self._scene_label(density),
            "density": density,
            "object_count": count,
            "avg_count_30f": round(avg_count, 1),
            "dominant_class": dominant,
            "anomaly": anomaly,
        }

        # Industry-specific KPIs
        if self.industry == "retail":
            meta["customer_count"] = sum(
                1 for d in detections if d["label"] == "person"
            )
            meta["avg_dwell_s"] = "N/A (needs tracking)"

        elif self.industry == "security":
            watch = self.rules.get("watch_classes", [])
            meta["watched_objects"] = sum(
                1 for d in detections if d["label"] in watch
            )
            meta["alert_objects"] = sum(
                1 for d in detections if d.get("is_alert")
            )

        elif self.industry == "automotive":
            vehicles = self.rules.get("vehicle_classes", [])
            meta["vehicle_count"] = sum(
                1 for d in detections if d["label"] in vehicles
            )

        return meta

    def _classify_density(self, count: int) -> str:
        thresholds = self.rules["thresholds"]
        label = list(thresholds.keys())[0]
        for name, threshold in thresholds.items():
            if count >= threshold:
                label = name
        return label

    def _scene_label(self, density: str) -> str:
        labels = {
            "general":    "environment",
            "retail":     "retail floor",
            "security":   "monitored zone",
            "industrial": "work floor",
            "automotive": "traffic scene",
        }
        return labels.get(self.industry, "scene")

    def _dominant_class(self, detections: List[Dict]) -> str:
        if not detections:
            return "none"
        counts: Dict[str, int] = {}
        for d in detections:
            counts[d["label"]] = counts.get(d["label"], 0) + 1
        return max(counts, key=counts.get)

    def _detect_anomaly(self, current: int, average: float) -> bool:
        """Flag if current count is >2x the rolling average (sudden crowd spike)."""
        if average < 2:
            return False
        return current > average * 2.5
