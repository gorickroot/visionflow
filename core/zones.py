"""
VisionFlow Zone Monitor
Loads zone configs from JSON and fires alerts when objects enter restricted areas.

Zone config format (zones.json):
[
  {
    "name": "Restricted Area",
    "bbox_rel": [0.0, 0.0, 0.5, 1.0],   // relative [x1,y1,x2,y2] 0-1
    "watch_classes": ["person"],
    "alert_on_enter": true
  }
]
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple


def iou_point_in_box(cx: float, cy: float, box: Tuple) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= cx <= x2 and y1 <= cy <= y2


class ZoneMonitor:
    """
    Monitors defined rectangular zones and alerts when watched classes enter.
    """

    def __init__(self, config_path: str, resolution: Tuple[int, int]):
        self.resolution = resolution
        self.zones = self._load(config_path)
        self._entry_state: Dict[str, set] = {}  # zone_name → set of track_ids

    def _load(self, path: str) -> List[Dict]:
        p = Path(path)
        if not p.exists():
            print(f"[ZoneMonitor] Config not found: {path} — zones disabled.")
            return []
        with open(p) as f:
            raw = json.load(f)

        w, h = self.resolution
        zones = []
        for z in raw:
            rx1, ry1, rx2, ry2 = z["bbox_rel"]
            zones.append({
                "name": z["name"],
                "bbox": (rx1 * w, ry1 * h, rx2 * w, ry2 * h),
                "watch_classes": z.get("watch_classes", ["person"]),
                "alert_on_enter": z.get("alert_on_enter", True),
            })
        print(f"[ZoneMonitor] Loaded {len(zones)} zone(s).")
        return zones

    def check(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return list of active zone alerts."""
        alerts = []
        for zone in self.zones:
            zone_name = zone["name"]
            zone_bbox = zone["bbox"]
            watch = zone["watch_classes"]

            intruders = []
            for det in detections:
                if det["label"] not in watch:
                    continue
                x1, y1, x2, y2 = det["bbox"]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if iou_point_in_box(cx, cy, zone_bbox):
                    intruders.append(det)

            if intruders:
                alerts.append({
                    "zone_name": zone_name,
                    "zone_bbox": tuple(map(int, zone_bbox)),
                    "intruder_count": len(intruders),
                    "intruder_labels": [d["label"] for d in intruders],
                })

        return alerts
