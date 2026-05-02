"""
VisionFlow Test Suite
Run: python -m pytest tests/ -v
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Tracker tests ──────────────────────────────────────────────
from core.tracker import ByteTracker, iou, Track


def make_det(label, bbox, conf=0.9):
    return {
        "label": label,
        "bbox": bbox,
        "conf": conf,
        "class_id": 0,
        "color": (0, 255, 200),
        "is_alert": False,
    }


class TestIoU:
    def test_perfect_overlap(self):
        box = (0, 0, 100, 100)
        assert iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert iou((0, 0, 50, 50), (100, 100, 200, 200)) == 0.0

    def test_half_overlap(self):
        score = iou((0, 0, 100, 100), (50, 0, 150, 100))
        assert 0.3 < score < 0.4  # expected ~0.333


class TestByteTracker:
    def test_assigns_id_on_first_detection(self):
        tracker = ByteTracker()
        dets = [make_det("person", (10, 10, 100, 200))]
        result = tracker.update(dets)
        # First frame: not confirmed yet (hits < 2)
        assert isinstance(result, list)

    def test_id_persists_across_frames(self):
        tracker = ByteTracker()
        det = make_det("person", (10, 10, 100, 200))
        tracker.update([det])           # frame 1
        result = tracker.update([det])  # frame 2 — now confirmed
        assert len(result) == 1
        assert "track_id" in result[0]

    def test_empty_detections(self):
        tracker = ByteTracker()
        result = tracker.update([])
        assert result == []

    def test_prunes_lost_tracks(self):
        tracker = ByteTracker(max_misses=2)
        det = make_det("person", (10, 10, 100, 200))
        tracker.update([det])
        tracker.update([det])
        # Now vanish the detection
        for _ in range(5):
            tracker.update([])
        assert len(tracker.tracks) == 0


# ── Scene Analyzer tests ───────────────────────────────────────
from core.scene import SceneAnalyzer


class TestSceneAnalyzer:
    def test_empty_scene(self):
        sa = SceneAnalyzer(industry="general")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        meta = sa.analyse(frame, [])
        assert meta["object_count"] == 0
        assert meta["dominant_class"] == "none"

    def test_density_label_crowded(self):
        sa = SceneAnalyzer(industry="general")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dets = [make_det("person", (i * 50, 0, i * 50 + 40, 100)) for i in range(12)]
        meta = sa.analyse(frame, dets)
        assert meta["density"] == "crowded"

    def test_dominant_class(self):
        sa = SceneAnalyzer()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dets = [
            make_det("person", (0, 0, 50, 100)),
            make_det("person", (60, 0, 110, 100)),
            make_det("car",    (120, 0, 200, 100)),
        ]
        meta = sa.analyse(frame, dets)
        assert meta["dominant_class"] == "person"

    def test_retail_kpis(self):
        sa = SceneAnalyzer(industry="retail")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dets = [make_det("person", (i * 50, 0, i * 50 + 40, 100)) for i in range(3)]
        meta = sa.analyse(frame, dets)
        assert "customer_count" in meta
        assert meta["customer_count"] == 3


# ── Heatmap tests ──────────────────────────────────────────────
from core.heatmap import HeatmapEngine


class TestHeatmapEngine:
    def test_overlay_returns_same_shape(self):
        hm = HeatmapEngine((640, 480))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [make_det("person", (100, 100, 200, 300))]
        hm.update(dets)
        result = hm.overlay(frame)
        assert result.shape == frame.shape

    def test_reset_clears_accumulator(self):
        hm = HeatmapEngine((640, 480))
        dets = [make_det("person", (100, 100, 200, 300))]
        hm.update(dets)
        hm.reset()
        assert hm._accumulator.max() == 0.0

    def test_empty_detections_no_crash(self):
        hm = HeatmapEngine((640, 480))
        hm.update([])  # should not raise
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = hm.overlay(frame)
        assert result.shape == frame.shape


# ── Zone Monitor tests ─────────────────────────────────────────
from core.zones import ZoneMonitor, iou_point_in_box
import tempfile, json


class TestZoneMonitor:
    def _make_config(self):
        cfg = [{"name": "Danger Zone", "bbox_rel": [0.0, 0.0, 0.5, 1.0],
                "watch_classes": ["person"], "alert_on_enter": True}]
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(cfg, tmp)
        tmp.close()
        return tmp.name

    def test_detects_intrusion(self):
        path = self._make_config()
        zm = ZoneMonitor(path, (1280, 720))
        # Person centroid at (200, 360) — inside left half zone
        dets = [make_det("person", (150, 310, 250, 410))]
        alerts = zm.check(dets)
        assert len(alerts) == 1
        assert alerts[0]["zone_name"] == "Danger Zone"

    def test_no_alert_outside_zone(self):
        path = self._make_config()
        zm = ZoneMonitor(path, (1280, 720))
        # Person centroid at (900, 360) — right half, outside zone
        dets = [make_det("person", (850, 310, 950, 410))]
        alerts = zm.check(dets)
        assert len(alerts) == 0

    def test_point_in_box(self):
        assert iou_point_in_box(50, 50, (0, 0, 100, 100)) is True
        assert iou_point_in_box(150, 50, (0, 0, 100, 100)) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
