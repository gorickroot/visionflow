"""
VisionFlow Core Pipeline
Manages the full detection + tracking + heatmap loop.
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from core.detector import YOLODetector
from core.tracker import ByteTracker
from core.heatmap import HeatmapEngine
from core.scene import SceneAnalyzer
from core.zones import ZoneMonitor
from utils.events import EventEmitter


class VisionPipeline:
    """
    Orchestrates the full VisionFlow processing chain:
      VideoCapture → Preprocess → Detect → Track → Annotate → Emit Events
    """

    def __init__(
        self,
        model_name: str = "yolov8n",
        conf: float = 0.35,
        iou: float = 0.45,
        device: str = "auto",
        industry: str = "general",
        enable_tracking: bool = False,
        enable_heatmap: bool = False,
        zones_config: Optional[str] = None,
        resolution: Tuple[int, int] = (1280, 720),
        save_output: bool = False,
        save_json: bool = False,
        logger=None,
    ):
        self.resolution = resolution
        self.industry = industry
        self.save_output = save_output
        self.save_json = save_json
        self.logger = logger
        self.enable_tracking = enable_tracking
        self.enable_heatmap = enable_heatmap
        self._running = False

        # Subsystems
        self.logger.info("Initialising detector...")
        self.detector = YOLODetector(
            model_name=model_name,
            conf=conf,
            iou=iou,
            device=device,
            industry=industry,
        )

        self.tracker = ByteTracker() if enable_tracking else None
        self.heatmap = HeatmapEngine(resolution) if enable_heatmap else None
        self.scene_analyzer = SceneAnalyzer(industry=industry)
        self.event_emitter = EventEmitter()

        self.zone_monitor = None
        if zones_config:
            self.zone_monitor = ZoneMonitor(zones_config, resolution)

        # Output setup
        self.output_dir = Path("outputs") / datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_output or save_json:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Outputs → {self.output_dir}")

        self._video_writer = None
        self._json_events = []
        self._session_id = str(uuid.uuid4())[:8]
        self._frame_count = 0

        self.logger.success("Pipeline ready.")

    # ───────────────────────────────────────── run ──
    def run(self, source, cam_id=0, display=None, benchmark=None):
        cap = self._open_source(source, cam_id)
        if cap is None:
            return

        if self.save_output:
            self._init_writer()

        self._running = True
        fps_timer = time.perf_counter()
        fps_display = 0.0
        frame_times = []

        self.logger.success("Pipeline running — press Q to quit.")

        while self._running:
            t0 = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                # Loop video files
                if source not in ("webcam",) and not source.startswith("rtsp"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame = cv2.resize(frame, self.resolution)
            self._frame_count += 1

            # ── Detect
            detections = self.detector.detect(frame)

            # ── Track
            if self.tracker and len(detections) > 0:
                detections = self.tracker.update(detections, frame)

            # ── Heatmap
            if self.heatmap:
                self.heatmap.update(detections)

            # ── Scene analysis
            scene_meta = self.scene_analyzer.analyse(frame, detections)

            # ── Zone alerts
            zone_alerts = []
            if self.zone_monitor:
                zone_alerts = self.zone_monitor.check(detections)

            # ── Annotate frame
            annotated = self._annotate(frame.copy(), detections, scene_meta, zone_alerts)

            if self.heatmap:
                annotated = self.heatmap.overlay(annotated)

            # ── FPS overlay
            t1 = time.perf_counter()
            frame_ms = (t1 - t0) * 1000
            frame_times.append(frame_ms)
            if len(frame_times) > 30:
                frame_times.pop(0)

            if time.perf_counter() - fps_timer >= 0.5:
                fps_display = 1000 / (sum(frame_times) / len(frame_times))
                fps_timer = time.perf_counter()

            annotated = self._draw_hud(annotated, fps_display, detections, scene_meta)

            # ── Save
            if self._video_writer:
                self._video_writer.write(annotated)

            if self.save_json:
                self._collect_event(detections, scene_meta, zone_alerts)

            if benchmark:
                benchmark.record(frame_ms)

            # ── Display
            if display:
                should_quit = display.show(annotated)
                if should_quit:
                    break

        # Cleanup
        cap.release()
        if self.save_json and self._json_events:
            self._flush_json()

    # ─────────────────────────────────── helpers ──
    def _open_source(self, source: str, cam_id: int):
        if source == "webcam":
            cap = cv2.VideoCapture(cam_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, 60)
        elif source.startswith("rtsp://") or source.startswith("http"):
            cap = cv2.VideoCapture(source)
        else:
            path = Path(source)
            if not path.exists():
                self.logger.error(f"Source not found: {source}")
                return None
            cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            self.logger.error(f"Could not open source: {source}")
            return None

        self.logger.info(f"Source opened: {source}")
        return cap

    def _init_writer(self):
        out_path = self.output_dir / f"visionflow_{self._session_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(
            str(out_path), fourcc, 30.0, self.resolution
        )
        self.logger.info(f"Recording → {out_path}")

    def _annotate(self, frame, detections, scene_meta, zone_alerts):
        """Draw bounding boxes, labels, confidence bars."""
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = det["label"]
            conf = det["conf"]
            track_id = det.get("track_id")
            color = det.get("color", (0, 255, 200))

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Corner accents
            corner_len = 12
            thick = 3
            for cx, cy, dx, dy in [
                (x1, y1, 1, 1), (x2, y1, -1, 1),
                (x1, y2, 1, -1), (x2, y2, -1, -1)
            ]:
                cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, thick)
                cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, thick)

            # Label pill
            id_str = f"#{track_id} " if track_id is not None else ""
            text = f"{id_str}{label}  {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            pill_y = max(y1 - 24, 0)
            cv2.rectangle(frame, (x1, pill_y), (x1 + tw + 10, pill_y + th + 8), color, -1)
            cv2.putText(frame, text, (x1 + 5, pill_y + th + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (10, 10, 10), 1, cv2.LINE_AA)

            # Confidence bar (bottom of box)
            bar_w = x2 - x1
            cv2.rectangle(frame, (x1, y2), (x2, y2 + 4), (30, 30, 30), -1)
            cv2.rectangle(frame, (x1, y2), (x1 + int(bar_w * conf), y2 + 4), color, -1)

        # Zone alert overlays
        for alert in zone_alerts:
            zx1, zy1, zx2, zy2 = alert["zone_bbox"]
            overlay = frame.copy()
            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)
            cv2.putText(frame, f"⚠ {alert['zone_name']} BREACH",
                        (zx1 + 6, zy1 + 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 255), 2, cv2.LINE_AA)

        return frame

    def _draw_hud(self, frame, fps, detections, scene_meta):
        """Transparent HUD overlay with stats panel."""
        h, w = frame.shape[:2]
        panel_w = 240
        panel_h = 160

        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_w, panel_h), (10, 10, 20), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (10, 10), (panel_w, panel_h), (0, 255, 200), 1)

        # Header
        cv2.putText(frame, "VISIONFLOW", (20, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)

        # FPS bar
        fps_color = (0, 255, 100) if fps >= 25 else (0, 180, 255) if fps >= 15 else (0, 60, 255)
        cv2.putText(frame, f"FPS  {fps:5.1f}", (20, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1, cv2.LINE_AA)

        # Object count
        obj_count = len(detections)
        cv2.putText(frame, f"OBJ  {obj_count:5d}", (20, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # Scene
        scene_label = scene_meta.get("scene", "—")
        density = scene_meta.get("density", "—")
        cv2.putText(frame, f"SCN  {scene_label[:12]}", (20, 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"DEN  {density}", (20, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1, cv2.LINE_AA)

        # Frame counter
        cv2.putText(frame, f"FRAME {self._frame_count:06d}", (20, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1, cv2.LINE_AA)

        # Industry badge
        badge_text = self.industry.upper()
        (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(frame, (w - bw - 22, 10), (w - 10, 10 + bh + 10), (0, 100, 80), -1)
        cv2.putText(frame, badge_text, (w - bw - 14, 10 + bh + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA)

        # Timestamp
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, ts, (w - 76, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA)

        return frame

    def _collect_event(self, detections, scene_meta, zone_alerts):
        event = {
            "frame": self._frame_count,
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self._session_id,
            "scene": scene_meta,
            "detections": [
                {
                    "label": d["label"],
                    "conf": round(d["conf"], 3),
                    "bbox": [round(v, 1) for v in d["bbox"]],
                    "track_id": d.get("track_id"),
                }
                for d in detections
            ],
            "zone_alerts": zone_alerts,
        }
        self._json_events.append(event)

    def _flush_json(self):
        out_path = self.output_dir / f"events_{self._session_id}.json"
        with open(out_path, "w") as f:
            json.dump(self._json_events, f, indent=2)
        self.logger.info(f"Events saved → {out_path} ({len(self._json_events)} frames)")

    def release(self):
        self._running = False
        if self._video_writer:
            self._video_writer.release()
