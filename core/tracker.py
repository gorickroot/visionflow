"""
VisionFlow Tracker
Lightweight ByteTrack-style multi-object tracker using IoU matching.
No external dependency — pure NumPy implementation.
For production, swap in: pip install lapx && use ultralytics ByteTrack.
"""

from typing import List, Dict, Any
import numpy as np


def iou(boxA, boxB) -> float:
    """Compute Intersection over Union of two boxes [x1,y1,x2,y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


class Track:
    _next_id = 1

    def __init__(self, detection: Dict[str, Any]):
        self.track_id = Track._next_id
        Track._next_id += 1
        self.bbox = detection["bbox"]
        self.label = detection["label"]
        self.hits = 1
        self.misses = 0
        self.age = 1
        # Simple Kalman-lite: exponential smoothed bbox
        self._smooth_bbox = list(detection["bbox"])

    def predict(self):
        """Persist current smoothed position as prediction."""
        self.age += 1
        self.misses += 1
        return self._smooth_bbox

    def update(self, detection: Dict[str, Any]):
        self.hits += 1
        self.misses = 0
        alpha = 0.6  # smoothing factor
        new = detection["bbox"]
        self._smooth_bbox = [
            alpha * n + (1 - alpha) * s
            for n, s in zip(new, self._smooth_bbox)
        ]
        self.bbox = tuple(self._smooth_bbox)
        return self

    def to_detection(self, original: Dict[str, Any]) -> Dict[str, Any]:
        return {**original, "track_id": self.track_id, "bbox": tuple(self._smooth_bbox)}


class ByteTracker:
    """
    IoU-based greedy multi-object tracker (ByteTrack-lite).
    Assigns persistent IDs across frames.

    For full ByteTrack with Kalman filter + LAPJV assignment:
      from ultralytics import YOLO
      model.track(source, tracker="bytetrack.yaml")
    """

    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 15):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.tracks: List[Track] = []

    def update(
        self, detections: List[Dict[str, Any]], frame: np.ndarray = None
    ) -> List[Dict[str, Any]]:
        if not detections:
            for t in self.tracks:
                t.misses += 1
            self._prune()
            return []

        # Predict all tracks
        predicted_bboxes = [t.predict() for t in self.tracks]

        # Greedy IoU matching
        matched_track_ids = set()
        matched_det_ids = set()
        track_det_pairs = []

        for ti, pred_box in enumerate(predicted_bboxes):
            best_iou = self.iou_threshold
            best_di = -1
            for di, det in enumerate(detections):
                if di in matched_det_ids:
                    continue
                if det["label"] != self.tracks[ti].label:
                    continue
                score = iou(pred_box, det["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_di = di
            if best_di >= 0:
                track_det_pairs.append((ti, best_di))
                matched_track_ids.add(ti)
                matched_det_ids.add(best_di)

        # Apply matches
        for ti, di in track_det_pairs:
            self.tracks[ti].update(detections[di])

        # New tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_det_ids:
                self.tracks.append(Track(det))

        # Prune dead tracks
        self._prune()

        # Build output — only confirmed tracks (hit > 1)
        output = []
        for t in self.tracks:
            if t.hits >= 2 or t.misses == 0:
                # Find matching detection for color/conf
                matched_det = next(
                    (d for d in detections if d["label"] == t.label),
                    None
                )
                if matched_det:
                    output.append(t.to_detection(matched_det))

        return output

    def _prune(self):
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
