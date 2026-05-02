"""
VisionFlow Event Emitter
Pub/sub system for detection events. Attach callbacks or webhook URLs.

Usage:
    emitter = EventEmitter()
    emitter.on("person_detected", my_callback)
    emitter.on("zone_breach", lambda e: requests.post(WEBHOOK_URL, json=e))
"""

from typing import Callable, Dict, List, Any
import threading


class EventEmitter:
    """
    Lightweight synchronous event bus.
    Runs callbacks in a background thread to avoid blocking the pipeline.
    """

    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()

    def on(self, event: str, callback: Callable):
        """Register a callback for an event type."""
        with self._lock:
            self._listeners.setdefault(event, []).append(callback)

    def off(self, event: str, callback: Callable):
        with self._lock:
            if event in self._listeners:
                self._listeners[event] = [
                    cb for cb in self._listeners[event] if cb != callback
                ]

    def emit(self, event: str, data: Any = None):
        """Fire all callbacks for the given event (non-blocking)."""
        with self._lock:
            callbacks = list(self._listeners.get(event, []))
        if callbacks:
            t = threading.Thread(
                target=self._dispatch, args=(callbacks, data), daemon=True
            )
            t.start()

    def _dispatch(self, callbacks: List[Callable], data: Any):
        for cb in callbacks:
            try:
                cb(data)
            except Exception as e:
                print(f"[EventEmitter] Callback error: {e}")

    def emit_detections(self, detections: List[Dict], scene_meta: Dict, zone_alerts: List[Dict]):
        """Convenience: fire typed events from pipeline output."""
        # Person detected
        persons = [d for d in detections if d["label"] == "person"]
        if persons:
            self.emit("person_detected", {"count": len(persons), "detections": persons})

        # Alert objects (weapons etc.)
        alerts = [d for d in detections if d.get("is_alert")]
        if alerts:
            self.emit("alert_object", {"detections": alerts})

        # Zone breaches
        for alert in zone_alerts:
            self.emit("zone_breach", alert)

        # Crowd spike anomaly
        if scene_meta.get("anomaly"):
            self.emit("crowd_anomaly", scene_meta)
