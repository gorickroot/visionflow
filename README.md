<div align="center">

```
 ██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗███████╗██╗      ██████╗ ██╗    ██╗
 ██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║██╔════╝██║     ██╔═══██╗██║    ██║
 ██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║█████╗  ██║     ██║   ██║██║ █╗ ██║
 ╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║██╔══╝  ██║     ██║   ██║██║███╗██║
  ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║██║     ███████╗╚██████╔╝╚███╔███╔╝
   ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝
```

**Real-time Object Detection · Multi-Object Tracking · Scene Understanding · Activity Heatmaps**

[![Python](https://img.shields.io/badge/Python-3.10%2B-00ffc8?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-7b61ff?style=flat-square)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-ff4d6d?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-00ffc8?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Building-fbbf24?style=flat-square)]()

<br/>

> *A production-grade CV pipeline — not a tutorial, not a notebook. A deployable system.*

</div>

---

## What is VisionFlow?

VisionFlow ingests live video streams (webcam, RTSP, file) and delivers **real-time object detection, multi-object tracking, activity heatmaps, and scene understanding** — all in one modular Python package built on YOLOv8 + OpenCV.

Industry-specific modes for **Retail · Security · Industrial · Automotive** with custom class filters, KPIs, and zone breach alerting.

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌────────────┐     ┌──────────────┐
│   OpenCV    │────▶│   YOLOv8     │────▶│    ONNX     │────▶│   NumPy    │────▶│   Output     │
│ Frame Input │     │  Detection   │     │  Inference  │     │  Post-proc │     │ Annotated    │
│ Webcam/RTSP │     │ 80 Classes   │     │  Optimised  │     │  NMS · IoU │     │ Stream/JSON  │
└─────────────┘     └──────────────┘     └─────────────┘     └────────────┘     └──────────────┘
                           │                                         │
                    ┌──────▼──────┐                         ┌───────▼───────┐
                    │  ByteTrack  │                         │   Heatmap     │
                    │  Tracker    │                         │   Engine      │
                    │  IoU-based  │                         │   Gaussian    │
                    └─────────────┘                         └───────────────┘
```

---

## Features

| Feature | Description |
|---|---|
| 🎯 **YOLOv8 Detection** | 5 model variants (nano → extra-large), 80 COCO classes |
| 🔄 **ByteTrack** | Pure-NumPy multi-object tracker with persistent IDs |
| 🌡️ **Activity Heatmap** | Gaussian accumulation → TURBO colormap, temporal decay |
| 🧠 **Scene Analyzer** | Crowd density, dominant class, anomaly spike detection |
| 🚨 **Zone Monitor** | JSON-configured restricted areas with breach alerting |
| 📡 **Event Emitter** | Pub/sub callbacks for `person_detected`, `zone_breach`, `crowd_anomaly` |
| 🏭 **Industry Modes** | Retail · Security · Industrial · Automotive class filters + KPIs |
| 💾 **JSON Events** | Per-frame structured output for downstream pipelines |
| ⚡ **ONNX Export** | Deploy without PyTorch via `model.export(format='onnx')` |
| 📊 **HUD Overlay** | Transparent stats panel: FPS · count · scene · density |

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/gorickroot/visionflow.git
cd visionflow
pip install -r requirements.txt
```

### 2. Run

```bash
# Webcam — default nano model
python run.py --source webcam

# Video file — medium model, retail mode
python run.py --source video.mp4 --model yolov8m --industry retail

# RTSP stream — tracking + heatmap + JSON events
python run.py --source rtsp://192.168.1.1:554/stream --track --heatmap --save-json

# Security mode with restricted zones
python run.py --source webcam --industry security --zones configs/zones_retail.json

# Benchmark FPS / latency
python run.py --source webcam --benchmark
```

### 3. Hotkeys

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `S` | Save snapshot |
| `F` | Toggle fullscreen |

---

## Industry Modes

### 🛒 Retail
```bash
python run.py --source webcam --industry retail --track --heatmap
```
Tracks `person`, `bottle`, `backpack`, `handbag` · KPIs: customer count, zone heatmaps

### 🔒 Security
```bash
python run.py --source webcam --industry security --zones configs/zones_retail.json
```
Tracks `person`, `car`, `knife` · Alerts: zone breach, suspicious object detection

### 🏭 Industrial
```bash
python run.py --source webcam --industry industrial --save-json
```
Tracks `person`, `truck`, `forklift` · KPIs: worker presence, overcrowding alerts

### 🚗 Automotive
```bash
python run.py --source traffic.mp4 --industry automotive --save
```
Tracks `car`, `truck`, `bus`, `motorcycle` · KPIs: vehicle count, congestion label

---

## Zone Configuration

```json
[
  {
    "name": "Restricted Area",
    "bbox_rel": [0.0, 0.0, 0.35, 1.0],
    "watch_classes": ["person"],
    "alert_on_enter": true
  },
  {
    "name": "Checkout Zone",
    "bbox_rel": [0.65, 0.4, 1.0, 1.0],
    "watch_classes": ["person"],
    "alert_on_enter": false
  }
]
```

`bbox_rel` = relative coordinates `[x1, y1, x2, y2]` from 0 to 1.

---

## Event System

```python
from core.pipeline import VisionPipeline

pipeline = VisionPipeline(industry="security")

# Webhook on zone breach
pipeline.event_emitter.on("zone_breach", lambda e:
    print(f"BREACH: {e['zone_name']} — {e['intruder_count']} intruder(s)")
)

# Crowd anomaly alert
pipeline.event_emitter.on("crowd_anomaly", lambda e:
    print(f"ANOMALY: {e['object_count']} objects detected")
)
```

Available events: `person_detected` · `alert_object` · `zone_breach` · `crowd_anomaly`

---

## Project Structure

```
visionflow/
├── run.py                    # Entry point + CLI
├── requirements.txt
├── core/
│   ├── pipeline.py           # Main orchestrator
│   ├── detector.py           # YOLOv8 wrapper + class filtering
│   ├── tracker.py            # ByteTrack-lite (pure NumPy)
│   ├── heatmap.py            # Gaussian accumulation heatmap
│   ├── scene.py              # Scene analyzer + density KPIs
│   ├── zones.py              # Restricted zone monitor
│   └── display.py            # OpenCV window + hotkeys
├── utils/
│   ├── logger.py             # Coloured terminal logger
│   ├── benchmark.py          # FPS/latency profiler
│   └── events.py             # Pub/sub event emitter
├── configs/
│   └── zones_retail.json     # Example zone config
├── outputs/                  # Saved videos + JSON events
└── tests/
    └── test_visionflow.py    # Full test suite
```

---

## Performance

| Model | Device | FPS | mAP@0.5 |
|---|---|---|---|
| YOLOv8n | CPU i7 | 25–35 | 0.52 |
| YOLOv8s | CPU i7 | 15–22 | 0.60 |
| YOLOv8m | CPU i7 | 8–12 | 0.67 |
| YOLOv8n | RTX 3060 | 120+ | 0.52 |
| YOLOv8x | RTX 3060 | 40–60 | 0.73 |

---

## Roadmap

- [ ] WebSocket real-time event streaming server
- [ ] Streamlit live dashboard
- [ ] Docker + docker-compose deployment
- [ ] DeepSORT tracker integration
- [ ] Custom YOLOv8 fine-tuning guide (PPE, retail objects)
- [ ] RTSP multi-camera support

---

## Author

**Gorick Nath** — BSc Computing Science · Griffith College Dublin, Ireland 🇮🇪

Building AI agents & automations 🤖 · Founding ZynthoAI

> *"Every line of code is a step forward."*

[![GitHub](https://img.shields.io/badge/GitHub-gorickroot-7b61ff?style=flat-square&logo=github)](https://github.com/gorickroot)

---

## License

MIT — use it, break it, ship it.

---

<div align="center">
Built with OpenCV · YOLOv8 · NumPy · PyTorch
</div>
