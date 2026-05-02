#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════╗
║          VisionFlow — CV Pipeline v1.0               ║
║   Real-time Object Detection & Scene Understanding   ║
║   Built with YOLOv8 · OpenCV · ONNX · NumPy         ║
╚══════════════════════════════════════════════════════╝
"""

import argparse
import sys
import time
from pathlib import Path

from core.pipeline import VisionPipeline
from core.display import DisplayEngine
from utils.logger import VisionLogger
from utils.benchmark import BenchmarkTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description="VisionFlow — Real-time CV Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --source webcam
  python run.py --source video.mp4 --model yolov8m --industry retail
  python run.py --source rtsp://192.168.1.1:554/stream --headless --save
  python run.py --source image.jpg --conf 0.4 --save
  python run.py --benchmark --source webcam
        """
    )

    # Source
    parser.add_argument("--source", default="webcam",
                        help="Input source: 'webcam', video path, image path, or RTSP URL")
    parser.add_argument("--cam-id", type=int, default=0,
                        help="Camera device ID (default: 0)")

    # Model
    parser.add_argument("--model", default="yolov8n",
                        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                        help="YOLOv8 model variant (n=nano, x=extra-large)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold (default: 0.35)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Inference device (default: auto)")

    # Industry mode
    parser.add_argument("--industry", default="general",
                        choices=["general", "retail", "security", "industrial", "automotive"],
                        help="Industry-specific detection mode")

    # Output
    parser.add_argument("--save", action="store_true",
                        help="Save annotated output to ./outputs/")
    parser.add_argument("--save-json", action="store_true",
                        help="Save detection events as JSON")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display window")
    parser.add_argument("--resolution", default="1280x720",
                        help="Output resolution WxH (default: 1280x720)")

    # Features
    parser.add_argument("--track", action="store_true",
                        help="Enable object tracking (ByteTrack)")
    parser.add_argument("--heatmap", action="store_true",
                        help="Generate crowd/activity heatmap overlay")
    parser.add_argument("--zones", type=str, default=None,
                        help="Path to zone config JSON for restricted area alerts")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run FPS / latency benchmark")

    return parser.parse_args()


def print_banner():
    banner = """
\033[38;5;51m
 ██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗███████╗██╗      ██████╗ ██╗    ██╗
 ██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║██╔════╝██║     ██╔═══██╗██║    ██║
 ██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║█████╗  ██║     ██║   ██║██║ █╗ ██║
 ╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║██╔══╝  ██║     ██║   ██║██║███╗██║
  ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║██║     ███████╗╚██████╔╝╚███╔███╔╝
   ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝
\033[0m
\033[38;5;245m  Real-time Object Detection & Scene Understanding  ·  YOLOv8 + OpenCV + ONNX\033[0m
\033[38;5;240m  ─────────────────────────────────────────────────────────────────────────\033[0m
"""
    print(banner)


def main():
    print_banner()
    args = parse_args()
    logger = VisionLogger(name="VisionFlow")

    # Parse resolution
    try:
        w, h = map(int, args.resolution.split("x"))
    except ValueError:
        logger.error(f"Invalid resolution format: {args.resolution}. Use WxH e.g. 1280x720")
        sys.exit(1)

    logger.info(f"Model      : {args.model}")
    logger.info(f"Source     : {args.source}")
    logger.info(f"Industry   : {args.industry}")
    logger.info(f"Device     : {args.device}")
    logger.info(f"Confidence : {args.conf}")
    logger.info(f"Tracking   : {'ON' if args.track else 'OFF'}")
    logger.info(f"Heatmap    : {'ON' if args.heatmap else 'OFF'}")
    print()

    # Build pipeline
    pipeline = VisionPipeline(
        model_name=args.model,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        industry=args.industry,
        enable_tracking=args.track,
        enable_heatmap=args.heatmap,
        zones_config=args.zones,
        resolution=(w, h),
        save_output=args.save,
        save_json=args.save_json,
        logger=logger,
    )

    display = DisplayEngine(
        resolution=(w, h),
        headless=args.headless,
        industry=args.industry,
    )

    tracker = BenchmarkTracker() if args.benchmark else None

    # Run
    try:
        pipeline.run(
            source=args.source,
            cam_id=args.cam_id,
            display=display,
            benchmark=tracker,
        )
    except KeyboardInterrupt:
        logger.info("Stopped by user (Ctrl+C)")
    finally:
        pipeline.release()
        display.close()
        if tracker:
            tracker.report()
        logger.success("VisionFlow shut down cleanly.")


if __name__ == "__main__":
    main()
