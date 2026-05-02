"""
VisionFlow Benchmark Tracker
Records per-frame latency and prints a report on exit.
"""

import time
import statistics
from typing import List


class BenchmarkTracker:
    def __init__(self):
        self._times: List[float] = []  # ms per frame
        self._start = time.perf_counter()

    def record(self, frame_ms: float):
        self._times.append(frame_ms)

    def report(self):
        if not self._times:
            print("\n[Benchmark] No data collected.")
            return

        total_s = time.perf_counter() - self._start
        n = len(self._times)
        avg_ms = statistics.mean(self._times)
        p50 = statistics.median(self._times)
        p95 = sorted(self._times)[int(n * 0.95)]
        p99 = sorted(self._times)[int(n * 0.99)]
        fps_avg = 1000 / avg_ms

        sep = "─" * 44
        print(f"\n\033[38;5;51m  ╔{sep}╗")
        print(f"  ║  VISIONFLOW BENCHMARK REPORT              ║")
        print(f"  ╠{sep}╣")
        print(f"  ║  Frames processed  : {n:>6d}               ║")
        print(f"  ║  Total runtime     : {total_s:>6.1f}s              ║")
        print(f"  ║  Avg FPS           : {fps_avg:>6.1f}               ║")
        print(f"  ║  Avg latency       : {avg_ms:>6.1f} ms            ║")
        print(f"  ║  P50 latency       : {p50:>6.1f} ms            ║")
        print(f"  ║  P95 latency       : {p95:>6.1f} ms            ║")
        print(f"  ║  P99 latency       : {p99:>6.1f} ms            ║")
        print(f"  ╚{sep}╝\033[0m\n")
