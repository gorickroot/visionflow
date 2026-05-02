"""
VisionFlow Display Engine
Handles OpenCV window rendering with keyboard shortcuts.
"""

import cv2
from typing import Tuple


class DisplayEngine:
    """
    Wraps cv2.imshow with keyboard handling and headless support.

    Hotkeys:
      Q / ESC  → quit
      S        → save snapshot
      H        → toggle heatmap label
      F        → toggle fullscreen
    """

    WINDOW_NAME = "VisionFlow"

    def __init__(
        self,
        resolution: Tuple[int, int] = (1280, 720),
        headless: bool = False,
        industry: str = "general",
    ):
        self.resolution = resolution
        self.headless = headless
        self.industry = industry
        self._snapshot_count = 0

        if not headless:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW_NAME, *resolution)
            # Window title bar
            cv2.setWindowTitle(
                self.WINDOW_NAME,
                f"VisionFlow — {industry.title()} Mode"
            )

    def show(self, frame) -> bool:
        """
        Display frame. Returns True if user requests quit.
        """
        if self.headless:
            return False

        cv2.imshow(self.WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):  # Q or ESC
            return True

        if key in (ord("s"), ord("S")):
            self._snapshot_count += 1
            path = f"outputs/snapshot_{self._snapshot_count:04d}.jpg"
            cv2.imwrite(path, frame)
            print(f"[Display] Snapshot saved → {path}")

        if key in (ord("f"), ord("F")):
            current = cv2.getWindowProperty(
                self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN
            )
            new_mode = (
                cv2.WINDOW_FULLSCREEN
                if current != cv2.WINDOW_FULLSCREEN
                else cv2.WINDOW_NORMAL
            )
            cv2.setWindowProperty(
                self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, new_mode
            )

        return False

    def close(self):
        if not self.headless:
            cv2.destroyAllWindows()
