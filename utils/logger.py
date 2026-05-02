"""
VisionFlow Logger — coloured terminal output.
"""

import sys
from datetime import datetime


class VisionLogger:
    CYAN   = "\033[38;5;51m"
    GREEN  = "\033[38;5;82m"
    YELLOW = "\033[38;5;220m"
    RED    = "\033[38;5;196m"
    GREY   = "\033[38;5;240m"
    RESET  = "\033[0m"
    BOLD   = "\033[1m"

    def __init__(self, name: str = "VisionFlow"):
        self.name = name

    def _ts(self):
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def _print(self, level: str, color: str, msg: str):
        ts = self.GREY + self._ts() + self.RESET
        tag = color + self.BOLD + f"[{level}]" + self.RESET
        print(f"  {ts}  {tag}  {msg}")

    def info(self, msg: str):
        self._print("INFO   ", self.CYAN, msg)

    def success(self, msg: str):
        self._print("OK     ", self.GREEN, msg)

    def warn(self, msg: str):
        self._print("WARN   ", self.YELLOW, msg)

    def error(self, msg: str):
        self._print("ERROR  ", self.RED, msg)
        sys.stderr.flush()

    def debug(self, msg: str):
        self._print("DEBUG  ", self.GREY, msg)
