from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

    def as_mss_dict(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


class ScreenGrabber:
    """Captures BGRA frames and converts them to the format used by detectors."""

    def __init__(self, region: CaptureRegion):
        self.region = region
        self._mss = None

    def __enter__(self) -> "ScreenGrabber":
        import mss

        self._mss = mss.mss()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._mss is not None:
            self._mss.close()
            self._mss = None

    def grab_bgr(self) -> np.ndarray:
        if self._mss is None:
            raise RuntimeError("ScreenGrabber нужно использовать внутри with")
        bgra = np.asarray(self._mss.grab(self.region.as_mss_dict()))
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

