from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenRegion:
    left: int = 0
    top: int = 0
    width: int = 1920
    height: int = 1080


def capture(region: ScreenRegion | None = None):
    """Capture a screen region lazily; returns a numpy BGRA array."""
    import mss
    import numpy as np

    target = region or ScreenRegion()
    with mss.mss() as camera:
        shot = camera.grab({"left": target.left, "top": target.top, "width": target.width, "height": target.height})
    return np.asarray(shot)


def find_color(region: ScreenRegion, color: tuple[int, int, int], tolerance: int = 20) -> tuple[int, int] | None:
    """Return the first pixel near an RGB color, or None when absent."""
    import cv2
    import numpy as np

    frame = capture(region)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    target = np.array(color, dtype=np.int16)
    delta = np.abs(rgb.astype(np.int16) - target)
    mask = np.all(delta <= tolerance, axis=2).astype(np.uint8)
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y = points[0][0]
    return int(x + region.left), int(y + region.top)

