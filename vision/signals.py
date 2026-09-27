from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ColorSignal:
    label: str
    rgb: tuple[int, int, int]
    tolerance: int = 24
    minimum_area: int = 20


class ColorSignalDetector:
    """Finds a colored progress marker without hard-coding screen coordinates."""

    def __init__(self, signals: list[ColorSignal]):
        self.signals = signals

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[ColorSignal, tuple[int, int], int]]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        found: list[tuple[ColorSignal, tuple[int, int], int]] = []
        for signal in self.signals:
            rgb = np.uint8([[signal.rgb]])
            target_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0, 0].astype(int)
            lower = np.array([max(0, target_hsv[0] - signal.tolerance), max(0, target_hsv[1] - 70), max(0, target_hsv[2] - 70)])
            upper = np.array([min(179, target_hsv[0] + signal.tolerance), 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = int(cv2.contourArea(contour))
                if area < signal.minimum_area:
                    continue
                moments = cv2.moments(contour)
                if not moments["m00"]:
                    continue
                center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
                found.append((signal, center, area))
        return sorted(found, key=lambda item: item[2], reverse=True)

