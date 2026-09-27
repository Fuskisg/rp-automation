"""Практический детектор мини-игр, перенесённый из предыдущего проекта.

Сохранены рабочие приёмы старой версии: mss для кадров, OpenCV
matchTemplate для подсказок и HSV/пиксельные маски для индикаторов.
Игровые шаблоны намеренно передаются снаружи и не входят в репозиторий.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .screen import CaptureRegion, ScreenGrabber


@dataclass(frozen=True)
class Prompt:
    name: str
    key: str
    score: float
    center: tuple[int, int]


class LegacyMiniGameDetector:
    """Detector compatible with the old template-driven mini-game loops."""

    def __init__(self, region: CaptureRegion, threshold: float = 0.90) -> None:
        self.region = region
        self.threshold = threshold
        self._templates: dict[str, tuple[str, np.ndarray]] = {}

    def add_template(self, name: str, key: str, image_path: str | Path) -> None:
        template = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if template is None:
            raise FileNotFoundError(image_path)
        self._templates[name] = (key, template)

    def scan_once(self) -> Prompt | None:
        with ScreenGrabber(self.region) as grabber:
            frame = grabber.grab_bgr()
        best: Prompt | None = None
        for name, (key, template) in self._templates.items():
            h, w = template.shape[:2]
            if h > frame.shape[0] or w > frame.shape[1]:
                continue
            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, point = cv2.minMaxLoc(result)
            if score < self.threshold:
                continue
            candidate = Prompt(name, key, float(score), (point[0] + w // 2 + self.region.left, point[1] + h // 2 + self.region.top))
            if best is None or candidate.score > best.score:
                best = candidate
        return best


@dataclass(frozen=True)
class ColorMarker:
    name: str
    lower_hsv: tuple[int, int, int]
    upper_hsv: tuple[int, int, int]
    min_area: int = 30


class LegacyColorDetector:
    """HSV contour detector used for colored progress bars and targets."""

    def __init__(self, markers: list[ColorMarker]) -> None:
        self.markers = markers

    def scan(self, frame_bgr: np.ndarray) -> list[tuple[str, tuple[int, int], float]]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        result: list[tuple[str, tuple[int, int], float]] = []
        for marker in self.markers:
            mask = cv2.inRange(hsv, np.array(marker.lower_hsv), np.array(marker.upper_hsv))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < marker.min_area:
                    continue
                x, y, width, height = cv2.boundingRect(contour)
                result.append((marker.name, (x + width // 2, y + height // 2), area))
        return sorted(result, key=lambda item: item[2], reverse=True)


def find_image_on_screen(path: str | Path, confidence: float = 0.90, region: tuple[int, int, int, int] | None = None):
    """Compatibility helper for old image-on-screen checks.

    Import is lazy so template matching can be tested on saved frames without
    installing GUI input packages.
    """
    import pyautogui

    return pyautogui.locateOnScreen(str(path), confidence=confidence, region=region)

