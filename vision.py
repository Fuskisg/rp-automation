from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    def __init__(self, region: CaptureRegion):
        self.region = region
        self._camera = None

    def __enter__(self):
        import mss
        self._camera = mss.mss()
        return self

    def __exit__(self, *_):
        if self._camera:
            self._camera.close()
            self._camera = None

    def grab_bgr(self) -> np.ndarray:
        if self._camera is None:
            raise RuntimeError("Используй ScreenGrabber внутри with")
        bgra = np.asarray(self._camera.grab(self.region.as_mss_dict()))
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)


@dataclass(frozen=True)
class Match:
    label: str
    score: float
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2


class TemplateMatcher:
    def __init__(self, threshold: float = 0.88):
        if not 0 < threshold <= 1:
            raise ValueError("threshold должен быть между 0 и 1")
        self.threshold = threshold
        self.templates: dict[str, np.ndarray] = {}

    def add(self, label: str, image: str | Path | np.ndarray) -> None:
        data = cv2.imread(str(image), cv2.IMREAD_COLOR) if isinstance(image, (str, Path)) else image
        if data is None or data.size == 0:
            raise ValueError(f"Шаблон не загружен: {label}")
        self.templates[label] = data

    def match_best(self, frame_bgr: np.ndarray) -> Match | None:
        found: list[Match] = []
        for label, template in self.templates.items():
            h, w = template.shape[:2]
            if h > frame_bgr.shape[0] or w > frame_bgr.shape[1]:
                continue
            scores = cv2.matchTemplate(frame_bgr, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, point = cv2.minMaxLoc(scores)
            if score >= self.threshold:
                found.append(Match(label, float(score), point[0], point[1], w, h))
        return max(found, key=lambda item: item.score, default=None)


@dataclass(frozen=True)
class ColorSignal:
    label: str
    rgb: tuple[int, int, int]
    tolerance: int = 24
    minimum_area: int = 20


class ColorSignalDetector:
    def __init__(self, signals: list[ColorSignal]):
        self.signals = signals

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[ColorSignal, tuple[int, int], int]]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        result = []
        for signal in self.signals:
            target = cv2.cvtColor(np.uint8([[signal.rgb]]), cv2.COLOR_RGB2HSV)[0, 0].astype(int)
            low = np.array([max(0, target[0]-signal.tolerance), max(0, target[1]-70), max(0, target[2]-70)])
            high = np.array([min(179, target[0]+signal.tolerance), 255, 255])
            mask = cv2.inRange(hsv, low, high)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = int(cv2.contourArea(contour))
                moments = cv2.moments(contour)
                if area < signal.minimum_area or not moments["m00"]:
                    continue
                center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
                result.append((signal, center, area))
        return sorted(result, key=lambda item: item[2], reverse=True)
