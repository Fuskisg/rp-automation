from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


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
    """Matches prompt images against a screenshot using normalized correlation.

    Templates are loaded once and reused. This keeps a mini-game loop cheap and
    makes it possible to test it with a regular PNG instead of the game window.
    """

    def __init__(self, threshold: float = 0.88):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold должен быть между 0 и 1")
        self.threshold = threshold
        self._templates: dict[str, np.ndarray] = {}

    def add(self, label: str, template: str | Path | np.ndarray) -> None:
        image = cv2.imread(str(template), cv2.IMREAD_COLOR) if isinstance(template, (str, Path)) else template
        if image is None or image.size == 0:
            raise ValueError(f"Не удалось загрузить шаблон: {label}")
        self._templates[label] = image

    def match_best(self, frame_bgr: np.ndarray) -> Match | None:
        candidates = [match for label in self._templates for match in self._find(label, frame_bgr)]
        return max(candidates, key=lambda item: item.score, default=None)

    def match_all(self, frame_bgr: np.ndarray) -> list[Match]:
        candidates = [match for label in self._templates for match in self._find(label, frame_bgr)]
        return sorted(candidates, key=lambda item: item.score, reverse=True)

    def _find(self, label: str, frame_bgr: np.ndarray) -> list[Match]:
        template = self._templates[label]
        height, width = template.shape[:2]
        frame_height, frame_width = frame_bgr.shape[:2]
        if height > frame_height or width > frame_width:
            return []
        scores = cv2.matchTemplate(frame_bgr, template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(scores >= self.threshold)
        matches = [Match(label, float(scores[y, x]), int(x), int(y), width, height) for y, x in zip(ys, xs)]
        return _suppress_overlaps(matches)


def _suppress_overlaps(matches: list[Match], iou_limit: float = 0.35) -> list[Match]:
    kept: list[Match] = []
    for candidate in sorted(matches, key=lambda item: item.score, reverse=True):
        if all(_iou(candidate, chosen) < iou_limit for chosen in kept):
            kept.append(candidate)
    return kept


def _iou(left: Match, right: Match) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.width, right.x + right.width)
    y2 = min(left.y + left.height, right.y + right.height)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union else 0.0

