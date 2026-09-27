from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import mss
import numpy as np

from core.input import InputAdapter
from modules.base import StoppableTask


class VisionMonitorTask(StoppableTask):
    """Continuously watch one template and optionally react to a match."""

    def __init__(
        self,
        input_adapter: InputAdapter,
        image_path: str | Path,
        threshold: float,
        interval: float,
        action: str,
        key: str,
        on_log,
    ) -> None:
        self.input_adapter = input_adapter
        self.image_path = Path(image_path)
        self.threshold = threshold
        self.interval = max(0.05, interval)
        self.action = action
        self.key = key.strip()
        super().__init__("Распознавание", self._work, on_log)

    def _work(self, stop: threading.Event) -> None:
        template = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if template is None:
            self.log(f"Не удалось открыть шаблон: {self.image_path}")
            return
        height, width = template.shape[:2]
        last_action = 0.0
        last_state = False
        with mss.mss() as capture:
            monitor = capture.monitors[1]
            self.log(f"Мониторинг запущен: {self.image_path.name}, порог {self.threshold:.2f}")
            while not stop.is_set():
                frame = cv2.cvtColor(np.asarray(capture.grab(monitor)), cv2.COLOR_BGRA2BGR)
                if height <= frame.shape[0] and width <= frame.shape[1]:
                    scores = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
                    _, score, _, point = cv2.minMaxLoc(scores)
                    found = score >= self.threshold
                    if found and not last_state:
                        x = monitor["left"] + point[0] + width // 2
                        y = monitor["top"] + point[1] + height // 2
                        self.log(f"Найдено: {self.image_path.name}, точность {score:.3f}, центр ({x}, {y})")
                    if found and time.monotonic() - last_action >= 0.8:
                        x = monitor["left"] + point[0] + width // 2
                        y = monitor["top"] + point[1] + height // 2
                        if self.action == "key" and self.key:
                            self.input_adapter.tap(self.key)
                            self.log(f"Действие: клавиша {self.key}")
                            last_action = time.monotonic()
                        elif self.action == "click":
                            self.input_adapter.click(x, y)
                            self.log(f"Действие: клик по ({x}, {y})")
                            last_action = time.monotonic()
                    last_state = found
                if stop.wait(self.interval):
                    break
        self.log("Мониторинг остановлен")
