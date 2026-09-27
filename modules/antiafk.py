from __future__ import annotations

import threading
import time

from core.input import InputAdapter
from modules.base import StoppableTask


PATTERNS: dict[str, tuple[tuple[str, float], ...]] = {
    "gentle": (("a", 0.45), ("d", 0.45), ("w", 0.35), ("s", 0.35)),
    "walk": (("w", 1.0), ("s", 0.8)),
    "turn": (("a", 0.75), ("d", 0.75)),
}


class AntiAfkTask(StoppableTask):
    def __init__(self, input_adapter: InputAdapter, interval: float, pattern: str, on_log):
        self.input_adapter = input_adapter
        self.interval = max(5.0, interval)
        self.pattern = pattern if pattern in PATTERNS else "gentle"
        super().__init__("Anti-AFK", self._work, on_log)

    def _work(self, stop: threading.Event) -> None:
        self.log(f"Anti-AFK запущен: {self.pattern}, интервал {self.interval:g} с")
        while not stop.wait(self.interval):
            for key, duration in PATTERNS[self.pattern]:
                if stop.is_set():
                    break
                self.input_adapter.hold(key, duration)
            self.log("Anti-AFK: цикл движения выполнен")
        self.log("Anti-AFK остановлен")

