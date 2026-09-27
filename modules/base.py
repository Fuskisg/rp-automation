from __future__ import annotations

import threading
from collections.abc import Callable


class StoppableTask(threading.Thread):
    def __init__(self, name: str, target: Callable[[threading.Event], None], on_log: Callable[[str], None]) -> None:
        super().__init__(name=name, daemon=True)
        self.stop_event = threading.Event()
        self._target_fn = target
        self._on_log = on_log

    def log(self, message: str) -> None:
        self._on_log(message)

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        try:
            self._target_fn(self.stop_event)
        except Exception as exc:  # keep the UI alive when a module fails
            self.log(f"Ошибка модуля {self.name}: {exc}")

