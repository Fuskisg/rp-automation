from __future__ import annotations

import threading
import time
from collections.abc import Iterable


class InputAdapter:
    """Small input layer shared by every module.

    Dry-run is enabled by default, so a scenario can be checked without sending
    keys to the active window. Real input is imported lazily only when needed.
    """

    def __init__(self, dry_run: bool = True, key_delay: float = 0.08, click_pause: float = 0.12) -> None:
        self.dry_run = dry_run
        self.key_delay = key_delay
        self.click_pause = click_pause
        self._lock = threading.RLock()
        self._pyautogui = None

    def _backend(self):
        if self._pyautogui is None:
            import pyautogui

            pyautogui.PAUSE = self.click_pause
            self._pyautogui = pyautogui
        return self._pyautogui

    def tap(self, key: str) -> None:
        with self._lock:
            if not self.dry_run:
                self._backend().press(key)
            time.sleep(self.key_delay)

    def tap_many(self, keys: Iterable[str]) -> None:
        for key in keys:
            self.tap(key)

    def key_down(self, key: str) -> None:
        with self._lock:
            if not self.dry_run:
                self._backend().keyDown(key)

    def key_up(self, key: str) -> None:
        with self._lock:
            if not self.dry_run:
                self._backend().keyUp(key)

    def hold(self, key: str, seconds: float) -> None:
        self.key_down(key)
        time.sleep(max(0.0, seconds))
        self.key_up(key)

    def click(self, x: int, y: int) -> None:
        with self._lock:
            if not self.dry_run:
                self._backend().click(x=x, y=y)
            time.sleep(self.click_pause)

