from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from core.input import InputAdapter
from modules.base import StoppableTask


@dataclass(frozen=True)
class Step:
    kind: str
    value: str
    seconds: float = 0.0


def parse_script(text: str) -> list[Step]:
    steps: list[Step] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        command = parts[0].lower()
        if command == "tap" and len(parts) == 2:
            steps.append(Step("tap", parts[1]))
        elif command == "hold" and len(parts) == 3:
            steps.append(Step("hold", parts[1], float(parts[2])))
        elif command == "wait" and len(parts) == 2:
            steps.append(Step("wait", "", float(parts[1])))
        else:
            raise ValueError(f"Строка {number}: используй tap KEY, hold KEY SEC или wait SEC")
    return steps


class MacroTask(StoppableTask):
    def __init__(self, input_adapter: InputAdapter, steps: list[Step], repeat: int, on_log):
        self.input_adapter = input_adapter
        self.steps = steps
        self.repeat = max(1, repeat)
        super().__init__("Macro", self._work, on_log)

    def _work(self, stop: threading.Event) -> None:
        self.log(f"Сценарий запущен: {len(self.steps)} действий × {self.repeat}")
        for cycle in range(self.repeat):
            for step in self.steps:
                if stop.is_set():
                    self.log("Сценарий остановлен")
                    return
                if step.kind == "tap":
                    self.input_adapter.tap(step.value)
                elif step.kind == "hold":
                    self.input_adapter.hold(step.value, step.seconds)
                else:
                    stop.wait(step.seconds)
            self.log(f"Сценарий: цикл {cycle + 1}/{self.repeat}")
        self.log("Сценарий завершён")

