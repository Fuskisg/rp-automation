from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Settings:
    dry_run: bool = True
    anti_afk_interval: float = 45.0
    anti_afk_pattern: str = "gentle"
    global_hotkey: str = "f8"
    key_delay: float = 0.08
    click_pause: float = 0.12
    window_title_hint: str = "GTA"
    custom_macros: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


class SettingsStore:
    def __init__(self, path: str | Path = "config.json") -> None:
        self.path = Path(path)

    def load(self) -> Settings:
        if not self.path.exists():
            return Settings()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            values = asdict(Settings())
            values.update({k: v for k, v in raw.items() if k in values})
            return Settings(**values)
        except (OSError, ValueError, TypeError):
            return Settings()

    def save(self, settings: Settings) -> None:
        self.path.write_text(
            json.dumps(asdict(settings), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

