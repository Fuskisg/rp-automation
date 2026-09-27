from __future__ import annotations

import logging
from pathlib import Path


def create_logger() -> logging.Logger:
    Path("logs").mkdir(exist_ok=True)
    logger = logging.getLogger("gta5rp_automation")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler("logs/automation.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

