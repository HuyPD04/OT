from __future__ import annotations

import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "src" / "resources" / "logs"

def setup_logging(level=logging.INFO):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    root = logging.getLogger()
    root.setLevel(level=level)

    if root.handlers:
        return
    console = logging.StreamHandler()
    console.setFormatter(formatter)

    file = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    file.setFormatter(formatter)

    root.addHandler(console)
    root.addHandler(file)
