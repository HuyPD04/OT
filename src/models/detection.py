from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    class_id: int
    track_id: int | None = None