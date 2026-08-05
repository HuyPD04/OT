from __future__ import annotations

from dataclasses import dataclass

from .framepacket import FramePacket


@dataclass(frozen=True, slots=True)
class PlateDetection:
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    class_id: int | None = None


@dataclass(frozen=True, slots=True)
class PlatePacket:
    frame: FramePacket
    plates: tuple[PlateDetection, ...]
    detection_frame_id: str | None = None
    inference_ms: float = 0.0
