from __future__ import annotations

from dataclasses import dataclass

from .framepacket import FramePacket

@dataclass(frozen=True, slots=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    class_id: int
    track_id: int | None = None

@dataclass(frozen=True, slots=False)
class DetectionPacket:
    frame: FramePacket
    detections: tuple[Detection, ...]
    inference_ms: float
