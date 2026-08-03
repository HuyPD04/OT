from __future__ import annotations

from dataclasses import dataclass

from .framepacket import FramePacket


@dataclass(frozen=True, slots=True)
class Track:
    x1: int
    y1: int
    x2: int
    y2: int
    track_id: int
    conf: float | None = None
    class_id: int | None = None


@dataclass(frozen=True, slots=True)
class TrackPacket:
    frame: FramePacket
    tracks: tuple[Track, ...]
    detection_frame_id: str | None = None
    tracking_ms: float = 0.0
