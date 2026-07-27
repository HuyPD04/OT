from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class FramePacket:
    camera_id: str
    frame_id: str

    received_ns: int
    pts_ns: int
    width: int
    height: int
    image: np.ndarray
    stream_epoch: int = 0