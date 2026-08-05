from __future__ import annotations

from ...models.detection import DetectionPacket
from .latestbuffer import LatestBuffer


class DetectionBuffer(LatestBuffer[DetectionPacket]):
    pass
