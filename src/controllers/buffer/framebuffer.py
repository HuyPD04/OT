from __future__ import annotations

from ...models.framepacket import FramePacket
from .latestbuffer import LatestBuffer


class FrameBuffer(LatestBuffer[FramePacket]):
    pass
