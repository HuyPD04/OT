from __future__ import annotations

from ...models.track import TrackPacket
from .latestbuffer import LatestBuffer


class TrackerBuffer(LatestBuffer[TrackPacket]):
    pass
