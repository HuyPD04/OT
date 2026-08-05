from __future__ import annotations

from ...models.plate import PlatePacket
from .latestbuffer import LatestBuffer


class PlateBuffer(LatestBuffer[PlatePacket]):
    pass
