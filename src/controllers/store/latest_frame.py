from __future__ import annotations

import threading
from ...models.framepacket import FramePacket

class LatestFrameStore:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame: FramePacket | None = None
        self._version = 0
        self._closed = False

    def publish(self, frame: FramePacket) -> None:
        with self._condition:
            if self._closed:
                return
            self._frame = frame
            self._version += 1

            self._condition.notify_all()

    def wait_next(self, pre_version: int, timeout: float) -> tuple[int, FramePacket | None]:
        with self._condition:
            updated = self._condition.wait_for(lambda: self._closed or self._version > pre_version, timeout=timeout)
            if not updated or self._closed:
                return pre_version, None

            return self._version, self._frame
        
    def snapshot(self) -> tuple[int, FramePacket | None]:
        with self._condition:
            return self._version, self._frame

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed