from __future__ import annotations

import threading
from typing import Generic, TypeVar


PacketT = TypeVar("PacketT")


class LatestBuffer(Generic[PacketT]):

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._packet: PacketT | None = None
        self._version = 0
        self._closed = False

    def publish(self, packet: PacketT) -> None:
        with self._condition:
            if self._closed:
                return
            self._packet = packet
            self._version += 1
            self._condition.notify_all()

    def wait_next(
            self,
            pre_version: int,
            timeout: float,
    ) -> tuple[int, PacketT | None]:
        with self._condition:
            updated = self._condition.wait_for(
                lambda: self._closed or self._version > pre_version,
                timeout=timeout,
            )
            if not updated or self._closed:
                return pre_version, None
            return self._version, self._packet

    def snapshot(self) -> tuple[int, PacketT | None]:
        with self._condition:
            return self._version, self._packet

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed
