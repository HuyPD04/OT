from abc import ABC, abstractmethod

from .framepacket import FramePacket
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class CameraConfig: 
    camera_id: str
    url: str
    codec: str = "h265"
    transport: str = "tcp"
    latency_ms: int = 500

class Camera(ABC):
    @abstractmethod
    def open(self) -> None:
        pass

    @abstractmethod
    def read(self, timeout: float) -> FramePacket | None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def is_open(self) -> bool:
        pass