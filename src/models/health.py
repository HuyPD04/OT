from __future__ import annotations

from dataclasses import dataclass,  field
from enum import Enum
import time
import threading

class HealthStatus(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"

@dataclass(frozen=True, slots=False)
class Health:
    name: str
    status: HealthStatus
    message: str = ""
    last_error: str | None = None
    update_ns: int = field(default_factory=time.monotonic_ns)

class WorkerHealthState:
    def __init__(self, name: str) -> None:
        self._name = name
        self._lock = threading.Lock()
        self._health = Health(
            name=name,
            status=HealthStatus.STOPPED,
            message="not started",
        )

    def set(
        self,
        status: HealthStatus,
        message: str = "",
        error: BaseException | None = None,
    ) -> None:
        last_error = None
        if error is not None:
            last_error = f"{error.__class__.__name__}: {error}"

        with self._lock:
            self._health = Health(
                name=self._name,
                status=status,
                message=message,
                last_error=last_error,
            )

    def snapshot(self) -> Health:
        with self._lock:
            return self._health