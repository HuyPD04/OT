from __future__ import annotations

import logging
import threading
import time
import random

from ..buffer.framebuffer import FrameBuffer
from ...models.camera import Camera
from ...models.health import HealthStatus, WorkerHealthState

logger = logging.getLogger(__name__)

class ThreadCapture:
    def __init__(
            self,
            camera: Camera,
            output: FrameBuffer,
            timeout_ms: int = 200,
            thread_name: str = "ThreadCapture",
            reconnect_initital: float = 1.0,
            reconnect_max: float = 30.0, 
            reconnect_stable_seconds: float = 30.0,
            reconnect_stable_frames: int = 100,
            minimum_capture_fps: float = 3.0,
            degraded_window_seconds: float = 3.0,
            startup_grace_seconds: float = 10.0
    ) -> None:
        self._camera = camera
        self._output = output
        self._timeout = timeout_ms
        self._thread_name = thread_name
        self._reconnect_initital = reconnect_initital
        self._reconnect_max = reconnect_max
        self._reconnect_stable_seconds = reconnect_stable_seconds
        self._reconnect_stable_frames = reconnect_stable_frames
        self._minimum_capture_fps = minimum_capture_fps
        self._degraded_window_seconds = degraded_window_seconds
        self._startup_grace_seconds = startup_grace_seconds

        self._health = WorkerHealthState(name=thread_name)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._health.set(HealthStatus.STARTING, "Thread capture starting")
        self._thread = threading.Thread(
            target=self._run,
            name = self._thread_name,
            daemon=False
        )
        self._thread.start()

    def _wait_or_stop(self, seconds: float) -> bool:
        return self._stop_event.wait(seconds)

    def _caculate_backoff(self, attempt: float) -> float:
        exp = self._reconnect_initital*(2**min(attempt, 10))
        base_delay = min(exp, self._reconnect_max)
        jitter = random.uniform(0, base_delay*0.2)
        return base_delay+jitter

    def _is_stable_connection(
            self,
            connect_at: float,
            stable_frames: int,
            now: float
    ) -> bool:
        return (now - connect_at >= self._reconnect_stable_seconds
                and stable_frames >= self._reconnect_stable_frames)

    def _run(self) -> None:
        reconnect_attempt = 0

        while not self._stop_event.is_set():
            camera = self._camera()
            last_error: BaseException | None = None
            try:
                logger.info("Camera connecting", extra={"attempt": reconnect_attempt})
                self._health.set(HealthStatus.STARTING, "Camera connecting")
                camera.open()
                logger.info("Camera connected")
                self._health.set(HealthStatus.RUNNING, "Camera connected")
                connected_at = time.monotonic()
                stable_frames = 0
                health_start_at = time.monotonic()
                health_frames = 0
                first_frame_received = False
                degraded_started_at: float | None = None
                while not self._stop_event.is_set():
                    packet = camera.read(timeout_ms=self._timeout)
                    if packet is not None:
                        self._output.publish(packet)
                        health_frames += 1
                        stable_frames += 1
                        first_frame_received = True
                    now = time.monotonic()
                    if reconnect_attempt > 0 and self._is_stable_connection(connect_at=connected_at, stable_frames=stable_frames, now=now):
                        reconnect_attempt = 0
                        logger.info("Camera connection stable")
                    health_elapsed = now - health_start_at
                    if health_elapsed >= 1.0:
                        capture_fps = health_frames / health_elapsed
                        if (
                            not first_frame_received
                            and now - connected_at < self._startup_grace_seconds
                        ):
                            self._health.set(
                                HealthStatus.STARTING,
                                "Waiting for first camera frame",
                            )
                            health_start_at = now
                            health_frames = 0
                            continue

                        if capture_fps < self._minimum_capture_fps:
                            if degraded_started_at is None:
                                degraded_started_at = now
                                logger.warning(f"Camera stream degraded: {capture_fps:.1f} FPS")
                                self._health.set(HealthStatus.DEGRADED,
                                                 f"Capture fps below target: {capture_fps:.1f}")
                            elif now - degraded_started_at >= self._degraded_window_seconds:
                                raise RuntimeError(f"RTSP capture remained degraded at {capture_fps:.1f}")
                        else:
                            degraded_started_at = None
                            self._health.set(HealthStatus.RUNNING, "Camera running")

                        health_start_at = now
                        health_frames = 0
            except Exception as error:
                last_error = error
                self._health.set(
                    HealthStatus.DEGRADED,
                    "camera stream failed; reconnect scheduled",
                    error,
                )
                logger.exception("Camera stream failed")
            finally:
                camera.close()

            if self._stop_event.is_set():
                break

            delay = self._caculate_backoff(reconnect_attempt)
            reconnect_attempt += 1
            logger.warning(f"Camera reconnect scheduled in {delay:.1f}s (attempt {reconnect_attempt})")

            if self._wait_or_stop(delay):
                break

        logger.info("Capture worker stopped")
        self._health.set(HealthStatus.STOPPED, "capture worker stopped")

    def stop(self) -> None:
        self._health.set(HealthStatus.STOPPED, "capture worker stopped")
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    @property
    def health(self) -> WorkerHealthState:
        return self._health.snapshot()
