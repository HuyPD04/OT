from __future__ import annotations

import threading
import time
import cv2
import numpy as np
import logging

from ..buffer.framebuffer import FrameBuffer
from ..buffer.detectionbuffer import DetectionBuffer
from ..buffer.trackerbuffer import TrackerBuffer
from ...utils.boxes import draw_detection, draw_track
from ...models.health import WorkerHealthState, HealthStatus

logger = logging.getLogger(__name__)

class ThreadDisplay:
    def __init__(
            self,
            frame_buffer: FrameBuffer,
            detection_buffer: DetectionBuffer | None = None,
            tracker_buffer: TrackerBuffer | None = None,
            window_name: str = "Object Tracking",
            thread_name: str = "ThreadDisplay"
    ) -> None:
        self._frame_buffer = frame_buffer
        self._detection_buffer = detection_buffer
        self._tracker_buffer = tracker_buffer
        self._window_name = window_name
        self._thread_name = thread_name

        self._health = WorkerHealthState(thread_name)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._health.set(HealthStatus.STARTING, "thread display starting")
        self._thread = threading.Thread(
            target=self._run,
            name=self._thread_name,
            daemon=False
        )
        self._thread.start()

    def _run(self) -> None:
        version = 0
        poll_timeout = 0.01
        last_frame: np.ndarray | None = None
        started_at = time.monotonic()
        waiting_warning_logged = False
        first_camera_frame_received = False

        try:
            self._health.set(HealthStatus.RUNNING, "thread display running")
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            last_frame = self._waiting_frame()
            cv2.imshow(self._window_name, last_frame)
            cv2.waitKey(1)

            while not self._stop_event.is_set():
                version, packet = self._frame_buffer.wait_next(
                    pre_version=version,
                    timeout=poll_timeout
                )
                if packet is not None:
                    last_frame = packet.image.copy()
                    first_camera_frame_received = True
                    self._draw_overlay(last_frame)

                cv2.imshow(self._window_name, last_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    self.stop()

                if (
                    not first_camera_frame_received
                    and not waiting_warning_logged
                    and time.monotonic() - started_at >= 5
                ):
                    logger.warning("Thread display is waiting for the first camera frame")
                    waiting_warning_logged = True
                if packet is None and self._frame_buffer.closed:
                    break

        except Exception:
            self._health.set(HealthStatus.FAILED, "thread display failed")
            logger.exception("Thread display failed")
        finally:
            cv2.destroyWindow(self._window_name)
            logger.info("Thread display stopped")
            if self._health.snapshot().status != HealthStatus.FAILED:
                self._health.set(HealthStatus.STOPPED, "preview worker stopped")

    @staticmethod
    def _waiting_frame() -> np.ndarray:
        image = np.full((540,960,3),32, dtype=np.uint8)
        cv2.putText(
            image,
            "Waiting for camera...",
            (245, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (220, 220, 220),
            2,
            cv2.LINE_AA
        )
        return image

    def _draw_overlay(self, image: np.ndarray) -> None:
        if self._tracker_buffer is not None:
            _, track_packet = self._tracker_buffer.snapshot()
            if track_packet is not None:
                for track in track_packet.tracks:
                    draw_track(image, track)
                return

        if self._detection_buffer is not None:
            _, detection_packet = self._detection_buffer.snapshot()
            if detection_packet is not None:
                for detection in detection_packet.detections:
                    draw_detection(image, detection)

    def stop(self) -> None:
        self._health.set(HealthStatus.STOPPING, "preview worker stopping")
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    @property
    def health(self) -> WorkerHealthState:
        return self._health.snapshot()
