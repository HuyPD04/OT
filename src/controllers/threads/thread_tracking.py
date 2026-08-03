from __future__ import annotations

import logging
import threading
import time

import numpy as np

from ...models.health import HealthStatus, WorkerHealthState
from ...models.track import Track, TrackPacket
from ...utils.sort import Sort
from ..buffer.detectionbuffer import DetectionBuffer
from ..buffer.framebuffer import FrameBuffer
from ..buffer.trackerbuffer import TrackerBuffer

logger = logging.getLogger(__name__)


class ThreadTracking:
    def __init__(
            self,
            frame_buffer: FrameBuffer,
            detection_buffer: DetectionBuffer,
            tracker_buffer: TrackerBuffer,
            max_age: int = 10,
            min_hits: int = 1,
            iou_threshold: float = 0.3,
            max_detection_lag_frames: int = 10,
            thread_name: str = "ThreadTracking",
    ) -> None:
        self._frame_buffer = frame_buffer
        self._detection_buffer = detection_buffer
        self._tracker_buffer = tracker_buffer
        self._tracker = Sort(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
        )
        self._max_detection_lag_frames = max_detection_lag_frames
        self._thread_name = thread_name

        self._health = WorkerHealthState(name=thread_name)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._health.set(HealthStatus.STARTING, "thread tracking starting")
        self._thread = threading.Thread(
            target=self._run,
            name=self._thread_name,
            daemon=False,
        )
        self._thread.start()

    def _run(self) -> None:
        frame_version = 0
        last_detection_version = 0
        self._health.set(HealthStatus.RUNNING, "thread tracking running")

        while not self._stop_event.is_set():
            frame_version, frame = self._frame_buffer.wait_next(
                pre_version=frame_version,
                timeout=0.5,
            )
            if frame is None:
                if self._frame_buffer.closed:
                    break
                continue

            start = time.perf_counter()
            detection_version, detection_packet = self._detection_buffer.snapshot()
            detections = np.empty((0, 5), dtype=float)
            detection_frame_id: str | None = None

            if (
                    detection_packet is not None
                    and detection_version > last_detection_version
                    and self._is_detection_usable(frame.frame_id, detection_packet.frame.frame_id)
            ):
                detection_frame_id = detection_packet.frame.frame_id
                last_detection_version = detection_version
                if detection_packet.detections:
                    detections = np.array(
                        [
                            [det.x1, det.y1, det.x2, det.y2, det.conf]
                            for det in detection_packet.detections
                        ],
                        dtype=float,
                    )

            try:
                self._tracker.update(detections)
                sort_tracks = self._tracker.get_tracks()
                tracks = tuple(
                    Track(
                        x1=max(0, int(round(track[0]))),
                        y1=max(0, int(round(track[1]))),
                        x2=min(frame.width - 1, int(round(track[2]))),
                        y2=min(frame.height - 1, int(round(track[3]))),
                        track_id=int(track[4]),
                    )
                    for track in sort_tracks
                    if track[2] > track[0] and track[3] > track[1]
                )
                tracking_ms = time.perf_counter() - start
                self._tracker_buffer.publish(
                    TrackPacket(
                        frame=frame,
                        tracks=tracks,
                        detection_frame_id=detection_frame_id,
                        tracking_ms=tracking_ms,
                    )
                )
            except Exception as error:
                logger.exception("Error during tracking: %s", error)
                self._health.set(HealthStatus.FAILED, f"Error during tracking: {error}")

        self._health.set(HealthStatus.STOPPED, "thread tracking stopped")

    def _is_detection_usable(self, frame_id: str, detection_frame_id: str) -> bool:
        try:
            current = int(frame_id)
            detected = int(detection_frame_id)
        except ValueError:
            return detection_frame_id == frame_id

        lag = current - detected
        return 0 <= lag <= self._max_detection_lag_frames

    def stop(self) -> None:
        self._stop_event.set()
        self._health.set(HealthStatus.STOPPING, "thread tracking stopping")

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    @property
    def health(self) -> WorkerHealthState:
        return self._health.snapshot()
