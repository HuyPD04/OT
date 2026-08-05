from __future__ import annotations

import logging
import threading
import time

import numpy as np

from ...models.health import HealthStatus, WorkerHealthState
from ...models.track import Track, TrackPacket
from ...utils.tracker import TrackedObject, Tracker
from ..buffer.detectionbuffer import DetectionBuffer
from ..buffer.trackerbuffer import TrackerBuffer

logger = logging.getLogger(__name__)


class ThreadTracking:
    def __init__(
            self,
            detection_buffer: DetectionBuffer,
            tracker_buffer: TrackerBuffer,
            max_age: int = 10,
            min_hits: int = 1,
            iou_threshold: float = 0.3,
            center_threshold: float = 1.5,
            center_weight: float = 0.35,
            thread_name: str = "ThreadTracking",
    ) -> None:
        self._detection_buffer = detection_buffer
        self._tracker_buffer = tracker_buffer
        self._tracker = Tracker(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            center_threshold=center_threshold,
            center_weight=center_weight,
        )
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
        detection_version = 0
        self._health.set(HealthStatus.RUNNING, "thread tracking running")

        while not self._stop_event.is_set():
            detection_version, detection_packet = self._detection_buffer.wait_next(
                pre_version=detection_version,
                timeout=0.5,
            )
            if detection_packet is None:
                if self._detection_buffer.closed:
                    break
                continue

            try:
                started_at = time.perf_counter()
                detections = np.asarray(
                    [
                        [det.x1, det.y1, det.x2, det.y2, det.conf, det.class_id]
                        for det in detection_packet.detections
                    ],
                    dtype=float,
                )
                tracked_objects = self._tracker.update(detections)
                tracks = tuple(
                    track
                    for tracked_object in tracked_objects
                    if (
                        track := self._build_track(
                            tracked_object,
                            detection_packet.frame.width,
                            detection_packet.frame.height,
                        )
                    ) is not None
                )
                self._tracker_buffer.publish(
                    TrackPacket(
                        frame=detection_packet.frame,
                        tracks=tracks,
                        detection_frame_id=detection_packet.frame.frame_id,
                        tracking_ms=time.perf_counter() - started_at,
                    )
                )
            except Exception as error:
                logger.exception("Error during tracking: %s", error)
                self._health.set(HealthStatus.FAILED, f"Error during tracking: {error}")

        self._health.set(HealthStatus.STOPPED, "thread tracking stopped")

    def _build_track(
            self,
            tracked_object: TrackedObject,
            frame_width: int,
            frame_height: int,
    ) -> Track | None:
        x1 = min(frame_width - 1, max(0, int(round(tracked_object.x1))))
        y1 = min(frame_height - 1, max(0, int(round(tracked_object.y1))))
        x2 = min(frame_width - 1, max(0, int(round(tracked_object.x2))))
        y2 = min(frame_height - 1, max(0, int(round(tracked_object.y2))))
        if x2 <= x1 or y2 <= y1:
            return None

        return Track(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            track_id=tracked_object.track_id,
            conf=tracked_object.score,
            class_id=tracked_object.class_id,
        )

    def stop(self) -> None:
        self._stop_event.set()
        self._health.set(HealthStatus.STOPPING, "thread tracking stopping")

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    @property
    def health(self) -> WorkerHealthState:
        return self._health.snapshot()
