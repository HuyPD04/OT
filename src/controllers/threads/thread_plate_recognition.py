from __future__ import annotations

import logging
import threading
import time

import numpy as np

from ...models.health import HealthStatus, WorkerHealthState
from ...models.plate import PlateDetection, PlatePacket
from ...models.track import Track
from ...utils.boxes import postprocess, preprocess
from ..backend.onnx_runtime import OnnxBackend
from ..buffer.platebuffer import PlateBuffer
from ..buffer.trackerbuffer import TrackerBuffer

logger = logging.getLogger(__name__)


class ThreadPlateRecognition:
    def __init__(
            self,
            tracker_buffer: TrackerBuffer,
            plate_buffer: PlateBuffer,
            backend: OnnxBackend,
            vehicle_class_ids: list[int] | None = None,
            roi_padding: float = 0.05,
            min_roi_size: int = 32,
            min_inference_interval_seconds: float = 0.5,
            thread_name: str = "ThreadPlateRecognition",
    ) -> None:
        self._tracker_buffer = tracker_buffer
        self._plate_buffer = plate_buffer
        self._backend = backend
        self._vehicle_class_ids = set(vehicle_class_ids or [2, 3, 5, 7])
        self._roi_padding = max(0.0, roi_padding)
        self._min_roi_size = max(1, min_roi_size)
        self._min_inference_interval_seconds = max(0.0, min_inference_interval_seconds)
        self._thread_name = thread_name
        self._last_inference_by_track: dict[int, float] = {}
        self._last_plates_by_track: dict[int, tuple[PlateDetection, ...]] = {}

        self._health = WorkerHealthState(name=thread_name)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._health.set(HealthStatus.STARTING, "thread plate recognition starting")
        self._thread = threading.Thread(
            target=self._run,
            name=self._thread_name,
            daemon=False,
        )
        self._thread.start()

    def _run(self) -> None:
        version = 0
        self._health.set(HealthStatus.RUNNING, "thread plate recognition running")

        while not self._stop_event.is_set():
            version, track_packet = self._tracker_buffer.wait_next(
                pre_version=version,
                timeout=0.5,
            )
            if track_packet is None:
                if self._tracker_buffer.closed:
                    break
                continue

            try:
                started_at = time.perf_counter()
                plates = []
                now = time.monotonic()
                active_track_ids = {track.track_id for track in track_packet.tracks}
                self._last_inference_by_track = {
                    track_id: inferred_at
                    for track_id, inferred_at in self._last_inference_by_track.items()
                    if track_id in active_track_ids
                }
                self._last_plates_by_track = {
                    track_id: track_plates
                    for track_id, track_plates in self._last_plates_by_track.items()
                    if track_id in active_track_ids
                }
                for track in track_packet.tracks:
                    if not self._is_vehicle_track(track):
                        self._last_plates_by_track.pop(track.track_id, None)
                        continue
                    if not self._should_infer(track, now):
                        plates.extend(self._last_plates_by_track.get(track.track_id, ()))
                        continue

                    crop, offset = self._crop_track(track_packet.frame.image, track)
                    if crop is None:
                        self._last_plates_by_track.pop(track.track_id, None)
                        continue

                    tensor, transport = preprocess(
                        crop,
                        input_size=self._backend.input_size,
                    )
                    outputs = self._backend.infer(tensor)
                    detections = postprocess(
                        outputs,
                        crop.shape[:2],
                        transport,
                        confidence_threshold=self._backend.confidence_threshold,
                        iou_threshold=self._backend.iou_threshold,
                        class_ids=self._backend.class_ids,
                    )
                    self._last_inference_by_track[track.track_id] = time.monotonic()

                    track_plates = []
                    for detection in detections:
                        track_plates.append(
                            PlateDetection(
                                track_id=track.track_id,
                                x1=detection.x1 + offset[0],
                                y1=detection.y1 + offset[1],
                                x2=detection.x2 + offset[0],
                                y2=detection.y2 + offset[1],
                                conf=detection.conf,
                                class_id=detection.class_id,
                            )
                        )
                    self._last_plates_by_track[track.track_id] = tuple(track_plates)
                    plates.extend(track_plates)

                self._plate_buffer.publish(
                    PlatePacket(
                        frame=track_packet.frame,
                        plates=tuple(plates),
                        detection_frame_id=track_packet.detection_frame_id,
                        inference_ms=time.perf_counter() - started_at,
                    )
                )
            except Exception as error:
                logger.exception("Error during plate recognition: %s", error)
                self._health.set(HealthStatus.FAILED, f"Error during plate recognition: {error}")

        self._health.set(HealthStatus.STOPPED, "thread plate recognition stopped")

    def _is_vehicle_track(self, track: Track) -> bool:
        return track.class_id in self._vehicle_class_ids

    def _should_infer(self, track: Track, now: float) -> bool:
        previous = self._last_inference_by_track.get(track.track_id)
        if previous is None:
            return True

        return now - previous >= self._min_inference_interval_seconds

    def _crop_track(self, image: np.ndarray, track: Track) -> tuple[np.ndarray, tuple[int, int]] | tuple[None, tuple[int, int]]:
        height, width = image.shape[:2]
        track_width = max(0, track.x2 - track.x1)
        track_height = max(0, track.y2 - track.y1)
        if track_width < self._min_roi_size or track_height < self._min_roi_size:
            return None, (0, 0)

        pad_x = int(round(track_width * self._roi_padding))
        pad_y = int(round(track_height * self._roi_padding))
        x1 = max(0, track.x1 - pad_x)
        y1 = max(0, track.y1 - pad_y)
        x2 = min(width - 1, track.x2 + pad_x)
        y2 = min(height - 1, track.y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None, (0, 0)

        return image[y1:y2, x1:x2], (x1, y1)

    def stop(self) -> None:
        self._stop_event.set()
        self._health.set(HealthStatus.STOPPING, "thread plate recognition stopping")

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    @property
    def health(self) -> WorkerHealthState:
        return self._health.snapshot()
