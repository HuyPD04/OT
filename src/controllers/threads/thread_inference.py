from __future__ import annotations

import logging
import threading
import time

from ...utils.boxes import preprocess, postprocess
from ...models.detection import DetectionPacket
from ..buffer.framebuffer import FrameBuffer
from ...models.health import HealthStatus, WorkerHealthState
from ..buffer.detectionbuffer import DetectionBuffer
from ..backend.onnx_runtime import OnnxBackend

logger = logging.getLogger(__name__)

class ThreadInference:
    def __init__(
            self,
            frame_buffer: FrameBuffer,
            detection_buffer: DetectionBuffer,
            backend: OnnxBackend,
            thread_name: str = "ThreadInference"
    ):
        self._frame_buffer = frame_buffer
        self._detection_buffer = detection_buffer
        self._backend = backend
        self._thread_name = thread_name

        self._health = WorkerHealthState(name=thread_name)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._health.set(HealthStatus.STARTING, "Thread inference starting")
        self._thread = threading.Thread(
            target=self._run,
            name = self._thread_name,
            daemon=False
        )
        self._thread.start()

    def _run(self) -> None:
        version = 0
        self._health.set(HealthStatus.RUNNING, "Thread inference running")

        while not self._stop_event.is_set():
            version, packet = self._frame_buffer.wait_next(
                pre_version=version,
                timeout=0.5
            )
            if packet is None:
                continue

            try:
                tensor, transport = preprocess(
                    packet.image,
                    input_size=self._backend.input_size,
                )
                infer_ms = time.perf_counter()
                outputs = self._backend.infer(tensor)
                infer_ms = (time.perf_counter() - infer_ms)
                detections = postprocess(
                    outputs,
                    packet.image.shape[:2],
                    transport,
                    confidence_threshold=self._backend.confidence_threshold,
                    iou_threshold=self._backend.iou_threshold,
                    class_ids=self._backend.class_ids,
                )

                self._detection_buffer.publish(
                    DetectionPacket(
                        frame=packet,
                        detections=tuple(detections),
                        inference_ms=infer_ms
                    )
                )
            except Exception as e:
                logger.exception("Error during inference: %s", e)
                self._health.set(HealthStatus.FAILED, f"Error during inference: {e}")

        self._health.set(HealthStatus.STOPPED, "Thread inference stopped")

    def stop(self) -> None:
        self._stop_event.set()
        self._health.set(HealthStatus.STOPPED, "Thread inference stopping")

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    @property
    def health(self) -> WorkerHealthState:
        return self._health.snapshot()

    
