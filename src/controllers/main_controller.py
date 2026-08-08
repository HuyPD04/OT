from __future__ import annotations

from ..models.health import Health

class APIController:
    def __init__(
            self,
            thread_capture,
            thread_output,
            thread_infer,
            thread_tracking,
            detection_buffer,
            frame_buffer,
            tracker_buffer,
            thread_plate=None,
            plate_buffer=None,
    ) -> None:
        self._thread_capture = thread_capture
        self._thread_output = thread_output
        self._frame_buffer = frame_buffer
        self._thread_infer = thread_infer
        self._thread_tracking = thread_tracking
        self._thread_plate = thread_plate
        self._detection_buffer = detection_buffer  
        self._tracker_buffer = tracker_buffer
        self._plate_buffer = plate_buffer

    def start(self) -> None:
        self._thread_capture.start()
        self._thread_infer.start()
        self._thread_tracking.start()
        if self._thread_plate is not None:
            self._thread_plate.start()

        for thread in self._thread_output:
            thread.start()

    def stop(self) -> None:
        self._thread_capture.stop()
        self._thread_infer.stop()
        self._thread_tracking.stop()
        if self._thread_plate is not None:
            self._thread_plate.stop()

        self._frame_buffer.close()
        self._detection_buffer.close()
        self._tracker_buffer.close()
        if self._plate_buffer is not None:
            self._plate_buffer.close()

        for thread in self._thread_output:
            thread.stop()

        self._thread_capture.join(timeout=10)
        self._thread_infer.join(timeout=10)
        self._thread_tracking.join(timeout=10)
        if self._thread_plate is not None:
            self._thread_plate.join(timeout=10)
        for thread in self._thread_output:
            thread.join(timeout=10)

    def health_snapshot(self) -> dict[str, Health]:
        workers = {
            "capture": self._thread_capture,
            "inference": self._thread_infer,
            "tracking": self._thread_tracking,
        }
        if self._thread_plate is not None:
            workers["plate_recognition"] = self._thread_plate
        workers.update(
            {
                f"output_{index}": thread
                for index, thread in enumerate(self._thread_output)
            }
        )
        return {
            name: worker.health
            for name, worker in workers.items()
        }

    def snapshot_detections(self):
        return self._detection_buffer.snapshot()

    def snapshot_tracks(self):
        return self._tracker_buffer.snapshot()

    def snapshot_plates(self):
        if self._plate_buffer is None:
            return 0, None
        return self._plate_buffer.snapshot()

    def wait_next_frame(self, pre_version: int, timeout: float):
        return self._frame_buffer.wait_next(pre_version=pre_version, timeout=timeout)
