from __future__ import annotations


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
