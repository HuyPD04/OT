from __future__ import annotations


class APIController:
    def __init__(
            self,
            thread_capture,
            thread_output,
            latest_frame_store
    ) -> None:
        self._thread_capture = thread_capture
        self._thread_output = thread_output
        self._latest_frame_store = latest_frame_store

    def start(self) -> None:
        self._thread_capture.start()

        for thread in self._thread_output:
            thread.start()

    def stop(self) -> None:
        self._thread_capture.stop()
        self._thread_capture.join(timeout=10)
        self._latest_frame_store.close()
        for thread in self._thread_output:
            thread.stop()
        for thread in self._thread_output:
            thread.join(timeout=10)
