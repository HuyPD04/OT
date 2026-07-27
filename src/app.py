from __future__ import annotations

import os
import sys
import cv2
import signal
import threading
import logging
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst # type: ignore

from .controllers.threads.thread_capture import ThreadCapture
from .controllers.threads.thread_display import ThreadDisplay
from .controllers.store.latest_frame import LatestFrameStore
from .controllers.main_controller import APIController
from .camera.gst_source import GstSource

logger = logging.getLogger(__name__)

def build_application(config):
    Gst.init(sys.argv)
    cv2.setNumThreads(1)

    frame_store = LatestFrameStore()
    camera = lambda: GstSource(config.camera)
    thread_output = []

    thread_capture = ThreadCapture(
        camera=camera,
        output=frame_store,
        timeout_ms=config.camera.timeout_ms,
        reconnect_initital=config.camera.reconnect_initital,
        reconnect_max=config.camera.reconnect_max,
        reconnect_stable_seconds=config.camera.reconnect_stable_seconds,
        reconnect_stable_frames=config.camera.reconnect_stable_frames,
        minimum_capture_fps=config.camera.minimum_capture_fps,
        degraded_window_seconds=config.camera.degraded_window_seconds,
        startup_grace_seconds=config.camera.startup_grace_seconds
    )

    thread_display = ThreadDisplay(frame_store=frame_store)
    thread_output = [thread_display]

    return APIController(
        thread_capture=thread_capture,
        latest_frame_store=frame_store,
        thread_output=thread_output
    )

def run_application(controller) -> None:
    shutdown_event = threading.Event()

    def request_shutdown(signum, frame) -> None:
        shutdown_event.set()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    controller.start()
    try:
        while not shutdown_event.wait(timeout=0.2):
            pass
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        shutdown_event.set()
    finally:
        logger.info("Stopping application")
        controller.stop()
        logger.info("Application stopped")
