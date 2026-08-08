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
from .controllers.threads.thread_inference import ThreadInference
from .controllers.threads.thread_plate_recognition import ThreadPlateRecognition
from .controllers.threads.thread_tracking import ThreadTracking
from .controllers.backend.onnx_runtime import OnnxBackend
from .controllers.buffer.detectionbuffer import DetectionBuffer
from .controllers.buffer.framebuffer import FrameBuffer
from .controllers.buffer.platebuffer import PlateBuffer
from .controllers.buffer.trackerbuffer import TrackerBuffer
from .controllers.main_controller import APIController
from .camera.gst_source import GstSource

logger = logging.getLogger(__name__)


def build_application(config, *, enable_display: bool = True):
    Gst.init(sys.argv)
    cv2.setNumThreads(1)

    frame_buffer = FrameBuffer()
    camera = lambda: GstSource(config.camera)
    thread_output = []

    thread_capture = ThreadCapture(
        camera=camera,
        output=frame_buffer,
        timeout_ms=config.camera.timeout_ms,
        reconnect_initital=config.camera.reconnect_initital,
        reconnect_max=config.camera.reconnect_max,
        reconnect_stable_seconds=config.camera.reconnect_stable_seconds,
        reconnect_stable_frames=config.camera.reconnect_stable_frames,
        minimum_capture_fps=config.camera.minimum_capture_fps,
        degraded_window_seconds=config.camera.degraded_window_seconds,
        startup_grace_seconds=config.camera.startup_grace_seconds
    )

    detection_buffer = DetectionBuffer()
    tracker_buffer = TrackerBuffer()
    plate_buffer = PlateBuffer()
    backend = OnnxBackend(
        model=config.detection.model_path,
        input_size=config.detection.input_size,
        confidence_threshold=config.detection.conf,
        iou_threshold=getattr(config.detection, "iou_threshold", 0.45),
        class_ids=getattr(config.detection, "class_ids", None),
    )
    thread_inference = ThreadInference(
        frame_buffer=frame_buffer,
        detection_buffer=detection_buffer,
        backend=backend,
    )
    thread_tracking = ThreadTracking(
        detection_buffer=detection_buffer,
        tracker_buffer=tracker_buffer,
        iou_threshold=getattr(config.detection, "tracking_iou_threshold", 0.3),
        center_threshold=getattr(config.detection, "tracking_center_threshold", 1.5),
        center_weight=getattr(config.detection, "tracking_center_weight", 0.35),
    )
    plate_backend = OnnxBackend(
        model=config.plate.model_path,
        input_size=config.plate.input_size,
        confidence_threshold=config.plate.conf,
        iou_threshold=getattr(config.plate, "iou_threshold", 0.45),
        class_ids=getattr(config.plate, "class_ids", None),
    )
    thread_plate = ThreadPlateRecognition(
        tracker_buffer=tracker_buffer,
        plate_buffer=plate_buffer,
        backend=plate_backend,
        vehicle_class_ids=getattr(config.plate, "vehicle_class_ids", None),
        roi_padding=getattr(config.plate, "roi_padding", 0.05),
        min_roi_size=getattr(config.plate, "min_roi_size", 32),
        min_inference_interval_seconds=getattr(
            config.plate,
            "min_inference_interval_seconds",
            0.5,
        ),
    )
    if enable_display:
        thread_output.append(
            ThreadDisplay(
                frame_buffer=frame_buffer,
                detection_buffer=detection_buffer,
                tracker_buffer=tracker_buffer,
                plate_buffer=plate_buffer,
            )
        )

    return APIController(
        thread_capture=thread_capture,
        thread_infer=thread_inference,
        thread_tracking=thread_tracking,
        detection_buffer=detection_buffer,
        frame_buffer=frame_buffer,
        tracker_buffer=tracker_buffer,
        thread_plate=thread_plate,
        plate_buffer=plate_buffer,
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
