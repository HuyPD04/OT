from __future__ import annotations

import time
import numpy as np
import logging
import cv2

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst # type: ignore

from ..models.framepacket import FramePacket
from ..models.camera import CameraConfig, Camera

logger = logging.getLogger(__name__)

class GstSource(Camera):
    def __init__(self, cfg: CameraConfig):
        self._cfg = cfg
        self._pipeline: Gst.Pipeline | None = None
        self._appsink: Gst.Element | None = None
        self._bus: Gst.Bus | None = None
        self._frame_id: int = 0
        self._last_frame_at_ns: int | None = None
        self._opened: bool = False

    def _build_pipeline_description(self) -> str:
        url = self._cfg.url
        codec = self._cfg.codec.lower()

        if codec == "h264":
            depayloader = "rtph264depay"
            parser = "h264parse"
            decoder = "avdec_h264"
            codec_chain = f"""
                {depayloader}
                !
                {parser}
                !
                {decoder}
            """
        elif codec == "h265":
            depayloader = "rtph265depay"
            parser = "h265parse"
            decoder = "avdec_h265"
            codec_chain = f"""
                {depayloader}
                !
                {parser}
                !
                {decoder}
            """
        else:
            raise RuntimeError(f"Unsupported RTSP codec: {codec}")

        return f"""
            rtspsrc
                name=rtsp_source
                location={url}
                protocols={self._cfg.transport}
                latency={self._cfg.latency_ms}
                drop-on-latency=false
            !
            {codec_chain}
            !
            queue max-size-buffers=2 leaky=downstream
            !
            videoconvert
            !
            appsink
                name=frame_sink
                emit-signals=false
                sync=false
                drop=true
                max-buffers=1
                wait-on-eos=false
        """

    def open(self) -> None:
        if self._opened:
            return
        description = self._build_pipeline_description()
        try:
            pipeline = Gst.parse_launch(description)
        except Exception as e:
            raise RuntimeError(f"Cannot create Gst pipeline: {e}")

        if not isinstance(pipeline, Gst.Pipeline):
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("GStreamer description did not create a Gst.Pipeline")
        pipeline.set_name(f"pipeline-{self._cfg.camera_id}")
        
        appsink = pipeline.get_by_name("frame_sink")
        if appsink is None:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("Cannot find appsink 'frame_sink'") 

        bus = pipeline.get_bus()
        if bus is None:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("Cannot get GstBus")
        
        state_result = pipeline.set_state(Gst.State.PLAYING)
        if state_result == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("Cannot change pipeline to PLAYING")
        
        self._pipeline = pipeline
        self._appsink = appsink
        self._bus = bus
        self._last_frame_at_ns = time.monotonic_ns()
        self._opened = True

    def _check_bus(self) -> None:
        if self._bus is None:
            return

        interesting_messages = (
            Gst.MessageType.ERROR
            | Gst.MessageType.EOS
            | Gst.MessageType.WARNING
        )

        while True:
            message = self._bus.pop_filtered(interesting_messages)
            if message is None:
                break

            source_name = message.src.get_name() if message.src else "unknown"

            if message.type == Gst.MessageType.WARNING:
                warning, debug = message.parse_warning()
                logger.warning(
                    "GStreamer warning from {}: {} debug={}",
                    source_name,
                    warning,
                    debug,
                )

            elif message.type == Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                raise RuntimeError(f"GStreamer error: {error}, debug={debug}")

            elif message.type == Gst.MessageType.EOS:
                raise RuntimeError("EOS")

    @staticmethod
    def _sample_to_numpy(sample: Gst.Sample) -> tuple[np.ndarray, int | None]:
        caps = sample.get_caps()
        buffer = sample.get_buffer()

        if caps is None or buffer is None:
            raise RuntimeError("GstSample has not caps or buffer")
        structure = caps.get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        fmt = structure.get_value("format")

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            raise RuntimeError("Cannot map GstBuffer")

        try:
            raw = np.frombuffer(map_info.data, dtype=np.uint8)
            if fmt == "BGR":
                required_row_bytes = width * 3
                row_stride = map_info.size // height

                if row_stride < required_row_bytes:
                    raise RuntimeError(
                        f"Invalid BGR buffer. "
                        f"Expected at least {required_row_bytes} bytes/row "
                        f"but got {row_stride}"
                    )

                rows = raw.reshape(height, row_stride)

                image = rows[:, :required_row_bytes].reshape(
                    height,
                    width,
                    3,
                ).copy()

            elif fmt == "NV12":
                expected_size = width * height * 3 // 2

                if map_info.size < expected_size:
                    raise RuntimeError(
                        f"Invalid NV12 buffer. "
                        f"Expected at least {expected_size} bytes "
                        f"but got {map_info.size}"
                    )

                nv12 = raw[:expected_size].reshape(
                    height * 3 // 2,
                    width,
                )

                image = cv2.cvtColor(
                    nv12,
                    cv2.COLOR_YUV2BGR_NV12,
                )
            elif fmt == "I420":
                expected_size = width * height * 3 // 2
                if map_info.size < expected_size:
                    raise RuntimeError("Invalid I420 buffer")
                i420 = raw[:expected_size].reshape(
                    height * 3 // 2,
                    width,
                )
                image = cv2.cvtColor(
                    i420,
                    cv2.COLOR_YUV2BGR_I420,
                )
            else:
                raise RuntimeError(f"Unsupported pixel format: {fmt}")

            pts_ns = (
                None
                if buffer.pts == Gst.CLOCK_TIME_NONE
                else int(buffer.pts)
            )

            return image, pts_ns

        finally:
            buffer.unmap(map_info)

    def read(self, timeout_ms: int) -> FramePacket | None:
        if not self._opened or self._appsink is None:
            raise RuntimeError("Camera source is not turned on") 
        self._check_bus()
        sample = self._appsink.emit("try-pull-sample", timeout_ms*Gst.MSECOND)

        now_ns = time.monotonic_ns()
        if sample is None:
            self._check_bus()
            assert self._last_frame_at_ns is not None

            stalled_seconds = (now_ns - self._last_frame_at_ns) / 1_000_000_000
            if stalled_seconds >= self._cfg.stall_timeout_seconds:
                raise RuntimeError(f"RTSP cannot create frame in {stalled_seconds:.2f}")
            return None
        image, pts_ns = self._sample_to_numpy(sample)
        packet = FramePacket(
            camera_id=self._cfg.camera_id,
            frame_id=self._frame_id,
            received_ns=now_ns,
            pts_ns=pts_ns,
            width=image.shape[1],
            height=image.shape[0],
            image=image,
        )
        self._frame_id += 1
        self._last_frame_at_ns = now_ns
        return packet

    def close(self) -> None:
        pipeline = self._pipeline
        self._opened = False
        self._appsink = None
        self._bus = None
        self._pipeline = None
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
            pipeline.get_state(2*Gst.SECOND)
    def is_open(self) -> bool:
        return self._opened