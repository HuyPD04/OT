import logging
from ..config.loader import Config
from ..camera.gst_source import GstSource
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst # type: ignore
import cv2
import time

logger = logging.getLogger(__name__)
cfg = Config()

def main():
    Gst.init(None)

    source = GstSource(cfg.camera)

    source.open()

    frame_count = 0
    fps_timer = time.perf_counter()

    try:

        while True:

            packet = source.read(timeout_ms=100)

            if packet is None:
                continue

            frame_count += 1

            cv2.imshow("RTSP", packet.image)

            if packet.pts_ns is not None:

                latency_ms = (
                    packet.received_ns - packet.pts_ns
                ) / 1_000_000

                logger.info(
                    "frame=%d pts=%.3fs latency=%.2fms shape=%s",
                    packet.frame_id,
                    packet.pts_ns / Gst.SECOND,
                    latency_ms,
                    packet.image.shape,
                )

            if time.perf_counter() - fps_timer >= 1:

                fps = frame_count / (time.perf_counter() - fps_timer)

                logger.info("FPS: %.2f", fps)

                fps_timer = time.perf_counter()
                frame_count = 0

            key = cv2.waitKey(1)

            if key == ord("q"):
                break

    finally:

        source.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
