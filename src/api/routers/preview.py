from __future__ import annotations

import asyncio

import cv2
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ...utils.boxes import draw_plate, draw_track
from ..dependencies import get_controller


router = APIRouter(tags=["preview"])

_MJPEG_BOUNDARY = "frame"
_JPEG_QUALITY = 85


def _encode_frame(controller, image, annotated: bool) -> bytes | None:
    output = image.copy()
    if annotated:
        _, track_packet = controller.snapshot_tracks()
        if track_packet is not None:
            for track in track_packet.tracks:
                draw_track(output, track)

        _, plate_packet = controller.snapshot_plates()
        if plate_packet is not None:
            for plate in plate_packet.plates:
                draw_plate(output, plate)

    encoded, jpeg = cv2.imencode(
        ".jpg",
        output,
        [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
    )
    return jpeg.tobytes() if encoded else None


async def _mjpeg_stream(request: Request, controller, annotated: bool):
    version = 0
    while not await request.is_disconnected():
        version, packet = await asyncio.to_thread(
            controller.wait_next_frame,
            version,
            0.5,
        )
        if packet is None:
            continue

        jpeg = await asyncio.to_thread(
            _encode_frame,
            controller,
            packet.image,
            annotated,
        )
        if jpeg is None:
            continue

        headers = (
            f"--{_MJPEG_BOUNDARY}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpeg)}\r\n\r\n"
        ).encode("ascii")
        yield headers + jpeg + b"\r\n"


def _preview_response(request: Request, controller, annotated: bool) -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_stream(request, controller, annotated),
        media_type=f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY}",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/preview.mjpeg", include_in_schema=False)
@router.get("/preview/raw.mjpeg")
def raw_preview(request: Request, controller=Depends(get_controller)) -> StreamingResponse:
    return _preview_response(request, controller, annotated=False)


@router.get("/preview/annotated.mjpeg")
def annotated_preview(
        request: Request,
        controller=Depends(get_controller),
) -> StreamingResponse:
    return _preview_response(request, controller, annotated=True)
