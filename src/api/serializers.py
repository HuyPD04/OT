from __future__ import annotations

from .schemas import (
    DetectionPacketResponse,
    DetectionResponse,
    FrameMetadata,
    PacketStatus,
    PlatePacketResponse,
    PlateResponse,
    TrackPacketResponse,
    TrackResponse,
    WorkerHealthResponse,
)


def serialize_health(health) -> WorkerHealthResponse:
    return WorkerHealthResponse(
        name=health.name,
        status=health.status.value,
        message=health.message,
        last_error=health.last_error,
        updated_monotonic_ns=health.update_ns,
    )


def serialize_frame(frame) -> FrameMetadata:
    return FrameMetadata(
        camera_id=frame.camera_id,
        frame_id=str(frame.frame_id),
        stream_epoch=frame.stream_epoch,
        received_ns=frame.received_ns,
        pts_ns=frame.pts_ns,
        width=frame.width,
        height=frame.height,
    )


def serialize_packet_status(version: int, packet) -> PacketStatus:
    if packet is None:
        return PacketStatus(version=version)
    return PacketStatus(
        version=version,
        frame_id=str(packet.frame.frame_id),
        stream_epoch=packet.frame.stream_epoch,
    )


def serialize_detections(version: int, packet) -> DetectionPacketResponse:
    return DetectionPacketResponse(
        version=version,
        frame=serialize_frame(packet.frame),
        inference_ms=packet.inference_ms * 1_000.0,
        detections=[
            DetectionResponse(
                x1=detection.x1,
                y1=detection.y1,
                x2=detection.x2,
                y2=detection.y2,
                confidence=detection.conf,
                class_id=detection.class_id,
                track_id=detection.track_id,
            )
            for detection in packet.detections
        ],
    )


def serialize_tracks(version: int, packet) -> TrackPacketResponse:
    return TrackPacketResponse(
        version=version,
        frame=serialize_frame(packet.frame),
        detection_frame_id=_serialize_frame_id(packet.detection_frame_id),
        tracking_ms=packet.tracking_ms * 1_000.0,
        tracks=[
            TrackResponse(
                x1=track.x1,
                y1=track.y1,
                x2=track.x2,
                y2=track.y2,
                track_id=track.track_id,
                confidence=track.conf,
                class_id=track.class_id,
            )
            for track in packet.tracks
        ],
    )


def serialize_plates(version: int, packet) -> PlatePacketResponse:
    return PlatePacketResponse(
        version=version,
        frame=serialize_frame(packet.frame),
        detection_frame_id=_serialize_frame_id(packet.detection_frame_id),
        inference_ms=packet.inference_ms * 1_000.0,
        plates=[
            PlateResponse(
                track_id=plate.track_id,
                x1=plate.x1,
                y1=plate.y1,
                x2=plate.x2,
                y2=plate.y2,
                confidence=plate.conf,
                class_id=plate.class_id,
            )
            for plate in packet.plates
        ],
    )


def _serialize_frame_id(frame_id) -> str | None:
    return str(frame_id) if frame_id is not None else None
