from __future__ import annotations

from pydantic import BaseModel


class FrameMetadata(BaseModel):
    camera_id: str
    frame_id: str
    stream_epoch: int
    received_ns: int
    pts_ns: int
    width: int
    height: int


class WorkerHealthResponse(BaseModel):
    name: str
    status: str
    message: str
    last_error: str | None
    updated_monotonic_ns: int


class LivenessResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    ready: bool
    workers: dict[str, WorkerHealthResponse]


class PacketStatus(BaseModel):
    version: int
    frame_id: str | None = None
    stream_epoch: int | None = None


class PipelineStatusResponse(BaseModel):
    workers: dict[str, WorkerHealthResponse]
    detections: PacketStatus
    tracks: PacketStatus
    plates: PacketStatus


class DetectionResponse(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    track_id: int | None


class DetectionPacketResponse(BaseModel):
    version: int
    frame: FrameMetadata
    inference_ms: float
    detections: list[DetectionResponse]


class TrackResponse(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    track_id: int
    confidence: float | None
    class_id: int | None


class TrackPacketResponse(BaseModel):
    version: int
    frame: FrameMetadata
    detection_frame_id: str | None
    tracking_ms: float
    tracks: list[TrackResponse]


class PlateResponse(BaseModel):
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int | None


class PlatePacketResponse(BaseModel):
    version: int
    frame: FrameMetadata
    detection_frame_id: str | None
    inference_ms: float
    plates: list[PlateResponse]
