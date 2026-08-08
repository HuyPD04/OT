from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_controller
from ..schemas import PipelineStatusResponse
from ..serializers import serialize_health, serialize_packet_status


router = APIRouter(tags=["status"])


@router.get("/status", response_model=PipelineStatusResponse)
def pipeline_status(controller=Depends(get_controller)) -> PipelineStatusResponse:
    detection_version, detection_packet = controller.snapshot_detections()
    track_version, track_packet = controller.snapshot_tracks()
    plate_version, plate_packet = controller.snapshot_plates()
    return PipelineStatusResponse(
        workers={
            name: serialize_health(health)
            for name, health in controller.health_snapshot().items()
        },
        detections=serialize_packet_status(detection_version, detection_packet),
        tracks=serialize_packet_status(track_version, track_packet),
        plates=serialize_packet_status(plate_version, plate_packet),
    )
