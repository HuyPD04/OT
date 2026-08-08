from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import get_controller
from ..schemas import (
    DetectionPacketResponse,
    PlatePacketResponse,
    TrackPacketResponse,
)
from ..serializers import serialize_detections, serialize_plates, serialize_tracks


router = APIRouter(tags=["tracking"])


@router.get("/detections/latest", response_model=DetectionPacketResponse)
def latest_detections(controller=Depends(get_controller)) -> DetectionPacketResponse:
    version, packet = controller.snapshot_detections()
    if packet is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no detection is available yet",
        )
    return serialize_detections(version, packet)


@router.get("/tracks/latest", response_model=TrackPacketResponse)
def latest_tracks(controller=Depends(get_controller)) -> TrackPacketResponse:
    version, packet = controller.snapshot_tracks()
    if packet is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no tracked frame is available yet",
        )
    return serialize_tracks(version, packet)


@router.get("/plates/latest", response_model=PlatePacketResponse)
def latest_plates(controller=Depends(get_controller)) -> PlatePacketResponse:
    version, packet = controller.snapshot_plates()
    if packet is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no plate result is available yet",
        )
    return serialize_plates(version, packet)
