from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from ..dependencies import get_controller
from ..schemas import LivenessResponse, ReadinessResponse
from ..serializers import serialize_health


router = APIRouter(tags=["health"])


@router.get("/live", response_model=LivenessResponse)
def liveness() -> LivenessResponse:
    return LivenessResponse(status="ok")


@router.get("/ready", response_model=ReadinessResponse)
def readiness(
        response: Response,
        controller=Depends(get_controller),
) -> ReadinessResponse:
    workers = {
        name: serialize_health(health)
        for name, health in controller.health_snapshot().items()
    }
    required_workers = {"capture", "inference", "tracking"}
    if "plate_recognition" in workers:
        required_workers.add("plate_recognition")
    is_ready = all(
        workers[name].status == "running"
        for name in required_workers
    )
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(ready=is_ready, workers=workers)
