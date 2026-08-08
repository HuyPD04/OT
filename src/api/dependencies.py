from __future__ import annotations

from fastapi import HTTPException, Request, status


def get_controller(request: Request):
    controller = getattr(request.app.state, "controller", None)
    if controller is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="pipeline is not ready",
        )
    return controller
