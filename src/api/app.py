from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routers import health, preview, status, tracking

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@asynccontextmanager
async def pipeline_lifespan(app: FastAPI):
    from ..app import build_application
    from ..config.loader import Config

    controller = build_application(
        Config(),
        enable_display=_env_flag("PREVIEW_WINDOW"),
    )
    app.state.controller = controller
    controller.start()
    try:
        yield
    finally:
        controller.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Object Tracking API",
        version="0.1.0",
        lifespan=pipeline_lifespan,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.include_router(health.router, prefix="/health")
    app.include_router(status.router, prefix="/api/v1")
    app.include_router(tracking.router, prefix="/api/v1")
    app.include_router(preview.router, prefix="/api/v1")
    return app
