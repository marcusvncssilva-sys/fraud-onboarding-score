"""Health check endpoints."""

import os

from fastapi import APIRouter, Request

from src.data.validators import HealthResponse

router = APIRouter(tags=["infra"])


@router.get("/health", response_model=HealthResponse)
async def health_check(http_request: Request) -> HealthResponse:
    """Health check para load balancer e monitoramento."""
    try:
        scorer = http_request.app.state.scorer
        store = http_request.app.state.feature_store
        model_version = scorer.version
        redis_ok = store.available
    except AttributeError:
        model_version = os.getenv("MODEL_VERSION", "dev")
        redis_ok = False

    return HealthResponse(
        status="healthy",
        model_version=model_version,
        environment=os.getenv("ENVIRONMENT", "development"),
        redis_connected=redis_ok,
    )


@router.get("/")
async def root() -> dict:  # type: ignore[type-arg]
    return {"service": "fraud-onboarding-score", "docs": "/docs"}
