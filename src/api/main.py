"""FastAPI application — Score Antifraude Onboarding.

Startup em src/api/lifespan.py: carrega modelo C-native + SHAP + Redis
antes de aceitar qualquer request. Zero I/O síncrono no hot path.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.lifespan import app_lifespan
from src.api.routers import health, score

app = FastAPI(
    title="Fraud Onboarding Score API",
    description=(
        "Sistema de score antifraude para onboarding de clientes. "
        "Score 0-1000 · p95 < 100ms · LGPD-compliant."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=app_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if os.getenv("ENVIRONMENT") == "development" else [],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(health.router)
app.include_router(score.router)
