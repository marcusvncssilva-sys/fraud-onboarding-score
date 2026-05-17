"""Lifespan da aplicação — carregamento de modelo e Redis antes de aceitar tráfego.

Separado de main.py para facilitar testes e reuso em workers adicionais.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

_log = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: carrega modelo (C-native) + SHAP + Redis antes do primeiro request.

    Todos os objetos ficam em app.state — imutáveis e thread-safe para leitura.
    Zero I/O síncrono no hot path após o startup.
    """
    # ── 1. Modelo LightGBM + persistência ────────────────────────────────────
    from src.models.champion.lgbm_model import LightGBMScorer
    from src.models.persistence import ModelPersistence

    scorer = LightGBMScorer.get_instance()
    model_path_str = os.getenv("MODEL_PATH", "models/champion.pkl")

    loaded = False
    try:
        payload = ModelPersistence.load(model_path_str)
        scorer._model = payload["model"]
        scorer._version = payload.get("version", "unknown")
        scorer._loaded = True
        # Warm-up: força JIT-compile das árvores LightGBM em C-nativo
        ModelPersistence.warm_up(scorer._model)
        loaded = True
        _log.info("Modelo carregado e aquecido: v%s", scorer.version)
    except FileNotFoundError:
        _log.warning("Modelo não encontrado em %s — modo heurístico ativo", model_path_str)
    except ValueError as exc:
        _log.error("Incompatibilidade de modelo: %s — modo heurístico ativo", exc)
    except Exception as exc:
        _log.error("Falha no carregamento do modelo: %s", exc)

    app.state.scorer = scorer

    # ── 2. SHAP (opcional — não bloqueia startup) ─────────────────────────────
    from src.models.champion.shap_explainer import FastShapExplainer

    shap_exp = FastShapExplainer()
    if loaded:
        booster = scorer.get_raw_booster()
        shap_exp.init(booster)
    app.state.shap_explainer = shap_exp

    # ── 3. Redis feature store ────────────────────────────────────────────────
    from src.features.store import FeatureStore

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    store = FeatureStore(redis_url)
    await store.connect()
    app.state.feature_store = store

    _log.info(
        "Startup OK — modelo: %s | SHAP: %s | Redis: %s",
        scorer.version,
        "ok" if shap_exp.available else "off",
        "ok" if store.available else "off",
    )

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    await store.close()
    _log.info("API encerrada.")
