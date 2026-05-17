"""Endpoint de scoring antifraude — hot path otimizado para p95 < 100ms.

Fluxo:
    1. Hard rules (early exit ~0.1ms)
    2. Redis cache lookup (~1ms, opcional)
    3. Feature extraction — numpy C-array (~0.05ms)
    4. model.predict_proba (~5ms com 50 árvores)
    5. SHAP top-3 cached (~0ms cache hit / ~8ms miss)
    6. Logging assíncrono (background task — fora do hot path)
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from src.data.validators import OnboardingScoreRequest, OnboardingScoreResponse, ShapFeature
from src.decision.engine import DecisionEngine, HardRules
from src.features.pipeline import extract_features
from src.models.score_transformer import ScoreTransformer
from src.utils.logger import AuditLogger

router = APIRouter(prefix="/v1/score", tags=["scoring"])

_decision_engine = DecisionEngine(threshold_approve=0.15, threshold_reject=0.60)
_audit_logger = AuditLogger()


@router.post("/onboarding", response_model=OnboardingScoreResponse)
async def score_onboarding(
    request: OnboardingScoreRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> OnboardingScoreResponse:
    """Score de risco para onboarding de cliente.

    Latência alvo: p95 < 100ms · Disponibilidade: 99.9%
    Score 0-1000 (maior = menor risco) + top-3 fatores SHAP (Art. 20 LGPD)
    """
    start = time.perf_counter()

    # ── 1. Hard rules — early exit antes do modelo ────────────────────────────
    rule = HardRules.check(request)
    if rule:
        latency_ms = (time.perf_counter() - start) * 1000
        background_tasks.add_task(
            _audit_logger.log_scoring_decision,
            request_id=request.request_id,
            cpf_hash=request.cpf_hash,
            score=0,
            risk_band="CRITICO",
            decision="REJECTED",
            model_version="hard-rule",
            latency_ms=latency_ms,
            top_risk_factors=[],
            hard_rule_triggered=rule,
        )
        return OnboardingScoreResponse(
            request_id=request.request_id,
            score=0,
            risk_band="CRITICO",
            decision="REJECTED",
            model_version="hard-rule",
            latency_ms=round(latency_ms, 2),
            top_risk_factors=[],
        )

    # ── 2. Resolver dependências da aplicação ─────────────────────────────────
    try:
        scorer = http_request.app.state.scorer
        store = http_request.app.state.feature_store
        shap_exp = http_request.app.state.shap_explainer
    except AttributeError:
        # Testes sem lifespan (TestClient direto sem startup)
        scorer = None
        store = None
        shap_exp = None

    # ── 3. Feature extraction — numpy C-array, zero cópia extra ──────────────
    # Tenta cache Redis primeiro
    features = None
    if store is not None and store.available:
        features = await store.get_features(request.cpf_hash)

    if features is None:
        features = extract_features(request)
        if store is not None and store.available:
            background_tasks.add_task(store.set_features, request.cpf_hash, features)

    # ── 4. Predição do modelo ─────────────────────────────────────────────────
    if scorer is not None and scorer.is_loaded:
        import numpy as np
        proba_fraud = float(scorer.predict_proba(features)[0])
        model_version = scorer.version
    else:
        proba_fraud = _heuristic_score(request)
        model_version = "heuristic-v0"

    # ── 5. Score + decisão ────────────────────────────────────────────────────
    import numpy as np

    score_val = int(ScoreTransformer.transform(np.array([proba_fraud]))[0])
    risk_band = ScoreTransformer.to_risk_band(score_val)
    decision = ScoreTransformer.proba_to_decision(
        proba_fraud,
        _decision_engine.threshold_approve,
        _decision_engine.threshold_reject,
    )

    # ── 6. SHAP top-3 (cache primeiro) ───────────────────────────────────────
    top_factors: list[ShapFeature] = []
    if shap_exp is not None and shap_exp.available:
        fkey = shap_exp.feature_key(features)
        cached_shap = await store.get_shap(fkey) if (store and store.available) else None

        if cached_shap is not None:
            raw_factors = cached_shap
        else:
            raw_factors = shap_exp.explain_top3(features)
            if store is not None and store.available and raw_factors:
                background_tasks.add_task(store.set_shap, fkey, raw_factors)

        top_factors = [ShapFeature(**f) for f in raw_factors]

    latency_ms = (time.perf_counter() - start) * 1000

    # ── 7. Auditoria assíncrona — totalmente fora do hot path ─────────────────
    background_tasks.add_task(
        _audit_logger.log_scoring_decision,
        request_id=request.request_id,
        cpf_hash=request.cpf_hash,
        score=score_val,
        risk_band=risk_band,
        decision=decision,
        model_version=model_version,
        latency_ms=latency_ms,
        top_risk_factors=[f.model_dump() for f in top_factors],
        hard_rule_triggered=None,
    )

    return OnboardingScoreResponse(
        request_id=request.request_id,
        score=score_val,
        risk_band=risk_band,
        decision=decision,
        model_version=model_version,
        latency_ms=round(latency_ms, 2),
        top_risk_factors=top_factors,
    )


def _heuristic_score(req: OnboardingScoreRequest) -> float:
    """Fallback heurístico quando o modelo ainda não foi treinado."""
    risk = 0.05
    if req.device_is_rooted:
        risk += 0.15
    if req.device_is_emulator:
        risk += 0.40
    if req.is_vpn:
        risk += 0.10
    if req.bio_liveness_score < 0.50:
        risk += 0.20
    if req.device_cpfs_30d > 3:
        risk += 0.15
    if req.cpf_onboardings_7d > 1:
        risk += 0.10
    if req.is_suspiciously_fast:
        risk += 0.10
    return min(1.0, risk)
