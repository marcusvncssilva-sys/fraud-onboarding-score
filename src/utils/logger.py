"""Logger estruturado para auditoria regulatória (Bacen Res. 85/2021)."""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional


def _setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class AuditLogger:
    """Logger de auditoria imutável (append-only) para decisões de scoring.

    Requisitos atendidos:
    - Retenção mínima: 5 anos (Res. BCB 85/2021)
    - Rastreabilidade completa de cada decisão
    - Sem dados PII nos logs (apenas hashes)
    - Exportável para auditoria regulatória
    """

    def __init__(self, service_name: str = "fraud-onboarding-score") -> None:
        self.service = service_name
        self._logger = _setup_logger(service_name)

    def log_scoring_decision(
        self,
        *,
        request_id: str,
        cpf_hash: str,
        score: int,
        risk_band: str,
        decision: str,
        model_version: str,
        latency_ms: float,
        top_risk_factors: list[dict[str, Any]],
        hard_rule_triggered: Optional[str] = None,
    ) -> None:
        """Registra cada decisão de scoring para auditoria."""
        event = {
            "event_type": "SCORING_DECISION",
            "event_id": str(uuid.uuid4()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "service": self.service,
            "request_id": request_id,
            "cpf_hash": cpf_hash,
            "score": score,
            "risk_band": risk_band,
            "decision": decision,
            "model_version": model_version,
            "latency_ms": round(latency_ms, 2),
            "top_risk_factors": top_risk_factors,
            "hard_rule_triggered": hard_rule_triggered,
        }
        self._logger.info(json.dumps(event, ensure_ascii=False))

    def log_provider_call(
        self,
        *,
        request_id: str,
        provider: str,
        success: bool,
        latency_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Registra chamadas a provedores externos."""
        event = {
            "event_type": "PROVIDER_CALL",
            "event_id": str(uuid.uuid4()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "service": self.service,
            "request_id": request_id,
            "provider": provider,
            "success": success,
            "latency_ms": round(latency_ms, 2),
            "error": error,
        }
        self._logger.info(json.dumps(event, ensure_ascii=False))

    def log_model_loaded(self, *, model_version: str, model_path: str) -> None:
        """Registra carregamento de modelo (rastreabilidade de versão)."""
        event = {
            "event_type": "MODEL_LOADED",
            "event_id": str(uuid.uuid4()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "service": self.service,
            "model_version": model_version,
            "model_path": model_path,
        }
        self._logger.info(json.dumps(event, ensure_ascii=False))
