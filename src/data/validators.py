"""Schemas Pydantic para validação de dados de entrada e saída."""

from typing import Optional
import uuid
from pydantic import BaseModel, Field, field_validator


class OnboardingScoreRequest(BaseModel):
    """Schema de requisição do endpoint de scoring.

    Todos os campos de PII já chegam pseudonimizados pelo cliente.
    """

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # ── Identificação (pseudonimizada — LGPD) ────────────────────────────────
    cpf_hash: str = Field(..., min_length=64, max_length=64)
    device_id_hash: str = Field(..., min_length=16, max_length=128)

    # ── Biometria ────────────────────────────────────────────────────────────
    bio_liveness_score: float = Field(..., ge=0.0, le=1.0)
    bio_face_match_score: float = Field(..., ge=0.0, le=1.0)
    bio_attempts: int = Field(..., ge=1, le=10)
    bio_liveness_passed: bool
    bio_face_match_passed: bool
    bio_failure_rate: float = Field(..., ge=0.0, le=1.0)

    # ── Dispositivo ──────────────────────────────────────────────────────────
    device_is_rooted: bool
    device_is_emulator: bool
    device_fraud_score: float = Field(..., ge=0.0, le=1.0)
    device_cpfs_30d: int = Field(..., ge=0)
    device_age_days: int = Field(..., ge=0)
    app_is_tampered: bool

    # ── Geolocalização ───────────────────────────────────────────────────────
    is_vpn: bool
    is_tor: bool
    is_proxy: bool
    is_foreign_ip: bool
    distance_from_cep_km: Optional[float] = Field(None, ge=0.0)
    anonymizer_score: int = Field(..., ge=0, le=3)

    # ── Sessão comportamental ────────────────────────────────────────────────
    session_completion_seconds: float = Field(..., ge=0.0)
    is_suspiciously_fast: bool
    is_night_session: bool
    copy_paste_count: int = Field(..., ge=0)
    typing_speed_cv: Optional[float] = Field(None, ge=0.0)

    # ── Velocidade histórica (feature store) ─────────────────────────────────
    cpf_onboardings_7d: int = Field(..., ge=0)
    cpf_onboardings_30d: int = Field(..., ge=0)
    device_cpfs_7d: int = Field(..., ge=0)
    ip_onboardings_24h: int = Field(..., ge=0)

    # ── DICT Bacen / Bureau ──────────────────────────────────────────────────
    has_pix_key: bool
    pix_key_age_days: Optional[int] = Field(None, ge=0)
    n_complaints_bacen: int = Field(..., ge=0)
    is_in_cadin: bool
    chargeback_ratio_90d: float = Field(..., ge=0.0)

    @field_validator("cpf_hash")
    @classmethod
    def validate_cpf_hash_format(cls, v: str) -> str:
        """Garante que o cpf_hash é um hex SHA-256 válido."""
        if not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError("cpf_hash deve ser um hash hexadecimal SHA-256")
        return v.lower()

    model_config = {"json_schema_extra": {
        "example": {
            "cpf_hash": "a" * 64,
            "device_id_hash": "dev_hash_example",
            "bio_liveness_score": 0.95,
            "bio_face_match_score": 0.92,
            "bio_attempts": 1,
            "bio_liveness_passed": True,
            "bio_face_match_passed": True,
            "bio_failure_rate": 0.0,
            "device_is_rooted": False,
            "device_is_emulator": False,
            "device_fraud_score": 0.05,
            "device_cpfs_30d": 1,
            "device_age_days": 365,
            "app_is_tampered": False,
            "is_vpn": False,
            "is_tor": False,
            "is_proxy": False,
            "is_foreign_ip": False,
            "distance_from_cep_km": 5.2,
            "anonymizer_score": 0,
            "session_completion_seconds": 180.0,
            "is_suspiciously_fast": False,
            "is_night_session": False,
            "copy_paste_count": 0,
            "typing_speed_cv": 0.3,
            "cpf_onboardings_7d": 0,
            "cpf_onboardings_30d": 0,
            "device_cpfs_7d": 1,
            "ip_onboardings_24h": 0,
            "has_pix_key": True,
            "pix_key_age_days": 300,
            "n_complaints_bacen": 0,
            "is_in_cadin": False,
            "chargeback_ratio_90d": 0.0,
        }
    }}


class ShapFeature(BaseModel):
    """Contribuição de uma feature individual para a decisão (SHAP)."""
    feature_name: str
    contribution: float
    direction: str  # "RISK" | "SAFE"
    feature_value: Optional[float] = None


class OnboardingScoreResponse(BaseModel):
    """Schema de resposta do scoring."""
    request_id: str
    score: int = Field(..., ge=0, le=1000, description="Score 0-1000. Maior = menor risco.")
    risk_band: str = Field(..., description="BAIXO | MEDIO_BAIXO | MEDIO_ALTO | ALTO | CRITICO")
    decision: str = Field(..., description="APPROVED | REVIEW | REJECTED")
    model_version: str
    latency_ms: float
    top_risk_factors: list[ShapFeature] = Field(
        default_factory=list,
        description="Top 3 fatores de risco (explicabilidade — Art. 20 LGPD)",
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_version: str
    environment: str
    redis_connected: bool
