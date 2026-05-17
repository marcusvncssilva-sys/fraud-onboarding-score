"""Configurações globais do pytest e fixtures compartilhadas."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("PSEUDONYM_SECRET_KEY", "test-secret-key-for-pytest-only")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
# Garante que o startup não tente carregar um modelo de disco nos testes unitários
os.environ.setdefault("MODEL_PATH", "models/champion.pkl")


@pytest.fixture
def sample_cpf_hash() -> str:
    return "a" * 64


@pytest.fixture
def sample_request_payload() -> dict:  # type: ignore[type-arg]
    return {
        "cpf_hash": "a" * 64,
        "device_id_hash": "dev_hash_test_001",
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
        "distance_from_cep_km": 5.0,
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
