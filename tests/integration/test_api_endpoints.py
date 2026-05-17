"""Testes de integração para os endpoints da API."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

VALID_REQUEST = {
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


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_version" in data
        assert data["status"] == "healthy"


class TestScoringEndpoint:
    def test_valid_request_returns_200(self):
        response = client.post("/v1/score/onboarding", json=VALID_REQUEST)
        assert response.status_code == 200

    def test_response_contains_score(self):
        response = client.post("/v1/score/onboarding", json=VALID_REQUEST)
        data = response.json()
        assert "score" in data
        assert 0 <= data["score"] <= 1000

    def test_response_contains_decision(self):
        response = client.post("/v1/score/onboarding", json=VALID_REQUEST)
        data = response.json()
        assert data["decision"] in {"APPROVED", "REVIEW", "REJECTED"}

    def test_response_contains_risk_band(self):
        response = client.post("/v1/score/onboarding", json=VALID_REQUEST)
        data = response.json()
        valid_bands = {"BAIXO", "MEDIO_BAIXO", "MEDIO_ALTO", "ALTO", "CRITICO"}
        assert data["risk_band"] in valid_bands

    def test_emulator_always_rejected(self):
        payload = {**VALID_REQUEST, "device_is_emulator": True}
        response = client.post("/v1/score/onboarding", json=payload)
        data = response.json()
        assert data["decision"] == "REJECTED"
        assert data["score"] == 0

    def test_tor_always_rejected(self):
        payload = {**VALID_REQUEST, "is_tor": True}
        response = client.post("/v1/score/onboarding", json=payload)
        data = response.json()
        assert data["decision"] == "REJECTED"

    def test_invalid_cpf_hash_returns_422(self):
        payload = {**VALID_REQUEST, "cpf_hash": "short"}
        response = client.post("/v1/score/onboarding", json=payload)
        assert response.status_code == 422

    def test_liveness_out_of_range_returns_422(self):
        payload = {**VALID_REQUEST, "bio_liveness_score": 1.5}
        response = client.post("/v1/score/onboarding", json=payload)
        assert response.status_code == 422

    def test_response_has_latency_field(self):
        response = client.post("/v1/score/onboarding", json=VALID_REQUEST)
        data = response.json()
        assert "latency_ms" in data
        assert data["latency_ms"] > 0
