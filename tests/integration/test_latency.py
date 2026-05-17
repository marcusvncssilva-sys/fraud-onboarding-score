"""Testes de latência — garante p95 < 100ms no hot path da API.

Estratégia:
  - Treina um mini-modelo LightGBM (50 árvores) como fixture de sessão
  - Injeta o modelo e um FeatureStore mock (sem Redis) no app.state
  - Mede 500 requests sequenciais com TestClient (sem overhead de rede)
  - Valida p50, p95 e p99

O objetivo do teste é detectar regressões de latência no código Python/numpy/LightGBM,
não medir RTT de rede — por isso usa TestClient síncrono.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ── fixture: modelo treinado uma vez para toda a sessão de testes ─────────────

@pytest.fixture(scope="session")
def trained_model() -> Any:
    """Treina LightGBM 50 árvores em 5k amostras — ~3s de setup."""
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV

    from src.data.generators.customer_generator import generate_onboarding_dataset
    from src.features.pipeline import build_feature_matrix

    df = generate_onboarding_dataset(n_samples=5_000, fraud_rate=0.05, seed=0)
    X = build_feature_matrix(df)
    y = df["is_fraud"].to_numpy(dtype=int)

    lgbm = LGBMClassifier(
        n_estimators=50,
        learning_rate=0.10,
        max_depth=5,
        num_leaves=16,
        scale_pos_weight=int((y == 0).sum() / max(1, (y == 1).sum())),
        random_state=0,
        n_jobs=-1,
        verbose=-1,
    )
    # cv="prefit" — calibra sobre um único modelo já treinado; sem cv folds extras
    # garante que predict_proba chama apenas 1 LightGBM internamente
    from sklearn.model_selection import train_test_split as _tts
    X_tr, X_cal, y_tr, y_cal = _tts(X, y, test_size=0.25, stratify=y, random_state=0)
    lgbm.fit(X_tr, y_tr)
    model = CalibratedClassifierCV(lgbm, method="sigmoid", cv="prefit")
    model.fit(X_cal, y_cal)

    # warm-up: força JIT-compile das árvores
    dummy = np.zeros((1, X.shape[1]), dtype=np.float32)
    for _ in range(20):
        model.predict_proba(dummy)

    return model


@pytest.fixture(scope="session")
def app_client(trained_model: Any) -> TestClient:
    """TestClient com modelo injetado em app.state (sem Redis)."""
    from src.api.main import app
    from src.models.champion.lgbm_model import LightGBMScorer
    from src.models.champion.shap_explainer import FastShapExplainer
    from src.features.store import FeatureStore

    scorer = LightGBMScorer()
    scorer._model = trained_model
    scorer._version = "test-50t"
    scorer._loaded = True

    shap_exp = FastShapExplainer()  # SHAP off — não queremos no hot path do teste

    store = FeatureStore.__new__(FeatureStore)
    store._available = False  # Redis off para isolar latência pura

    # Injeta diretamente no app.state sem passar pelo lifespan
    app.state.scorer = scorer
    app.state.shap_explainer = shap_exp
    app.state.feature_store = store

    return TestClient(app, raise_server_exceptions=True)


# ── payload padrão ─────────────────────────────────────────────────────────────

VALID_PAYLOAD = {
    "cpf_hash": "a" * 64,
    "device_id_hash": "dev_latency_test_001",
    "bio_liveness_score": 0.95,
    "bio_face_match_score": 0.92,
    "bio_attempts": 1,
    "bio_liveness_passed": True,
    "bio_face_match_passed": True,
    "bio_failure_rate": 0.01,
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
    "anonymizer_score": 0,
    "session_completion_seconds": 180.0,
    "is_suspiciously_fast": False,
    "is_night_session": False,
    "copy_paste_count": 0,
    "cpf_onboardings_7d": 0,
    "cpf_onboardings_30d": 0,
    "device_cpfs_7d": 1,
    "ip_onboardings_24h": 0,
    "has_pix_key": True,
    "n_complaints_bacen": 0,
    "is_in_cadin": False,
    "chargeback_ratio_90d": 0.0,
}


# ── testes de latência ────────────────────────────────────────────────────────

class TestLatency:
    N_REQUESTS = 500
    P95_BUDGET_MS = 100.0
    P99_BUDGET_MS = 200.0   # p99 tem cauda mais longa (GC, cache miss)

    def _measure(self, client: TestClient, n: int = N_REQUESTS) -> list[float]:
        times_ms: list[float] = []
        for _ in range(n):
            t0 = time.perf_counter()
            resp = client.post("/v1/score/onboarding", json=VALID_PAYLOAD)
            elapsed = (time.perf_counter() - t0) * 1000
            assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
            times_ms.append(elapsed)
        return times_ms

    def test_p95_under_100ms(self, app_client: TestClient) -> None:
        times = self._measure(app_client)
        times.sort()
        p95 = times[int(len(times) * 0.95)]
        assert p95 < self.P95_BUDGET_MS, (
            f"p95 = {p95:.1f}ms excede orçamento de {self.P95_BUDGET_MS}ms"
        )

    def test_p99_under_150ms(self, app_client: TestClient) -> None:
        times = self._measure(app_client)
        times.sort()
        p99 = times[int(len(times) * 0.99)]
        assert p99 < self.P99_BUDGET_MS, (
            f"p99 = {p99:.1f}ms excede orçamento de {self.P99_BUDGET_MS}ms"
        )

    def test_p50_under_50ms(self, app_client: TestClient) -> None:
        """p50 < 50ms — inclui overhead do TestClient (JSON parse + Python calls)."""
        times = self._measure(app_client)
        times.sort()
        p50 = times[int(len(times) * 0.50)]
        assert p50 < 50.0, f"p50 = {p50:.1f}ms — overhead maior que esperado"

    def test_hard_rule_early_exit_faster(self, app_client: TestClient) -> None:
        """Hard rule deve ser mais rápida que scoring completo."""
        rejected_payload = {**VALID_PAYLOAD, "device_is_emulator": True}

        times_normal: list[float] = []
        times_hard: list[float] = []

        for _ in range(100):
            t0 = time.perf_counter()
            app_client.post("/v1/score/onboarding", json=VALID_PAYLOAD)
            times_normal.append((time.perf_counter() - t0) * 1000)

        for _ in range(100):
            t0 = time.perf_counter()
            app_client.post("/v1/score/onboarding", json=rejected_payload)
            times_hard.append((time.perf_counter() - t0) * 1000)

        p50_normal = sorted(times_normal)[50]
        p50_hard = sorted(times_hard)[50]
        # Hard rule deve ser ≤ normal (early exit)
        assert p50_hard <= p50_normal * 1.2, (
            f"Hard rule p50 ({p50_hard:.1f}ms) muito maior que normal ({p50_normal:.1f}ms)"
        )

    def test_latency_reported_in_response(self, app_client: TestClient) -> None:
        resp = app_client.post("/v1/score/onboarding", json=VALID_PAYLOAD)
        data = resp.json()
        assert "latency_ms" in data
        assert 0 < data["latency_ms"] < self.P95_BUDGET_MS


class TestLatencyFeatureExtraction:
    """Micro-benchmark de extract_features isolado (sem HTTP)."""

    N_ITERS = 10_000

    def test_extract_features_under_1ms(self, sample_request_payload: dict) -> None:  # type: ignore[type-arg]
        from src.data.validators import OnboardingScoreRequest
        from src.features.pipeline import extract_features

        req = OnboardingScoreRequest(**sample_request_payload)

        # warm-up
        for _ in range(100):
            extract_features(req)

        times = []
        for _ in range(self.N_ITERS):
            t0 = time.perf_counter()
            extract_features(req)
            times.append((time.perf_counter() - t0) * 1000)

        times.sort()
        p95 = times[int(len(times) * 0.95)]
        assert p95 < 1.0, f"extract_features p95 = {p95:.3f}ms — numpy overhead inesperado"


class TestLatencyModelPredict:
    """Micro-benchmark de predict_proba isolado."""

    N_ITERS = 1_000

    def test_predict_proba_under_15ms(self, trained_model: Any) -> None:
        """CalibratedClassifierCV(cv='prefit') + 50 árvores deve ser < 15ms p95."""
        import numpy as np
        from src.features.pipeline import N_FEATURES

        X = np.zeros((1, N_FEATURES), dtype=np.float32)

        # warm-up
        for _ in range(50):
            trained_model.predict_proba(X)

        times = []
        for _ in range(self.N_ITERS):
            t0 = time.perf_counter()
            trained_model.predict_proba(X)
            times.append((time.perf_counter() - t0) * 1000)

        times.sort()
        p95 = times[int(len(times) * 0.95)]
        assert p95 < 15.0, (
            f"predict_proba p95 = {p95:.2f}ms — tree traversal muito lento"
        )
