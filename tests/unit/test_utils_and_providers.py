"""Testes unitários para utilitários: crypto, metrics e providers."""

import os
import asyncio
import numpy as np
import pytest

os.environ.setdefault("PSEUDONYM_SECRET_KEY", "test-secret-key-for-pytest-only")


# ─── DataPseudonymizer ────────────────────────────────────────────────────────

class TestDataPseudonymizer:
    def setup_method(self):
        from src.utils.crypto import DataPseudonymizer
        self.p = DataPseudonymizer("test-secret-key-32-chars-minimum!")

    def test_cpf_hash_is_64_chars(self):
        h = self.p.pseudonymize_cpf("123.456.789-09")
        assert len(h) == 64

    def test_cpf_hash_is_deterministic(self):
        h1 = self.p.pseudonymize_cpf("12345678909")
        h2 = self.p.pseudonymize_cpf("12345678909")
        assert h1 == h2

    def test_different_cpfs_give_different_hashes(self):
        h1 = self.p.pseudonymize_cpf("11111111111")
        h2 = self.p.pseudonymize_cpf("22222222222")
        assert h1 != h2

    def test_cpf_with_dots_and_dash_same_as_digits_only(self):
        h1 = self.p.pseudonymize_cpf("123.456.789-09")
        h2 = self.p.pseudonymize_cpf("12345678909")
        assert h1 == h2

    def test_device_hash_is_deterministic(self):
        h1 = self.p.pseudonymize_device("device-001")
        h2 = self.p.pseudonymize_device("device-001")
        assert h1 == h2

    def test_missing_secret_raises(self):
        from src.utils.crypto import DataPseudonymizer
        original = os.environ.pop("PSEUDONYM_SECRET_KEY", None)
        try:
            with pytest.raises(ValueError):
                DataPseudonymizer()
        finally:
            if original is not None:
                os.environ["PSEUDONYM_SECRET_KEY"] = original

    def test_biometric_hash_returns_string(self):
        h = self.p.hash_biometric_reference("base64imagedata")
        assert isinstance(h, str)
        assert len(h) == 64


class TestDataClassifier:
    def test_pii_fields_are_redacted(self):
        from src.utils.crypto import DataClassifier
        data = {"cpf": "123.456.789-09", "score": 850, "nome": "João"}
        result = DataClassifier.sanitize_for_logging(data)
        assert result["cpf"] == "[REDACTED_PII]"
        assert result["nome"] == "[REDACTED_PII]"
        assert result["score"] == 850

    def test_sensitive_fields_are_redacted(self):
        from src.utils.crypto import DataClassifier
        data = {"bio_image_b64": "abc123", "geo_latitude": -23.5}
        result = DataClassifier.sanitize_for_logging(data)
        assert result["bio_image_b64"] == "[REDACTED_SENSITIVE]"
        assert result["geo_latitude"] == "[REDACTED_SENSITIVE]"

    def test_non_sensitive_fields_pass_through(self):
        from src.utils.crypto import DataClassifier
        data = {"cpf_hash": "a" * 64, "score": 700}
        result = DataClassifier.sanitize_for_logging(data)
        assert result["cpf_hash"] == "a" * 64
        assert result["score"] == 700


# ─── Metrics ─────────────────────────────────────────────────────────────────

class TestMetrics:
    def _make_data(self, n=1000, fraud_rate=0.1, seed=42):
        rng = np.random.default_rng(seed)
        y_true = (rng.random(n) < fraud_rate).astype(int)
        y_proba = np.where(y_true == 1,
                           rng.beta(8, 2, n),
                           rng.beta(2, 8, n))
        return y_true, y_proba

    def test_ks_between_0_and_1(self):
        from src.utils.metrics import compute_ks
        y_true, y_proba = self._make_data()
        ks = compute_ks(y_true, y_proba)
        assert 0.0 <= ks <= 1.0

    def test_perfect_model_ks_near_1(self):
        from src.utils.metrics import compute_ks
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_proba = np.array([0.9, 0.85, 0.8, 0.2, 0.15, 0.1])
        ks = compute_ks(y_true, y_proba)
        assert ks > 0.8

    def test_psi_stable_same_distribution(self):
        from src.utils.metrics import compute_psi
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 5000)
        cur = rng.normal(0, 1, 5000)
        psi = compute_psi(ref, cur)
        assert psi < 0.10  # mesma distribuição → estável

    def test_full_report_keys(self):
        from src.utils.metrics import compute_full_report
        y_true, y_proba = self._make_data()
        report = compute_full_report(y_true, y_proba)
        for key in ("roc_auc", "pr_auc", "ks", "precision", "recall", "f1"):
            assert key in report

    def test_full_report_values_in_range(self):
        from src.utils.metrics import compute_full_report
        y_true, y_proba = self._make_data()
        report = compute_full_report(y_true, y_proba)
        assert 0 <= report["roc_auc"] <= 1
        assert 0 <= report["ks"] <= 1
        assert 0 <= report["precision"] <= 1

    def test_business_impact_keys(self):
        from src.utils.metrics import compute_business_impact
        y_true, y_proba = self._make_data()
        impact = compute_business_impact(y_true, y_proba)
        for key in ("net_benefit_brl", "fraud_capture_rate", "false_positive_rate"):
            assert key in impact

    def test_business_impact_capture_rate_in_range(self):
        from src.utils.metrics import compute_business_impact
        y_true, y_proba = self._make_data()
        impact = compute_business_impact(y_true, y_proba)
        assert 0.0 <= impact["fraud_capture_rate"] <= 1.0
        assert 0.0 <= impact["false_positive_rate"] <= 1.0


# ─── Providers ───────────────────────────────────────────────────────────────

class TestMockBiometricProvider:
    def test_analyze_returns_response(self):
        from src.providers.biometric_provider import MockBiometricProvider
        provider = MockBiometricProvider()
        response = asyncio.get_event_loop().run_until_complete(
            provider.analyze("a" * 64)
        )
        assert 0.0 <= response.liveness_score <= 1.0
        assert 0.0 <= response.face_match_score <= 1.0
        assert response.provider_decision in {"APPROVED", "REJECTED", "REVIEW"}

    def test_analyze_is_deterministic(self):
        from src.providers.biometric_provider import MockBiometricProvider
        provider = MockBiometricProvider()
        cpf_hash = "b" * 64
        r1 = asyncio.get_event_loop().run_until_complete(provider.analyze(cpf_hash))
        r2 = asyncio.get_event_loop().run_until_complete(provider.analyze(cpf_hash))
        assert r1.liveness_score == r2.liveness_score
        assert r1.face_match_score == r2.face_match_score

    def test_different_cpfs_give_different_scores(self):
        from src.providers.biometric_provider import MockBiometricProvider
        provider = MockBiometricProvider()
        r1 = asyncio.get_event_loop().run_until_complete(provider.analyze("a" * 64))
        r2 = asyncio.get_event_loop().run_until_complete(provider.analyze("b" * 64))
        assert r1.liveness_score != r2.liveness_score


class TestMockDICTBacenProvider:
    def test_query_returns_response(self):
        from src.providers.dict_bacen_provider import MockDICTBacenProvider
        provider = MockDICTBacenProvider()
        response = asyncio.get_event_loop().run_until_complete(
            provider.query("a" * 64)
        )
        assert isinstance(response.has_pix_key, bool)
        assert response.n_complaints_bacen >= 0
        assert response.chargeback_ratio_90d >= 0.0

    def test_pix_key_age_is_none_when_no_pix(self):
        from src.providers.dict_bacen_provider import MockDICTBacenProvider
        provider = MockDICTBacenProvider()
        # Testar com hash que resulte em has_pix_key=False
        # (determinístico pelo seed do hash)
        response = asyncio.get_event_loop().run_until_complete(
            provider.query("a" * 64)
        )
        if not response.has_pix_key:
            assert response.pix_key_age_days is None
