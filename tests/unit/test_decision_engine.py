"""Testes unitários para o Decision Engine e Score Transformer."""

import pytest

from src.decision.engine import DecisionEngine, HardRules
from src.models.score_transformer import ScoreTransformer


# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_request(**overrides):  # type: ignore[no-untyped-def]
    """Cria um request mínimo válido para testes."""
    from src.data.validators import OnboardingScoreRequest
    base = {
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
    base.update(overrides)
    return OnboardingScoreRequest(**base)


# ─── ScoreTransformer ─────────────────────────────────────────────────────────

class TestScoreTransformer:
    def test_zero_fraud_probability_gives_max_score(self):
        import numpy as np
        scores = ScoreTransformer.transform(np.array([0.0]))
        assert scores[0] == 1000

    def test_full_fraud_probability_gives_min_score(self):
        import numpy as np
        scores = ScoreTransformer.transform(np.array([1.0]))
        assert scores[0] == 0

    def test_score_range_always_between_0_and_1000(self):
        import numpy as np
        probas = np.linspace(0, 1, 100)
        scores = ScoreTransformer.transform(probas)
        assert scores.min() >= 0
        assert scores.max() <= 1000

    def test_higher_fraud_proba_gives_lower_score(self):
        import numpy as np
        scores = ScoreTransformer.transform(np.array([0.1, 0.5, 0.9]))
        assert scores[0] > scores[1] > scores[2]

    def test_risk_band_mapping(self):
        assert ScoreTransformer.to_risk_band(900) == "BAIXO"
        assert ScoreTransformer.to_risk_band(700) == "MEDIO_BAIXO"
        assert ScoreTransformer.to_risk_band(500) == "MEDIO_ALTO"
        assert ScoreTransformer.to_risk_band(300) == "ALTO"
        assert ScoreTransformer.to_risk_band(100) == "CRITICO"

    def test_decision_approved(self):
        assert ScoreTransformer.proba_to_decision(0.05) == "APPROVED"

    def test_decision_rejected(self):
        assert ScoreTransformer.proba_to_decision(0.80) == "REJECTED"

    def test_decision_review(self):
        assert ScoreTransformer.proba_to_decision(0.40) == "REVIEW"


# ─── HardRules ───────────────────────────────────────────────────────────────

class TestHardRules:
    def test_clean_request_triggers_no_rule(self):
        req = make_request()
        assert HardRules.check(req) is None

    def test_emulator_triggers_hard_rule(self):
        req = make_request(device_is_emulator=True)
        assert HardRules.check(req) == "HARD_RULE_EMULATOR_DETECTED"

    def test_tor_triggers_hard_rule(self):
        req = make_request(is_tor=True)
        assert HardRules.check(req) == "HARD_RULE_TOR_EXIT_NODE"

    def test_tampered_app_triggers_hard_rule(self):
        req = make_request(app_is_tampered=True)
        assert HardRules.check(req) == "HARD_RULE_APP_TAMPERED"

    def test_cpf_velocity_triggers_hard_rule(self):
        req = make_request(cpf_onboardings_7d=3)
        assert HardRules.check(req) == "HARD_RULE_CPF_VELOCITY_7D"

    def test_device_velocity_triggers_hard_rule(self):
        req = make_request(device_cpfs_7d=5)
        assert HardRules.check(req) == "HARD_RULE_DEVICE_VELOCITY_7D"


# ─── DecisionEngine ──────────────────────────────────────────────────────────

class TestDecisionEngine:
    def setup_method(self):
        self.engine = DecisionEngine(threshold_approve=0.15, threshold_reject=0.60)

    def test_low_risk_approved(self):
        req = make_request()
        result = self.engine.decide(req, proba_fraud=0.05)
        assert result.decision == "APPROVED"
        assert result.score >= 800
        assert result.hard_rule_triggered is None

    def test_high_risk_rejected(self):
        req = make_request()
        result = self.engine.decide(req, proba_fraud=0.80)
        assert result.decision == "REJECTED"
        assert result.score <= 200

    def test_medium_risk_goes_to_review(self):
        req = make_request()
        result = self.engine.decide(req, proba_fraud=0.35)
        assert result.decision == "REVIEW"

    def test_hard_rule_overrides_score(self):
        """Emulador deve ser REJECTED mesmo com probabilidade baixa."""
        req = make_request(device_is_emulator=True)
        result = self.engine.decide(req, proba_fraud=0.01)
        assert result.decision == "REJECTED"
        assert result.score == 0
        assert result.risk_band == "CRITICO"
        assert result.hard_rule_triggered == "HARD_RULE_EMULATOR_DETECTED"
