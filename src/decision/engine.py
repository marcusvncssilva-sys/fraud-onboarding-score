"""Decision Engine — combina hard rules com ML score para decisão final."""

from dataclasses import dataclass
from typing import Optional

from src.data.validators import OnboardingScoreRequest
from src.models.score_transformer import ScoreTransformer


@dataclass
class DecisionResult:
    decision: str           # APPROVED | REVIEW | REJECTED
    score: int              # 0-1000
    risk_band: str
    hard_rule_triggered: Optional[str]  # Nome da regra se ativada


class HardRules:
    """Regras determinísticas que sobrepõem o score do modelo.

    Hard rules são aplicadas ANTES do score de ML.
    Fraudes com sinais fortíssimos não precisam de modelo — são bloqueadas diretamente.
    """

    @staticmethod
    def check(req: OnboardingScoreRequest) -> Optional[str]:
        """Retorna nome da regra violada ou None se nenhuma."""
        if req.device_is_emulator:
            return "HARD_RULE_EMULATOR_DETECTED"

        if req.is_tor:
            return "HARD_RULE_TOR_EXIT_NODE"

        if req.app_is_tampered:
            return "HARD_RULE_APP_TAMPERED"

        if req.cpf_onboardings_7d >= 3:
            return "HARD_RULE_CPF_VELOCITY_7D"

        if req.device_cpfs_7d >= 5:
            return "HARD_RULE_DEVICE_VELOCITY_7D"

        if req.bio_attempts >= 5 and req.bio_failure_rate >= 0.8:
            return "HARD_RULE_BIOMETRIC_REPEATED_FAILURE"

        return None


class DecisionEngine:
    """Combina hard rules + ML score para produzir decisão final.

    Fluxo:
        1. Hard rules: se ativadas → REJECTED imediato (score = 0)
        2. ML score: determina APPROVED | REVIEW | REJECTED
    """

    def __init__(
        self,
        threshold_approve: float = 0.15,
        threshold_reject: float = 0.60,
    ) -> None:
        self.threshold_approve = threshold_approve
        self.threshold_reject = threshold_reject

    def decide(
        self,
        request: OnboardingScoreRequest,
        proba_fraud: float,
    ) -> DecisionResult:
        # 1. Verificar hard rules primeiro
        rule = HardRules.check(request)
        if rule:
            return DecisionResult(
                decision="REJECTED",
                score=0,
                risk_band="CRITICO",
                hard_rule_triggered=rule,
            )

        # 2. ML score decide
        import numpy as np
        score = int(ScoreTransformer.transform(np.array([proba_fraud]))[0])
        risk_band = ScoreTransformer.to_risk_band(score)
        decision = ScoreTransformer.proba_to_decision(
            proba_fraud, self.threshold_approve, self.threshold_reject
        )

        return DecisionResult(
            decision=decision,
            score=score,
            risk_band=risk_band,
            hard_rule_triggered=None,
        )
