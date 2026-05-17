"""Transformação de probabilidade calibrada em score 0-1000."""

import numpy as np


class ScoreTransformer:
    """Converte probabilidade de fraude [0,1] em score [0,1000].

    Convenção de mercado financeiro (padrão bureau de crédito):
        Score ALTO → BAIXO risco (cliente bom)
        Score BAIXO → ALTO risco (possível fraude)

    Isso é intencional e facilita comunicação com negócio/crédito.
    """

    SCORE_MIN = 0
    SCORE_MAX = 1000

    BANDS = {
        "BAIXO":        (800, 1000),
        "MEDIO_BAIXO":  (600, 799),
        "MEDIO_ALTO":   (400, 599),
        "ALTO":         (200, 399),
        "CRITICO":      (0,   199),
    }

    @staticmethod
    def transform(proba_fraud: np.ndarray) -> np.ndarray:
        """proba_fraud=1.0 → score 0; proba_fraud=0.0 → score 1000."""
        score = (1.0 - proba_fraud) * 1000.0
        return np.clip(score, ScoreTransformer.SCORE_MIN, ScoreTransformer.SCORE_MAX).astype(int)

    @staticmethod
    def to_risk_band(score: int) -> str:
        """Converte score numérico em faixa de risco."""
        for band, (low, high) in ScoreTransformer.BANDS.items():
            if low <= score <= high:
                return band
        return "CRITICO"

    @staticmethod
    def proba_to_decision(
        proba_fraud: float,
        threshold_approve: float = 0.15,
        threshold_reject: float = 0.60,
    ) -> str:
        """Converte probabilidade em decisão de negócio (dual-threshold)."""
        if proba_fraud < threshold_approve:
            return "APPROVED"
        elif proba_fraud >= threshold_reject:
            return "REJECTED"
        return "REVIEW"
