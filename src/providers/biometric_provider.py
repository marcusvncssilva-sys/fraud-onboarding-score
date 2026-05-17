"""Mock do provedor de biometria facial (liveness + face match).

Em produção: substituir por httpx.AsyncClient chamando Idwall, unico IDPay,
Neurotech ou provedor homologado pelo banco.
"""

import random
from dataclasses import dataclass

from src.providers.base_provider import BaseProvider, ProviderTimeoutError


@dataclass
class BiometricResponse:
    request_id: str
    liveness_score: float
    face_match_score: float
    liveness_passed: bool
    face_match_passed: bool
    provider_decision: str      # APPROVED | REJECTED | REVIEW
    response_time_ms: float
    provider_version: str


class MockBiometricProvider(BaseProvider):
    """Simula provedor de biometria facial com latência e falhas realistas.

    Parâmetros simulados baseados em benchmarks públicos de provedores
    como Idwall (~200-600ms p95) e Neurotech (~300-800ms p95).
    """

    name = "biometric_mock"
    LATENCY_RANGE_MS = (200.0, 700.0)
    LIVENESS_THRESHOLD = 0.70
    FACE_MATCH_THRESHOLD = 0.75

    async def analyze(self, cpf_hash: str) -> BiometricResponse:
        """Analisa biometria e retorna scores de liveness e face match."""
        self._maybe_raise_error()
        latency_ms = await self._simulate_latency(*self.LATENCY_RANGE_MS)

        # Seed determinístico baseado no hash (reprodutível para testes)
        seed = int(cpf_hash[:8], 16) % (2**31)
        rng = random.Random(seed)

        liveness = max(0.0, min(1.0, rng.gauss(0.85, 0.12)))
        face_match = max(0.0, min(1.0, rng.gauss(0.88, 0.10)))

        liveness_passed = liveness >= self.LIVENESS_THRESHOLD
        face_match_passed = face_match >= self.FACE_MATCH_THRESHOLD

        if liveness_passed and face_match_passed:
            decision = "APPROVED"
        elif liveness < 0.40 or face_match < 0.40:
            decision = "REJECTED"
        else:
            decision = "REVIEW"

        return BiometricResponse(
            request_id=f"bio_{cpf_hash[:12]}",
            liveness_score=round(liveness, 4),
            face_match_score=round(face_match, 4),
            liveness_passed=liveness_passed,
            face_match_passed=face_match_passed,
            provider_decision=decision,
            response_time_ms=round(latency_ms, 2),
            provider_version="mock-v1.0.0",
        )
