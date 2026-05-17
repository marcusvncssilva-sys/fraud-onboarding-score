"""Mock do provedor DICT Bacen e dados regulatórios.

Em produção: consultar API do Bacen (https://www.bcb.gov.br/estabilidadefinanceira/dict)
e bureau parceiro para histórico de reclamações e chargeback.
"""

import random
from dataclasses import dataclass
from typing import Optional

from src.providers.base_provider import BaseProvider


@dataclass
class DICTResponse:
    """Dados do Diretório de Identificadores de Contas Transacionais."""
    cpf_hash: str
    has_pix_key: bool
    pix_key_age_days: Optional[int]
    n_complaints_bacen: int
    is_in_cadin: bool
    chargeback_ratio_90d: float
    account_age_days: Optional[int]


class MockDICTBacenProvider(BaseProvider):
    """Simula consulta ao DICT Bacen + dados de risco regulatório.

    Latência simulada baseada em SLA público da API Pix do Bacen (~100-400ms).
    """

    name = "dict_bacen_mock"
    LATENCY_RANGE_MS = (100.0, 400.0)

    async def query(self, cpf_hash: str) -> DICTResponse:
        """Consulta histórico regulatório do CPF."""
        self._maybe_raise_error()
        await self._simulate_latency(*self.LATENCY_RANGE_MS)

        seed = int(cpf_hash[:8], 16) % (2**31)
        rng = random.Random(seed)

        has_pix = rng.random() > 0.15  # ~85% dos brasileiros têm Pix

        return DICTResponse(
            cpf_hash=cpf_hash,
            has_pix_key=has_pix,
            pix_key_age_days=rng.randint(30, 1500) if has_pix else None,
            n_complaints_bacen=rng.choices(
                [0, 1, 2, 3], weights=[0.85, 0.10, 0.04, 0.01]
            )[0],
            is_in_cadin=rng.random() < 0.08,
            chargeback_ratio_90d=max(0.0, rng.gauss(0.01, 0.02)),
            account_age_days=rng.randint(30, 3650) if has_pix else None,
        )
