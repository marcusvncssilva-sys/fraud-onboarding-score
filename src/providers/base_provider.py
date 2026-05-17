"""Classe base para provedores externos (biometria, device, geo, DICT)."""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


class ProviderError(Exception):
    """Erro genérico de provedor externo."""


class ProviderTimeoutError(ProviderError):
    """Timeout na chamada ao provedor."""


class ProviderUnavailableError(ProviderError):
    """Provedor indisponível (circuit breaker)."""


@dataclass
class ProviderCallMetrics:
    provider_name: str
    latency_ms: float
    success: bool
    error: Optional[str] = None


class BaseProvider(ABC):
    """Interface base para todos os provedores externos."""

    name: str = "base"
    timeout_ms: float = 1000.0
    error_rate: float = 0.01  # 1% de falha simulada

    async def _simulate_latency(self, min_ms: float, max_ms: float) -> float:
        """Simula latência realista de chamada de rede."""
        latency_s = random.uniform(min_ms, max_ms) / 1000
        await asyncio.sleep(latency_s)
        return latency_s * 1000

    def _maybe_raise_error(self) -> None:
        """Simula falhas ocasionais do provedor."""
        if random.random() < self.error_rate:
            raise ProviderTimeoutError(
                f"{self.name}: timeout após {self.timeout_ms:.0f}ms"
            )
