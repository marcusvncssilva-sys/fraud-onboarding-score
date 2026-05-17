"""Redis feature store — cache assíncrono de vetores e SHAP (TTL 5min)."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import numpy as np

from src.features.pipeline import N_FEATURES

_log = logging.getLogger(__name__)

_FEATURES_PREFIX = "feat:"
_SHAP_PREFIX = "shap:"
_TTL = 300  # 5 minutos


class FeatureStore:
    """Cache online de features por cpf_hash usando Redis async.

    Degradação graciosa: se Redis estiver indisponível, retorna None
    e o hot path calcula tudo on-the-fly sem lançar exceção.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379") -> None:
        self._url = redis_url
        self._client: Any = None
        self._available = False

    async def connect(self) -> None:
        try:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(
                self._url,
                decode_responses=False,
                socket_connect_timeout=1.0,
                socket_timeout=0.5,
            )
            await self._client.ping()
            self._available = True
            _log.info("Redis conectado: %s", self._url)
        except Exception as exc:
            _log.warning("Redis indisponível (modo degradado): %s", exc)
            self._available = False

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass

    @property
    def available(self) -> bool:
        return self._available

    # ── features ─────────────────────────────────────────────────────────────

    async def get_features(self, cpf_hash: str) -> Optional[np.ndarray]:
        if not self._available:
            return None
        try:
            raw: Optional[bytes] = await self._client.get(f"{_FEATURES_PREFIX}{cpf_hash}")
            if raw is None:
                return None
            arr = np.frombuffer(raw, dtype=np.float32)
            if arr.shape[0] != N_FEATURES:
                return None
            return arr.reshape(1, -1)
        except Exception:
            return None

    async def set_features(self, cpf_hash: str, features: np.ndarray) -> None:
        if not self._available:
            return
        try:
            await self._client.setex(
                f"{_FEATURES_PREFIX}{cpf_hash}",
                _TTL,
                features.astype(np.float32).flatten().tobytes(),
            )
        except Exception:
            pass

    # ── SHAP cache ────────────────────────────────────────────────────────────

    async def get_shap(self, feature_key: str) -> Optional[list[dict[str, Any]]]:
        if not self._available:
            return None
        try:
            raw = await self._client.get(f"{_SHAP_PREFIX}{feature_key}")
            if raw is None:
                return None
            return json.loads(raw)  # type: ignore[no-any-return]
        except Exception:
            return None

    async def set_shap(self, feature_key: str, factors: list[dict[str, Any]]) -> None:
        if not self._available:
            return
        try:
            await self._client.setex(
                f"{_SHAP_PREFIX}{feature_key}",
                _TTL,
                json.dumps(factors, ensure_ascii=False),
            )
        except Exception:
            pass

    # ── batch pipeline (para leituras multi-key no hot path) ─────────────────

    async def mget_features(self, cpf_hashes: list[str]) -> list[Optional[np.ndarray]]:
        """Lê múltiplas chaves com um único round-trip Redis."""
        if not self._available or not cpf_hashes:
            return [None] * len(cpf_hashes)
        try:
            keys = [f"{_FEATURES_PREFIX}{h}" for h in cpf_hashes]
            raws = await self._client.mget(*keys)
            results: list[Optional[np.ndarray]] = []
            for raw in raws:
                if raw is None:
                    results.append(None)
                    continue
                arr = np.frombuffer(raw, dtype=np.float32)
                results.append(arr.reshape(1, -1) if arr.shape[0] == N_FEATURES else None)
            return results
        except Exception:
            return [None] * len(cpf_hashes)
