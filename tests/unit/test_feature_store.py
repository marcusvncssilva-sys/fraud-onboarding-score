"""Testes do FeatureStore em modo degradado (Redis indisponível)."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.pipeline import N_FEATURES
from src.features.store import FeatureStore


@pytest.fixture
def offline_store() -> FeatureStore:
    """FeatureStore com Redis marcado como indisponível (sem conexão real)."""
    store = FeatureStore.__new__(FeatureStore)
    store._url = "redis://localhost:6379"
    store._client = None
    store._available = False
    return store


class TestFeatureStoreDegradedMode:
    """Quando Redis está indisponível todas as operações devem retornar None/pass."""

    @pytest.mark.asyncio
    async def test_get_features_returns_none(self, offline_store: FeatureStore) -> None:
        result = await offline_store.get_features("a" * 64)
        assert result is None

    @pytest.mark.asyncio
    async def test_set_features_does_not_raise(self, offline_store: FeatureStore) -> None:
        X = np.zeros((1, N_FEATURES), dtype=np.float32)
        await offline_store.set_features("a" * 64, X)  # deve ser silencioso

    @pytest.mark.asyncio
    async def test_get_shap_returns_none(self, offline_store: FeatureStore) -> None:
        result = await offline_store.get_shap("deadbeef")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_shap_does_not_raise(self, offline_store: FeatureStore) -> None:
        factors = [{"feature_name": "f", "contribution": 0.1, "direction": "RISK"}]
        await offline_store.set_shap("deadbeef", factors)

    @pytest.mark.asyncio
    async def test_mget_features_returns_nones(self, offline_store: FeatureStore) -> None:
        result = await offline_store.mget_features(["a" * 64, "b" * 64])
        assert result == [None, None]

    @pytest.mark.asyncio
    async def test_mget_features_empty_list(self, offline_store: FeatureStore) -> None:
        result = await offline_store.mget_features([])
        assert result == []

    @pytest.mark.asyncio
    async def test_close_does_not_raise(self, offline_store: FeatureStore) -> None:
        await offline_store.close()

    def test_available_false(self, offline_store: FeatureStore) -> None:
        assert offline_store.available is False

    @pytest.mark.asyncio
    async def test_connect_unreachable_sets_unavailable(self) -> None:
        store = FeatureStore("redis://127.0.0.1:19999")  # porta inexistente
        await store.connect()
        assert store.available is False
