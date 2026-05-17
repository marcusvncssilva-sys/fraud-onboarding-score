"""Persistência do modelo — save/load com protocol pickle mais rápido.

Mantém contrato estrito de versão para detectar incompatibilidades
entre o modelo serializado e o código de inferência atual.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from src.features.pipeline import FEATURE_NAMES, N_FEATURES

_log = logging.getLogger(__name__)

_PROTOCOL = pickle.HIGHEST_PROTOCOL
_MAGIC = b"FRAUD_SCORE_MODEL_V1"


class ModelPersistence:
    """Salva e carrega modelos com verificação de integridade de features."""

    @staticmethod
    def save(
        model: Any,
        path: str | Path,
        version: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Serializa modelo + metadados de features para disco."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "_magic": _MAGIC,
            "model": model,
            "version": version,
            "features": FEATURE_NAMES,
            "n_features": N_FEATURES,
        }
        if extra:
            payload.update(extra)

        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=_PROTOCOL)
        tmp.replace(path)  # rename atômico — evita arquivo corrompido
        _log.info("Modelo salvo: %s (v%s)", path, version)

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        """Carrega modelo do disco e valida compatibilidade de features."""
        path = Path(path)
        with open(path, "rb") as f:
            payload: dict[str, Any] = pickle.load(f)  # noqa: S301

        if payload.get("_magic") != _MAGIC:
            _log.warning("Magic bytes ausentes em %s — arquivo pode ser legado", path)

        saved_features: list[str] = payload.get("features", [])
        if saved_features and saved_features != FEATURE_NAMES:
            diff = set(saved_features) ^ set(FEATURE_NAMES)
            raise ValueError(
                f"Incompatibilidade de features entre modelo e código: {diff}"
            )

        saved_n = payload.get("n_features", len(saved_features))
        if saved_n and saved_n != N_FEATURES:
            raise ValueError(
                f"n_features do modelo ({saved_n}) ≠ pipeline atual ({N_FEATURES})"
            )

        return payload

    @staticmethod
    def warm_up(model: Any, n_features: int = N_FEATURES, n_iters: int = 10) -> None:
        """Executa predições de aquecimento para JIT-compile das árvores."""
        dummy = np.zeros((1, n_features), dtype=np.float32)
        for _ in range(n_iters):
            model.predict_proba(dummy)
        _log.info("Warm-up concluído (%d iterações)", n_iters)
