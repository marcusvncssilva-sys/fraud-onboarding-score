"""SHAP aproximado — top-3 fatores, resultado cacheável no Redis."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

import numpy as np

from src.features.pipeline import FEATURE_NAMES

_log = logging.getLogger(__name__)


class FastShapExplainer:
    """TreeExplainer com approximate=True para LightGBM.

    Computa apenas top-3 features para minimizar latência.
    O resultado é cacheável via FeatureStore.set_shap().
    """

    def __init__(self) -> None:
        self._explainer: Any = None

    def init(self, booster: Any) -> None:
        """Inicializa com o booster LightGBM nativo."""
        if booster is None:
            return
        try:
            import shap

            self._explainer = shap.TreeExplainer(
                booster,
                feature_perturbation="tree_path_dependent",
            )
            _log.info("SHAP TreeExplainer inicializado")
        except Exception as exc:
            _log.warning("SHAP indisponível (respostas sem explicabilidade): %s", exc)

    @property
    def available(self) -> bool:
        return self._explainer is not None

    def feature_key(self, X: np.ndarray) -> str:
        """Chave de cache derivada do vetor de features (primeiros 16 bytes MD5)."""
        return hashlib.md5(X.tobytes(), usedforsecurity=False).hexdigest()[:16]  # noqa: S324

    def explain_top3(self, X: np.ndarray) -> list[dict[str, Any]]:
        """Retorna top-3 fatores de risco em < 10ms (approximate=True)."""
        if not self.available or self._explainer is None:
            return []
        try:
            # approximate=True usa caminho da árvore — ~5x mais rápido que exact
            sv = self._explainer.shap_values(X, approximate=True)
            # LightGBM binary: lista [classe0, classe1] ou array 2D
            if isinstance(sv, list):
                values = sv[1][0]
            else:
                values = sv[0] if sv.ndim == 2 else sv  # type: ignore[union-attr]

            top_idx = np.argsort(np.abs(values))[-3:][::-1]
            return [
                {
                    "feature_name": FEATURE_NAMES[i],
                    "contribution": round(float(values[i]), 4),
                    "direction": "RISK" if values[i] > 0 else "SAFE",
                    "feature_value": float(X[0, i]),
                }
                for i in top_idx
            ]
        except Exception as exc:
            _log.debug("SHAP error (ignorado): %s", exc)
            return []
