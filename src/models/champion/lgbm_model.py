"""LightGBM champion — singleton carregado no startup, predict vetorizado."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np

_log = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/champion.pkl"))

MODEL_VERSION_FALLBACK = "heuristic-v0"


class LightGBMScorer:
    """Wrapper thread-safe do modelo LightGBM calibrado.

    Singleton — instanciado uma vez no startup da aplicação e compartilhado
    entre todos os workers do Uvicorn dentro do mesmo processo.
    """

    _instance: Optional["LightGBMScorer"] = None

    def __init__(self) -> None:
        self._model: Any = None
        self._version: str = MODEL_VERSION_FALLBACK
        self._loaded: bool = False

    # ── singleton ─────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "LightGBMScorer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Usado nos testes para forçar reload."""
        cls._instance = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self, path: Optional[Path] = None) -> bool:
        model_path = path or _DEFAULT_MODEL_PATH
        try:
            with open(model_path, "rb") as f:
                data: dict[str, Any] = pickle.load(f)
            self._model = data["model"]
            self._version = data.get("version", "unknown")
            self._loaded = True
            _log.info("Modelo carregado: %s (versão: %s)", model_path, self._version)
            return True
        except FileNotFoundError:
            _log.warning("Modelo não encontrado em %s — modo heurístico ativo", model_path)
            return False
        except Exception as exc:
            _log.error("Falha ao carregar modelo: %s", exc)
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def version(self) -> str:
        return self._version

    # ── inferência ────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidade de fraude — shape (n,)."""
        if not self._loaded or self._model is None:
            raise RuntimeError("Modelo não carregado — chame load() antes de predict_proba()")
        return self._model.predict_proba(X)[:, 1]  # type: ignore[no-any-return]

    def get_raw_booster(self) -> Any:
        """Retorna o booster LightGBM nativo para o SHAP TreeExplainer.

        Navega por wrappers de calibração (CalibratedClassifierCV)
        para expor o modelo base.
        """
        if not self._loaded or self._model is None:
            return None
        m = self._model
        # CalibratedClassifierCV (sklearn >= 1.2) expõe calibrated_classifiers_
        if hasattr(m, "calibrated_classifiers_"):
            base = m.calibrated_classifiers_[0].estimator
        elif hasattr(m, "base_estimator"):
            base = m.base_estimator
        else:
            base = m
        # LGBMClassifier expõe booster_ após fit
        return getattr(base, "booster_", base)
