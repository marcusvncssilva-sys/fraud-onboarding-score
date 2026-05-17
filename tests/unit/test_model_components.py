"""Testes para LightGBMScorer, FastShapExplainer e ModelPersistence."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.features.pipeline import FEATURE_NAMES, N_FEATURES


# ── fixture: mini-modelo treinado ────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_model() -> Any:
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split

    from src.data.generators.customer_generator import generate_onboarding_dataset
    from src.features.pipeline import build_feature_matrix

    df = generate_onboarding_dataset(n_samples=500, fraud_rate=0.10, seed=7)
    X = build_feature_matrix(df)
    y = df["is_fraud"].to_numpy(dtype=int)

    X_tr, X_cal, y_tr, y_cal = train_test_split(X, y, test_size=0.25, stratify=y, random_state=7)

    lgbm = LGBMClassifier(n_estimators=10, max_depth=3, random_state=7, verbose=-1, n_jobs=1)
    lgbm.fit(X_tr, y_tr)

    model = CalibratedClassifierCV(lgbm, method="sigmoid", cv="prefit")
    model.fit(X_cal, y_cal)
    return model


@pytest.fixture
def saved_model_path(tiny_model: Any, tmp_path: Path) -> Path:
    from src.models.persistence import ModelPersistence

    path = tmp_path / "test_model.pkl"
    ModelPersistence.save(tiny_model, path, version="test-v0.1")
    return path


# ── ModelPersistence ──────────────────────────────────────────────────────────

class TestModelPersistence:
    def test_save_creates_file(self, tiny_model: Any, tmp_path: Path) -> None:
        from src.models.persistence import ModelPersistence

        path = tmp_path / "model.pkl"
        ModelPersistence.save(tiny_model, path, version="v1")
        assert path.exists()

    def test_load_returns_model(self, saved_model_path: Path) -> None:
        from src.models.persistence import ModelPersistence

        payload = ModelPersistence.load(saved_model_path)
        assert "model" in payload
        assert "version" in payload
        assert payload["version"] == "test-v0.1"

    def test_load_validates_features(self, saved_model_path: Path) -> None:
        from src.models.persistence import ModelPersistence

        payload = ModelPersistence.load(saved_model_path)
        assert payload["features"] == FEATURE_NAMES

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        from src.models.persistence import ModelPersistence

        with pytest.raises(FileNotFoundError):
            ModelPersistence.load(tmp_path / "nonexistent.pkl")

    def test_load_incompatible_features_raises(self, tiny_model: Any, tmp_path: Path) -> None:
        from src.models.persistence import ModelPersistence

        # Salva um payload com lista de features diferente
        path = tmp_path / "bad_model.pkl"
        payload = {
            "_magic": b"FRAUD_SCORE_MODEL_V1",
            "model": tiny_model,
            "version": "v-bad",
            "features": ["wrong_feature_a", "wrong_feature_b"],
            "n_features": 2,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

        with pytest.raises(ValueError, match="Incompatibilidade de features"):
            ModelPersistence.load(path)

    def test_warm_up_does_not_raise(self, tiny_model: Any) -> None:
        from src.models.persistence import ModelPersistence

        ModelPersistence.warm_up(tiny_model, n_features=N_FEATURES, n_iters=3)

    def test_save_load_roundtrip(self, tiny_model: Any, tmp_path: Path) -> None:
        from src.models.persistence import ModelPersistence

        path = tmp_path / "roundtrip.pkl"
        ModelPersistence.save(tiny_model, path, version="rt-v1", extra={"custom": 42})
        payload = ModelPersistence.load(path)
        assert payload["custom"] == 42

        X = np.zeros((1, N_FEATURES), dtype=np.float32)
        proba = payload["model"].predict_proba(X)
        assert proba.shape == (1, 2)


# ── LightGBMScorer ────────────────────────────────────────────────────────────

class TestLightGBMScorer:
    def test_singleton(self) -> None:
        from src.models.champion.lgbm_model import LightGBMScorer

        LightGBMScorer.reset()
        a = LightGBMScorer.get_instance()
        b = LightGBMScorer.get_instance()
        assert a is b

    def test_not_loaded_initially(self) -> None:
        from src.models.champion.lgbm_model import LightGBMScorer

        LightGBMScorer.reset()
        s = LightGBMScorer.get_instance()
        assert not s.is_loaded

    def test_load_missing_file_returns_false(self, tmp_path: Path) -> None:
        from src.models.champion.lgbm_model import LightGBMScorer

        LightGBMScorer.reset()
        s = LightGBMScorer()
        result = s.load(tmp_path / "missing.pkl")
        assert result is False
        assert not s.is_loaded

    def test_load_valid_model(self, saved_model_path: Path) -> None:
        from src.models.champion.lgbm_model import LightGBMScorer

        s = LightGBMScorer()
        result = s.load(saved_model_path)
        assert result is True
        assert s.is_loaded
        assert s.version == "test-v0.1"

    def test_predict_proba_shape(self, saved_model_path: Path) -> None:
        from src.models.champion.lgbm_model import LightGBMScorer

        s = LightGBMScorer()
        s.load(saved_model_path)
        X = np.zeros((3, N_FEATURES), dtype=np.float32)
        out = s.predict_proba(X)
        assert out.shape == (3,)

    def test_predict_proba_range(self, saved_model_path: Path) -> None:
        from src.models.champion.lgbm_model import LightGBMScorer

        s = LightGBMScorer()
        s.load(saved_model_path)
        X = np.random.default_rng(0).random((10, N_FEATURES)).astype(np.float32)
        out = s.predict_proba(X)
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_predict_proba_not_loaded_raises(self) -> None:
        from src.models.champion.lgbm_model import LightGBMScorer

        s = LightGBMScorer()
        X = np.zeros((1, N_FEATURES), dtype=np.float32)
        with pytest.raises(RuntimeError, match="não carregado"):
            s.predict_proba(X)

    def test_get_raw_booster_not_loaded_returns_none(self) -> None:
        from src.models.champion.lgbm_model import LightGBMScorer

        s = LightGBMScorer()
        assert s.get_raw_booster() is None

    def test_get_raw_booster_loaded(self, saved_model_path: Path) -> None:
        from src.models.champion.lgbm_model import LightGBMScorer

        s = LightGBMScorer()
        s.load(saved_model_path)
        booster = s.get_raw_booster()
        assert booster is not None


# ── FastShapExplainer ─────────────────────────────────────────────────────────

class TestFastShapExplainer:
    def test_not_available_before_init(self) -> None:
        from src.models.champion.shap_explainer import FastShapExplainer

        exp = FastShapExplainer()
        assert not exp.available

    def test_init_with_none_stays_unavailable(self) -> None:
        from src.models.champion.shap_explainer import FastShapExplainer

        exp = FastShapExplainer()
        exp.init(None)
        assert not exp.available

    def test_init_with_booster(self, saved_model_path: Path) -> None:
        from src.models.champion.lgbm_model import LightGBMScorer
        from src.models.champion.shap_explainer import FastShapExplainer

        s = LightGBMScorer()
        s.load(saved_model_path)
        booster = s.get_raw_booster()

        exp = FastShapExplainer()
        exp.init(booster)
        assert exp.available

    def test_explain_top3_returns_list(self, saved_model_path: Path) -> None:
        from src.models.champion.lgbm_model import LightGBMScorer
        from src.models.champion.shap_explainer import FastShapExplainer

        s = LightGBMScorer()
        s.load(saved_model_path)
        booster = s.get_raw_booster()

        exp = FastShapExplainer()
        exp.init(booster)

        X = np.zeros((1, N_FEATURES), dtype=np.float32)
        result = exp.explain_top3(X)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_explain_top3_schema(self, saved_model_path: Path) -> None:
        from src.models.champion.lgbm_model import LightGBMScorer
        from src.models.champion.shap_explainer import FastShapExplainer

        s = LightGBMScorer()
        s.load(saved_model_path)
        exp = FastShapExplainer()
        exp.init(s.get_raw_booster())

        X = np.ones((1, N_FEATURES), dtype=np.float32)
        result = exp.explain_top3(X)
        for item in result:
            assert "feature_name" in item
            assert "contribution" in item
            assert item["direction"] in ("RISK", "SAFE")
            assert item["feature_name"] in FEATURE_NAMES

    def test_explain_top3_unavailable_returns_empty(self) -> None:
        from src.models.champion.shap_explainer import FastShapExplainer

        exp = FastShapExplainer()
        X = np.zeros((1, N_FEATURES), dtype=np.float32)
        assert exp.explain_top3(X) == []

    def test_feature_key_deterministic(self) -> None:
        from src.models.champion.shap_explainer import FastShapExplainer

        exp = FastShapExplainer()
        X = np.ones((1, N_FEATURES), dtype=np.float32)
        assert exp.feature_key(X) == exp.feature_key(X)

    def test_feature_key_different_for_different_inputs(self) -> None:
        from src.models.champion.shap_explainer import FastShapExplainer

        exp = FastShapExplainer()
        X1 = np.zeros((1, N_FEATURES), dtype=np.float32)
        X2 = np.ones((1, N_FEATURES), dtype=np.float32)
        assert exp.feature_key(X1) != exp.feature_key(X2)
