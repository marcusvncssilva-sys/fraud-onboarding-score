"""Testes unitários do feature pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.validators import OnboardingScoreRequest
from src.features.pipeline import (
    FEATURE_NAMES,
    N_FEATURES,
    build_feature_matrix,
    extract_features,
)


@pytest.fixture
def valid_request(sample_request_payload: dict) -> OnboardingScoreRequest:  # type: ignore[type-arg]
    return OnboardingScoreRequest(**sample_request_payload)


class TestExtractFeatures:
    def test_shape(self, valid_request: OnboardingScoreRequest) -> None:
        X = extract_features(valid_request)
        assert X.shape == (1, N_FEATURES)

    def test_dtype_float32(self, valid_request: OnboardingScoreRequest) -> None:
        X = extract_features(valid_request)
        assert X.dtype == np.float32

    def test_no_nan(self, valid_request: OnboardingScoreRequest) -> None:
        X = extract_features(valid_request)
        assert not np.any(np.isnan(X))

    def test_no_inf(self, valid_request: OnboardingScoreRequest) -> None:
        X = extract_features(valid_request)
        assert not np.any(np.isinf(X))

    def test_feature_count_matches_names(self, valid_request: OnboardingScoreRequest) -> None:
        X = extract_features(valid_request)
        assert X.shape[1] == len(FEATURE_NAMES)

    def test_bio_quality_derived(self, valid_request: OnboardingScoreRequest) -> None:
        X = extract_features(valid_request)
        liveness = valid_request.bio_liveness_score
        face_match = valid_request.bio_face_match_score
        idx = FEATURE_NAMES.index("bio_quality")
        assert abs(float(X[0, idx]) - liveness * face_match) < 1e-5

    def test_device_risk_derived(self, valid_request: OnboardingScoreRequest) -> None:
        X = extract_features(valid_request)
        expected = float(valid_request.device_is_rooted) + valid_request.device_fraud_score
        idx = FEATURE_NAMES.index("device_risk")
        assert abs(float(X[0, idx]) - expected) < 1e-5

    def test_velocity_composite_non_negative(self, valid_request: OnboardingScoreRequest) -> None:
        X = extract_features(valid_request)
        idx = FEATURE_NAMES.index("velocity_composite")
        assert float(X[0, idx]) >= 0.0

    def test_bool_fields_are_float(self, valid_request: OnboardingScoreRequest) -> None:
        X = extract_features(valid_request)
        bool_features = [
            "bio_liveness_passed", "bio_face_match_passed",
            "device_is_rooted", "app_is_tampered",
            "is_vpn", "is_proxy", "is_foreign_ip",
            "is_suspiciously_fast", "is_night_session",
        ]
        for name in bool_features:
            idx = FEATURE_NAMES.index(name)
            val = float(X[0, idx])
            assert val in (0.0, 1.0), f"{name} = {val} not in {{0.0, 1.0}}"

    def test_high_risk_request(self, sample_request_payload: dict) -> None:  # type: ignore[type-arg]
        payload = {
            **sample_request_payload,
            "device_is_rooted": True,
            "device_fraud_score": 0.95,
            "is_tor": True,
            "bio_liveness_score": 0.10,
        }
        req = OnboardingScoreRequest(**payload)
        X = extract_features(req)
        device_risk_idx = FEATURE_NAMES.index("device_risk")
        assert float(X[0, device_risk_idx]) > 1.0


class TestBuildFeatureMatrix:
    def test_shape(self) -> None:
        from src.data.generators.customer_generator import generate_onboarding_dataset

        df = generate_onboarding_dataset(n_samples=100, seed=0)
        X = build_feature_matrix(df)
        assert X.shape == (100, N_FEATURES)

    def test_dtype(self) -> None:
        from src.data.generators.customer_generator import generate_onboarding_dataset

        df = generate_onboarding_dataset(n_samples=50, seed=1)
        X = build_feature_matrix(df)
        assert X.dtype == np.float32

    def test_no_nan(self) -> None:
        from src.data.generators.customer_generator import generate_onboarding_dataset

        df = generate_onboarding_dataset(n_samples=200, seed=2)
        X = build_feature_matrix(df)
        assert not np.any(np.isnan(X))

    def test_consistent_with_extract_features(self, sample_request_payload: dict) -> None:  # type: ignore[type-arg]
        """build_feature_matrix e extract_features devem produzir os mesmos valores."""
        import pandas as pd

        req = OnboardingScoreRequest(**sample_request_payload)
        X_req = extract_features(req)

        # Cria DataFrame com as mesmas features
        row = {name: float(X_req[0, i]) for i, name in enumerate(FEATURE_NAMES[:25])}
        df = pd.DataFrame([row])
        X_df = build_feature_matrix(df)

        np.testing.assert_allclose(X_req, X_df, rtol=1e-5)
