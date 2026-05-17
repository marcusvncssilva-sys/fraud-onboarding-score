"""Testes unitários do gerador de dados sintéticos."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.generators.customer_generator import generate_onboarding_dataset


@pytest.fixture(scope="module")
def dataset():
    return generate_onboarding_dataset(n_samples=1_000, fraud_rate=0.02, seed=42)


def test_row_count(dataset) -> None:
    assert len(dataset) == 1_000


def test_fraud_rate_approx(dataset) -> None:
    rate = dataset["is_fraud"].mean()
    assert 0.01 <= rate <= 0.04, f"taxa de fraude inesperada: {rate:.3f}"


def test_required_columns_present(dataset) -> None:
    from src.features.pipeline import FEATURE_NAMES

    raw_feature_cols = FEATURE_NAMES[:25]
    for col in raw_feature_cols:
        assert col in dataset.columns, f"coluna ausente: {col}"


def test_no_nan_in_numeric_features(dataset) -> None:
    from src.features.pipeline import FEATURE_NAMES

    for col in FEATURE_NAMES[:25]:
        if dataset[col].dtype in (float, np.float64, np.float32):
            nulls = dataset[col].isna().sum()
            # apenas pix_key_age_days é nullable
            if col != "pix_key_age_days":
                assert nulls == 0, f"{col} tem {nulls} NaN inesperados"


def test_bio_scores_in_range(dataset) -> None:
    assert dataset["bio_liveness_score"].between(0, 1).all()
    assert dataset["bio_face_match_score"].between(0, 1).all()


def test_anonymizer_score_clipped(dataset) -> None:
    assert dataset["anonymizer_score"].between(0, 3).all()


def test_reproducibility() -> None:
    df1 = generate_onboarding_dataset(n_samples=100, seed=99)
    df2 = generate_onboarding_dataset(n_samples=100, seed=99)
    assert (df1["bio_liveness_score"].values == df2["bio_liveness_score"].values).all()


def test_fraud_vs_legit_separation(dataset) -> None:
    fraud = dataset[dataset["is_fraud"] == 1]
    legit = dataset[dataset["is_fraud"] == 0]
    # fraudadores devem ter mais CPFs distintos por device
    assert fraud["device_cpfs_30d"].mean() > legit["device_cpfs_30d"].mean()
    # clientes legítimos devem ter melhor liveness
    assert legit["bio_liveness_score"].mean() > fraud["bio_liveness_score"].mean()
