"""Feature engineering — pipeline vetorizado e Redis feature store."""

from src.features.pipeline import (
    FEATURE_NAMES,
    N_FEATURES,
    build_feature_matrix,
    extract_features,
)
from src.features.store import FeatureStore

__all__ = [
    "FEATURE_NAMES",
    "N_FEATURES",
    "extract_features",
    "build_feature_matrix",
    "FeatureStore",
]
