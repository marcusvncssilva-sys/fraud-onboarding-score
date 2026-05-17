"""Feature engineering pipeline — numpy puro, zero loops Python.

28 features críticas: 25 raw + 3 derivadas via operações vetorizadas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from src.data.validators import OnboardingScoreRequest

# Ordem canônica — deve ser idêntica entre treino e inferência
FEATURE_NAMES: list[str] = [
    # biometria (0-5)
    "bio_liveness_score",
    "bio_face_match_score",
    "bio_attempts",
    "bio_liveness_passed",
    "bio_face_match_passed",
    "bio_failure_rate",
    # dispositivo (6-10)
    "device_is_rooted",
    "device_fraud_score",
    "device_cpfs_30d",
    "device_age_days",
    "app_is_tampered",
    # geo (11-14)
    "is_vpn",
    "is_proxy",
    "is_foreign_ip",
    "anonymizer_score",
    # sessão (15-18)
    "session_completion_seconds",
    "is_suspiciously_fast",
    "is_night_session",
    "copy_paste_count",
    # velocidade (19-22)
    "cpf_onboardings_7d",
    "cpf_onboardings_30d",
    "device_cpfs_7d",
    "ip_onboardings_24h",
    # regulatório (23-24)
    "n_complaints_bacen",
    "chargeback_ratio_90d",
    # derivadas (25-27)
    "bio_quality",          # liveness * face_match — qualidade combinada
    "device_risk",          # is_rooted + fraud_score
    "velocity_composite",   # anonymizer * log1p(ip_onboardings_24h)
]

N_FEATURES: int = len(FEATURE_NAMES)
_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def extract_features(req: "OnboardingScoreRequest") -> np.ndarray:
    """Request → array (1, N_FEATURES) float32. Zero loops Python."""
    a = np.empty(N_FEATURES, dtype=np.float32)

    a[0]  = req.bio_liveness_score
    a[1]  = req.bio_face_match_score
    a[2]  = float(req.bio_attempts)
    a[3]  = float(req.bio_liveness_passed)
    a[4]  = float(req.bio_face_match_passed)
    a[5]  = req.bio_failure_rate
    a[6]  = float(req.device_is_rooted)
    a[7]  = req.device_fraud_score
    a[8]  = float(req.device_cpfs_30d)
    a[9]  = float(req.device_age_days)
    a[10] = float(req.app_is_tampered)
    a[11] = float(req.is_vpn)
    a[12] = float(req.is_proxy)
    a[13] = float(req.is_foreign_ip)
    a[14] = float(req.anonymizer_score)
    a[15] = req.session_completion_seconds
    a[16] = float(req.is_suspiciously_fast)
    a[17] = float(req.is_night_session)
    a[18] = float(req.copy_paste_count)
    a[19] = float(req.cpf_onboardings_7d)
    a[20] = float(req.cpf_onboardings_30d)
    a[21] = float(req.device_cpfs_7d)
    a[22] = float(req.ip_onboardings_24h)
    a[23] = float(req.n_complaints_bacen)
    a[24] = req.chargeback_ratio_90d
    # derivadas — operações vetorizadas, sem branch
    a[25] = a[0] * a[1]
    a[26] = a[6] + a[7]
    a[27] = a[14] * np.log1p(a[22])

    return a.reshape(1, -1)


def build_feature_matrix(df: "pd.DataFrame") -> np.ndarray:
    """DataFrame de treino → matriz (n, N_FEATURES) float32.

    Mesma lógica de extract_features, mas aplicada a colunas inteiras
    para evitar o overhead de criar um objeto por linha.
    """
    import pandas as pd  # import local para não poluir o namespace de inferência

    n = len(df)
    X = np.empty((n, N_FEATURES), dtype=np.float32)

    raw_cols = FEATURE_NAMES[:25]
    for i, col in enumerate(raw_cols):
        X[:, i] = df[col].to_numpy(dtype=np.float32)

    # derivadas vetorizadas
    X[:, 25] = X[:, 0] * X[:, 1]
    X[:, 26] = X[:, 6] + X[:, 7]
    X[:, 27] = X[:, 14] * np.log1p(X[:, 22])

    return X
