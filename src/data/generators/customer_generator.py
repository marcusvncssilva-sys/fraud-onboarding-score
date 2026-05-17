"""Gerador de dados sintéticos de onboarding bancário.

Produz amostras realistas para treino do modelo LightGBM.
Clientes legítimos e fraudadores têm distribuições distintas e correlacionadas.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

import numpy as np
import pandas as pd


def generate_onboarding_dataset(
    n_samples: int = 100_000,
    fraud_rate: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Gera DataFrame com n_samples linhas e coluna `is_fraud` (0/1).

    Proporção de fraude é controlada por fraud_rate.
    Seed garante reprodutibilidade completa.
    """
    rng = np.random.default_rng(seed)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    legit = _gen_legitimate(n_legit, rng)
    fraud = _gen_fraudulent(n_fraud, rng)

    df = pd.concat([legit, fraud], ignore_index=True)
    # shuffle sem viés de ordenação
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _cpf_hashes(n: int, rng: np.random.Generator) -> list[str]:
    ids = rng.integers(10_000_000_000, 99_999_999_999, size=n)
    return [hashlib.sha256(str(v).encode()).hexdigest() for v in ids]


def _device_hashes(n: int, rng: np.random.Generator) -> list[str]:
    ids = rng.integers(100_000, 999_999, size=n)
    return [f"dev_{v:06d}" for v in ids]


# ── geradores ─────────────────────────────────────────────────────────────────

def _gen_legitimate(n: int, rng: np.random.Generator) -> pd.DataFrame:
    d: dict[str, Any] = {}

    d["cpf_hash"] = _cpf_hashes(n, rng)
    d["device_id_hash"] = _device_hashes(n, rng)

    # biometria — alta qualidade
    d["bio_liveness_score"]    = rng.beta(20, 2, n).clip(0, 1)
    d["bio_face_match_score"]  = rng.beta(18, 2, n).clip(0, 1)
    d["bio_attempts"]          = rng.choice([1, 1, 1, 1, 2, 2, 3], size=n).astype(int)
    d["bio_liveness_passed"]   = (np.array(d["bio_liveness_score"]) > 0.5).astype(bool)
    d["bio_face_match_passed"] = (np.array(d["bio_face_match_score"]) > 0.5).astype(bool)
    d["bio_failure_rate"]      = rng.beta(1, 10, n)

    # dispositivo — baixo risco
    d["device_is_rooted"]   = rng.random(n) < 0.02
    d["device_is_emulator"] = np.zeros(n, dtype=bool)
    d["device_fraud_score"] = rng.beta(1, 20, n)
    d["device_cpfs_30d"]    = rng.choice([1, 1, 1, 2, 3], size=n).astype(int)
    d["device_age_days"]    = rng.integers(30, 2000, n)
    d["app_is_tampered"]    = np.zeros(n, dtype=bool)

    # geo — baixo anonimizador
    d["is_vpn"]       = rng.random(n) < 0.05
    d["is_tor"]       = np.zeros(n, dtype=bool)
    d["is_proxy"]     = rng.random(n) < 0.03
    d["is_foreign_ip"] = rng.random(n) < 0.02
    d["distance_from_cep_km"] = rng.exponential(5, n)
    anon = (
        np.array(d["is_vpn"], dtype=int)
        + np.array(d["is_tor"], dtype=int)
        + np.array(d["is_proxy"], dtype=int)
    )
    d["anonymizer_score"] = anon.clip(0, 3).astype(int)

    # sessão — comportamento humano normal
    d["session_completion_seconds"] = rng.normal(200, 60, n).clip(30, 600)
    d["is_suspiciously_fast"] = (np.array(d["session_completion_seconds"]) < 30).astype(bool)
    d["is_night_session"]     = rng.random(n) < 0.12
    d["copy_paste_count"]     = rng.choice([0, 0, 0, 1, 2], size=n, p=[0.70, 0.15, 0.10, 0.03, 0.02]).astype(int)
    d["typing_speed_cv"]      = rng.beta(5, 5, n)

    # velocidade — baixa
    d["cpf_onboardings_7d"]  = rng.choice([0, 1, 2], size=n, p=[0.90, 0.08, 0.02]).astype(int)
    d["cpf_onboardings_30d"] = (
        np.array(d["cpf_onboardings_7d"])
        + rng.choice([0, 1, 2], size=n, p=[0.80, 0.15, 0.05]).astype(int)
    )
    d["device_cpfs_7d"]     = rng.choice([1, 2, 3], size=n, p=[0.85, 0.12, 0.03]).astype(int)
    d["ip_onboardings_24h"] = rng.choice([0, 1, 2, 3], size=n, p=[0.80, 0.12, 0.05, 0.03]).astype(int)

    # regulatório — perfil saudável
    pix = rng.random(n) > 0.30
    d["has_pix_key"]         = pix
    d["pix_key_age_days"]    = np.where(pix, rng.integers(30, 1500, n), np.nan)
    d["n_complaints_bacen"]  = rng.choice([0, 0, 0, 1, 2], size=n, p=[0.85, 0.08, 0.04, 0.02, 0.01]).astype(int)
    d["is_in_cadin"]         = rng.random(n) < 0.05
    d["chargeback_ratio_90d"] = rng.beta(1, 50, n)

    d["is_fraud"] = np.zeros(n, dtype=int)
    return pd.DataFrame(d)


def _gen_fraudulent(n: int, rng: np.random.Generator) -> pd.DataFrame:
    d: dict[str, Any] = {}

    d["cpf_hash"] = _cpf_hashes(n, rng)
    d["device_id_hash"] = _device_hashes(n, rng)

    # biometria — baixa qualidade, múltiplas tentativas
    d["bio_liveness_score"]    = rng.beta(3, 5, n).clip(0, 1)
    d["bio_face_match_score"]  = rng.beta(3, 7, n).clip(0, 1)
    d["bio_attempts"]          = rng.choice([2, 3, 4, 5], size=n, p=[0.30, 0.30, 0.20, 0.20]).astype(int)
    d["bio_liveness_passed"]   = (np.array(d["bio_liveness_score"]) > 0.5).astype(bool)
    d["bio_face_match_passed"] = (np.array(d["bio_face_match_score"]) > 0.5).astype(bool)
    d["bio_failure_rate"]      = rng.beta(5, 3, n)

    # dispositivo — alto risco
    d["device_is_rooted"]   = rng.random(n) < 0.40
    d["device_is_emulator"] = rng.random(n) < 0.55
    d["device_fraud_score"] = rng.beta(8, 3, n)
    d["device_cpfs_30d"]    = rng.integers(3, 20, n)
    d["device_age_days"]    = rng.integers(0, 90, n)
    d["app_is_tampered"]    = rng.random(n) < 0.45

    # geo — anonimizadores frequentes
    d["is_vpn"]       = rng.random(n) < 0.50
    d["is_tor"]       = rng.random(n) < 0.35
    d["is_proxy"]     = rng.random(n) < 0.40
    d["is_foreign_ip"] = rng.random(n) < 0.30
    d["distance_from_cep_km"] = rng.exponential(80, n)
    anon = (
        np.array(d["is_vpn"], dtype=int)
        + np.array(d["is_tor"], dtype=int)
        + np.array(d["is_proxy"], dtype=int)
    )
    d["anonymizer_score"] = anon.clip(0, 3).astype(int)

    # sessão — automatizada, rápida
    d["session_completion_seconds"] = rng.normal(20, 8, n).clip(5, 120)
    d["is_suspiciously_fast"] = (np.array(d["session_completion_seconds"]) < 30).astype(bool)
    d["is_night_session"]     = rng.random(n) < 0.55
    d["copy_paste_count"]     = rng.integers(3, 20, n)
    d["typing_speed_cv"]      = rng.beta(2, 8, n)

    # velocidade — alta
    d["cpf_onboardings_7d"]  = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.20, 0.20, 0.30, 0.20, 0.10]).astype(int)
    d["cpf_onboardings_30d"] = np.array(d["cpf_onboardings_7d"]) + rng.integers(2, 8, n)
    d["device_cpfs_7d"]      = rng.integers(3, 15, n)
    d["ip_onboardings_24h"]  = rng.integers(2, 30, n)

    # regulatório — perfil irregular
    pix = rng.random(n) > 0.70
    d["has_pix_key"]         = pix
    d["pix_key_age_days"]    = np.where(pix, rng.integers(0, 30, n), np.nan)
    d["n_complaints_bacen"]  = rng.integers(2, 10, n)
    d["is_in_cadin"]         = rng.random(n) < 0.50
    d["chargeback_ratio_90d"] = rng.beta(5, 3, n)

    d["is_fraud"] = np.ones(n, dtype=int)
    return pd.DataFrame(d)
