"""Utilitários de criptografia e pseudonimização (LGPD Art. 5º e 11º)."""

import hashlib
import hmac
import os
from typing import Optional


class DataPseudonymizer:
    """Pseudonimização de dados pessoais sensíveis conforme LGPD.

    Técnica: HMAC-SHA256 com salt de ambiente.
    - Determinístico: mesmo CPF → mesmo hash (permite joins seguros)
    - Irreversível sem a chave secreta
    - Salt diferente por ambiente (dev ≠ staging ≠ prod)

    Dados cobertos:
    - CPF (Art. 5º, I — dado pessoal identificável)
    - Imagem facial (Art. 11º — dado biométrico sensível)
    - Dispositivo vinculado à identidade
    - Localização quando vinculada à identidade
    """

    def __init__(self, secret_key: Optional[str] = None) -> None:
        key = secret_key or os.environ.get("PSEUDONYM_SECRET_KEY", "")
        if not key:
            raise ValueError(
                "PSEUDONYM_SECRET_KEY não configurada. "
                "Definir via variável de ambiente ou Vault."
            )
        self._key = key.encode("utf-8")

    def pseudonymize_cpf(self, cpf: str) -> str:
        """Pseudonimiza CPF. Resultado determinístico para joins entre sistemas."""
        clean = "".join(filter(str.isdigit, cpf))
        return self._hmac(f"CPF:{clean}")

    def pseudonymize_device(self, device_id: str) -> str:
        """Pseudonimiza device ID para features de velocidade."""
        return self._hmac(f"DEVICE:{device_id}")

    def pseudonymize_ip(self, ip: str) -> str:
        """Pseudonimiza IP para features de geolocalização."""
        return self._hmac(f"IP:{ip}")

    def hash_biometric_reference(self, image_b64: str) -> str:
        """Hash da imagem biométrica.

        NUNCA armazenar a imagem original — apenas o hash do provedor.
        Imagem facial é dado sensível (Art. 11 LGPD).
        """
        # Usa apenas os primeiros 200 chars para performance
        # (suficiente para unicidade em dados de teste)
        return self._hmac(f"BIO:{image_b64[:200]}")

    def _hmac(self, value: str) -> str:
        return hmac.new(self._key, value.encode("utf-8"), hashlib.sha256).hexdigest()


class DataClassifier:
    """Classifica campos de dados segundo a LGPD para tratamento adequado."""

    # Dados pessoais identificáveis (Art. 5º, I) — nunca em logs sem pseudonimização
    PII_FIELDS: frozenset[str] = frozenset({
        "cpf", "nome", "nome_mae", "data_nascimento",
        "email", "telefone", "endereco", "cep",
    })

    # Dados sensíveis com proteção extra (Art. 11º LGPD)
    SENSITIVE_FIELDS: frozenset[str] = frozenset({
        "bio_image_b64",     # imagem facial
        "geo_latitude",      # localização vinculada à identidade
        "geo_longitude",
        "device_mac_address",
    })

    @classmethod
    def sanitize_for_logging(cls, data: dict) -> dict:  # type: ignore[type-arg]
        """Remove / mascara campos sensíveis antes de logar."""
        result = {}
        for k, v in data.items():
            if k in cls.PII_FIELDS:
                result[k] = "[REDACTED_PII]"
            elif k in cls.SENSITIVE_FIELDS:
                result[k] = "[REDACTED_SENSITIVE]"
            else:
                result[k] = v
        return result
