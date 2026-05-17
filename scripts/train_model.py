"""Script de treinamento do modelo champion LightGBM.

Uso:
    python scripts/train_model.py [--data PATH] [--output PATH] [--n-estimators N]

O modelo treinado é salvo em models/champion.pkl e está pronto para
ser carregado pela API no startup.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o modelo champion LightGBM")
    parser.add_argument("--data", default="data/raw/onboarding_dataset.parquet")
    parser.add_argument("--output", default="models/champion.pkl")
    parser.add_argument("--n-estimators", type=int, default=50,
                        help="Máximo de árvores — 50 garante tree traversal < 10ms")
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--fraud-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── dados ─────────────────────────────────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        _log.info("Dataset não encontrado — gerando %d amostras...", args.samples)
        from src.data.generators.customer_generator import generate_onboarding_dataset
        import pandas as pd

        df = generate_onboarding_dataset(args.samples, args.fraud_rate, args.seed)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_path, index=False)
        _log.info("Dataset salvo em %s (%d linhas)", data_path, len(df))
    else:
        import pandas as pd
        df = pd.read_parquet(data_path)
        _log.info("Dataset carregado: %s (%d linhas)", data_path, len(df))

    fraud_pct = df["is_fraud"].mean() * 100
    _log.info("Taxa de fraude: %.2f%%", fraud_pct)

    # ── features ──────────────────────────────────────────────────────────────
    from src.features.pipeline import build_feature_matrix

    X = build_feature_matrix(df)
    y = df["is_fraud"].to_numpy(dtype=int)
    _log.info("Feature matrix: %s, labels: %s", X.shape, y.shape)

    # ── split ────────────────────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=args.seed
    )
    _log.info("Train: %d | Test: %d", len(X_train), len(X_test))

    # ── modelo ────────────────────────────────────────────────────────────────
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV

    scale_pos = int((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    _log.info("scale_pos_weight: %d", scale_pos)

    lgbm = LGBMClassifier(
        n_estimators=args.n_estimators,
        learning_rate=0.10,
        max_depth=5,
        num_leaves=16,
        min_child_samples=50,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=0.10,
        scale_pos_weight=scale_pos,
        objective="binary",
        metric="average_precision",
        random_state=args.seed,
        n_jobs=-1,
        verbose=-1,
    )

    _log.info("Treinando LightGBM (n_estimators=%d)...", args.n_estimators)
    t0 = time.perf_counter()

    # Fit do LightGBM base primeiro
    from sklearn.model_selection import train_test_split as _tts
    X_tr, X_cal, y_tr, y_cal = _tts(X_train, y_train, test_size=0.20, stratify=y_train, random_state=args.seed)
    lgbm.fit(X_tr, y_tr)

    # Calibração Platt com cv="prefit" — usa um único modelo, sem overhead de cv folds
    model = CalibratedClassifierCV(lgbm, method="sigmoid", cv="prefit")
    model.fit(X_cal, y_cal)
    elapsed = time.perf_counter() - t0
    _log.info("Treinamento concluído em %.1fs", elapsed)

    # ── avaliação ─────────────────────────────────────────────────────────────
    _evaluate(model, X_test, y_test)

    # ── latência de inferência ────────────────────────────────────────────────
    _benchmark_latency(model, X_test[:1])

    # ── salvar com persistência validada ─────────────────────────────────────
    from src.models.persistence import ModelPersistence

    ModelPersistence.save(
        model=model,
        path=args.output,
        version=f"lgbm-{args.n_estimators}t-v1.0.0",
        extra={
            "n_estimators": args.n_estimators,
            "train_samples": int(len(X_train)),
            "fraud_rate": float(fraud_pct / 100),
        },
    )
    _log.info("Modelo pronto para uso em %s", args.output)


def _evaluate(model: object, X_test: object, y_test: object) -> None:
    import numpy as np
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
    )

    proba = model.predict_proba(X_test)[:, 1]  # type: ignore[union-attr]
    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)
    _log.info("ROC-AUC: %.4f | PR-AUC: %.4f", roc, pr)

    # KS
    from scipy.stats import ks_2samp
    scores_fraud = proba[y_test == 1]  # type: ignore[index]
    scores_legit = proba[y_test == 0]  # type: ignore[index]
    ks_stat, _ = ks_2samp(scores_fraud, scores_legit)
    _log.info("KS: %.4f", ks_stat)

    if roc < 0.88:
        _log.warning("ROC-AUC %.4f < quality gate 0.88", roc)
    if ks_stat < 0.40:
        _log.warning("KS %.4f < quality gate 0.40", ks_stat)


def _benchmark_latency(model: object, X_single: object) -> None:
    import numpy as np

    # warm-up
    for _ in range(5):
        model.predict_proba(X_single)  # type: ignore[union-attr]

    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        model.predict_proba(X_single)  # type: ignore[union-attr]
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    p50 = times[int(len(times) * 0.50)]
    p95 = times[int(len(times) * 0.95)]
    p99 = times[int(len(times) * 0.99)]
    _log.info("Latência modelo — p50: %.2fms | p95: %.2fms | p99: %.2fms", p50, p95, p99)
    if p95 > 10:
        _log.warning("p95 %.2fms > 10ms (tree traversal budget)", p95)


if __name__ == "__main__":
    main()
