"""Métricas de avaliação de modelo e impacto de negócio."""

from typing import Optional

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def compute_ks(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Kolmogorov-Smirnov — principal métrica de discriminação no mercado bancário.

    Mede a separação máxima entre distribuição de scores de bons e maus.

    Referência de qualidade:
        > 0.40 → Excelente
        0.30–0.40 → Bom
        0.20–0.30 → Aceitável
        < 0.20 → Fraco
    """
    fraud_scores = y_proba[y_true == 1]
    legit_scores = y_proba[y_true == 0]
    ks_stat, _ = stats.ks_2samp(legit_scores, fraud_scores)
    return float(ks_stat)


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Population Stability Index (PSI) — estabilidade da distribuição de features.

    Referência:
        PSI < 0.10 → Estável
        0.10–0.20 → Mudança moderada (monitorar)
        > 0.20 → Mudança significativa (investigar / retreinar)
    """
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bins)

    ref_pct = ref_counts / len(reference) + 1e-10
    cur_pct = cur_counts / len(current) + 1e-10

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_full_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:  # type: ignore[type-arg]
    """Suite completa de métricas para modelo antifraude."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_proba)

    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "ks": round(compute_ks(y_true, y_proba), 4),
        "threshold": threshold,
        "precision": round(tp / (tp + fp) if (tp + fp) > 0 else 0.0, 4),
        "recall": round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4),
        "specificity": round(tn / (tn + fp) if (tn + fp) > 0 else 0.0, 4),
        "f1": round(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0, 4),
        "false_positive_rate": round(fp / (fp + tn) if (fp + tn) > 0 else 0.0, 4),
        "false_negative_rate": round(fn / (fn + tp) if (fn + tp) > 0 else 0.0, 4),
        "fraud_capture_rate": round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def compute_business_impact(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    avg_fraud_value_brl: float = 5_000.0,
    avg_customer_ltv_brl: float = 150.0,
    review_cost_brl: float = 25.0,
    review_fraud_capture_rate: float = 0.80,
    threshold_approve: float = 0.15,
    threshold_reject: float = 0.60,
) -> dict:  # type: ignore[type-arg]
    """Simula impacto financeiro das decisões do modelo.

    Essencial para apresentação executiva e justificativa de threshold.
    """
    approve_mask = y_proba < threshold_approve
    reject_mask = y_proba >= threshold_reject
    review_mask = ~approve_mask & ~reject_mask

    approved_fraud = int(y_true[approve_mask].sum())
    rejected_fraud = int(y_true[reject_mask].sum())
    rejected_legit = int((1 - y_true)[reject_mask].sum())
    review_fraud = int(y_true[review_mask].sum())

    review_fraud_captured = review_fraud * review_fraud_capture_rate

    fraud_prevented_brl = (rejected_fraud + review_fraud_captured) * avg_fraud_value_brl
    fraud_loss_brl = approved_fraud * avg_fraud_value_brl
    revenue_lost_brl = rejected_legit * avg_customer_ltv_brl
    review_cost_total_brl = int(review_mask.sum()) * review_cost_brl
    net_benefit_brl = fraud_prevented_brl - fraud_loss_brl - revenue_lost_brl - review_cost_total_brl

    total_fraud = max(int(y_true.sum()), 1)
    total_legit = max(int((1 - y_true).sum()), 1)

    return {
        "total_cases": len(y_true),
        "auto_approved": int(approve_mask.sum()),
        "auto_rejected": int(reject_mask.sum()),
        "sent_to_review": int(review_mask.sum()),
        "fraud_prevented_brl": round(fraud_prevented_brl, 2),
        "fraud_loss_brl": round(fraud_loss_brl, 2),
        "revenue_lost_brl": round(revenue_lost_brl, 2),
        "review_cost_brl": round(review_cost_total_brl, 2),
        "net_benefit_brl": round(net_benefit_brl, 2),
        "fraud_capture_rate": round((rejected_fraud + review_fraud_captured) / total_fraud, 4),
        "false_positive_rate": round(rejected_legit / total_legit, 4),
    }
