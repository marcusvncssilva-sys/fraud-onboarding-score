# ADR-001: Escolha do modelo champion — LightGBM Ensemble

**Status:** Aceito
**Data:** 2024-01
**Decisores:** Time de Data Science — Prevenção a Fraudes

---

## Contexto

Precisamos de um modelo de classificação binária (fraude/legítimo) para onboarding
com os seguintes requisitos não-negociáveis:

- Latência de inferência p95 < 100ms (para servir em real-time dentro do SLA de 500ms da API)
- Explicabilidade por requisição (Art. 20 LGPD — direito à explicação de decisões automatizadas)
- Alta discriminação em dados fortemente desbalanceados (~2% fraude)
- Suporte nativo a features mistas (numéricas + categóricas de alta cardinalidade)

---

## Alternativas avaliadas

| Modelo | ROC-AUC | Latência p99 | Explicabilidade | Veredicto |
|---|---|---|---|---|
| Regressão Logística | 0.82 | 3ms | Coeficientes nativos | Descartado — performance insuficiente |
| Random Forest | 0.90 | 45ms | SHAP (lento) | Descartado — latência alta + sem regularização L1/L2 |
| **LightGBM** | **0.94** | **12ms** | **SHAP nativo (TreeExplainer)** | **Champion** |
| XGBoost | 0.93 | 18ms | SHAP nativo | Challenger no ensemble |
| CatBoost | 0.92 | 22ms | SHAP nativo | Challenger (melhor em categóricas) |
| MLP (Rede Neural) | 0.93 | 35ms | SHAP (aproximado) | Descartado — explicabilidade limitada |

---

## Decisão

**LightGBM como modelo base do ensemble**, com XGBoost e CatBoost como challengers.

Arquitetura final:
- `VotingClassifier(lgbm=0.5, xgb=0.3, catboost=0.2)` — soft voting
- `CalibratedClassifierCV(method='sigmoid', cv=5)` — Platt Scaling pós-ensemble
- `TreeExplainer(lgbm_fitted)` — SHAP sobre o LightGBM base (mais rápido que ensemble completo)

Tratamento de desbalanceamento via `scale_pos_weight=49` (sem SMOTE — dado sintético
introduz ruído em features de alta dimensionalidade como biometria).

---

## Consequências

**Positivas:**
- Melhor trade-off performance/latência entre os candidatos
- SHAP TreeExplainer é O(n·log n) — compatível com latência < 100ms
- `scale_pos_weight` nativo elimina necessidade de oversampling
- MLflow tracking simplificado (LightGBM tem serialização nativa)

**Negativas:**
- Ensemble aumenta complexidade de deploy vs. modelo único (~3x o tamanho em disco)
- Retraining do ensemble é ~2.5x mais lento que modelo único (mitiga: pipeline paralelo)

**Riscos e mitigações:**
- Ensemble propenso a overfitting: mitigado por calibração e `min_child_samples=100`
- SHAP do ensemble completo seria lento: mitigado usando explainer apenas no LightGBM base

---

## Revisão

Esta decisão deve ser revisada se:
- KS em produção cair consistentemente abaixo de 0.40 por 2+ semanas
- Novos tipos de fraude emergirem que não sejam capturáveis por gradient boosting tabular
- Latência p99 ultrapassar 150ms em produção (possível com feature store lento)
