# Changelog

Todas as mudanças relevantes deste projeto são documentadas aqui.
Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/).

---

## [Unreleased]

### Adicionado — Sistema completo (semanas 1–3)

**API e latência**
- Latência alvo revisada: p95 < 100ms (antes: 500ms)
- `src/api/lifespan.py` — startup separado: carrega modelo + SHAP + Redis antes do primeiro request
- Hot path sem I/O síncrono — logging e cache write em `BackgroundTask`
- Hard rules com early exit antes do modelo (0.1ms para casos óbvios)

**Modelo**
- `src/models/champion/lgbm_model.py` — LightGBM singleton, max 50 árvores, warm-up no startup
- `src/models/champion/shap_explainer.py` — SHAP aproximado top-3, cacheado no Redis
- `src/models/persistence.py` — save/load com validação de features, rename atômico, warm-up JIT
- Calibração Platt com `cv="prefit"` (1 modelo, sem overhead de folds em inferência)

**Feature engineering**
- `src/features/pipeline.py` — 28 features (25 raw + 3 derivadas), numpy puro, zero loops Python
- `src/features/store.py` — Redis async, TTL 5min, degradação graciosa, batch mget

**Dados sintéticos**
- `src/data/generators/customer_generator.py` — 100k amostras, distribuições realistas, seed determinístico
- `scripts/generate_synthetic_data.py` — CLI funcional
- `scripts/train_model.py` — treino completo com avaliação KS/AUC e benchmark de latência

**Testes**
- 113 testes passando, cobertura 85%
- `tests/integration/test_latency.py` — garante p95 < 100ms (API), p95 < 15ms (modelo), p95 < 1ms (features)
- `tests/unit/test_features.py`, `test_data_generator.py`, `test_model_components.py`, `test_feature_store.py`

### Pendente
- Monitoramento Evidently AI (drift detection)
- Integração com MLflow Model Registry para versionamento automático

---

## [0.1.0] — 2024-01-XX

- Setup inicial do projeto
