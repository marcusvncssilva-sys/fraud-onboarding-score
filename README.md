# 🏦 Fraud Onboarding Score

> Sistema de score antifraude em tempo real para onboarding de clientes bancários.
> Score 0–1000 · p95 < 100ms · Explicável via SHAP · LGPD-compliant

[![CI](https://github.com/seu-usuario/fraud-onboarding-score/actions/workflows/ci.yml/badge.svg)](https://github.com/seu-usuario/fraud-onboarding-score/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen)](htmlcov/index.html)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![LGPD](https://img.shields.io/badge/LGPD-compliant-green)](docs/lgpd/)
[![Model KS](https://img.shields.io/badge/KS-0.58-blue)](docs/model_card.md)

---

## O que este projeto resolve

Instituições financeiras enfrentam uma janela crítica de risco no onboarding:
o momento em que um novo cliente se identifica e solicita acesso a produtos bancários.
Fraudadores exploram esse momento com identidades falsas, dispositivos compartilhados
e biometrias adulteradas.

Este sistema integra sinais de **biometria facial**, **device intelligence**,
**geolocalização** e **dados regulatórios (DICT Bacen)** em um score único e
interpretável, servido em tempo real com explicabilidade por decisão.

---

## Performance do modelo

| Métrica | Valor | Referência |
|---|---|---|
| KS (Kolmogorov-Smirnov) | 0.58 | > 0.40 = Excelente |
| ROC-AUC | 0.94 | > 0.90 = Excelente |
| PR-AUC | 0.81 | Baseline = taxa de fraude |
| Fraudes capturadas (threshold otimizado) | 87% | Alvo: > 80% |
| Taxa de falso positivo | 4.2% | Alvo: < 10% |
| Benefício líquido simulado | R$ 2.1M / 100k clientes | — |

---

## Arquitetura

```
App Mobile / Web
       │
       ▼
 API Gateway (FastAPI)          ← autenticação JWT, rate limiting, logging
       │
  ┌────┴─────────────────────────────────┐
  │         Feature Engineering          │
  │  Biometria · Device · Geo · DICT    │
  └────────────────┬─────────────────────┘
                   │
            Feature Store
          Redis (online < 5ms)
          S3/Parquet (offline)
                   │
           Scoring Engine
      LightGBM Ensemble Calibrado
         SHAP Explainer
                   │
          Decision Engine
     Hard Rules + Dual-Threshold
          ┌────┬────┐
      APPROVED REVIEW REJECTED
```

Detalhes: [docs/architecture/system_design.md](docs/architecture/system_design.md)

---

## Stack tecnológica

| Camada | Tecnologia |
|---|---|
| API | FastAPI + Uvicorn |
| Modelo | LightGBM + XGBoost + CatBoost (ensemble) |
| Calibração | Platt Scaling (CalibratedClassifierCV) |
| Explicabilidade | SHAP TreeExplainer |
| Feature Store | Feast + Redis (online) + S3 (offline) |
| MLOps | MLflow (tracking + registry) |
| Monitoramento | Evidently AI (drift) + Prometheus + Grafana |
| Containerização | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Qualidade | ruff, black, mypy strict, bandit, pytest |

---

## Quick start

### Pré-requisitos

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- Docker + Docker Compose

### 1. Instalar dependências

```bash
git clone https://github.com/seu-usuario/fraud-onboarding-score.git
cd fraud-onboarding-score
make install
```

### 2. Configurar ambiente

```bash
cp configs/env/dev.env .env
# Editar .env se necessário (padrões funcionam para dev local)
```

### 3. Subir serviços (API + Redis + MLflow)

```bash
make docker-run
```

### 4. Testar a API

```bash
# Health check
curl http://localhost:8000/health

# Score de onboarding
curl -X POST http://localhost:8000/v1/score/onboarding \
  -H "Content-Type: application/json" \
  -d '{
    "cpf_hash": "'"$(python -c "print('a'*64)")"'",
    "device_id_hash": "dev_abc123",
    "bio_liveness_score": 0.95,
    "bio_face_match_score": 0.92,
    "bio_attempts": 1,
    "bio_liveness_passed": true,
    "bio_face_match_passed": true,
    "bio_failure_rate": 0.0,
    "device_is_rooted": false,
    "device_is_emulator": false,
    "device_fraud_score": 0.05,
    "device_cpfs_30d": 1,
    "device_age_days": 365,
    "app_is_tampered": false,
    "is_vpn": false,
    "is_tor": false,
    "is_proxy": false,
    "is_foreign_ip": false,
    "anonymizer_score": 0,
    "session_completion_seconds": 180,
    "is_suspiciously_fast": false,
    "is_night_session": false,
    "copy_paste_count": 0,
    "cpf_onboardings_7d": 0,
    "cpf_onboardings_30d": 0,
    "device_cpfs_7d": 1,
    "ip_onboardings_24h": 0,
    "has_pix_key": true,
    "n_complaints_bacen": 0,
    "is_in_cadin": false,
    "chargeback_ratio_90d": 0.0
  }'
```

**Resposta esperada:**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "score": 923,
  "risk_band": "BAIXO",
  "decision": "APPROVED",
  "model_version": "lgbm-ensemble-v1.0.0",
  "latency_ms": 47.3,
  "top_risk_factors": []
}
```

Documentação interativa: **http://localhost:8000/docs**

---

## Desenvolvimento

```bash
make test          # Roda todos os testes com coverage
make lint          # Lint + type check + security scan
make format        # Auto-formata o código
make generate-data # Gera 100k amostras sintéticas
make train         # Treina o modelo champion
make mlflow-ui     # Abre MLflow em http://localhost:5000
make drift-check   # Relatório de data drift
```

---

## Estrutura do projeto

```
fraud-onboarding-score/
├── .github/           # CI/CD (GitHub Actions) + PR template
├── configs/           # Hiperparâmetros, features, thresholds, envs
├── data/              # Dados sintéticos (nunca dados reais no repo)
├── docs/
│   ├── architecture/  # Diagrama de sistema + ADRs
│   ├── lgpd/          # Mapa de dados sensíveis + política de retenção
│   └── model_card.md  # Documentação completa do modelo
├── notebooks/         # EDA, feature engineering, análise de negócio
├── src/
│   ├── api/           # FastAPI (routers, middleware, schemas)
│   ├── data/          # Geradores sintéticos + validadores Pydantic
│   ├── decision/      # Decision Engine (hard rules + score ML)
│   ├── features/      # Feature engineering pipeline
│   ├── models/        # Baseline, champion, calibração, SHAP
│   ├── monitoring/    # Drift detector, PSI, performance tracker
│   ├── providers/     # Mocks de APIs externas (biometria, device, DICT)
│   └── utils/         # Crypto (LGPD), logger de auditoria, métricas
├── tests/             # unit + integration (coverage > 75%)
├── scripts/           # CLI: generate-data, train, evaluate, deploy
└── docker/            # Dockerfile.api + docker-compose.yml
```

---

## Decisões arquiteturais

As decisões de design relevantes estão documentadas como Architecture Decision Records (ADRs):

- [ADR-001: Escolha do modelo champion (LightGBM Ensemble)](docs/architecture/adr/ADR-001-model-choice.md)
- [ADR-002: Estratégia de feature store (Feast + Redis)](docs/architecture/adr/ADR-002-feature-store.md)

---

## Conformidade LGPD

| Requisito | Implementação |
|---|---|
| Art. 5º — Dados pessoais | CPF pseudonimizado via HMAC-SHA256 antes de qualquer processamento |
| Art. 11º — Dados sensíveis | Imagem biométrica nunca armazenada — apenas score do provedor |
| Art. 20º — Explicabilidade | SHAP values disponíveis por requisição (top 3 fatores na resposta) |
| Res. BCB 85/2021 — Auditoria | Logs imutáveis JSON com retenção configurável (padrão: 5 anos) |

Detalhes: [docs/lgpd/](docs/lgpd/)

---

## Monitoramento em produção

| Sinal | Ferramenta | Alerta |
|---|---|---|
| Data drift por feature | Evidently AI | PSI > 0.20 |
| Performance do modelo | KS semanal | Queda > 15% |
| Latência da API | Prometheus + Grafana | p95 > 100ms |
| Erros de provedor | Structured logs | Taxa > 5% |

---

## Model Card

Documentação completa do modelo incluindo performance, limitações, vieses e
conformidade regulatória: **[docs/model_card.md](docs/model_card.md)**

---

## Licença

Projeto de portfólio — uso educacional e demonstração técnica.
