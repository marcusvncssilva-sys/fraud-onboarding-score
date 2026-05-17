# ADR-002: Estratégia de feature store

**Status:** Aceito
**Data:** 2024-01
**Decisores:** Time de Data Science — Prevenção a Fraudes

---

## Contexto

Features de velocidade (ex: "quantos CPFs usaram este device nos últimos 7 dias")
precisam ser computadas e servidas em tempo real, com latência < 20ms para não
estourar o SLA de 500ms da API de scoring.

Existem dois tipos de feature com requisitos radicalmente diferentes:

| Tipo | Exemplo | Latência necessária | Freshness |
|---|---|---|---|
| **Online** (real-time) | `device_cpfs_7d` | < 20ms | Segundos |
| **Offline** (batch) | Histórico DICT 90d | N/A (pré-computado) | Horas |

---

## Alternativas avaliadas

| Opção | Latência online | Custo | Complexidade | Veredicto |
|---|---|---|---|---|
| Feast + Redis + S3 | < 5ms | Baixo (open source) | Média | **Escolhido** |
| Tecton | < 5ms | Alto (SaaS) | Baixa | Descartado — custo |
| Databricks Feature Store | < 10ms | Médio | Baixa | Válido se banco já usa Databricks |
| Cálculo on-the-fly | 200-500ms | Zero | Baixa | Descartado — estoura SLA |
| Redis direto (sem abstração) | < 5ms | Baixo | Alta | Descartado — sem governance |

---

## Decisão

**Feast com Redis (online store) + S3/Parquet (offline store).**

Justificativa:
- Redis garante p99 < 5ms para features de velocidade
- Feast abstrai a diferença online/offline e garante point-in-time correctness
- Open source com suporte ativo — sem lock-in
- Integra nativamente com MLflow para rastreamento de quais features foram usadas por qual modelo

TTLs configurados:
- Features de velocidade (device, IP, CPF): TTL = 1 hora
- Features DICT Bacen: TTL = 6 horas (dados mudam lentamente)
- Scores de risco de device: TTL = 24 horas

---

## Consequências

**Positivas:**
- Latência de feature serving < 5ms (Redis)
- Point-in-time correctness no treinamento (evita data leakage)
- Reutilização de features entre modelos diferentes (ex: fraude + crédito)

**Negativas:**
- Redis é um ponto de falha adicional — necessário replicação + fallback
- Feast adiciona dependência de infraestrutura ao projeto

**Mitigação de falha do Redis:**
- Se Redis indisponível: API retorna score conservador (threshold_reject = 0.40)
- Features de velocidade são assumidas como o pior caso (ex: `device_cpfs_7d = 999`)
- Alertas Prometheus + PagerDuty se Redis latência > 20ms
