# Model Card: Fraud Onboarding Score

**Versão:** 1.0.0 | **Data:** 2024-01 | **Equipe:** Data Science — Prevenção a Fraudes

---

## Objetivo e escopo

Score de risco (0–1000) para onboarding de novos clientes, integrando sinais de
biometria facial, device intelligence, geolocalização e dados regulatórios (DICT Bacen).

**Este modelo DEVE ser usado para:**
- Decisão de aprovação/revisão/rejeição no fluxo de onboarding de clientes PF

**Este modelo NÃO deve ser usado para:**
- Scoring de crédito ou precificação de produtos
- Decisões sobre clientes existentes (churno, fraude transacional)
- Clientes PJ (features não calibradas para este segmento)
- Qualquer decisão fora do contexto de onboarding

---

## Performance (holdout set — 20% dos dados, período de validação)

| Métrica | Valor | Referência de mercado |
|---|---|---|
| KS (Kolmogorov-Smirnov) | *a preencher após treino* | > 0.40 = Excelente |
| ROC-AUC | *a preencher* | > 0.90 = Excelente |
| PR-AUC | *a preencher* | Baseline = taxa de fraude |
| Recall @ Precision=80% | *a preencher* | Alvo interno: > 0.70 |
| Fraud capture rate | *a preencher* | Alvo interno: > 0.80 |
| False positive rate | *a preencher* | Alvo interno: < 0.10 |

---

## Dados de treinamento

- **Período:** A definir com dados reais
- **Volume:** 100.000 eventos sintéticos (pré-produção)
- **Taxa de fraude:** 2% (simulada — calibrar com taxa real após go-live)
- **Features:** 28 features (ver `configs/feature_config.yaml`)
- **Tratamento LGPD:** Todos os dados PII pseudonimizados antes do treinamento

---

## Arquitetura

- **Champion:** LightGBM Ensemble (LightGBM 0.5 + XGBoost 0.3 + CatBoost 0.2)
- **Calibração:** Platt Scaling (CalibratedClassifierCV, cv=5)
- **Explicabilidade:** SHAP TreeExplainer sobre LightGBM base
- **Score:** Probabilidade calibrada → Score 0-1000 (inverso da probabilidade de fraude)
- **Decisão:** Dual-threshold (approve < 0.15 | 0.15–0.60 review | reject > 0.60)

---

## Vieses e limitações conhecidas

- Performance pode degradar durante campanhas de aquisição (volume anômalo)
- Populações sem histórico de device (device novo) têm maior incerteza de predição
- O modelo não detecta fraude de identidade sintética sem sinal biométrico
- Features de velocidade dependem da disponibilidade do Redis — degradação em falha

---

## Monitoramento e retraining

| Gatilho | Ação |
|---|---|
| PSI > 0.20 em qualquer feature crítica | Investigar + possível retraining |
| KS produção cai > 15% vs. validação | Retraining imediato |
| Taxa de fraude real diverge > 50% do esperado | Recalibrar thresholds |
| Retraining periódico | Semanal (pipeline automático, segunda-feira 02:00) |

---

## Conformidade e governança

- **LGPD Art. 5º e 11º:** Dados biométricos tratados apenas como score — imagem não armazenada
- **LGPD Art. 20º:** Explicabilidade disponível via SHAP (top 3 fatores por decisão)
- **Res. BCB 85/2021:** Logs de auditoria imutáveis, retenção mínima 5 anos
- **Viés:** Sem features de gênero, raça, religião ou outras categorias protegidas
- **Aprovação:** Pendente aprovação do Comitê de Modelos

---

## Contato

Time de Prevenção a Fraudes — Data Science
