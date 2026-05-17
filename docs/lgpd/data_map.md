# Mapa de dados sensíveis (LGPD)

**Referência legal:** Lei 13.709/2018 (LGPD) · Arts. 5º, 11º, 16º e 20º

---

## Dados coletados e tratamento

| Campo | Classificação LGPD | Armazenado? | Tratamento |
|---|---|---|---|
| CPF | Dado pessoal (Art. 5º, I) | Nunca | Pseudonimizado via HMAC-SHA256 antes de qualquer uso |
| Nome completo | Dado pessoal (Art. 5º, I) | Nunca | Não entra no pipeline de ML |
| Data de nascimento | Dado pessoal (Art. 5º, I) | Nunca | Apenas idade derivada (não identificável) |
| Imagem facial | **Dado sensível biométrico (Art. 11º)** | **Nunca** | Score do provedor armazenado; imagem descartada imediatamente |
| Localização GPS | Dado pessoal (contextual) | Nunca | Convertida em features derivadas (distância, país) |
| Device ID | Dado pessoal (vinculado à identidade) | Hash | Pseudonimizado via HMAC-SHA256 |
| IP | Dado pessoal (contextual) | Hash | Pseudonimizado; geolocalização convertida em features |
| Score de biometria | Dado derivado (não pessoal) | Sim | Retorno do provedor — sem PII |
| Score de device | Dado derivado (não pessoal) | Sim | Retorno do provedor — sem PII |

---

## Finalidade e base legal

- **Finalidade:** Prevenção à fraude no onboarding (legítimo interesse — Art. 7º, IX)
- **Titular:** Cliente pessoa física em processo de onboarding
- **Controlador:** Instituição financeira
- **Operadores:** Provedores de biometria, device intelligence e geolocalização

---

## Direito de explicação (Art. 20º)

Toda decisão automatizada de rejeição pode ser explicada ao titular mediante
solicitação. A API retorna os top 3 fatores de risco via SHAP values na resposta
padrão, sem exposição de dados pessoais.

---

## Retenção de dados

| Tipo de dado | Retenção | Base legal |
|---|---|---|
| Logs de auditoria de decisão | 5 anos | Res. BCB 85/2021 |
| Features pseudonimizadas (Redis) | TTL 1–24h | Necessidade operacional |
| Artefatos de modelo (MLflow) | Indefinido | Rastreabilidade regulatória |
| Dados de treinamento pseudonimizados | 2 anos | Retraining e auditoria |

---

## Incidentes e notificação

Em caso de incidente de segurança envolvendo dados de titulares, a ANPD deve
ser notificada em até 2 dias úteis (Art. 48º LGPD), com descrição dos dados
afetados, número de titulares e medidas adotadas.
