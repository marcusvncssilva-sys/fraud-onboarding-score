## Descrição

<!-- O que esse PR faz? Por que essa mudança é necessária? -->

## Tipo de mudança

- [ ] Bug fix
- [ ] Nova feature
- [ ] Refactoring (sem mudança de comportamento)
- [ ] Feature engineering (nova feature no modelo)
- [ ] Mudança no modelo (hiperparâmetros, arquitetura)
- [ ] Documentação
- [ ] CI/CD

## Checklist

- [ ] Código segue os padrões do projeto (black, ruff, mypy passando)
- [ ] Testes adicionados/atualizados para cobrir as mudanças
- [ ] Coverage mantida acima de 75%
- [ ] Documentação atualizada (docstrings, README se necessário)
- [ ] Sem dados pessoais reais no código ou testes
- [ ] Sem secrets ou chaves hardcoded
- [ ] ADR criado se houve decisão arquitetural relevante

## Para mudanças no modelo

- [ ] MLflow tracking configurado para o experimento
- [ ] Métricas documentadas (KS, AUC, PR-AUC)
- [ ] Comparação com o modelo atual (champion vs challenger)
- [ ] SHAP analysis realizada
- [ ] Impacto de negócio estimado

## Evidências

<!-- Screenshots, gráficos, outputs de terminal relevantes -->

## Como testar

```bash
# Comandos para testar as mudanças
make test
```
