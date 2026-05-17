"""Script de avaliação do modelo.

Uso:
    python scripts/evaluate_model.py \
        --model-version latest \
        --min-ks 0.40 \
        --output reports/evaluation_report.html
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia modelo e verifica quality gates")
    parser.add_argument("--model-version", default="latest")
    parser.add_argument("--min-ks", type=float, default=0.40)
    parser.add_argument("--output", default="reports/evaluation_report.html")
    args = parser.parse_args()

    print(f"Avaliando modelo versão: {args.model_version}")
    print(f"Quality gate mínimo — KS: {args.min_ks}")

    # TODO: implementar na Semana 3
    # 1. Carregar modelo do MLflow Registry
    # 2. Carregar holdout set
    # 3. Gerar predições
    # 4. Calcular métricas completas (compute_full_report)
    # 5. Calcular impacto de negócio (compute_business_impact)
    # 6. Verificar quality gates — falhar se KS < min_ks
    # 7. Gerar relatório HTML com Evidently

    print("⚠️  Script de avaliação ainda não implementado — tarefa da Semana 3")


if __name__ == "__main__":
    main()
