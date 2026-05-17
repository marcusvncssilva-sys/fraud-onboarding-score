.PHONY: install lint test train serve docker-build docker-run generate-data evaluate drift-check ci clean

# ─── Setup ────────────────────────────────────────────────────────────────────
install:
	poetry install
	poetry run pre-commit install
	@echo "✅ Ambiente configurado com sucesso"

# ─── Qualidade de código ──────────────────────────────────────────────────────
lint:
	poetry run ruff check src/ tests/
	poetry run black --check src/ tests/
	poetry run isort --check-only src/ tests/
	poetry run mypy src/ --strict
	poetry run bandit -r src/ -ll
	@echo "✅ Lint passou"

format:
	poetry run black src/ tests/
	poetry run isort src/ tests/
	poetry run ruff check src/ tests/ --fix

# ─── Testes ───────────────────────────────────────────────────────────────────
test:
	poetry run pytest tests/ -v
	@echo "✅ Testes passaram"

test-unit:
	poetry run pytest tests/unit/ -v

test-integration:
	poetry run pytest tests/integration/ -v

# ─── Dados ────────────────────────────────────────────────────────────────────
generate-data:
	poetry run python scripts/generate_synthetic_data.py \
		--samples 100000 \
		--fraud-rate 0.02 \
		--output data/raw/onboarding_dataset.parquet
	@echo "✅ Dataset sintético gerado: 100k amostras, 2% fraude"

# ─── Treinamento ──────────────────────────────────────────────────────────────
train:
	poetry run python scripts/train_model.py \
		--config configs/model_config.yaml \
		--data data/processed/features.parquet
	@echo "✅ Modelo treinado. Acesse o MLflow em http://localhost:5000"

mlflow-ui:
	poetry run mlflow ui --backend-store-uri mlflow/mlruns --port 5000 &
	@echo "🚀 MLflow UI: http://localhost:5000"

# ─── Avaliação ────────────────────────────────────────────────────────────────
evaluate:
	poetry run python scripts/evaluate_model.py \
		--model-version latest \
		--output reports/evaluation_report.html

# ─── Serving ──────────────────────────────────────────────────────────────────
serve:
	poetry run uvicorn src.api.main:app \
		--reload \
		--port 8000 \
		--log-level info
	@echo "🚀 API rodando em http://localhost:8000/docs"

# ─── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker build -f docker/Dockerfile.api -t fraud-score-api:latest .
	@echo "✅ Imagem fraud-score-api:latest gerada"

docker-run:
	docker-compose -f docker/docker-compose.yml up -d
	@echo "🚀 Serviços iniciados. API: http://localhost:8000"

docker-down:
	docker-compose -f docker/docker-compose.yml down

# ─── Monitoramento ────────────────────────────────────────────────────────────
drift-check:
	poetry run python -m src.monitoring.drift_detector \
		--reference data/processed/reference_dataset.parquet \
		--current data/processed/current_dataset.parquet \
		--output reports/drift_report.html
	@echo "📊 Relatório de drift: reports/drift_report.html"

# ─── CI completo (usado no GitHub Actions) ───────────────────────────────────
ci: lint test
	@echo "✅ CI passou — pronto para merge"

# ─── Limpeza ──────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/ .mypy_cache/ .ruff_cache/ dist/ build/
	@echo "✅ Limpeza concluída"
