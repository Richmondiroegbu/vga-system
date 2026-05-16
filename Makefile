.PHONY: install test lint check-arch dev download-models clean

install:
	pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/unit/ -v --tb=short
	pytest tests/integration/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

test-chaos:
	pytest tests/chaos/ -v --tb=short

lint:
	ruff check vga/ tests/
	black --check vga/ tests/
	mypy vga/ --ignore-missing-imports

check-arch:
	python -m vga.devtools.architecture_linter --check-all

dev:
	uvicorn vga.api.main:app --host 0.0.0.0 --port 8000 --reload &
	streamlit run vga/ui/app.py --server.port 8501

download-models:
	bash scripts/download_all_models.sh

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null; true
