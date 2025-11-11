.PHONY: help install install-dev test test-cov format lint clean build

help:
	@echo "EDM Cue Analyzer - Development Commands"
	@echo ""
	@echo "  make install      - Install package"
	@echo "  make install-dev  - Install package with dev dependencies"
	@echo "  make test         - Run tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make format       - Format code with black and isort"
	@echo "  make lint         - Run linting checks"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make build        - Build distribution packages"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=edm_cue_analyzer --cov-report=html --cov-report=term

format:
	black src/ tests/ examples/
	isort src/ tests/ examples/

lint:
	flake8 src/ tests/ examples/
	black --check src/ tests/ examples/
	isort --check-only src/ tests/ examples/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/

build: clean
	python -m build
