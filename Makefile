# Makefile for ta-numba development

.PHONY: help install test lint format clean build upload-test upload docs

help:
	@echo "Available commands:"
	@echo "  install     Install development dependencies"
	@echo "  test        Run all tests"
	@echo "  lint        Run linting checks"
	@echo "  format      Format code with black and isort"
	@echo "  clean       Clean up build artifacts"
	@echo "  build       Build distribution packages"
	@echo "  upload-test Upload to test PyPI"
	@echo "  upload      Upload to PyPI"
	@echo "  benchmark   Run performance benchmarks"

install:
	pip install -e .[dev]
	pre-commit install

test:
	python -m pytest tests/ -v

test-unit:
	python -m pytest tests/unit/ -v

test-integration:
	python -m pytest tests/integration/ -v

test-performance:
	python -m pytest tests/performance/ -v

lint:
	flake8 src/ tests/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload-test: build
	python -m twine upload --repository testpypi dist/*

upload: build
	python -m twine upload dist/*

benchmark:
	python tests/performance/test_comprehensive_comparison.py

organize:
	python scripts/organize_project.py

prepare-release:
	python scripts/prepare_release.py --version $(VERSION)

# Development shortcuts
dev-setup:
	python scripts/setup_dev.py

warmup:
	python -c "import ta_numba.warmup; ta_numba.warmup.warmup_all()"
