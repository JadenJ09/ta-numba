# Makefile for ta-numba project management

# Use python3 from the environment
PYTHON=python3
PIP=pip3

# Default target
.PHONY: help
help:
	@echo "Makefile for ta-numba"
	@echo ""
	@echo "Usage:"
	@echo "  make install    - Install dependencies for development"
	@echo "  make build      - Build the wheel and sdist packages"
	@echo "  make test       - Run the test suite (requires pytest)"
	@echo "  make clean      - Remove build artifacts and caches"
	@echo "  make publish    - Publish to PyPI (requires twine)"
	@echo "  make test-publish - Publish to TestPyPI (requires twine)"
	@echo ""

# Install the package in editable mode with development dependencies
.PHONY: install
install:
	@echo "--- Installing dependencies in editable mode ---"
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]
	@echo "Done."

# Build the sdist and wheel packages
.PHONY: build
build:
	@echo "--- Building package (sdist and wheel) ---"
	$(PYTHON) -m build
	@echo "Build complete. Artifacts are in the 'dist/' directory."

# Run tests using pytest
.PHONY: test
test:
	@echo "--- Running tests with pytest ---"
	$(PYTHON) -m pytest tests/
	@echo "Tests complete."

# Clean up build artifacts and caches
.PHONY: clean
clean:
	@echo "--- Cleaning up build artifacts ---"
	rm -rf build dist src/ta_numba.egg-info .pytest_cache
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	@echo "Clean complete."

# Publish to PyPI
.PHONY: publish
publish: clean build
	@echo "--- Publishing to PyPI ---"
	$(PYTHON) -m twine upload dist/*
	@echo "Published to PyPI."

# Publish to TestPyPI for testing
.PHONY: test-publish
test-publish: clean build
	@echo "--- Publishing to TestPyPI ---"
	$(PYTHON) -m twine upload --repository testpypi dist/*
	@echo "Published to TestPyPI."

# Install build and publish tools
.PHONY: install-tools
install-tools:
	@echo "--- Installing build and publish tools ---"
	$(PIP) install --upgrade pip build twine
	@echo "Tools installed."

