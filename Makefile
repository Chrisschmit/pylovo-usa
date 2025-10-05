.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: check-uv
check-uv: ## Check if uv is installed
	@command -v uv >/dev/null 2>&1 || { echo "❌ uv is not installed. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

.PHONY: venv
venv: check-uv ## Create virtual environment using uv
	@if [ -d .venv ]; then \
		echo "⚠️  Virtual environment already exists at .venv"; \
		echo "  Use 'make venv-clean' to recreate it"; \
	else \
		uv venv .venv; \
		echo ""; \
		echo "✓ Virtual environment created successfully with uv!"; \
		echo "  Activate with: source .venv/bin/activate"; \
	fi

.PHONY: venv-clean
venv-clean: check-uv ## Remove and recreate virtual environment
	@echo "Removing existing virtual environment..."
	rm -rf .venv
	uv venv .venv
	@echo ""
	@echo "✓ Virtual environment recreated successfully with uv!"
	@echo "  Activate with: source .venv/bin/activate"

.PHONY: setup
setup: check-uv ## Create venv and install production dependencies (one-step setup)
	@if [ ! -d .venv ]; then \
		uv venv .venv; \
		echo ""; \
	fi
	@echo "Installing production dependencies with uv..."
	uv pip install -r requirements.txt
	@echo ""
	@echo "✓ Setup complete!"
	@echo "  Activate with: source .venv/bin/activate"

.PHONY: setup-dev
setup-dev: check-uv ## Create venv and install ALL dependencies (production + dev)
	@if [ ! -d .venv ]; then \
		uv venv .venv; \
		echo ""; \
	fi
	@echo "Installing all dependencies (prod + dev) with uv..."
	uv pip install -r requirements-dev.txt
	@echo ""
	@echo "Installing pre-commit hooks..."
	.venv/bin/pre-commit install
	@echo ""
	@echo "✓ Development setup complete!"
	@echo "  Activate with: source .venv/bin/activate"
	@echo "  Pre-commit hooks are now active!"

.PHONY: install
install: check-uv ## Install dependencies using uv (use after activating venv)
	uv pip install -r requirements.txt
	@echo "✓ Dependencies installed successfully with uv"

.PHONY: install-dev
install-dev: check-uv ## Install all dependencies (production + development) with uv
	uv pip install -r requirements-dev.txt
	@echo "✓ All dependencies (production + dev) installed successfully"

.PHONY: pre-commit-install
pre-commit-install: ## Install pre-commit hooks
	pre-commit install
	@echo "✓ Pre-commit hooks installed"

.PHONY: lint
lint: ## Run all linting and formatting checks via pre-commit
	@echo "Running all linting checks (pre-commit)..."
	pre-commit run --all-files
	@echo ""
	@echo "✓ All linting checks complete!"

.PHONY: clean
clean: clean-build clean-pyc ## remove all build artifacts and Python file artifacts

.PHONY: clean-build
clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +

.PHONY: clean-pyc
clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
