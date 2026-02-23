.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync --extra dev
	@uv run pre-commit install

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running ruff"
	@uv run ruff format
	@uv run ruff check --fix

.PHONY: test
test: ## Run tests (skips e2e unless DOCLING_SERVE_URL is set)
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest

.PHONY: test-e2e
test-e2e: ## Run end-to-end tests (requires DOCLING_SERVE_URL)
	@echo "🚀 Running e2e tests"
	@uv run python -m pytest tests/e2e -m e2e -v

.PHONY: docker-build
docker-build: ## Build the patched docling-serve image locally
	@echo "🐳 Building docling-serve image"
	@docker build -t docling-serve-plugins:latest -f plugins/Dockerfile.docling-serve plugins/

.PHONY: docker-up
docker-up: ## Start the Docker Compose stack
	@echo "🐳 Running docker compose"
	@docker compose up -d

.PHONY: docker-down
docker-down: ## Stop the Docker Compose stack
	@echo "🐳 Stopping docker compose"
	@docker compose down

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
