ENV_FILES := $(wildcard .env .env.local)

ifneq ($(ENV_FILES),)
    include $(ENV_FILES)
    export
    # Strip optional quotes from exported variables loaded from env files
    $(foreach v,$(shell grep -h -E '^[A-Za-z_][A-Za-z0-9_]*=' $(ENV_FILES) 2>/dev/null | cut -d= -f1),$(eval $(v) := $(patsubst "%",%,$(patsubst '%',%,$($(v))))))
endif

ENV_FILE_ARGS := $(foreach f,$(ENV_FILES),--env-file $(f))

DOCLING_HOST_PORT ?= 5001
DOCLING_SERVE_URL ?= http://localhost:$(DOCLING_HOST_PORT)
export DOCLING_SERVE_URL

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
test: ## Run tests
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest

.PHONY: docker-build
docker-build: ## Build the patched docling-serve image locally
	@echo "🐳 Building docling-serve image"
	@docker build --no-cache  -t docling-serve-plugins:latest -f plugins/Dockerfile.docling-serve plugins/

.PHONY: docker-up
docker-up: ## Start the Docker Compose stack
	@echo "🐳 Running docker compose"
	@docker compose $(ENV_FILE_ARGS) up -d
	@echo "🌐 docling-serve is available at: $(DOCLING_SERVE_URL)"

.PHONY: docker-down
docker-down: ## Stop the Docker Compose stack
	@echo "🐳 Stopping docker compose"
	@docker compose $(ENV_FILE_ARGS) down

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
