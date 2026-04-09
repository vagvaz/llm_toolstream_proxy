.PHONY: install dev lint typecheck test ci ruff-format

VENV := .venv/bin
PYTHON := $(VENV)/python
PIP := $(VENV)/pip

install:
	python3 -m venv .venv
	$(PIP) install -e ".[dev]"

dev: install

lint:
	$(VENV)/ruff check src tests

ruff-format:
	$(VENV)/ruff format src tests

typecheck:
	$(VENV)/mypy src tests

test:
	$(VENV)/pytest tests/ -v

ci: lint typecheck test
	@echo "All checks passed."