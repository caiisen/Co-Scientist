.PHONY: install test lint

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
