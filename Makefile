.PHONY: install lint format format-check test test-cov run clean docker-build docker-run

install:
	uv sync

lint:
	uv run ruff check src test

format:
	uv run ruff format src test

format-check:
	uv run ruff format --check src test

test:
	uv run pytest test -v

test-cov:
	uv run pytest test --cov=src --cov-report=term-missing

run:
	uv run python -m src.view.detector_view

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage

docker-build:
	docker build -t uao-neumonia .

docker-run: docker-build
	docker run --rm \
		-e DISPLAY=$(DISPLAY) \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $(CURDIR)/model:/app/model \
		uao-neumonia
