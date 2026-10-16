.PHONY: install test lint run predict docker-build docker-run clean

install:
	pip install -r requirements.txt -r requirements-dev.txt

test:
	python -m pytest -q

lint:
	ruff check .

run:
	python -m pipeline.run_pipeline

predict:
	python -m pipeline.predict

docker-build:
	docker build -t healthcare-ml-pipeline .

docker-run:
	docker run --rm -v $(PWD)/output:/app/output -v $(PWD)/docs/images:/app/docs/images healthcare-ml-pipeline

clean:
	rm -rf output .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
