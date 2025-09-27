.PHONY: install test lint clean docs

install:
	pip install -e ".[dev]"

test:
	pytest tests/

lint:
	flake8 src/zauriscore
	black src/zauriscore
	isort src/zauriscore

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	mkdocs build
