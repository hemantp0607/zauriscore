# Contributing to ZauriScore

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Windows: `.\\venv\\Scripts\\activate`)
4. Install dev dependencies: `pip install -e '.[dev]'`

## Code Style

We use:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

Run all checks:
```bash
make lint
```

## Testing

1. Write tests for new features
2. Ensure all tests pass: `pytest`
3. Check coverage: `pytest --cov=zauriscore`

## Pull Request Process

1. Create a feature branch
2. Write clear commit messages
3. Add/update tests
4. Update documentation
5. Submit PR with description

## Documentation

- Update relevant docs in `docs/`
- Add docstrings to new code
- Include example usage

## Release Process

1. Update version in `src/zauriscore/__init__.py`
2. Update CHANGELOG.md
3. Create release PR
4. Tag release after merge

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Follow project guidelines
