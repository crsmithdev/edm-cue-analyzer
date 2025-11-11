# Contributing to EDM Cue Analyzer

Thank you for your interest in contributing to EDM Cue Analyzer! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/crsmithdev/edm-cue-analyzer.git
   cd edm-cue-analyzer
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

## Running Tests

Run the test suite with pytest:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=edm_cue_analyzer --cov-report=html
```

## Code Style

This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting

Format your code before committing:

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

## Project Structure

```
edm-cue-analyzer/
├── src/
│   └── edm_cue_analyzer/     # Main package
│       ├── __init__.py
│       ├── analyzer.py        # Audio analysis
│       ├── cli.py             # CLI interface
│       ├── config.py          # Configuration
│       ├── cue_generator.py   # Cue generation
│       ├── display.py         # Terminal display
│       ├── rekordbox.py       # XML export
│       └── default_config.yaml
├── tests/                     # Test suite
├── examples/                  # Example scripts
├── docs/                      # Documentation
└── pyproject.toml             # Project metadata
```

## Making Changes

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and add tests

3. Run tests and formatting:
   ```bash
   pytest tests/
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   ```

4. Commit your changes with clear messages:
   ```bash
   git commit -m "Add feature: description"
   ```

5. Push and create a pull request

## Pull Request Guidelines

- Include tests for new features
- Update documentation as needed
- Follow existing code style
- Write clear commit messages
- Keep changes focused and atomic

## Reporting Issues

When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Any error messages

## Questions?

Feel free to open an issue for questions or discussions!
