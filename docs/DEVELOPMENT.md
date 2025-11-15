# Development Guide

Quick reference for common development tasks in the EDM Cue Analyzer project.

## Initial Setup

```bash
# Clone the repository
git clone https://github.com/crsmithdev/edm-cue-analyzer.git
cd edm-cue-analyzer

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

## Common Tasks

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_analyzer.py -v

# Run specific test
pytest tests/test_analyzer.py::TestConfig::test_load_default_config -v

# Run with coverage
pytest tests/ --cov=edm_cue_analyzer --cov-report=html

# Open coverage report
# Windows:
start htmlcov/index.html
# Linux/Mac:
open htmlcov/index.html
```

### Code Formatting

```bash
# Format all code with black
black src/ tests/ examples/

# Sort imports with isort
isort src/ tests/ examples/

# Do both at once
make format
```

### Linting

```bash
# Run flake8
flake8 src/ tests/ examples/

# Check formatting (without changing)
black --check src/ tests/ examples/

# Check import sorting (without changing)
isort --check-only src/ tests/ examples/

# Run all linting checks
make lint
```

### Building Package

```bash
# Clean build artifacts
make clean

# Build distribution
make build

# Or manually:
python -m build
```

### Running the CLI

```bash
# After installation, use the command
edm-cue-analyzer track.mp3

# Or run directly from source
python -m edm_cue_analyzer.cli track.mp3

# With custom config
edm-cue-analyzer track.mp3 -c examples/custom_config.yaml

# Export to XML
edm-cue-analyzer track.flac -o output.xml
```

### Testing Manually

```bash
# Start Python REPL in the project
python

# Then in REPL:
>>> from pathlib import Path
# Removed

This document has been removed. See `docs/ARCHITECTURE.md` for the
consolidated architecture and developer guidance. For contribution
instructions, refer to `CONTRIBUTING.md` in the repository root.

# Check if package is importable
python -c "import edm_cue_analyzer; print(edm_cue_analyzer.__version__)"
```

### Test Failures

```bash
# Run with verbose output
pytest tests/ -vv

# Run with print statements
pytest tests/ -s

# Run with debugger on failure
pytest tests/ --pdb
```

### Dependency Issues

```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Reinstall package
pip uninstall edm-cue-analyzer
pip install -e ".[dev]"
```

## Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [pytest Documentation](https://docs.pytest.org/)
- [black Documentation](https://black.readthedocs.io/)
- [librosa Documentation](https://librosa.org/doc/latest/)

## Quick Reference Commands

```bash
make help          # Show all available commands
make install-dev   # Install with dev dependencies
make test          # Run tests
make test-cov      # Run tests with coverage
make format        # Format code (black + isort)
make lint          # Run all linting checks
make clean         # Clean build artifacts
make build         # Build distribution package
```
