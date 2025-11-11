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
>>> from edm_cue_analyzer import load_config, AudioAnalyzer, CueGenerator
>>> 
>>> config = load_config()
>>> analyzer = AudioAnalyzer(config.analysis)
>>> # Test with your audio file
```

## Project Structure Reference

```
src/edm_cue_analyzer/
├── __init__.py          # Package exports
├── analyzer.py          # Audio analysis (librosa)
├── cli.py               # Command-line interface
├── config.py            # Configuration management
├── cue_generator.py     # Cue point generation
├── display.py           # Terminal display
├── rekordbox.py         # Rekordbox XML export
└── default_config.yaml  # Default configuration

tests/
├── __init__.py
└── test_analyzer.py     # Test suite

examples/
├── README.md
├── example_usage.py     # Usage examples
└── custom_config.yaml   # Example config
```

## Adding New Features

### 1. Write Tests First (TDD)

```python
# tests/test_analyzer.py
def test_new_feature():
    """Test the new feature."""
    # Arrange
    config = load_config()
    
    # Act
    result = new_feature(config)
    
    # Assert
    assert result is not None
```

### 2. Implement Feature

```python
# src/edm_cue_analyzer/module.py
def new_feature(config):
    """Implement new feature."""
    # Your code here
    pass
```

### 3. Update Exports

```python
# src/edm_cue_analyzer/__init__.py
from .module import new_feature

__all__ = [
    # ... existing exports
    'new_feature',
]
```

### 4. Run Tests

```bash
pytest tests/ -v
```

### 5. Format and Lint

```bash
make format
make lint
```

## Configuration

### Loading Config

```python
from edm_cue_analyzer import load_config, get_default_config

# Load default config
config = load_config()

# Load custom config
config = load_config(Path("custom_config.yaml"))

# Get default config object
default = get_default_config()
```

### Creating Custom Config

```yaml
# my_config.yaml
hot_cues:
  A:
    label: "Custom Cue"
    position_percent: 0.25
    loop_bars: 4
    color: "PINK"

analysis:
  tempo:
    start_bpm: 128
```

## Debugging

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Using Python Debugger

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or in Python 3.7+
breakpoint()
```

### VSCode Debug Configuration

Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: CLI",
            "type": "python",
            "request": "launch",
            "module": "edm_cue_analyzer.cli",
            "args": ["path/to/test/track.mp3"],
            "console": "integratedTerminal"
        },
        {
            "name": "Python: Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "console": "integratedTerminal"
        }
    ]
}
```

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes
# ... edit files ...

# Run tests
make test

# Format code
make format

# Commit
git add .
git commit -m "Add new feature: description"

# Push
git push origin feature/new-feature

# Create Pull Request on GitHub
```

## Troubleshooting

### Import Errors

```bash
# Make sure package is installed in development mode
pip install -e .

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
