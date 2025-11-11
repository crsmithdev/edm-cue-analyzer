# Virtual Environment Setup

This document describes the virtual environment setup for the EDM Cue Analyzer project.

## Virtual Environment Created

A Python virtual environment has been created in the `venv/` directory.

## Activation

### Windows PowerShell
```powershell
.\venv\Scripts\Activate.ps1
```

### Windows CMD
```cmd
venv\Scripts\activate.bat
```

### Linux/Mac
```bash
source venv/bin/activate
```

## Deactivation

To deactivate the virtual environment:
```bash
deactivate
```

## Installed Packages

The following packages have been installed:

### Core Dependencies
- librosa>=0.10.0 - Audio analysis
- numpy>=1.24.0 - Numerical operations
- pyyaml>=6.0 - Configuration parsing
- colorama>=0.4.6 - Terminal colors
- soundfile>=0.12.0 - Audio I/O

### Development Dependencies
- pytest>=7.0 - Testing framework
- pytest-cov>=4.0 - Coverage reporting
- black>=23.0 - Code formatting
- isort>=5.0 - Import sorting
- flake8>=6.0 - Linting

## Package Installation

The project is installed in editable mode:
```bash
pip install -e ".[dev]"
```

This means changes to the source code are immediately reflected without reinstalling.

## Verifying Installation

Test that the package is installed correctly:
```bash
# Test CLI command
edm-cue-analyzer --help

# Test import
python -c "import edm_cue_analyzer; print(edm_cue_analyzer.__version__)"
```

## Running Tests

```bash
pytest tests/ -v
```

## Updating Dependencies

To update all dependencies:
```bash
pip install --upgrade -r requirements.txt
pip install -e ".[dev]"
```

## Reinstalling

If you need to reinstall from scratch:
```bash
# Deactivate if active
deactivate

# Remove venv directory
Remove-Item -Recurse -Force venv

# Recreate venv
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# Install
pip install --upgrade pip
pip install -e ".[dev]"
```

## Notes

- The `venv/` directory is ignored by git (see `.gitignore`)
- Always activate the virtual environment before working on the project
- All development should be done within the virtual environment
