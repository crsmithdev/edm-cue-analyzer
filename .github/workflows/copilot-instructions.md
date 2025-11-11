# Copilot Instructions

You are an expert on python programming, machine learning, electronic dance music, and software audio analysis.

# Project

The project is an early-stage software application for DJs of EDM to automate or improve aspects of preparation for live sets.  Currently the only feature is generation of cue points on tracks and export of that data for DJ applications (currently Rekordbox).  More features are planned.

# Source Control

Generate minimal, information-dense commit messages. No explanations unless critical.

## Format
```
<type>: <what> [<where>]
```

## Rules
1. **50 chars max** for subject line
2. No body unless breaking change
3. Use present tense, imperative mood
4. No punctuation at end
5. Omit obvious context ("the", "a", unnecessary words)
6. Use abbreviations: cfg/config, docs, deps, init, rm, mv, fmt, ref, perf, chore

## Types (use shortest)
- `fix` - bug fixes
- `feat` - new features  
- `perf` - performance
- `ref` - refactoring (no behavior change)
- `docs` - documentation only
- `style` - formatting, whitespace
- `test` - add/modify tests
- `chore` - tooling, deps, config
- `rm` - remove code/files
- `init` - initial commit

## Examples
```
fix: null check in auth middleware
feat: add dark mode toggle
perf: cache user queries
ref: extract validation logic
chore: bump deps
rm: unused imports
docs: update API examples
style: format with prettier
test: add edge cases for parser
```

## Multi-file commits
```
feat: user profile edit
- add edit form component
- wire up API endpoint
- add validation
```

## Breaking changes (only exception to terse rule)
```
feat!: change auth API to use tokens

BREAKING: session cookies replaced with JWT
```

## Never include
- Filler words: "this commit", "now we", "basically"
- Issue references in subject (put in body if needed)
- Excessive detail
- Emojis
- Timestamps or author info (git has this)

# Code Style and Formatting

## PEP 8 Compliance
- Follow PEP 8 style guide for all Python code
- Use 4 spaces for indentation (never tabs)
- Limit lines to 88 characters (Black's default) or 79 for strict PEP 8
- Use snake_case for functions, variables, and module names
- Use PascalCase for class names
- Use UPPER_CASE for constants
- Separate top-level functions and classes with two blank lines
- Separate method definitions inside classes with one blank line

## Formatting Tools
- Use **Ruff** for linting and auto-fixing (replaces flake8, isort, pyupgrade, autoflake)
- Use **Black** or **Ruff format** for code formatting
- Configure formatting in `pyproject.toml`

## Type Hints and Static Type Checking

### Use Modern Type Hints
- **Always** add type hints to function signatures (parameters and return types)
- Use native types for Python 3.9+ (e.g., `list[str]` instead of `List[str]`)
- Use `|` for union types in Python 3.10+ (e.g., `int | float` instead of `Union[int, float]`)
- Add type hints to class attributes and instance variables
- Use `TypeAlias` for complex type definitions

### Static Type Checking
- Use **mypy** or **pyright** for static type checking
- Configure mypy in `pyproject.toml`

### When to Add Type Hints
- **Always:** Function/method signatures (parameters and return types)
- **Always:** Public API boundaries
- **Optional but recommended:** Local variables when type isn't obvious from context
- **Not needed:** When mypy can easily infer the type from assignment

## Project Structure and Organization

- Use the `src` layout for packages
- **Use `pyproject.toml`** as the single source of configuration (PEP 621)
- Include all tool configurations in `pyproject.toml`
- Avoid separate config files unless absolutely necessary
- Use **uv** (fast, Rust-based) or **Poetry** for dependency management
- Use **pipx** or **uvx** to run Python tools
- Always use virtual environments for projects
- Pin major versions, allow minor/patch updates: `package>=1.2.0,<2.0.0`

## Documentation and Comments

- Use docstrings for **all** public modules, functions, classes, and methods
- Follow Google or NumPy docstring style
- Include parameter types and return types in docstrings when helpful
- Write comments to explain **why**, not **what**
- Keep comments up-to-date with code changes
- Use inline comments sparingly (only for complex logic)
- Prefer self-documenting code with clear variable and function names

## Testing
- Write tests using **pytest** (modern standard)
- Follow naming conventions: `test_*.py` or `*_test.py`
- Test functions should start with `test_`
- Use fixtures for setup and teardown
- Aim for high test coverage (80%+) but focus on meaningful tests

## Error Handling and Logging

- Use specific exception types, not bare `except:`
- Create custom exceptions for domain-specific errors
- Use context managers (`with` statements) for resource management
- Include helpful error messages
- Use the `logging` module, not `print()` statements
- Configure logging with appropriate levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Include context in log messages

## Modern Python Features

### Use Python 3.10+ Features
- **Structural pattern matching** (match/case)
- **Parenthesized context managers**
- Use `@dataclass` for simple data containers
- Use Pydantic for data validation and settings management
- Always use context managers for file I/O and resource management
- Create custom context managers for cleanup logic

## Code Quality and Practices

### General Principles
- **DRY (Don't Repeat Yourself)**: Extract common logic into functions
- **SOLID principles**: Follow for object-oriented design
- **Keep functions small**: Aim for single responsibility
- **Use descriptive names**: Variables and functions should be self-documenting
- **Avoid magic numbers**: Use named constants

### Use List/Dict Comprehensions
- Prefer comprehensions over loops for transformations
- Keep comprehensions readable (don't nest too deeply)

### Use Pathlib for File Operations
- Use `pathlib.Path` instead of `os.path`

## Pre-commit Hooks and CI/CD

### Pre-commit Configuration
Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-added-large-files

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
```

### GitHub Actions CI
```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install uv
        run: pip install uv
      
      - name: Install dependencies
        run: uv sync --all-extras
      
      - name: Run ruff
        run: uv run ruff check .
      
      - name: Run mypy
        run: uv run mypy src/
      
      - name: Run tests
        run: uv run pytest
```

## Security and Performance

### Security Best Practices
- Never hardcode secrets or credentials
- Use environment variables for sensitive data
- Validate and sanitize all user input
- Use secure random for cryptographic purposes: `secrets` module
- Keep dependencies updated

```python
import os
from secrets import token_urlsafe

# Good - read from environment
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable not set")

# Generate secure random tokens
secure_token = token_urlsafe(32)
```

### Performance Considerations
- Use generators for large datasets
- Profile before optimizing (`cProfile`, `line_profiler`)
- Use `functools.lru_cache` for memoization
- Consider async/await for I/O-bound operations

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(n: int) -> int:
    """Cache results of expensive computation."""
    # ... complex calculation
    return result

# Generator for memory efficiency
def read_large_file(path: str):
    """Read large file line by line."""
    with open(path) as f:
        for line in f:
            yield line.strip()
```

## Summary Checklist

When generating Python code, ensure:
- ✅ Type hints on all function signatures
- ✅ Docstrings for public APIs
- ✅ PEP 8 compliant formatting
- ✅ Modern Python 3.10+ syntax (native types, match/case)
- ✅ Proper error handling with specific exceptions
- ✅ Logging instead of print statements
- ✅ Use of context managers for resources
- ✅ Descriptive variable and function names
- ✅ Tests with pytest for new functionality
- ✅ No hardcoded secrets or magic numbers
- ✅ Configuration in pyproject.toml
- ✅ Use pathlib for file operations
- ✅ Dataclasses or Pydantic for data structures

## Tools Reference

**Essential Tools:**
- **Ruff**: Linting and formatting (`pip install ruff`)
- **mypy**: Static type checking (`pip install mypy`)
- **pytest**: Testing framework (`pip install pytest`)
- **uv**: Package manager (`pip install uv`)
- **pre-commit**: Git hooks (`pip install pre-commit`)

**Configuration File:** `pyproject.toml` (single source of truth)

**Python Version:** Target Python 3.12+ for new projects