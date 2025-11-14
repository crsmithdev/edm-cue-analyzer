# EDM Cue Analyzer

[![CI](https://github.com/crsmithdev/edm-cue-analyzer/workflows/CI/badge.svg)](https://github.com/crsmithdev/edm-cue-analyzer/actions)
[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automated cue point generation for DJ performance. Analyzes EDM tracks and generates intelligent hot cues and memory cues for seamless mixing in Rekordbox.

**Features advanced BPM detection with Essentia** - the state-of-the-art beat tracking algorithm optimized for EDM.

## Features

- 🎵 **Multi-format support** - MP3, FLAC, WAV, and all librosa-supported formats
- 🎯 **Smart cue detection** - Automatic detection of drops, breakdowns, and builds
- 🎨 **Color-coded cues** - 8 configurable hot cues (A-H) with CDJ colors
- 📍 **Memory cues** - Visual reference points for track structure
- 📊 **Advanced analysis** - BPM detection, energy analysis, spectral analysis
- 🎛️ **Rekordbox export** - Direct XML export for Rekordbox import
- ⚙️ **Fully configurable** - YAML configuration for complete customization
- 🖥️ **Beautiful CLI** - Color-coded terminal display with ASCII waveforms
- 📚 **Library + CLI** - Use as a Python library or command-line tool

## Installation

**Requirements:** Python 3.10-3.12 (Python 3.13 not yet supported due to numba compatibility)

### Option 1: Dev Container (Recommended)

Get the best BPM detection with Essentia and a consistent Linux environment:

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Install [VS Code](https://code.visualstudio.com/) + [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
3. Open this project in VS Code
4. Click "Reopen in Container" when prompted

See [DEV_CONTAINER_SETUP.md](DEV_CONTAINER_SETUP.md) for details.

**Benefits:**
- ✅ **Essentia** - State-of-the-art BPM detection (MIREX 2013 winner)
- ✅ **Aubio** - Advanced beat tracking fallback
- ✅ No compilation issues
- ✅ Works on Windows, macOS, Linux
- ✅ Isolated environment with all dependencies

### Option 2: Local Installation

#### Basic Installation (Aubio + Librosa)

```bash
git clone https://github.com/crsmithdev/edm-cue-analyzer.git
cd edm-cue-analyzer
pip install -e .
```

This installs with aubio for improved BPM detection (better than librosa alone).

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Command Line

```bash
# Analyze and display cue points
edm-cue-analyzer track.mp3

# Export to Rekordbox XML
edm-cue-analyzer track.flac -o cues.xml

# Use custom configuration
edm-cue-analyzer track.mp3 -c custom_config.yaml
```

### Python Library

```python
from pathlib import Path
from edm_cue_analyzer import (
    load_config,
    AudioAnalyzer,
    CueGenerator,
    export_to_rekordbox,
    display_results
)

# Load configuration
config = load_config()

# Analyze audio
analyzer = AudioAnalyzer(config.analysis)
structure = analyzer.analyze_file(Path("track.mp3"))

# Generate cues
generator = CueGenerator(config)
cues = generator.generate_cues(structure)

# Display and export
display_results(cues, structure)
export_to_rekordbox(Path("track.mp3"), cues, structure, Path("output.xml"))
```

## Default Cue Layout

**Hot Cues:**
- **A** - Early Intro (8%, 8 bars, BLUE)
- **B** - Late Intro (20%, 16 bars, GREEN)
- **C** - Pre-Drop (before 1st drop, 4 bars, ORANGE)
- **D** - Post-Drop (after 1st drop, 8 bars, GREEN)
- **E** - Breakdown (1st breakdown, 16 bars, TEAL)
- **F** - Second Energy (2nd drop, 8 bars, YELLOW)
- **G** - Outro Start (85%, 8 bars, GREEN)
- **H** - Safe Hold (50%, 16 bars, BLUE)

**Memory Cues:**
- Track Start, First Drop, First Breakdown, Second Drop, Outro

## Configuration

Customize cue positions, colors, and analysis parameters with YAML:

```yaml
hot_cues:
  A:
    label: "Early Intro"
    position_percent: 0.08
    loop_bars: 8
    color: "BLUE"
  # ... more cues

analysis:
  tempo:
    hop_length: 512
    start_bpm: 120
  # ... more settings
```

See `examples/custom_config.yaml` for a complete example.

## Project Structure

```
edm-cue-analyzer/
├── src/
│   └── edm_cue_analyzer/     # Main package
│       ├── analyzer.py        # Audio analysis
│       ├── cli.py             # CLI interface
│       ├── config.py          # Configuration management
│       ├── cue_generator.py   # Cue generation logic
│       ├── display.py         # Terminal display
│       ├── rekordbox.py       # Rekordbox XML export
│       └── default_config.yaml
├── tests/                     # Test suite
├── examples/                  # Usage examples
├── docs/                      # Documentation
├── pyproject.toml             # Project metadata
└── README.md
```

## Development

### Setup

```bash
# Clone and install
git clone https://github.com/crsmithdev/edm-cue-analyzer.git
cd edm-cue-analyzer
pip install -e ".[dev]"
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=edm_cue_analyzer --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/
```

### Using Make

```bash
make help          # Show all commands
make install-dev   # Install with dev dependencies
make test          # Run tests
make test-cov      # Run tests with coverage
make format        # Format code
make lint          # Run linting
make clean         # Clean build artifacts
```

## Dependencies

**Core:**
- **librosa** - Audio analysis and feature extraction
- **numpy** - Numerical operations
- **pyyaml** - Configuration file parsing
- **colorama** - Cross-platform terminal colors
- **soundfile** - Audio file I/O
- **aubio** - Beat tracking and BPM detection

**Optional (Dev Container):**
- **essentia** - Advanced BPM detection optimized for EDM (Linux only)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Documentation

- [Quick Start Guide](docs/QUICK_START.md)
- [Project Summary](docs/PROJECT_SUMMARY.md)
- [Examples](examples/)

## Acknowledgments

Built with:
- [librosa](https://librosa.org/) for audio analysis
- [essentia](https://essentia.upf.edu/) for advanced BPM detection
- [aubio](https://aubio.org/) for beat tracking
- [colorama](https://github.com/tartley/colorama) for terminal colors

## Support

- [Report Issues](https://github.com/crsmithdev/edm-cue-analyzer/issues)
- [View Documentation](docs/)

---

Made with ❤️ for DJs
