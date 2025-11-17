# Dev Container Setup

This project uses VS Code Dev Containers to provide a consistent Linux development environment with all dependencies pre-installed, including essentia for accurate BPM detection.

## Prerequisites

1. **Docker Desktop** - [Download here](https://www.docker.com/products/docker-desktop/)
2. **VS Code** - [Download here](https://code.visualstudio.com/)
3. **Dev Containers Extension** - Install from VS Code marketplace

## Quick Start

1. Install Docker Desktop and start it
2. Open this project in VS Code
3. When prompted, click "Reopen in Container" (or press F1 → "Dev Containers: Reopen in Container")
4. Wait for the container to build (first time takes ~5 minutes)
5. You're ready! All commands now run in the Linux container

## What's Included

### Audio/ML Libraries
- ✅ **essentia** - State-of-the-art BPM detection (MIREX winner)
- ✅ **aubio** - Fast beat tracking (fallback)
- ✅ **librosa** - General audio analysis
- ✅ **FFmpeg** - Audio format support

### Development Tools
- ✅ Python 3.12
- ✅ Git
- ✅ Zsh with Oh My Zsh
- ✅ All dev dependencies (pytest, ruff, mypy)

## Usage

### Running the Analyzer

```bash
# Inside the container terminal
edm-cue-analyzer analyze tests/your-track.flac -a bpm
```

### Running Tests

```bash
pytest
```

### BPM Detection Methods

The analyzer will automatically use the best available method:

1. **Essentia** (best) - MIREX 2013 winner, optimized for EDM
2. **Aubio** (good) - Fast, reliable fallback
3. **Librosa** (basic) - Last resort

## Benefits

### For This Project
- ✅ No Windows compilation issues
- ✅ Essentia works perfectly
- ✅ Easy to add ML libraries later (TensorFlow, PyTorch, Demucs)
- ✅ Consistent environment for all developers

### For Development
- ✅ Work in VS Code on Windows as normal
- ✅ Code runs in Linux automatically
- ✅ GPU passthrough support for future ML models
- ✅ Same container can be used for deployment

## Troubleshooting

### Container won't build
- Make sure Docker Desktop is running
- Check Docker Desktop has enough resources (Settings → Resources → 4GB RAM minimum)

### Extensions not loading
- The first build installs everything - wait for "postCreateCommand" to finish
- Check the "Dev Container" output panel in VS Code

### Audio files not accessible
- Audio files in `./tests` are automatically mounted
- Add more mounts in `.devcontainer/devcontainer.json` if needed

## Switching Between Environments

### Use Dev Container (Recommended)
- Best BPM accuracy with essentia
- Future-proof for ML features
- Click "Reopen in Container" in VS Code

### Use Local Windows Environment
- Falls back to aubio or librosa
- Click "Reopen Locally" in VS Code
- Some features may have reduced accuracy

## Next Steps

Once in the container, try:

```bash
# Test essentia is working
edm-cue-analyzer analyze tests/your-track.flac -a bpm --verbose

# Run all tests
pytest -v

# Check code quality
ruff check .
```

The container is persistent - your work is saved locally, only the execution happens in Linux!
