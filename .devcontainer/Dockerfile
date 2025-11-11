# Dockerfile for EDM Cue Analyzer Development
# Optimized for audio/ML work with essentia, aubio, and future ML libraries
# Note: Using Python 3.12 due to numba (librosa dependency) not yet supporting 3.13

FROM python:3.12-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for audio processing and ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    pkg-config \
    git \
    curl \
    # Audio libraries
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    # Essentia dependencies
    libyaml-dev \
    libfftw3-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libsamplerate0-dev \
    libtag1-dev \
    libchromaprint-dev \
    # Aubio dependencies  
    libaubio5 \
    libaubio-dev \
    # Python dev tools
    python3-dev \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy and install Python dependencies
COPY pyproject.toml ./
COPY README.md ./

# Upgrade pip and install essentia + other dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install essentia && \
    pip install aubio

# Switch to non-root user
USER $USERNAME

# Install the package in development mode (will be done by postCreateCommand)
# This layer just ensures the base dependencies are cached

# Set default command
CMD ["/bin/bash"]
