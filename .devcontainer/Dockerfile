# Minimal dev container for EDM Cue Analyzer
FROM python:3.12-slim-bookworm

# Install system dependencies needed for audio libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    git \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libsamplerate0-dev \
    libaubio5 \
    libaubio-dev \
    libyaml-dev \
    libfftw3-dev \
    libtag1-dev \
    libchromaprint-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pre-install essentia and aubio at the system level
RUN pip install --no-cache-dir essentia aubio

CMD ["/bin/bash"]