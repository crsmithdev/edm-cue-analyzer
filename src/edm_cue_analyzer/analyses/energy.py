"""Energy curve analysis."""

import logging
from dataclasses import dataclass

import librosa
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EnergyResult:
    """Result from energy analysis."""

    times: np.ndarray  # Time points for energy curve
    curve: np.ndarray  # RMS energy values
    window_seconds: float  # Window size used


async def analyze_energy(context: dict) -> EnergyResult:
    """
    Calculate RMS energy curve over time.

    Args:
        context: Dictionary containing:
            - y: Audio signal (mono, float32)
            - sr: Sample rate
            - config: Optional config with energy_window_seconds

    Returns:
        EnergyResult with energy curve and time points
    """
    y = context["y"]
    sr = context["sr"]
    config = context.get("config")

    # Get window size from config or use default
    window_seconds = 0.1
    if config and hasattr(config, "energy_window_seconds"):
        window_seconds = config.energy_window_seconds

    logger.debug("Calculating energy curve with window=%.2fs", window_seconds)

    hop_length = int(sr * window_seconds)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    logger.debug("Energy curve calculated: %d points", len(rms))

    return EnergyResult(times=times, curve=rms, window_seconds=window_seconds)
