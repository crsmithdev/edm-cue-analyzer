"""Build detection analysis."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


async def analyze_builds(context: dict) -> list[float]:
    """
    Detect build-up sections (gradual energy increases).

    Args:
        context: Dictionary containing:
            - energy: EnergyResult from energy analysis
            - config: Optional analysis config

    Returns:
        List of timestamps where builds occur
    """
    energy_result = context["energy"]
    config = context.get("config")

    energy = energy_result.curve
    times = energy_result.times

    # Get config values
    build_window_size = 20
    energy_threshold_increase = 0.2

    if config:
        build_window_size = getattr(config, "build_window_size", 20)
        energy_threshold_increase = getattr(config, "energy_threshold_increase", 0.2)

    builds = []

    logger.debug(
        "Build detection: window_size=%d, threshold=%.2f%%",
        build_window_size,
        energy_threshold_increase * 100,
    )

    # Look for sustained energy increases
    for i in range(len(energy) - build_window_size):
        # Check if energy consistently increases over window
        window = energy[i : i + build_window_size]
        if np.all(np.diff(window) > 0) and window[-1] > window[0] * (
            1 + energy_threshold_increase
        ):
            builds.append(float(times[i]))

    return builds
