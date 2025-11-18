"""Drop detection analysis using Wolfram paper's energy-based approach."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


async def analyze_drops(context: dict) -> list[float]:
    """
    Detect drop points using energy drop detection (Wolfram paper method).

    Based on "[WSC22] Analyze 'the drop' in EDM songs" by Suhaan Mobhani.
    
    Method:
    1. Calculate Short-Time Energy from audio
    2. Compute scale factor using median energy ratios
    3. Find positions where: energy[i] > scale * energy[i+1]
    4. Convert sample positions to timestamps
    5. Remove adjacent duplicates within 3 seconds

    Args:
        context: Dictionary containing:
            - bpm: BpmResult from BPM analysis
            - energy: EnergyResult from energy analysis
            - features: Optional dict of extracted features (for RMS energy)
            - config: Optional analysis config

    Returns:
        List of timestamps where drops occur
    """
    bpm_result = context["bpm"]
    energy_result = context["energy"]
    config = context.get("config")
    metadata = context.get("metadata")

    # Step 1: Get Short-Time Energy
    # Use RMS energy curve as our "Short-Time Energy"
    energy = energy_result.curve
    times = energy_result.times
    bar_duration = bpm_result.bar_duration

    # Get config values
    drop_min_spacing_bars = 8
    if config:
        drop_min_spacing_bars = getattr(config, "drop_min_spacing_bars", 8)

    min_spacing = bar_duration * drop_min_spacing_bars

    if len(energy) < 2:
        logger.warning("Energy array too short for drop detection")
        return []

    logger.debug(
        "Wolfram drop detection: %d energy samples, time range: %.1fs to %.1fs",
        len(energy), times[0], times[-1]
    )

    # Step 2: Determine drop threshold
    # Use a fixed threshold approach inspired by Wolfram paper
    # Threshold represents how much energy must drop (e.g., 1.3 = 23% decrease)
    
    # If we have Spotify energy metadata, use it to calibrate the threshold
    # Higher energy tracks (more intense) should use higher thresholds
    base_threshold = 1.5  # Default: 33% energy decrease
    
    if metadata and hasattr(metadata, 'energy') and metadata.energy is not None:
        # Spotify energy is 0.0 to 1.0
        # Scale threshold: low energy (0.3) -> 1.4, high energy (0.9) -> 1.7
        spotify_energy = metadata.energy
        base_threshold = 1.4 + (spotify_energy * 0.5)
        logger.debug(
            "Using Spotify energy %.2f to set drop threshold: %.2f (%.0f%% decrease)",
            spotify_energy, base_threshold, (1 - 1/base_threshold) * 100
        )
    else:
        logger.debug(
            "Using default drop threshold: %.2f (%.0f%% decrease)",
            base_threshold, (1 - 1/base_threshold) * 100
        )

    # Step 3: Find drop positions
    # A drop occurs when: energy[i] > threshold * energy[i+1]
    # This identifies sharp energy decreases
    
    drop_candidates = []
    for i in range(len(energy) - 1):
        if energy[i] > base_threshold * energy[i + 1]:
            drop_candidates.append(i)
    
    logger.debug("Found %d initial drop candidates", len(drop_candidates))

    if not drop_candidates:
        return []

    # Step 4: Convert sample positions to timestamps
    drop_timings = []
    for idx in drop_candidates:
        # Map index to time
        drop_time = times[idx]
        drop_timings.append(drop_time)

    # Step 5: Remove adjacent duplicates
    # Paper uses 3-second threshold - drops must be >3 seconds apart
    final_drops = []
    duplicate_threshold = 3.0  # seconds
    
    for i, drop_time in enumerate(drop_timings):
        # Check if this is far enough from the previous drop
        if i == 0 or (drop_time - drop_timings[i-1]) > duplicate_threshold:
            final_drops.append(float(drop_time))
    
    logger.debug(
        "After removing duplicates: %d drops (removed %d adjacent duplicates)",
        len(final_drops), len(drop_timings) - len(final_drops)
    )

    # Apply minimum spacing constraint (from config)
    spaced_drops = []
    for drop in final_drops:
        if not spaced_drops or (drop - spaced_drops[-1]) > min_spacing:
            spaced_drops.append(drop)

    logger.debug(
        "After spacing constraint (%.1fs): %d drops",
        min_spacing, len(spaced_drops)
    )

    return spaced_drops
