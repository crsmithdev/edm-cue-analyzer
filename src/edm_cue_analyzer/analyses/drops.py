"""Drop detection analysis using Wolfram paper's energy-based approach."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


async def analyze_drops(context: dict) -> list[float]:
    """
    Detect drop points using multi-signal approach.

    Combines energy-based detection with onset strength and structural timing.
    
    Signals:
    1. Energy drops: Sharp decreases in RMS energy
    2. Onset strength: Strong percussive events
    3. Structural timing: Drops occur at predictable intervals (every 32-64 bars)

    Args:
        context: Dictionary containing:
            - bpm: BpmResult from BPM analysis
            - energy: EnergyResult from energy analysis
            - features: Dict with onset_strength, onset_times
            - config: Optional analysis config
            - metadata: Optional track metadata

    Returns:
        List of timestamps where drops occur
    """
    bpm_result = context["bpm"]
    energy_result = context["energy"]
    features = context.get("features", {})
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

    # Step 4: Filter by onset strength
    # Real drops have strong percussive onsets
    onset_strength = features.get("onset_strength")
    onset_times_from_features = features.get("onset_times")
    
    if onset_strength is not None and len(onset_strength) > 0:
        # Get onset strength config
        onset_std_threshold = 1.5
        if config:
            onset_std_threshold = getattr(config, "drop_onset_strength_std", 1.5)
        
        # Calculate onset strength statistics
        mean_onset = np.mean(onset_strength)
        std_onset = np.std(onset_strength)
        onset_threshold = mean_onset + (onset_std_threshold * std_onset)
        
        # Convert onset strength frames to time
        # librosa uses hop_length=512 by default at 22050 Hz
        hop_length = 512
        sr = 22050
        onset_strength_times = np.arange(len(onset_strength)) * hop_length / sr
        
        # Filter drops: must have high onset strength nearby (±0.5s window)
        filtered_candidates = []
        window = 0.5  # seconds
        
        for idx in drop_candidates:
            drop_time = times[idx]
            
            # Find onset strength at this time
            time_idx = np.searchsorted(onset_strength_times, drop_time)
            if time_idx >= len(onset_strength):
                time_idx = len(onset_strength) - 1
            
            # Check window around drop time
            window_start = max(0, time_idx - int(window * sr / hop_length))
            window_end = min(len(onset_strength), time_idx + int(window * sr / hop_length))
            
            # If any onset in window exceeds threshold, keep this drop
            if np.max(onset_strength[window_start:window_end]) > onset_threshold:
                filtered_candidates.append(idx)
        
        logger.debug(
            "After onset filter (threshold=%.2f): %d drops (removed %d)",
            onset_threshold, len(filtered_candidates), len(drop_candidates) - len(filtered_candidates)
        )
        drop_candidates = filtered_candidates
    
    if not drop_candidates:
        return []

    # Step 5: Convert sample positions to timestamps
    drop_timings = []
    for idx in drop_candidates:
        drop_time = times[idx]
        drop_timings.append(drop_time)

    # Step 6: Structural timing (currently disabled - needs tuning)
    # TODO: Re-enable after analyzing why it filters out real drops
    structural_drops = drop_timings

    # Step 7: Remove adjacent duplicates (3s threshold)
    final_drops = []
    duplicate_threshold = 3.0
    
    for i, drop_time in enumerate(structural_drops):
        if i == 0 or (drop_time - structural_drops[i-1]) > duplicate_threshold:
            final_drops.append(float(drop_time))
    
    logger.debug(
        "After deduplication: %d drops (removed %d)",
        len(final_drops), len(structural_drops) - len(final_drops)
    )

    # Step 8: Apply minimum spacing constraint
    spaced_drops = []
    for drop in final_drops:
        if not spaced_drops or (drop - spaced_drops[-1]) > min_spacing:
            spaced_drops.append(drop)

    logger.debug(
        "Final result after spacing (%.1fs): %d drops",
        min_spacing, len(spaced_drops)
    )

    return spaced_drops
