"""Drop detection analysis."""

import logging

import numpy as np
from scipy import interpolate

logger = logging.getLogger(__name__)


async def analyze_drops(context: dict) -> list[float]:
    """
    Detect drop points (beat/bass returns after breakdowns or track start).

    In EDM, a "drop" is when the beat/bass returns in full force, either:
    1. Initial drop: First strong beat establishment
    2. Post-breakdown drop: Beat returns after being stripped away

    Args:
        context: Dictionary containing:
            - bpm: BpmResult from BPM analysis
            - energy: EnergyResult from energy analysis
            - features: Optional dict of extracted features (onsets, HPSS, etc.)
            - config: Optional analysis config

    Returns:
        List of timestamps where drops occur
    """
    bpm_result = context["bpm"]
    energy_result = context["energy"]
    features = context.get("features", {})
    config = context.get("config")

    energy = energy_result.curve
    times = energy_result.times
    bar_duration = bpm_result.bar_duration

    # Get config values
    drop_min_spacing_bars = 8
    energy_window_seconds = 0.1
    if config:
        drop_min_spacing_bars = getattr(config, "drop_min_spacing_bars", 8)
        energy_window_seconds = getattr(config, "energy_window_seconds", 0.1)

    drops = []
    min_spacing = bar_duration * drop_min_spacing_bars

    # Check available features
    has_onsets = "onset_times" in features and "onset_strength" in features

    if not has_onsets:
        logger.warning("No onset detection available, drop detection may be inaccurate")
        return drops

    # Get features
    onset_times = features["onset_times"]
    onset_strength = features["onset_strength"]
    low_energy = features.get("low_energy", energy)  # Fallback to overall energy
    percussive_energy = features.get("percussive_energy", energy)

    # Align low_energy and percussive_energy to times array
    if len(low_energy) != len(times):
        low_energy_interp = interpolate.interp1d(
            np.linspace(0, times[-1], len(low_energy)),
            low_energy,
            kind="linear",
            fill_value="extrapolate",
        )
        low_energy = low_energy_interp(times)

    if len(percussive_energy) != len(times):
        perc_energy_interp = interpolate.interp1d(
            np.linspace(0, times[-1], len(percussive_energy)),
            percussive_energy,
            kind="linear",
            fill_value="extrapolate",
        )
        percussive_energy = perc_energy_interp(times)

    # Calculate statistics
    mean_low = np.mean(low_energy)
    std_low = np.std(low_energy)
    mean_perc = np.mean(percussive_energy)
    std_perc = np.std(percussive_energy)
    mean_onset = np.mean(onset_strength)
    std_onset = np.std(onset_strength)

    # Thresholds for "significant" bass/beat presence
    low_threshold = mean_low + 0.3 * std_low
    perc_threshold = mean_perc + 0.3 * std_perc
    onset_threshold = mean_onset + 0.8 * std_onset  # Strong onset required

    logger.debug(
        "Drop detection: Low threshold=%.4f, Perc threshold=%.4f, Onset threshold=%.4f",
        low_threshold,
        perc_threshold,
        onset_threshold,
    )

    # Strategy: Find strong onsets where bass/percussion RETURNS after being low
    lookback_seconds = 8.0  # Look back 8 seconds to see if bass was low
    lookback_frames = int(lookback_seconds / energy_window_seconds)

    for onset_time in onset_times:
        # Find closest time index
        idx = np.argmin(np.abs(times - onset_time))

        if idx < lookback_frames or idx >= len(times) - 2:
            continue

        # Get onset strength
        onset_idx = int(onset_time * len(onset_strength) / times[-1])
        if onset_idx >= len(onset_strength):
            continue

        # Check if this is a strong onset
        if onset_strength[onset_idx] < onset_threshold:
            continue

        # Check if low-frequency (bass) and percussion are present at drop
        has_bass = low_energy[idx] > low_threshold
        has_beat = percussive_energy[idx] > perc_threshold

        if not (has_bass and has_beat):
            continue

        # KEY: Check if bass/percussion were LOW recently (breakdown before drop)
        # or if this is the first strong bass/beat (initial drop)
        lookback_start = max(0, idx - lookback_frames)
        recent_low_avg = np.mean(low_energy[lookback_start:idx])
        recent_perc_avg = np.mean(percussive_energy[lookback_start:idx])

        # Drop detected if:
        # 1. Bass/beat are NOW strong, AND
        # 2. Bass/beat were recently weak (indicating a return/drop moment)
        bass_increase = low_energy[idx] > recent_low_avg * 1.4
        beat_increase = percussive_energy[idx] > recent_perc_avg * 1.3

        # Also allow early track drops where bass simply starts strong
        is_early_drop = onset_time < 60  # First minute
        bass_strong = low_energy[idx] > mean_low + std_low

        # Check minimum spacing and add drop
        is_valid_drop = (bass_increase and beat_increase) or (
            is_early_drop and bass_strong and has_beat
        )
        if is_valid_drop and (not drops or (onset_time - drops[-1]) > min_spacing):
            drops.append(float(onset_time))
            logger.debug(
                "Drop detected at %.2fs (low: %.4f, perc: %.4f, onset: %.4f)",
                onset_time,
                low_energy[idx],
                percussive_energy[idx],
                onset_strength[onset_idx],
            )

    return drops
