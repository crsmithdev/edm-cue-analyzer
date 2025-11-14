"""Breakdown detection analysis."""

import logging

import numpy as np
from scipy import interpolate

logger = logging.getLogger(__name__)


async def analyze_breakdowns(context: dict) -> list[float]:
    """
    Detect breakdown points using combined energy and available features.

    Uses spectral complexity and HFC from Essentia if available for better accuracy.

    Args:
        context: Dictionary containing:
            - bpm: BpmResult from BPM analysis
            - energy: EnergyResult from energy analysis
            - features: Optional dict of extracted features (spectral, HPSS, etc.)
            - config: Optional analysis config

    Returns:
        List of timestamps where breakdowns occur
    """
    bpm_result = context["bpm"]
    energy_result = context["energy"]
    features = context.get("features", {})
    config = context.get("config")

    energy = energy_result.curve
    times = energy_result.times
    bar_duration = bpm_result.bar_duration

    # Get config values
    breakdown_energy_threshold = 0.35
    breakdown_min_duration_bars = 4
    breakdown_min_spacing_bars = 8
    breakdown_perc_threshold = 0.5
    breakdown_ratio_threshold = 1.5

    if config:
        breakdown_energy_threshold = getattr(config, "breakdown_energy_threshold", 0.35)
        breakdown_min_duration_bars = getattr(config, "breakdown_min_duration_bars", 4)
        breakdown_min_spacing_bars = getattr(config, "breakdown_min_spacing_bars", 8)
        breakdown_perc_threshold = getattr(config, "breakdown_perc_threshold", 0.5)
        breakdown_ratio_threshold = getattr(config, "breakdown_ratio_threshold", 1.5)

    breakdowns = []

    max_energy = np.max(energy)
    mean_energy = np.mean(energy)

    # Check for Essentia spectral features (best for breakdown detection)
    use_spectral = "spectral_complexity" in features and "hfc" in features
    use_hpss = "harmonic_energy" in features and "percussive_energy" in features

    if use_spectral:
        logger.debug("Breakdown detection mode: Essentia spectral-based")
    elif use_hpss:
        logger.debug("Breakdown detection mode: HPSS-based")
    else:
        logger.debug("Breakdown detection mode: energy-based")

    # Setup thresholds based on available features
    if use_spectral:
        spectral_complexity = features["spectral_complexity"]
        hfc = features["hfc"]

        # Align spectral features with energy curve
        if len(spectral_complexity) != len(energy):
            old_indices = np.linspace(0, len(times) - 1, len(spectral_complexity))
            new_indices = np.arange(len(times))
            f_complexity = interpolate.interp1d(
                old_indices, spectral_complexity, kind="linear", fill_value="extrapolate"
            )
            f_hfc = interpolate.interp1d(
                old_indices, hfc, kind="linear", fill_value="extrapolate"
            )
            spectral_complexity = f_complexity(new_indices)
            hfc = f_hfc(new_indices)

        mean_complexity = np.mean(spectral_complexity)
        mean_hfc = np.mean(hfc)

        # Breakdowns have low complexity and low HFC
        complexity_threshold = mean_complexity * 0.7
        hfc_threshold = mean_hfc * 0.6
        energy_threshold = min(max_energy * breakdown_energy_threshold, mean_energy * 0.85)

    elif use_hpss:
        harmonic_energy = features["harmonic_energy"]
        percussive_energy = features["percussive_energy"]
        mean_perc = np.mean(percussive_energy)

        # Calculate ratio of harmonic to percussive
        epsilon = 1e-10
        harmonic_ratio = harmonic_energy / (percussive_energy + epsilon)
        mean_ratio = np.mean(harmonic_ratio)

        # Breakdown thresholds
        energy_threshold = min(max_energy * breakdown_energy_threshold, mean_energy * 0.85)
        perc_threshold = mean_perc * breakdown_perc_threshold
        ratio_threshold = mean_ratio * breakdown_ratio_threshold
    else:
        # Fallback to simple energy-based detection
        energy_threshold = min(max_energy * breakdown_energy_threshold, mean_energy * 0.85)

    in_breakdown = False
    breakdown_start = None
    breakdown_start_idx = None

    for i in range(len(energy)):
        # Determine if this is breakdown-like based on available features
        if use_spectral:
            # Essentia-based: low energy + low complexity + low HFC = breakdown
            is_breakdown_like = (
                energy[i] < energy_threshold
                and spectral_complexity[i] < complexity_threshold
                and hfc[i] < hfc_threshold
            )
        elif use_hpss:
            is_breakdown_like = energy[i] < energy_threshold and (
                percussive_energy[i] < perc_threshold or harmonic_ratio[i] > ratio_threshold
            )
        else:
            is_breakdown_like = energy[i] < energy_threshold

        if is_breakdown_like and not in_breakdown:
            breakdown_start = times[i]
            breakdown_start_idx = i
            in_breakdown = True
        elif not is_breakdown_like and in_breakdown:
            # Check if breakdown was long enough
            min_duration = bar_duration * breakdown_min_duration_bars
            if breakdown_start and (times[i] - breakdown_start) >= min_duration:
                # Verify it's a significant breakdown
                breakdown_avg = np.mean(energy[breakdown_start_idx:i])
                valid_breakdown = breakdown_avg < mean_energy * 0.9

                if use_spectral:
                    # Essentia validation: breakdown region should have low complexity
                    breakdown_complexity_avg = np.mean(spectral_complexity[breakdown_start_idx:i])
                    valid_breakdown = valid_breakdown and (
                        breakdown_complexity_avg < mean_complexity * 0.8
                    )
                elif use_hpss:
                    breakdown_perc_avg = np.mean(percussive_energy[breakdown_start_idx:i])
                    valid_breakdown = valid_breakdown and (breakdown_perc_avg < mean_perc * 0.8)

                if valid_breakdown:
                    # Ensure minimum spacing between breakdowns
                    min_spacing = bar_duration * breakdown_min_spacing_bars
                    if not breakdowns or (breakdown_start - breakdowns[-1]) > min_spacing:
                        breakdowns.append(float(breakdown_start))
            in_breakdown = False
            breakdown_start = None
            breakdown_start_idx = None

    return breakdowns
