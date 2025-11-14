"""Configuration management for EDM Cue Analyzer."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CueConfig:
    """Configuration for a single cue point."""

    name: str
    description: str = ""
    position_percent: float | None = None
    position_method: str | None = None
    offset_bars: int = 0
    loop_bars: int | None = None
    color: str = "GREEN"


@dataclass
class AnalysisConfig:
    """Configuration for audio analysis parameters."""

    # Energy analysis
    energy_window_seconds: float = 1.0
    energy_threshold_increase: float = 0.15

    # Frequency bands
    low_freq_max: int = 250
    mid_freq_max: int = 4000

    # Section detection
    min_section_duration: float = 8.0

    # Drop detection thresholds
    drop_energy_multiplier: float = 1.3
    drop_onset_strength_std: float = 1.5  # Standard deviations above mean for onset
    drop_energy_std: float = 0.7  # Standard deviations above mean for energy
    drop_max_energy_threshold: float = 0.60  # Percentage of max energy
    drop_min_spacing_bars: int = 28  # Minimum bars between drops
    drop_lookback_seconds: float = 5.0  # How far back to look for energy average

    # Breakdown detection thresholds
    breakdown_energy_threshold: float = 0.6
    breakdown_min_duration_bars: int = 4
    breakdown_min_spacing_bars: int = 16
    breakdown_perc_threshold: float = 0.6  # Percussive energy threshold
    breakdown_ratio_threshold: float = 1.1  # Harmonic/percussive ratio

    # Build detection
    build_window_size: int = 5  # Number of samples to check for sustained increase


@dataclass
class Config:
    """Main configuration object."""

    hot_cues: dict[str, CueConfig] = field(default_factory=dict)
    memory_cues: list = field(default_factory=list)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    rekordbox_colors: dict[str, str] = field(default_factory=dict)
    display: dict[str, Any] = field(default_factory=dict)


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to custom config file. If None, uses default.

    Returns:
        Config object with loaded settings.
    """
    if config_path is None:
        # Use default config
        default_path = Path(__file__).parent / "default_config.yaml"
        config_path = default_path

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # Parse hot cues
    hot_cues = {}
    for cue_id, cue_data in raw_config.get("hot_cues", {}).items():
        hot_cues[cue_id] = CueConfig(
            name=cue_data["name"],
            description=cue_data.get("description", ""),
            position_percent=cue_data.get("position_percent"),
            position_method=cue_data.get("position_method"),
            offset_bars=cue_data.get("offset_bars", 0),
            loop_bars=cue_data.get("loop_bars"),
            color=cue_data.get("color", "GREEN"),
        )

    # Parse memory cues
    memory_cues = raw_config.get("memory_cues", [])

    # Parse analysis config
    analysis_data = raw_config.get("analysis", {})
    analysis = AnalysisConfig(
        energy_window_seconds=analysis_data.get("energy_window_seconds", 1.0),
        energy_threshold_increase=analysis_data.get("energy_threshold_increase", 0.15),
        low_freq_max=analysis_data.get("low_freq_max", 250),
        mid_freq_max=analysis_data.get("mid_freq_max", 4000),
        min_section_duration=analysis_data.get("min_section_duration", 8.0),
        drop_energy_multiplier=analysis_data.get("drop_energy_multiplier", 1.3),
        drop_onset_strength_std=analysis_data.get("drop_onset_strength_std", 1.5),
        drop_energy_std=analysis_data.get("drop_energy_std", 0.7),
        drop_max_energy_threshold=analysis_data.get("drop_max_energy_threshold", 0.60),
        drop_min_spacing_bars=analysis_data.get("drop_min_spacing_bars", 28),
        breakdown_energy_threshold=analysis_data.get("breakdown_energy_threshold", 0.6),
        breakdown_min_duration_bars=analysis_data.get("breakdown_min_duration_bars", 4),
        breakdown_min_spacing_bars=analysis_data.get("breakdown_min_spacing_bars", 16),
        breakdown_perc_threshold=analysis_data.get("breakdown_perc_threshold", 0.6),
        breakdown_ratio_threshold=analysis_data.get("breakdown_ratio_threshold", 1.1),
        build_window_size=analysis_data.get("build_window_size", 5),
    )

    # Parse Rekordbox colors
    rekordbox_colors = raw_config.get("rekordbox", {}).get("colors", {})

    # Parse display settings
    display = raw_config.get("display", {})

    return Config(
        hot_cues=hot_cues,
        memory_cues=memory_cues,
        analysis=analysis,
        rekordbox_colors=rekordbox_colors,
        display=display,
    )


def get_default_config() -> Config:
    """Get default configuration."""
    return load_config()
