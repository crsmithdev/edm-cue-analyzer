"""Configuration management for EDM Cue Analyzer."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class CueConfig:
    """Configuration for a single cue point."""
    name: str
    description: str = ""
    position_percent: Optional[float] = None
    position_method: Optional[str] = None
    offset_bars: int = 0
    loop_bars: Optional[int] = None
    color: str = "GREEN"


@dataclass
class AnalysisConfig:
    """Configuration for audio analysis parameters."""
    energy_window_seconds: float = 5.0
    energy_threshold_increase: float = 0.15
    low_freq_max: int = 250
    mid_freq_max: int = 4000
    min_section_duration: float = 8.0
    drop_energy_multiplier: float = 1.3
    breakdown_energy_threshold: float = 0.6


@dataclass
class Config:
    """Main configuration object."""
    hot_cues: Dict[str, CueConfig] = field(default_factory=dict)
    memory_cues: list = field(default_factory=list)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    rekordbox_colors: Dict[str, str] = field(default_factory=dict)
    display: Dict[str, Any] = field(default_factory=dict)


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to custom config file. If None, uses default.
        
    Returns:
        Config object with loaded settings.
    """
    if config_path is None:
        # Use default config
        default_path = Path(__file__).parent.parent / "default_config.yaml"
        config_path = default_path
    
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    # Parse hot cues
    hot_cues = {}
    for cue_id, cue_data in raw_config.get('hot_cues', {}).items():
        hot_cues[cue_id] = CueConfig(
            name=cue_data['name'],
            description=cue_data.get('description', ''),
            position_percent=cue_data.get('position_percent'),
            position_method=cue_data.get('position_method'),
            offset_bars=cue_data.get('offset_bars', 0),
            loop_bars=cue_data.get('loop_bars'),
            color=cue_data.get('color', 'GREEN')
        )
    
    # Parse memory cues
    memory_cues = raw_config.get('memory_cues', [])
    
    # Parse analysis config
    analysis_data = raw_config.get('analysis', {})
    analysis = AnalysisConfig(
        energy_window_seconds=analysis_data.get('energy_window_seconds', 5.0),
        energy_threshold_increase=analysis_data.get('energy_threshold_increase', 0.15),
        low_freq_max=analysis_data.get('low_freq_max', 250),
        mid_freq_max=analysis_data.get('mid_freq_max', 4000),
        min_section_duration=analysis_data.get('min_section_duration', 8.0),
        drop_energy_multiplier=analysis_data.get('drop_energy_multiplier', 1.3),
        breakdown_energy_threshold=analysis_data.get('breakdown_energy_threshold', 0.6)
    )
    
    # Parse Rekordbox colors
    rekordbox_colors = raw_config.get('rekordbox', {}).get('colors', {})
    
    # Parse display settings
    display = raw_config.get('display', {})
    
    return Config(
        hot_cues=hot_cues,
        memory_cues=memory_cues,
        analysis=analysis,
        rekordbox_colors=rekordbox_colors,
        display=display
    )


def get_default_config() -> Config:
    """Get default configuration."""
    return load_config()
