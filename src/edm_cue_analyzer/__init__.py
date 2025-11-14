"""EDM Cue Analyzer - Automated cue point generation for DJ performance."""

__version__ = "1.0.0"
__author__ = "EDM Cue Analyzer Project"

from .analyzer import (
    AudioAnalyzer,
    FeatureExtractor,
    HPSSFeatureExtractor,
    OnsetFeatureExtractor,
    SpectralFeatureExtractor,
    TrackStructure,
)
from .config import Config, get_default_config, load_config
from .cue_generator import CueGenerator, CuePoint
from .display import TerminalDisplay, display_results
from .rekordbox import RekordboxXMLExporter, export_to_rekordbox

__all__ = [
    # Config
    "Config",
    "load_config",
    "get_default_config",
    # Analysis
    "AudioAnalyzer",
    "TrackStructure",
    "FeatureExtractor",
    "HPSSFeatureExtractor",
    "SpectralFeatureExtractor",
    "OnsetFeatureExtractor",
    # Cue generation
    "CueGenerator",
    "CuePoint",
    # Export
    "RekordboxXMLExporter",
    "export_to_rekordbox",
    # Display
    "TerminalDisplay",
    "display_results",
]
