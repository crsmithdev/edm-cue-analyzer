"""EDM Cue Analyzer - Automated cue point generation for DJ performance."""

__version__ = "1.0.0"
__author__ = "EDM Cue Analyzer Project"

from .config import Config, load_config, get_default_config
from .analyzer import AudioAnalyzer, TrackStructure
from .cue_generator import CueGenerator, CuePoint
from .rekordbox import RekordboxXMLExporter, export_to_rekordbox
from .display import TerminalDisplay, display_results

__all__ = [
    # Config
    'Config',
    'load_config',
    'get_default_config',
    
    # Analysis
    'AudioAnalyzer',
    'TrackStructure',
    
    # Cue generation
    'CueGenerator',
    'CuePoint',
    
    # Export
    'RekordboxXMLExporter',
    'export_to_rekordbox',
    
    # Display
    'TerminalDisplay',
    'display_results',
]
