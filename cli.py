"""Command-line interface for EDM Cue Analyzer."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .config import load_config, Config
from .analyzer import AudioAnalyzer
from .cue_generator import CueGenerator
from .rekordbox import export_to_rekordbox
from .display import display_results


def analyze_track(
    filepath: Path,
    config: Config,
    output_xml: Optional[Path] = None,
    display: bool = True
) -> int:
    """
    Analyze a single track and generate cues.
    
    Args:
        filepath: Path to audio file
        config: Configuration object
        output_xml: Optional path for XML output
        display: Whether to display results in terminal
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        print(f"Analyzing: {filepath}")
        
        # Analyze audio
        analyzer = AudioAnalyzer(config.analysis)
        structure = analyzer.analyze_file(filepath)
        
        # Generate cues
        generator = CueGenerator(config)
        cues = generator.generate_cues(structure)
        
        # Display results
        if display:
            display_results(
                str(filepath),
                structure,
                cues,
                color_enabled=config.display.get('color_enabled', True),
                waveform_width=config.display.get('waveform_width', 80)
            )
        
        # Export to XML if requested
        if output_xml:
            export_to_rekordbox(
                filepath,
                cues,
                structure,
                output_xml,
                color_mapping=config.rekordbox_colors
            )
            print(f"\n✓ Exported to: {output_xml}\n")
        
        return 0
        
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}", file=sys.stderr)
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze EDM tracks and generate DJ cue points',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a track with default config
  edm-cue-analyzer track.mp3

  # Analyze and export to Rekordbox XML
  edm-cue-analyzer track.flac -o track_cues.xml

  # Use custom configuration
  edm-cue-analyzer track.mp3 -c my_config.yaml

  # Analyze without terminal display
  edm-cue-analyzer track.mp3 --no-display -o output.xml
        """
    )
    
    parser.add_argument(
        'input',
        type=Path,
        help='Audio file to analyze (.mp3, .flac, .wav, etc.)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output Rekordbox XML file path'
    )
    
    parser.add_argument(
        '-c', '--config',
        type=Path,
        help='Custom configuration YAML file'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable terminal display of results'
    )
    
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='EDM Cue Analyzer 1.0.0'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        return 1
    
    # Load configuration
    try:
        if args.config:
            config = load_config(args.config)
        else:
            config = load_config()
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1
    
    # Analyze track
    return analyze_track(
        args.input,
        config,
        output_xml=args.output,
        display=not args.no_display
    )


if __name__ == '__main__':
    sys.exit(main())
