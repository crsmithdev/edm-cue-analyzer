"""Command-line interface for EDM Cue Analyzer."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .analyzer import AudioAnalyzer
from .config import Config, load_config
from .cue_generator import CueGenerator
from .display import display_results
from .rekordbox import export_to_rekordbox

logger = logging.getLogger(__name__)


async def analyze_track(
    filepath: Path,
    config: Config,
    output_xml: Path | None = None,
    display: bool = True,
    bpm_only: bool = False,
) -> int:
    """
    Analyze a single track and generate cues.

    Args:
        filepath: Path to audio file
        config: Configuration object
        output_xml: Optional path for XML output
        display: Whether to display results in terminal
        bpm_only: Only detect and display BPM

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        logger.info("Analyzing: %s", filepath)

        # Analyze audio
        analyzer = AudioAnalyzer(config.analysis)

        if bpm_only:
            # Just detect BPM
            structure = await analyzer.analyze_file(filepath)
            print(f"\n{'=' * 80}\n")
            print(f"Track: {filepath}")
            print(f"BPM: {structure.bpm:.1f}")
            print(f"Duration: {int(structure.duration // 60)}:{int(structure.duration % 60):02d}")
            print(f"\n{'=' * 80}\n")
            return 0

        structure = await analyzer.analyze_file(filepath)

        # Generate cues
        generator = CueGenerator(config)
        cues = generator.generate_cues(structure)

        # Display results
        if display:
            display_results(
                str(filepath),
                structure,
                cues,
                color_enabled=config.display.get("color_enabled", True),
                waveform_width=config.display.get("waveform_width", 80),
            )

        # Export to XML if requested
        if output_xml:
            export_to_rekordbox(
                filepath, cues, structure, output_xml, color_mapping=config.rekordbox_colors
            )
            print(f"\n✓ Exported to: {output_xml}\n")

        return 0

    except Exception as e:
        logger.error("Error analyzing %s: %s", filepath, e, exc_info=True)
        return 1


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze EDM tracks and generate DJ cue points",
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

  # Enable verbose debug logging
  edm-cue-analyzer track.mp3 --verbose
        """,
    )

    parser.add_argument("input", type=Path, help="Audio file to analyze (.mp3, .flac, .wav, etc.)")

    parser.add_argument("-o", "--output", type=Path, help="Output Rekordbox XML file path")

    parser.add_argument("-c", "--config", type=Path, help="Custom configuration YAML file")

    parser.add_argument(
        "--no-display", action="store_true", help="Disable terminal display of results"
    )

    parser.add_argument(
        "--bpm-only", action="store_true", help="Only detect and display BPM (skip full analysis)"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")

    parser.add_argument("-v", "--version", action="version", version="EDM Cue Analyzer 1.0.0")

    args = parser.parse_args()

    # Configure logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        if args.verbose
        else "%(levelname)s: %(message)s"
    )
    logging.basicConfig(level=log_level, format=log_format)

    # Set log level for edm_cue_analyzer package
    logging.getLogger("edm_cue_analyzer").setLevel(log_level)

    # Suppress overly verbose libraries
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # Validate input file
    if not args.input.exists():
        logger.error("File not found: %s", args.input)
        return 1

    # Load configuration
    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as e:
        logger.error("Error loading configuration: %s", e, exc_info=True)
        return 1

    # Analyze track
    return asyncio.run(
        analyze_track(
            args.input,
            config,
            output_xml=args.output,
            display=not args.no_display,
            bpm_only=args.bpm_only,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
