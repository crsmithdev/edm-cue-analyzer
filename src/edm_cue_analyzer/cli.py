"""Command-line interface for EDM Cue Analyzer."""

import argparse
import asyncio
import glob
import logging
import sys
import time
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
    log_file: Path | None = None,
    verbose: bool = False,
    track_num: int | None = None,
    total_tracks: int | None = None,
) -> tuple[int, float]:
    """
    Analyze a single track and generate cues.

    Args:
        filepath: Path to audio file
        config: Configuration object
        output_xml: Optional path for XML output
        display: Whether to display results in terminal
        bpm_only: Only detect and display BPM
        log_file: Optional path to append log output to
        verbose: Enable verbose output
        track_num: Current track number (for progress display)
        total_tracks: Total number of tracks (for progress display)

    Returns:
        Tuple of (exit_code, elapsed_time)
    """
    start_time = time.perf_counter()
    
    try:
        # Progress prefix
        progress = f"[{track_num}/{total_tracks}] " if track_num and total_tracks else ""
        
        if verbose:
            logger.info("%sAnalyzing: %s", progress, filepath)
        else:
            # Compact progress for non-verbose
            print(f"{progress}Analyzing: {filepath.name}...", end=" ", flush=True)

        # Analyze audio
        analyzer = AudioAnalyzer(config.analysis)

        if bpm_only:
            # Just detect BPM
            structure = await analyzer.analyze_file(filepath)
            elapsed = time.perf_counter() - start_time
            
            if verbose:
                print(f"\n{'=' * 80}\n")
                print(f"Track: {filepath}")
                print(f"BPM: {structure.bpm:.1f}")
                print(f"Duration: {int(structure.duration // 60)}:{int(structure.duration % 60):02d}")
                print(f"Analysis time: {elapsed:.2f}s")
                print(f"\n{'=' * 80}\n")
            else:
                print(f"BPM: {structure.bpm:.1f} ({elapsed:.1f}s)")
            
            return 0, elapsed

        structure = await analyzer.analyze_file(filepath)

        # Generate cues
        generator = CueGenerator(config)
        cues = generator.generate_cues(structure)
        
        elapsed = time.perf_counter() - start_time

        # Display results
        if display and verbose:
            display_results(
                str(filepath),
                structure,
                cues,
                color_enabled=config.display.get("color_enabled", True),
                waveform_width=config.display.get("waveform_width", 80),
            )
        elif not verbose:
            # Compact summary
            print(f"✓ BPM: {structure.bpm:.1f}, Drops: {len(structure.drops)}, "
                  f"Breakdowns: {len(structure.breakdowns)} ({elapsed:.1f}s)")

        # Export to XML if requested
        if output_xml:
            export_to_rekordbox(
                filepath, cues, structure, output_xml, color_mapping=config.rekordbox_colors
            )
            if verbose:
                print(f"\n✓ Exported to: {output_xml}\n")

        return 0, elapsed

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        if verbose:
            logger.error("Error analyzing %s: %s", filepath, e, exc_info=True)
        else:
            print(f"✗ Error: {e}")
        return 1, elapsed


async def batch_analyze(
    files: list[Path],
    config: Config,
    output_xml: Path | None = None,
    display: bool = True,
    bpm_only: bool = False,
    log_file: Path | None = None,
    verbose: bool = False,
) -> int:
    """
    Analyze multiple tracks.

    Args:
        files: List of audio files to analyze
        config: Configuration object
        output_xml: Optional path for XML output
        display: Whether to display results in terminal
        bpm_only: Only detect and display BPM
        log_file: Optional path to append log output to
        verbose: Enable verbose output

    Returns:
        Exit code (0 for success, 1 if any errors)
    """
    total_files = len(files)
    batch_start = time.perf_counter()
    
    print(f"\n{'=' * 80}")
    print(f"Analyzing {total_files} track{'s' if total_files != 1 else ''}")
    print(f"{'=' * 80}\n")
    
    results = []
    errors = 0
    
    for i, filepath in enumerate(files, 1):
        exit_code, elapsed = await analyze_track(
            filepath,
            config,
            output_xml=output_xml if total_files == 1 else None,  # Only single file XML
            display=display,
            bpm_only=bpm_only,
            log_file=log_file,
            verbose=verbose,
            track_num=i,
            total_tracks=total_files,
        )
        
        results.append((filepath, exit_code, elapsed))
        if exit_code != 0:
            errors += 1
    
    batch_elapsed = time.perf_counter() - batch_start
    
    # Summary
    print(f"\n{'=' * 80}")
    print(f"Batch complete: {total_files - errors}/{total_files} successful")
    print(f"Total time: {batch_elapsed:.1f}s | Avg: {batch_elapsed/total_files:.1f}s/track")
    print(f"{'=' * 80}\n")
    
    if verbose and results:
        print("\nPer-file timing:")
        for filepath, exit_code, elapsed in results:
            status = "✓" if exit_code == 0 else "✗"
            print(f"  {status} {filepath.name}: {elapsed:.2f}s")
        print()
    
    return 1 if errors > 0 else 0


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze EDM tracks and generate DJ cue points",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a track with default config
  edm-cue-analyzer track.mp3

  # Analyze multiple files
  edm-cue-analyzer track1.flac track2.flac track3.flac

  # Analyze with glob pattern
  edm-cue-analyzer "/music/*.flac"

  # Analyze and export to Rekordbox XML (single file only)
  edm-cue-analyzer track.flac -o track_cues.xml

  # Use custom configuration
  edm-cue-analyzer track.mp3 -c my_config.yaml

  # Quiet mode (compact progress)
  edm-cue-analyzer *.flac

  # Verbose mode (detailed per-file output)
  edm-cue-analyzer *.flac --verbose
        """,
    )

    parser.add_argument(
        "input",
        type=str,
        nargs="+",
        help="Audio file(s) to analyze (.mp3, .flac, .wav, etc.) or glob pattern",
    )

    parser.add_argument("-o", "--output", type=Path, help="Output Rekordbox XML file path")

    parser.add_argument("-c", "--config", type=Path, help="Custom configuration YAML file")

    parser.add_argument(
        "--no-display", action="store_true", help="Disable terminal display of results"
    )

    parser.add_argument(
        "--bpm-only", action="store_true", help="Only detect and display BPM (skip full analysis)"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")

    parser.add_argument(
        "--log-file",
        type=Path,
        help="Append all output (stdout and logs) to specified file",
    )

    parser.add_argument("-v", "--version", action="version", version="EDM Cue Analyzer 1.0.0")

    args = parser.parse_args()

    # Configure logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        if args.verbose
        else "%(levelname)s: %(message)s"
    )

    # Setup logging handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        # Append to log file
        file_handler = logging.FileHandler(args.log_file, mode='a')
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)

    # Set log level for edm_cue_analyzer package
    logging.getLogger("edm_cue_analyzer").setLevel(log_level)

    # Suppress overly verbose libraries
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # Expand input files (handle globs)
    files = []
    for pattern in args.input:
        # Try as glob pattern first
        matches = glob.glob(pattern, recursive=True)
        if matches:
            files.extend([Path(f) for f in matches])
        else:
            # Try as direct path
            p = Path(pattern)
            if p.exists():
                files.append(p)
            else:
                logger.error("File not found: %s", pattern)
                return 1
    
    # Remove duplicates, keep order
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    files = unique_files
    
    if not files:
        logger.error("No files to analyze")
        return 1

    # Load configuration
    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as e:
        logger.error("Error loading configuration: %s", e, exc_info=True)
        return 1

    # Analyze tracks
    return asyncio.run(
        batch_analyze(
            files,
            config,
            output_xml=args.output,
            display=not args.no_display,
            bpm_only=args.bpm_only,
            log_file=args.log_file,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
