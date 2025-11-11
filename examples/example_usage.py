#!/usr/bin/env python3
"""
Example script demonstrating library usage.

This shows how to use edm_cue_analyzer as a Python library
rather than a command-line tool.
"""

import asyncio
from pathlib import Path

from edm_cue_analyzer import (
    AudioAnalyzer,
    CueGenerator,
    display_results,
    export_to_rekordbox,
    load_config,
)


async def analyze_single_track(audio_file: str, output_xml: str = None):
    """Analyze a single track and optionally export."""

    print(f"Analyzing: {audio_file}\n")

    # 1. Load configuration (default or custom)
    config = load_config()  # Or load_config(Path("custom_config.yaml"))

    # 2. Analyze the audio file
    analyzer = AudioAnalyzer(config.analysis)
    structure = await analyzer.analyze_file(Path(audio_file))

    print(f"Detected BPM: {structure.bpm:.1f}")
    print(f"Duration: {structure.duration:.1f}s")
    print(f"Drops found: {len(structure.drops)}")
    print(f"Breakdowns found: {len(structure.breakdowns)}")
    print()

    # 3. Generate cue points
    generator = CueGenerator(config)
    cues = generator.generate_cues(structure)

    print(f"Generated {len(cues)} cue points")

    # 4. Display results in terminal
    display_results(audio_file, structure, cues)

    # 5. Export to Rekordbox XML (optional)
    if output_xml:
        export_to_rekordbox(
            Path(audio_file),
            cues,
            structure,
            Path(output_xml)
        )
        print(f"\n✓ Exported to: {output_xml}")


async def batch_analyze_tracks(audio_directory: str, output_directory: str):
    """Analyze multiple tracks in a directory concurrently."""

    audio_dir = Path(audio_directory)
    output_dir = Path(output_directory)
    output_dir.mkdir(exist_ok=True)

    # Find all audio files
    audio_files = []
    for ext in ['*.mp3', '*.flac', '*.wav']:
        audio_files.extend(audio_dir.glob(ext))

    print(f"Found {len(audio_files)} audio files\n")

    # Load config once
    config = load_config()
    analyzer = AudioAnalyzer(config.analysis)
    generator = CueGenerator(config)

    async def process_file(audio_file: Path):
        """Process a single file."""
        try:
            print(f"Processing: {audio_file.name}")

            # Analyze
            structure = await analyzer.analyze_file(audio_file)

            # Generate cues
            cues = generator.generate_cues(structure)

            # Export XML
            output_xml = output_dir / f"{audio_file.stem}_cues.xml"
            export_to_rekordbox(audio_file, cues, structure, output_xml)

            print(f"  ✓ Exported to: {output_xml.name}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

        print()

    # Process files concurrently
    await asyncio.gather(*[process_file(f) for f in audio_files])


async def custom_cue_workflow(audio_file: str):
    """Example of customizing cue generation programmatically."""

    from edm_cue_analyzer.config import AnalysisConfig, Config, CueConfig

    # Create custom configuration programmatically
    config = Config()

    # Define custom hot cues
    config.hot_cues = {
        'A': CueConfig(
            name="Intro",
            position_percent=0.05,
            loop_bars=16,
            color="BLUE"
        ),
        'B': CueConfig(
            name="Main Drop",
            position_method="first_drop",
            offset_bars=0,
            loop_bars=8,
            color="RED"
        ),
        'C': CueConfig(
            name="Breakdown",
            position_method="first_breakdown",
            offset_bars=0,
            loop_bars=16,
            color="TEAL"
        ),
        'D': CueConfig(
            name="Outro",
            position_percent=0.85,
            loop_bars=16,
            color="GREEN"
        ),
    }

    # Custom analysis parameters
    config.analysis = AnalysisConfig(
        energy_window_seconds=4.0,
        energy_threshold_increase=0.20,
        drop_energy_multiplier=1.4
    )

    # Analyze with custom config
    analyzer = AudioAnalyzer(config.analysis)
    structure = await analyzer.analyze_file(Path(audio_file))

    generator = CueGenerator(config)
    cues = generator.generate_cues(structure)

    print(f"Generated {len(cues)} custom cues for {audio_file}")
    for cue in cues:
        print(f"  {cue.label}: {cue.position:.1f}s")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python example_usage.py <audio_file> [output.xml]")
        print("  python example_usage.py --batch <audio_dir> <output_dir>")
        sys.exit(1)

    if sys.argv[1] == '--batch':
        if len(sys.argv) < 4:
            print("Batch mode requires audio directory and output directory")
            sys.exit(1)
        asyncio.run(batch_analyze_tracks(sys.argv[2], sys.argv[3]))
    else:
        audio_file = sys.argv[1]
        output_xml = sys.argv[2] if len(sys.argv) > 2 else None
        asyncio.run(analyze_single_track(audio_file, output_xml))
