# Quick Start Guide - EDM Cue Analyzer

## Installation (30 seconds)

```bash
cd edm_cue_analyzer
pip install -e .
```

That's it! The `edm-cue-analyzer` command is now available.

## Your First Analysis (1 minute)

```bash
# Basic analysis with terminal display
edm-cue-analyzer your_track.mp3

# Generate Rekordbox XML
edm-cue-analyzer your_track.mp3 -o cues.xml
```

You'll see:
- Track info (BPM, duration)
- Detected structure (drops, breakdowns, builds)
- 8 hot cues (A-H) with colors and positions
- 5 memory cues
- ASCII timeline showing cue positions

## Import to Rekordbox (30 seconds)

1. In Rekordbox: File → Import → XML
2. Select the generated XML file
3. Cues appear on your track automatically!

## Common Workflows

### Single Track Analysis
```bash
edm-cue-analyzer track.flac -o track_cues.xml
```

### Batch Process a Folder
```bash
# Use the example script
python examples/example_usage.py --batch /path/to/music /path/to/output
```

### Custom Configuration
```bash
# Create my_config.yaml (see examples/custom_config.yaml)
edm-cue-analyzer track.mp3 -c my_config.yaml -o output.xml
```

## Using as a Python Library

```python
from pathlib import Path
from edm_cue_analyzer import (
    load_config,
    AudioAnalyzer, 
    CueGenerator,
    export_to_rekordbox
)

# Load config
config = load_config()

# Analyze
analyzer = AudioAnalyzer(config.analysis)
structure = analyzer.analyze_file(Path("track.mp3"))

# Generate cues
generator = CueGenerator(config)
cues = generator.generate_cues(structure)

# Export
export_to_rekordbox(
    Path("track.mp3"),
    cues,
    structure,
    Path("output.xml")
)
```

## Understanding the Output

### Hot Cue Colors
- **BLUE**: Minimal/safest (low complexity)
- **GREEN**: Stable loop (safe for live play)
- **TEAL**: Structured phrase (needs timing)
- **YELLOW**: Melodic highlight
- **ORANGE**: Build/timing tool (headphones only)
- **RED**: Instant jump (drop swaps)

### Hot Cue Layout
- **A**: Early intro blend (8%)
- **B**: Late intro blend (20%)
- **C**: Pre-drop setup (detected)
- **D**: Post-drop exit (detected)
- **E**: Breakdown (detected)
- **F**: Second energy point (detected)
- **G**: Outro start (85%)
- **H**: Safe hold / emergency (50%)

## Customization

Edit `default_config.yaml` or create your own:

```yaml
hot_cues:
  A:
    name: "My Custom Cue"
    position_percent: 0.10  # 10% into track
    loop_bars: 16
    color: "GREEN"
```

Position methods:
- `position_percent`: Fixed percentage (0.0-1.0)
- `position_method`: Structure-based
  - `first_drop`, `second_drop`
  - `first_breakdown`
  - `before_first_drop`, `after_first_drop`
- `offset_bars`: Shift from detected position (±bars)

## Troubleshooting

**"No drops detected"**
- Normal for minimal/ambient tracks
- Falls back to percentage-based positioning
- Adjust `analysis.drop_energy_multiplier` in config

**"BPM seems wrong"**
- librosa sometimes detects half/double BPM
- Future versions will add manual BPM override

**"Cues not at exact moments I want"**
- Use `offset_bars` to fine-tune
- Customize `position_percent` for fixed positions
- Edit config to match your mixing style

## Examples Directory

Check `examples/` for:
- `custom_config.yaml`: Alternative cue layout
- `example_usage.py`: Batch processing, library usage

## Need Help?

1. Check `README.md` for full documentation
2. Check `PROJECT_SUMMARY.md` for architecture details
3. Run tests: `pytest tests/ -v`
4. View help: `edm-cue-analyzer --help`

## Supported Formats

- MP3 (320kbps, 128kbps, any bitrate)
- FLAC (lossless)
- WAV (uncompressed)
- M4A, OGG, and most formats librosa supports

## Next Steps

1. Analyze a few tracks to see how it performs
2. Adjust `default_config.yaml` to match your style
3. Create custom configs for different genres
4. Integrate into your DJ workflow!

Happy mixing! 🎧
