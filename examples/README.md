# EDM Cue Analyzer Examples

This directory contains example scripts and configurations for using EDM Cue Analyzer.

## Files

- `example_usage.py` - Demonstrates how to use the library programmatically
- `custom_config.yaml` - Example custom configuration file

## Basic Usage

### Command Line

```bash
# Analyze a single track
edm-cue-analyzer track.mp3

# Export to Rekordbox XML
edm-cue-analyzer track.flac -o cues.xml

# Use custom configuration
edm-cue-analyzer track.mp3 -c custom_config.yaml
```

### Python Library

```python
from pathlib import Path
from edm_cue_analyzer import load_config, AudioAnalyzer, CueGenerator
from edm_cue_analyzer import export_to_rekordbox, display_results

# Load configuration
config = load_config()

# Analyze audio file
analyzer = AudioAnalyzer(config.analysis)
structure = analyzer.analyze_file(Path("track.mp3"))

# Generate cue points
generator = CueGenerator(config)
cues = generator.generate_cues(structure)

# Display results
display_results(cues, structure)

# Export to Rekordbox
export_to_rekordbox(Path("track.mp3"), cues, structure, Path("output.xml"))
```

## Custom Configuration

See `custom_config.yaml` for an example of how to customize:
- Hot cue positions and colors
- Memory cue positions
- Analysis parameters
- Loop lengths

## Batch Processing

See `example_usage.py` for a complete example of batch processing multiple audio files.
