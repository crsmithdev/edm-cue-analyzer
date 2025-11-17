# BPM Validation Tool

Comprehensive validation tool to test BPM detection accuracy by comparing analyzer results against online BPM databases.

## Features

- 🌐 **Multi-source lookup**: Fetches reference BPMs from GetSongBPM, Tunebat, and Beatport
- 💾 **Smart caching**: Caches online lookups to avoid repeated requests
- 📊 **Detailed statistics**: Accuracy metrics, error analysis, performance benchmarks
- 🎯 **Error detection**: Identifies and highlights tracks with largest detection errors
- 📝 **Comprehensive logging**: Full logs saved for debugging and review

## Installation

Install additional dependencies:

```bash
pip install aiohttp beautifulsoup4 lxml
```

## Usage

### Basic Usage

```bash
python validate_bpm_accuracy.py /path/to/music/library
```

### Limit Files

Test only first 20 files:

```bash
python validate_bpm_accuracy.py /path/to/music/library --max-files 20
```

### Skip Cache

Re-fetch all BPM data (useful if previous lookups failed):

```bash
python validate_bpm_accuracy.py /path/to/music/library --skip-cache
```

## Expected Filename Format

The tool works best with files named in these formats:

- `Artist - Title.flac`
- `01. Artist - Title.mp3`
- `Artist_-_Title.wav`

Track numbers are automatically removed. If no artist/title separator is found, the entire filename is used as the title.

## Output

### Console Report

The tool prints a comprehensive report including:

- **Summary**: Total files, successful validations
- **Accuracy Statistics**: Mean error, accuracy within thresholds (±1 BPM, ±2 BPM, ±5 BPM, ±5%)
- **Performance**: Analysis times
- **Reference Sources**: Breakdown by database (GetSongBPM, Tunebat, Beatport)
- **Top 10 Largest Errors**: Tracks with worst detection accuracy
- **Perfect Matches**: Tracks within ±0.5 BPM

### Example Output

```
================================================================================
BPM VALIDATION REPORT
================================================================================

📊 Summary
  Total files scanned: 50
  Successfully validated: 42
  No reference found: 8

🎯 Accuracy Statistics
  Mean error: 1.23 BPM (0.95%)
  Within ±1 BPM: 28/42 (66.7%)
  Within ±2 BPM: 36/42 (85.7%)
  Within ±5 BPM: 41/42 (97.6%)
  Within ±5%: 40/42 (95.2%)

⏱️  Performance
  Mean analysis time: 16.42s per track
  Total analysis time: 689.6s

🔍 Reference Sources
  GetSongBPM: 25 tracks (59.5%)
  Tunebat: 12 tracks (28.6%)
  Beatport: 5 tracks (11.9%)

❌ Top 10 Largest Errors

  1. Deadmau5 - Strobe
     Detected: 128.0 BPM | Reference: 130.0 BPM (GetSongBPM)
     Error: -2.0 BPM (-1.5%)

  2. Eric Prydz - Opus
     Detected: 126.0 BPM | Reference: 128.0 BPM (Beatport)
     Error: -2.0 BPM (-1.6%)

✅ Perfect Matches (±0.5 BPM): 18
  • Sasha - Xpander: 124.0 BPM
  • Adam Beyer - Your Mind: 130.0 BPM
  • Amelie Lens - Feel It: 138.0 BPM

================================================================================

💾 Detailed results saved to: bpm_validation_results.json
📝 Log file: bpm_validation.log
```

### Generated Files

1. **`bpm_validation_results.json`**: Complete validation data in JSON format
2. **`bpm_validation.log`**: Detailed processing log
3. **`bpm_cache.json`**: Cached online BPM lookups (speeds up re-runs)

## JSON Output Format

```json
[
  {
    "filepath": "/path/to/track.flac",
    "artist": "Artist Name",
    "title": "Track Title",
    "detected_bpm": 124.5,
    "reference_bpm": 125.0,
    "source": "GetSongBPM",
    "error_bpm": -0.5,
    "error_percent": -0.4,
    "analysis_time": 16.23,
    "success": true,
    "error_message": null
  }
]
```

## How It Works

1. **Find Files**: Scans directory for audio files (.flac, .mp3, .wav, .m4a, .aac, .ogg)
2. **Parse Filenames**: Extracts artist and title from filename
3. **Lookup Reference**: Queries online databases for known BPM (with caching)
4. **Analyze Audio**: Runs BPM analysis using the EDM Cue Analyzer (use `-a bpm`)
5. **Compare Results**: Calculates error and detects octave errors (half/double time)
6. **Generate Report**: Produces comprehensive statistics and error analysis

## Rate Limiting

The tool includes automatic rate limiting:
- 0.5s delay between API calls to each source
- 1s delay between processing files
- Caching to minimize redundant requests

## Tips

- **First Run**: May take a while due to online lookups (cached for subsequent runs)
- **Best Results**: Use well-tagged files with clear artist/title in filename
- **Large Libraries**: Use `--max-files` to test a sample first
- **Cache Issues**: Use `--skip-cache` if you suspect cached data is wrong

## Troubleshooting

### No Reference BPM Found

If many files show "No reference found":
- Check filename format (should include artist and title)
- Tracks might not be in online databases (obscure/underground tracks)
- Try searching manually on GetSongBPM.com to verify availability

### Analysis Errors

Check `bpm_validation.log` for detailed error messages and stack traces.

### Network Issues

- Ensure internet connection is stable
- Some databases may block requests if rate limits are exceeded
- Wait a few minutes and try again
