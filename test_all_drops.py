#!/usr/bin/env python3
"""
Test drop detection on all annotated tracks from Annotations.md.
"""

import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(name)-30s %(levelname)-8s %(message)s'
)

# Only show debug for drops
logging.getLogger('src.edm_cue_analyzer.analyses.drops').setLevel(logging.WARNING)

from src.edm_cue_analyzer.analyzer import AudioAnalyzer
from src.edm_cue_analyzer.config import AnalysisConfig


@dataclass
class Track:
    """Track with expected drop times."""
    filename: str
    expected_drops: list[float]


# Parse annotations from Annotations.md
TRACKS = [
    Track("AUTOFLOWER - Dimension.flac", [2*60 + 8]),
    Track("AUTOFLOWER - THE ONLY ONE.flac", [2*60 + 3]),
    Track("AUTOFLOWER - When It's Over (Extended Mix).flac", [31, 2*60 + 35]),
    Track("AUTOFLOWER - Wallflower.flac", [1*60 + 2, 2*60 + 5]),
    Track("Activa - Get On With It (Extended Mix).flac", [27, 4*60 + 3]),
    Track("Adana Twins - Maya.flac", [2*60 + 3, 4*60 + 21]),
    Track("Agents Of Time - Zodiac.flac", [1*60 + 47, 4*60 + 3]),
    Track("3LAU, Dnmo - Falling.flac", [44, 2*60 + 40]),
]


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m{secs:02d}s"


async def test_track(track: Track, analyzer: AudioAnalyzer) -> dict:
    """Test drop detection on a single track."""
    
    audio_path = Path("/music") / track.filename
    
    if not audio_path.exists():
        return {
            'track': track.filename,
            'status': 'MISSING',
            'expected': len(track.expected_drops),
            'detected': 0,
            'errors': []
        }
    
    try:
        # Run analysis
        structure = await analyzer.analyze_with(audio_path, analyses='drops')
        
        # Calculate accuracy
        errors = []
        for exp_time in track.expected_drops:
            if structure.drops:
                closest = min(structure.drops, key=lambda d: abs(d - exp_time))
                error = abs(closest - exp_time)
                errors.append(error)
            else:
                errors.append(None)  # Not detected
        
        # Determine status
        if len(structure.drops) == 0:
            status = 'NONE'
        elif len(structure.drops) != len(track.expected_drops):
            status = 'PARTIAL'
        elif all(e is not None and e < 10.0 for e in errors):
            status = 'GOOD'
        else:
            status = 'POOR'
        
        return {
            'track': track.filename,
            'status': status,
            'expected': len(track.expected_drops),
            'detected': len(structure.drops),
            'errors': errors,
            'drops': structure.drops,
            'expected_times': track.expected_drops,
            'bpm': structure.detected_bpm,
            'duration': structure.duration
        }
        
    except Exception as e:
        return {
            'track': track.filename,
            'status': 'ERROR',
            'expected': len(track.expected_drops),
            'detected': 0,
            'errors': [],
            'error_msg': str(e)
        }


async def main():
    """Run tests on all tracks."""
    
    print("\n" + "="*100)
    print("DROP DETECTION VALIDATION - ANNOTATED TRACKS")
    print("="*100)
    
    # Initialize analyzer
    config = AnalysisConfig()
    config.drop_min_spacing_bars = 8
    analyzer = AudioAnalyzer(config)
    
    results = []
    
    for i, track in enumerate(TRACKS, 1):
        print(f"\n[{i}/{len(TRACKS)}] Testing: {track.filename}")
        print("-"*100)
        
        result = await test_track(track, analyzer)
        results.append(result)
        
        # Show result
        status_icon = {
            'GOOD': '✓',
            'PARTIAL': '⚠',
            'POOR': '✗',
            'NONE': '✗',
            'ERROR': '❌',
            'MISSING': '❌'
        }.get(result['status'], '?')
        
        print(f"Status: {status_icon} {result['status']}")
        print(f"Expected: {result['expected']} drops  |  Detected: {result['detected']} drops")
        
        if result.get('drops'):
            print(f"\nDetected drops:")
            for drop in result['drops']:
                print(f"  - {format_time(drop)} ({drop:.2f}s)")
        
        if result.get('errors'):
            print(f"\nAccuracy:")
            for exp_time, error in zip(result['expected_times'], result['errors']):
                if error is not None:
                    status = '✓' if error < 10.0 else '✗'
                    print(f"  {status} Expected {format_time(exp_time)} -> Error: {error:.2f}s")
                else:
                    print(f"  ✗ Expected {format_time(exp_time)} -> Not detected")
        
        if result.get('error_msg'):
            print(f"Error: {result['error_msg']}")
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    total = len(results)
    good = sum(1 for r in results if r['status'] == 'GOOD')
    partial = sum(1 for r in results if r['status'] == 'PARTIAL')
    poor = sum(1 for r in results if r['status'] == 'POOR')
    none = sum(1 for r in results if r['status'] == 'NONE')
    error = sum(1 for r in results if r['status'] in ['ERROR', 'MISSING'])
    
    print(f"\nTracks tested: {total}")
    print(f"  ✓ Good (all drops within 10s):  {good}")
    print(f"  ⚠ Partial (some drops detected): {partial}")
    print(f"  ✗ Poor (detected but inaccurate): {poor}")
    print(f"  ✗ None (no drops detected):      {none}")
    print(f"  ❌ Error/Missing:                 {error}")
    
    # Calculate overall accuracy
    all_errors = [e for r in results for e in r.get('errors', []) if e is not None]
    if all_errors:
        avg_error = sum(all_errors) / len(all_errors)
        max_error = max(all_errors)
        min_error = min(all_errors)
        
        print(f"\nDrop detection accuracy:")
        print(f"  Average error: {avg_error:.2f}s")
        print(f"  Min error:     {min_error:.2f}s")
        print(f"  Max error:     {max_error:.2f}s")
        print(f"  Drops within 10s: {sum(1 for e in all_errors if e < 10.0)}/{len(all_errors)}")
    
    print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
