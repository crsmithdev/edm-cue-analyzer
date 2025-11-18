#!/usr/bin/env python3
"""
Validate drop detection on test tracks.
Compares before/after metrics for Change 7 (Energy Derivative Analysis).
"""

import json
import subprocess
import sys
from pathlib import Path

# Test tracks from /music directory
TEST_TRACKS = [
    "/music/3LAU, Dnmo - Falling.flac",
    "/music/AUTOFLOWER - Dimension.flac",
    "/music/AUTOFLOWER - THE ONLY ONE.flac",
    "/music/AUTOFLOWER - Wallflower.flac",
    "/music/AUTOFLOWER - When It's Over (Extended Mix).flac",
    "/music/Activa - Get On With It (Extended Mix).flac",
    "/music/Adam Beyer - Pilot.flac",
    "/music/Adana Twins - Maya.flac",
    "/music/Agents Of Time - Zodiac.flac",
    "/music/Artbat - Artefact.flac",
]

def analyze_track(file_path: str):
    """Analyze a single track and return drop times."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {Path(file_path).name}")
    print(f"{'='*80}")
    
    try:
        # Run Python code directly
        code = f"""
import asyncio
from pathlib import Path
from edm_cue_analyzer.analyzer import AudioAnalyzer
from edm_cue_analyzer.config import Config

async def analyze():
    config = Config()
    analyzer = AudioAnalyzer(config)
    result = await analyzer.analyze_with(Path('{file_path}'), analyses='drops')
    drops = result.drops if hasattr(result, 'drops') else []
    for drop in drops:
        print(f'DROP:{{drop:.2f}}')

asyncio.run(analyze())
"""
        
        result = subprocess.run(
            ['python', '-c', code],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        # Parse output to extract drops
        drops = []
        for line in result.stdout.split('\n'):
            if line.startswith('DROP:'):
                try:
                    drop_time = float(line.split(':')[1])
                    drops.append(drop_time)
                except:
                    pass
        
        print(f"✓ Detected {len(drops)} drop(s)")
        
        if drops:
            print(f"  Drop times: {[f'{d:.1f}s' for d in drops]}")
        
        return {
            'file': Path(file_path).name,
            'drops': drops,
            'num_drops': len(drops),
            'success': True
        }
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return {
            'file': Path(file_path).name,
            'drops': [],
            'num_drops': 0,
            'success': False,
            'error': str(e)
        }

def main():
    print("="*80)
    print("DROP DETECTION VALIDATION - Change 7: Energy Derivative Analysis")
    print("="*80)
    print(f"\nTesting {len(TEST_TRACKS)} tracks with new derivative-based validation")
    print("This should reduce false positives by verifying energy 'rapidly falls' after peak")
    
    results = []
    for track in TEST_TRACKS:
        if not Path(track).exists():
            print(f"\n⚠ Skipping (not found): {Path(track).name}")
            continue
        
        result = analyze_track(track)
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r['success']]
    total_drops = sum(r['num_drops'] for r in successful)
    
    print(f"\nTracks analyzed: {len(successful)}/{len(TEST_TRACKS)}")
    print(f"Total drops detected: {total_drops}")
    print(f"Average drops per track: {total_drops/len(successful):.1f}")
    
    print("\nPer-track breakdown:")
    for r in successful:
        print(f"  {r['file']}: {r['num_drops']} drop(s)")
    
    if any(not r['success'] for r in results):
        print("\nErrors:")
        for r in results:
            if not r['success']:
                print(f"  {r['file']}: {r.get('error', 'Unknown error')}")
    
    print("\n" + "="*80)
    print("NOTE: Compare these results with baseline to evaluate Change 7 impact")
    print("Expected: Reduction in total drops detected (fewer false positives)")
    print("="*80)
    
    # Save results
    output_file = Path('/workspaces/edm-cue-analyzer/drop_validation_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'test_name': 'Change 7: Energy Derivative Analysis',
            'tracks_tested': len(successful),
            'total_drops': total_drops,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()
