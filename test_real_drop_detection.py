#!/usr/bin/env python3
"""
Test the new drop detection system on a real EDM track.

Uses: Autoflower - When It's Over (Extended Mix)
Expected drops: 31s, 2m35s (155s)
"""

import asyncio
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)-30s %(levelname)-8s %(message)s'
)

# Enable debug for drop detection specifically
logging.getLogger('src.edm_cue_analyzer.analyses.drops').setLevel(logging.DEBUG)

from src.edm_cue_analyzer.analyzer import AudioAnalyzer
from src.edm_cue_analyzer.config import AnalysisConfig


async def test_real_track():
    """Test drop detection on a real track."""
    
    # Track details
    audio_path = Path("/music/AUTOFLOWER - When It's Over (Extended Mix).flac")
    expected_drops = [31.0, 155.0]  # 31s, 2m35s
    
    print("\n" + "="*80)
    print("TESTING DROP DETECTION ON REAL TRACK")
    print("="*80)
    print(f"\nTrack: {audio_path.name}")
    print(f"Expected drops: {expected_drops[0]:.0f}s, {expected_drops[1]:.0f}s")
    
    if not audio_path.exists():
        print(f"\n❌ ERROR: File not found: {audio_path}")
        return
    
    # Initialize analyzer
    config = AnalysisConfig()
    config.drop_min_spacing_bars = 8  # Allow closer drops for testing
    analyzer = AudioAnalyzer(config)
    
    print("\n" + "-"*80)
    print("RUNNING ANALYSIS...")
    print("-"*80 + "\n")
    
    # Run full analysis
    try:
        structure = await analyzer.analyze_with(audio_path, analyses='drops')
        
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"\nTrack duration: {structure.duration:.1f}s")
        print(f"Detected BPM: {structure.detected_bpm:.1f}")
        print(f"\nDetected drops: {len(structure.drops)}")
        
        if structure.drops:
            print("\nDrop times:")
            for i, drop_time in enumerate(structure.drops, 1):
                # Find closest expected drop
                closest_expected = min(expected_drops, key=lambda d: abs(d - drop_time))
                error = abs(drop_time - closest_expected)
                
                # Format time as MM:SS
                minutes = int(drop_time // 60)
                seconds = int(drop_time % 60)
                
                status = "✓" if error < 10.0 else "✗"
                print(f"  {status} Drop {i}: {minutes}m{seconds:02d}s ({drop_time:.2f}s) - "
                      f"Expected: {closest_expected:.0f}s (error: {error:.2f}s)")
            
            # Calculate accuracy
            print(f"\nAccuracy check (tolerance: ±10s):")
            for exp_time in expected_drops:
                if structure.drops:
                    closest = min(structure.drops, key=lambda d: abs(d - exp_time))
                    error = abs(closest - exp_time)
                    status = "✓" if error < 10.0 else "✗"
                    
                    exp_min = int(exp_time // 60)
                    exp_sec = int(exp_time % 60)
                    det_min = int(closest // 60)
                    det_sec = int(closest % 60)
                    
                    print(f"  {status} Expected {exp_min}m{exp_sec:02d}s ({exp_time:.0f}s) -> "
                          f"Detected {det_min}m{det_sec:02d}s ({closest:.2f}s) "
                          f"(error: {error:.2f}s)")
                else:
                    print(f"  ✗ Expected {exp_time:.0f}s -> Not detected")
        else:
            print("\n  ⚠ No drops detected!")
        
        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_real_track())
