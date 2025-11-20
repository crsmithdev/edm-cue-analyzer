#!/usr/bin/env python3
"""
Quick test of the new Method 2.2 drop detection on a single track
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from edm_cue_analyzer.analyzer import AudioAnalyzer
from edm_cue_analyzer.config import Config

async def test_single_track():
    """Test new combined drop detection on AUTOFLOWER track"""
    music_file = Path("/music/AUTOFLOWER - When It's Over (Extended Mix).flac")
    
    if not music_file.exists():
        print(f"File not found: {music_file}")
        return
    
    print("Testing combined drop detection (Methods 2.1 + 2.2)")
    print(f"Track: {music_file.name}")
    print("Manual annotations: 31s, 155s")
    
    config = Config()
    analyzer = AudioAnalyzer(config)
    
    try:
        print("\nRunning analysis...")
        structure = await analyzer.analyze_with(music_file, analyses={'drops'})
        
        if hasattr(structure, 'drops') and structure.drops:
            detected_times = [round(drop, 1) for drop in structure.drops]
            print(f"Detected drops: {detected_times}")
            
            # Compare with manual annotations
            manual_drops = [31, 155]
            for manual_time in manual_drops:
                closest_detected = None
                min_error = float('inf')
                
                for det_time in detected_times:
                    error = abs(det_time - manual_time)
                    if error < min_error:
                        min_error = error
                        closest_detected = det_time
                
                if closest_detected and min_error <= 10:
                    print(f"✓ Manual {manual_time}s → Detected {closest_detected}s (error: ±{min_error:.1f}s)")
                else:
                    print(f"✗ Manual {manual_time}s → MISSED")
            
        else:
            print("No drops detected")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_single_track())