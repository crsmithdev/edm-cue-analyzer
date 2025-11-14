#!/usr/bin/env python3
"""
Test consensus BPM detection on Adam Beyer - Pilot track.
Expected: Should detect 133 BPM (not 122.2 BPM from single method).
"""

import asyncio
import logging
from pathlib import Path

from src.edm_cue_analyzer.config import load_config
from src.edm_cue_analyzer.analyzer import AudioAnalyzer

# Set up logging to see consensus details
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_adam_beyer():
    """Test consensus BPM detection on Adam Beyer - Pilot."""
    
    audio_file = Path("/workspaces/edm-cue-analyzer/tests/Adam Beyer - Pilot.flac")
    
    if not audio_file.exists():
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    print("=" * 80)
    print("Testing Consensus BPM Detection on Adam Beyer - Pilot")
    print("=" * 80)
    print(f"Audio file: {audio_file}")
    print(f"Official BPM: 133 BPM (from Beatport)")
    print(f"Previous detection: 122.2 BPM (octave error)")
    print(f"Expected consensus: 133 BPM (corrected)")
    print()
    
    # Load config and create analyzer
    config = load_config()
    analyzer = AudioAnalyzer(config.analysis)
    
    # Analyze the track
    print("Analyzing track (this will show all consensus details)...")
    print()
    
    structure = await analyzer.analyze_file(audio_file)
    
    # Display results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Detected BPM: {structure.bpm:.1f} BPM")
    print(f"Official BPM: 133 BPM")
    print(f"Error: {structure.bpm - 133:.1f} BPM ({((structure.bpm - 133) / 133) * 100:.2f}%)")
    print()
    
    if abs(structure.bpm - 133) < 2:
        print("✅ SUCCESS! Consensus BPM is correct!")
    else:
        print("❌ FAILED! BPM still incorrect")
    
    print()
    print(f"Track duration: {structure.duration:.1f} seconds")
    print(f"Bar duration: {structure.bar_duration:.3f} seconds")
    print(f"Number of bars: {structure.num_bars}")
    print()

if __name__ == "__main__":
    asyncio.run(test_adam_beyer())
