#!/usr/bin/env python3
"""Quick test of consensus BPM detection on Adam Beyer - Pilot."""

import logging
import numpy as np
import soundfile as sf
from pathlib import Path

from src.edm_cue_analyzer.consensus import ConsensusBpmDetector

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(name)s - %(message)s'
)

def test_consensus():
    """Test consensus BPM detection."""
    
    audio_file = Path("/workspaces/edm-cue-analyzer/tests/Adam Beyer - Pilot.flac")
    
    print("=" * 80)
    print("Quick Consensus BPM Test: Adam Beyer - Pilot")
    print("=" * 80)
    print(f"Audio file: {audio_file.name}")
    print(f"Official BPM: 133 BPM (Beatport)")
    print(f"Previous single-method: 122.2 BPM (octave error)")
    print()
    
    # Load audio
    print("Loading audio...")
    y, sr = sf.read(str(audio_file))
    
    # Convert to mono if needed
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    
    y = y.astype(np.float32)
    
    print(f"Duration: {len(y) / sr:.1f} seconds")
    print(f"Sample rate: {sr} Hz")
    print()
    
    # Create consensus detector
    print("Running consensus BPM detection...")
    detector = ConsensusBpmDetector(
        min_bpm=60.0,
        max_bpm=200.0,
        expected_range=(120.0, 145.0),
        octave_tolerance=0.1,
    )
    
    # Detect BPM
    result = detector.detect(y, sr)
    
    # Display results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Consensus BPM: {result.bpm:.1f} BPM")
    print(f"Confidence: {result.confidence * 100:.1f}%")
    print(f"Methods used: {result.metadata.get('num_methods', 0)}")
    print(f"Agreement: {result.metadata.get('agreement_count', 0)}/{result.metadata.get('num_methods', 0)} methods")
    print()
    print(f"Official BPM: 133 BPM")
    print(f"Error: {result.bpm - 133:.1f} BPM ({((result.bpm - 133) / 133) * 100:.2f}%)")
    print()
    
    if abs(result.bpm - 133) < 2:
        print("✅ SUCCESS! Consensus correctly detected 133 BPM!")
    else:
        print(f"❌ FAILED! Detected {result.bpm:.1f} BPM instead of 133 BPM")
    
    print()
    print(f"Beat positions available: {'Yes' if result.beats is not None else 'No'}")
    if result.beats is not None:
        print(f"Number of beats: {len(result.beats)}")
    print()

if __name__ == "__main__":
    test_consensus()
