#!/usr/bin/env python3
"""Quick test without the slow RhythmDescriptors method."""

import logging
import numpy as np
import soundfile as sf
from pathlib import Path

from src.edm_cue_analyzer.consensus import ConsensusBpmDetector

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

def test_fast_consensus():
    """Test consensus BPM without slow methods."""
    
    audio_file = Path("/workspaces/edm-cue-analyzer/tests/Adam Beyer - Pilot.flac")
    
    print("=" * 80)
    print("Fast Consensus Test (skipping RhythmDescriptors)")
    print("=" * 80)
    print("Official BPM: 133 BPM")
    print()
    
    # Load audio
    y, sr = sf.read(str(audio_file))
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)
    
    # Manually test the fast methods
    try:
        import essentia.standard as es
        
        # Test multifeature
        print("Testing Essentia multifeature...")
        extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, _, conf, _, _ = extractor(y)
        print(f"  Result: {bpm:.1f} BPM (confidence: {np.mean(conf):.2f})")
        
        # Test degara
        print("Testing Essentia degara...")
        extractor = es.RhythmExtractor2013(method="degara")
        bpm, _, _, _, _ = extractor(y)
        print(f"  Result: {bpm:.1f} BPM")
        
        # Test Percival
        print("Testing Essentia Percival...")
        estimator = es.PercivalBpmEstimator()
        bpm = estimator(y)
        print(f"  Result: {bpm:.1f} BPM")
        
    except Exception as e:
        print(f"Essentia tests failed: {e}")
    
    # Test Aubio
    try:
        import aubio
        print("Testing Aubio...")
        hop_size = 512
        win_s = 1024
        tempo = aubio.tempo("default", win_s, hop_size, sr)
        
        beat_times = []
        for i in range(0, len(y), hop_size):
            chunk = y[i:i+hop_size]
            if len(chunk) < hop_size:
                chunk = np.pad(chunk, (0, hop_size - len(chunk)))
            if tempo(chunk):
                beat_times.append(tempo.get_last_s())
        
        if len(beat_times) >= 2:
            bpm = 60.0 / np.median(np.diff(beat_times))
            print(f"  Result: {bpm:.1f} BPM ({len(beat_times)} beats)")
    except Exception as e:
        print(f"Aubio test failed: {e}")
    
    # Test Librosa
    try:
        import librosa
        print("Testing Librosa...")
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        print(f"  Result: {tempo:.1f} BPM")
    except Exception as e:
        print(f"Librosa test failed: {e}")
    
    print()
    print("=" * 80)
    print("Analysis:")
    print("If most methods detect ~122 BPM, that explains the consensus result.")
    print("The fix requires RhythmDescriptors to add 133 BPM as second peak.")
    print("=" * 80)

if __name__ == "__main__":
    test_fast_consensus()
