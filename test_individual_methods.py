#!/usr/bin/env python3
"""Debug individual BPM detection methods on Adam Beyer - Pilot."""

import logging
import numpy as np
import soundfile as sf
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # Suppress debug logs
    format='%(message)s'
)

def test_individual_methods():
    """Test each BPM detection method individually."""
    
    audio_file = Path("/workspaces/edm-cue-analyzer/tests/Adam Beyer - Pilot.flac")
    
    print("=" * 80)
    print("Individual Method Testing: Adam Beyer - Pilot")
    print("=" * 80)
    print(f"Official BPM: 133 BPM (Beatport)")
    print()
    
    # Load audio
    y, sr = sf.read(str(audio_file))
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)
    
    results = []
    
    # Test Essentia methods
    try:
        import essentia.standard as es
        
        # Method 1: RhythmDescriptors
        try:
            rhythm_desc = es.RhythmDescriptors()
            bpm, beats, confidence, _, _, first_peak_bpm, _, _, second_peak_bpm, second_peak_weight, _, _ = rhythm_desc(y)
            results.append(("Essentia RhythmDescriptors", float(bpm), f"2nd peak: {second_peak_bpm:.1f} (weight: {second_peak_weight:.2f})"))
        except Exception as e:
            results.append(("Essentia RhythmDescriptors", None, str(e)))
        
        # Method 2: RhythmExtractor2013 multifeature
        try:
            extractor = es.RhythmExtractor2013(method="multifeature")
            bpm, _, confidence, _, _ = extractor(y)
            results.append(("Essentia multifeature", float(bpm), f"conf: {np.mean(confidence):.2f}"))
        except Exception as e:
            results.append(("Essentia multifeature", None, str(e)))
        
        # Method 3: RhythmExtractor2013 degara
        try:
            extractor = es.RhythmExtractor2013(method="degara")
            bpm, _, _, _, _ = extractor(y)
            results.append(("Essentia degara", float(bpm), ""))
        except Exception as e:
            results.append(("Essentia degara", None, str(e)))
        
        # Method 4: PercivalBpmEstimator
        try:
            estimator = es.PercivalBpmEstimator()
            bpm = estimator(y)
            results.append(("Essentia Percival", float(bpm), ""))
        except Exception as e:
            results.append(("Essentia Percival", None, str(e)))
            
    except ImportError:
        print("⚠️  Essentia not available")
    
    # Test Aubio
    try:
        import aubio
        hop_size = 512
        win_s = 1024
        tempo = aubio.tempo("default", win_s, hop_size, sr)
        
        beat_times = []
        for i in range(0, len(y), hop_size):
            chunk = y[i:i+hop_size]
            if len(chunk) < hop_size:
                chunk = np.pad(chunk, (0, hop_size - len(chunk)))
            is_beat = tempo(chunk.astype(np.float32))
            if is_beat:
                beat_times.append(tempo.get_last_s())
        
        if len(beat_times) >= 2:
            intervals = np.diff(beat_times)
            bpm = 60.0 / np.median(intervals)
            results.append(("Aubio", float(bpm), f"{len(beat_times)} beats"))
        else:
            results.append(("Aubio", None, "Too few beats"))
            
    except Exception as e:
        results.append(("Aubio", None, str(e)))
    
    # Test Librosa
    try:
        import librosa
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        results.append(("Librosa", float(tempo), ""))
    except Exception as e:
        results.append(("Librosa", None, str(e)))
    
    # Display results
    print("METHOD RESULTS:")
    print("-" * 80)
    print(f"{'Method':<30} {'BPM':>10} {'Error':>10} {'Notes'}")
    print("-" * 80)
    
    for method, bpm, notes in results:
        if bpm is not None:
            error = bpm - 133
            error_pct = (error / 133) * 100
            print(f"{method:<30} {bpm:>10.1f} {error:>+9.1f} ({error_pct:+.1f}%)  {notes}")
        else:
            print(f"{method:<30} {'FAILED':>10}            {notes}")
    
    print("-" * 80)
    print()
    
    # Analyze clustering
    valid_bpms = [bpm for _, bpm, _ in results if bpm is not None]
    if valid_bpms:
        print(f"Valid detections: {len(valid_bpms)}")
        print(f"Mean: {np.mean(valid_bpms):.1f} BPM")
        print(f"Median: {np.median(valid_bpms):.1f} BPM")
        print(f"Std dev: {np.std(valid_bpms):.1f} BPM")
        print(f"Range: {min(valid_bpms):.1f} - {max(valid_bpms):.1f} BPM")
        print()
        
        # Check for octaves
        print("Checking for octave relationships:")
        for i, (m1, bpm1, _) in enumerate(results):
            if bpm1 is None:
                continue
            for m2, bpm2, _ in results[i+1:]:
                if bpm2 is None:
                    continue
                ratio = bpm1 / bpm2
                print(f"  {m1[:20]:20} / {m2[:20]:20} = {ratio:.3f}")

if __name__ == "__main__":
    test_individual_methods()
