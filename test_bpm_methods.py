#!/usr/bin/env python3
"""
Test alternative Essentia BPM detection methods.
Compare different algorithms to see which performs best.
"""

import essentia.standard as es
import soundfile as sf
import numpy as np
from pathlib import Path
import time


def test_all_bpm_methods(audio_path):
    """Test all available BPM detection methods in Essentia."""
    
    print(f"\n{'='*80}")
    print(f"Testing BPM Detection Methods")
    print(f"File: {Path(audio_path).name}")
    print(f"{'='*80}\n")
    
    # Load audio
    print("Loading audio...")
    audio, sr = sf.read(audio_path, dtype='float32')
    
    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample to 44100 if needed (Essentia requirement)
    if sr != 44100:
        print(f"Resampling from {sr}Hz to 44100Hz...")
        resampler = es.Resample(inputSampleRate=sr, outputSampleRate=44100)
        audio = resampler(audio)
    
    results = {}
    
    # Method 1: RhythmExtractor2013 - multifeature (current default)
    print("\n1. RhythmExtractor2013 (multifeature)...")
    start = time.time()
    try:
        extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, confidence, estimates, intervals = extractor(audio)
        elapsed = time.time() - start
        results['RhythmExtractor2013-multifeature'] = {
            'bpm': bpm,
            'confidence': confidence,
            'estimates': estimates[:5] if len(estimates) > 5 else estimates,
            'time': elapsed,
            'status': 'success'
        }
        print(f"   BPM: {bpm:.1f}, Confidence: {confidence:.2f}, Time: {elapsed:.2f}s")
    except Exception as e:
        results['RhythmExtractor2013-multifeature'] = {'status': 'error', 'error': str(e)}
        print(f"   ERROR: {e}")
    
    # Method 2: RhythmExtractor2013 - degara (faster)
    print("\n2. RhythmExtractor2013 (degara)...")
    start = time.time()
    try:
        extractor = es.RhythmExtractor2013(method="degara")
        bpm, beats, confidence, estimates, intervals = extractor(audio)
        elapsed = time.time() - start
        results['RhythmExtractor2013-degara'] = {
            'bpm': bpm,
            'confidence': confidence,
            'estimates': estimates[:5] if len(estimates) > 5 else estimates,
            'time': elapsed,
            'status': 'success'
        }
        print(f"   BPM: {bpm:.1f}, Confidence: {confidence:.2f}, Time: {elapsed:.2f}s")
    except Exception as e:
        results['RhythmExtractor2013-degara'] = {'status': 'error', 'error': str(e)}
        print(f"   ERROR: {e}")
    
    # Method 3: PercivalBpmEstimator
    print("\n3. PercivalBpmEstimator...")
    start = time.time()
    try:
        estimator = es.PercivalBpmEstimator()
        bpm = estimator(audio)
        elapsed = time.time() - start
        results['PercivalBpmEstimator'] = {
            'bpm': bpm,
            'time': elapsed,
            'status': 'success'
        }
        print(f"   BPM: {bpm:.1f}, Time: {elapsed:.2f}s")
    except Exception as e:
        results['PercivalBpmEstimator'] = {'status': 'error', 'error': str(e)}
        print(f"   ERROR: {e}")
    
    # Method 4: LoopBpmEstimator (designed for loops/repetitive music)
    print("\n4. LoopBpmEstimator...")
    start = time.time()
    try:
        estimator = es.LoopBpmEstimator()
        bpm = estimator(audio)
        elapsed = time.time() - start
        results['LoopBpmEstimator'] = {
            'bpm': bpm,
            'time': elapsed,
            'status': 'success',
            'note': 'Returns 0.0 if confidence < threshold'
        }
        if bpm == 0.0:
            print(f"   BPM: {bpm:.1f} (LOW CONFIDENCE - unreliable), Time: {elapsed:.2f}s")
        else:
            print(f"   BPM: {bpm:.1f}, Time: {elapsed:.2f}s")
    except Exception as e:
        results['LoopBpmEstimator'] = {'status': 'error', 'error': str(e)}
        print(f"   ERROR: {e}")
    
    # Method 5: RhythmDescriptors (combines RhythmExtractor + BpmHistogramDescriptors)
    print("\n5. RhythmDescriptors (with histogram analysis)...")
    start = time.time()
    try:
        descriptor = es.RhythmDescriptors()
        beats_pos, confidence, bpm, estimates, intervals, \
        first_peak_bpm, first_peak_spread, first_peak_weight, \
        second_peak_bpm, second_peak_spread, second_peak_weight, \
        histogram = descriptor(audio)
        elapsed = time.time() - start
        
        results['RhythmDescriptors'] = {
            'bpm': bpm,
            'confidence': confidence,
            'first_peak_bpm': first_peak_bpm,
            'first_peak_weight': first_peak_weight,
            'second_peak_bpm': second_peak_bpm,
            'second_peak_weight': second_peak_weight,
            'time': elapsed,
            'status': 'success'
        }
        print(f"   Primary BPM: {bpm:.1f}, Confidence: {confidence:.2f}")
        print(f"   1st Peak: {first_peak_bpm:.1f} BPM (weight: {first_peak_weight:.2f}, spread: {first_peak_spread:.2f})")
        print(f"   2nd Peak: {second_peak_bpm:.1f} BPM (weight: {second_peak_weight:.2f}, spread: {second_peak_spread:.2f})")
        print(f"   Time: {elapsed:.2f}s")
        
        # Check for harmonic relationships (octave errors)
        if second_peak_bpm > 0:
            ratio = first_peak_bpm / second_peak_bpm if second_peak_bpm != 0 else 0
            print(f"   BPM Ratio (1st/2nd): {ratio:.2f}x", end="")
            if 1.9 < ratio < 2.1:
                print(" → Possible OCTAVE relationship!")
            elif 0.45 < ratio < 0.55:
                print(" → Possible HALF-TIME relationship!")
            else:
                print()
        
    except Exception as e:
        results['RhythmDescriptors'] = {'status': 'error', 'error': str(e)}
        print(f"   ERROR: {e}")
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Method':<40} {'BPM':<10} {'Confidence':<12} {'Time (s)':<10}")
    print("-" * 80)
    
    for method, data in results.items():
        if data['status'] == 'success':
            bpm_str = f"{data['bpm']:.1f}" if 'bpm' in data else "N/A"
            conf_str = f"{data['confidence']:.2f}" if 'confidence' in data else "N/A"
            time_str = f"{data['time']:.2f}"
            print(f"{method:<40} {bpm_str:<10} {conf_str:<12} {time_str:<10}")
        else:
            print(f"{method:<40} ERROR: {data.get('error', 'Unknown')}")
    
    # Check for consensus
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")
    
    bpms = [data['bpm'] for data in results.values() 
            if data['status'] == 'success' and 'bpm' in data and data['bpm'] > 0]
    
    if bpms:
        avg_bpm = np.mean(bpms)
        std_bpm = np.std(bpms)
        min_bpm = np.min(bpms)
        max_bpm = np.max(bpms)
        
        print(f"BPM Range: {min_bpm:.1f} - {max_bpm:.1f}")
        print(f"Average BPM: {avg_bpm:.1f}")
        print(f"Std Dev: {std_bpm:.2f}")
        
        if std_bpm < 2:
            print("✅ STRONG CONSENSUS - All methods agree within 2 BPM")
        elif std_bpm < 5:
            print("✅ GOOD CONSENSUS - Most methods agree within 5 BPM")
        elif std_bpm < 10:
            print("⚠️ WEAK CONSENSUS - Methods differ by up to 10 BPM")
        else:
            print("❌ NO CONSENSUS - Large disagreement between methods")
            print("   → Check for octave errors (half-time/double-time)")
            print("   → Try manual verification")
        
        # Check for octave relationships
        for i, bpm1 in enumerate(bpms):
            for bpm2 in bpms[i+1:]:
                ratio = bpm1 / bpm2 if bpm2 != 0 else 0
                if 1.9 < ratio < 2.1 or 0.45 < ratio < 0.55:
                    print(f"\n⚠️ OCTAVE ERROR DETECTED: {bpm1:.1f} and {bpm2:.1f} BPM")
                    print(f"   Ratio: {ratio:.2f}x (suggests half-time/double-time confusion)")
                    if ratio > 1:
                        print(f"   → Consider {bpm2:.1f} BPM might be correct")
                    else:
                        print(f"   → Consider {bpm1:.1f} BPM might be correct")
    
    return results


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_bpm_methods.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not Path(audio_file).exists():
        print(f"Error: File not found: {audio_file}")
        sys.exit(1)
    
    results = test_all_bpm_methods(audio_file)
