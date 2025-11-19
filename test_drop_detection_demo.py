#!/usr/bin/env python3
"""
Demonstration of the new three-stage drop detection system.

This script creates a synthetic EDM track with known drop points and tests
the new drop detection algorithm to verify it works correctly.
"""

import asyncio
import numpy as np
import logging

# Set up logging to see the detailed drop detection process
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)-30s %(levelname)-8s %(message)s'
)

from src.edm_cue_analyzer.analyzer import AudioAnalyzer
from src.edm_cue_analyzer.config import AnalysisConfig

# Suppress Essentia logs
logging.getLogger('essentia').setLevel(logging.WARNING)


def create_synthetic_edm_track(sr=22050, duration_sec=60, bpm=128):
    """
    Create a synthetic EDM track with characteristic drop patterns.
    
    Structure:
    - 0-15s: Intro (minimal)
    - 15-25s: Buildup (increasing energy, filter sweep)
    - 25s: DROP 1 (bass surge)
    - 25-50s: Main section (high energy)
    - 50-60s: Outro
    """
    print(f"\nGenerating synthetic EDM track:")
    print(f"  Duration: {duration_sec}s")
    print(f"  Sample rate: {sr} Hz")
    print(f"  BPM: {bpm}")
    print(f"  Expected drops: 25s")
    
    samples = sr * duration_sec
    t = np.linspace(0, duration_sec, samples)
    
    # Initialize track
    track = np.zeros(samples)
    
    # Bass frequency (sub-bass)
    bass_freq = 60  # Hz
    
    # Kick drum pattern (4/4 time)
    beat_interval = 60.0 / bpm  # seconds per beat
    for beat_time in np.arange(0, duration_sec, beat_interval):
        beat_sample = int(beat_time * sr)
        if beat_sample < samples:
            # Create kick drum (bass impulse)
            kick_length = int(0.1 * sr)
            kick_env = np.exp(-np.linspace(0, 5, kick_length))
            kick = kick_env * np.sin(2 * np.pi * bass_freq * np.linspace(0, 0.1, kick_length))
            end_idx = min(beat_sample + kick_length, samples)
            track[beat_sample:end_idx] += kick[:end_idx - beat_sample] * 0.3
    
    # Add harmonic content (chords for structure detection)
    chord_freqs = [261.63, 329.63, 392.00]  # C major chord
    for freq in chord_freqs:
        track += 0.1 * np.sin(2 * np.pi * freq * t)
    
    # Section-specific elements
    
    # INTRO (0-15s): Minimal - just high-hats
    intro_mask = (t >= 0) & (t < 15)
    hihat = np.random.normal(0, 0.05, samples)
    hihat = hihat * intro_mask
    track += hihat
    
    # BUILDUP 1 (15-25s): Increasing energy, filter sweep
    buildup1_mask = (t >= 15) & (t < 25)
    buildup1_t = (t - 15) / 10  # 0 to 1 over buildup
    buildup_len = np.sum(buildup1_mask)
    if buildup_len > 0:
        # Rising filter (increasing spectral centroid)
        for freq in np.linspace(500, 4000, 10):
            track[buildup1_mask] += 0.05 * buildup1_t[buildup1_mask] * np.sin(2 * np.pi * freq * t[buildup1_mask])
        # Increasing energy
        track[buildup1_mask] *= (1 + buildup1_t[buildup1_mask])
    
    # DROP 1 (25s onwards): Massive bass surge
    drop1_mask = (t >= 25) & (t < 60)
    # Strong bass
    track[drop1_mask] += 0.8 * np.sin(2 * np.pi * bass_freq * t[drop1_mask])
    # Sub-bass
    track[drop1_mask] += 0.6 * np.sin(2 * np.pi * (bass_freq / 2) * t[drop1_mask])
    # Full spectrum energy
    for freq in [bass_freq * i for i in range(2, 8)]:
        track[drop1_mask] += 0.2 * np.sin(2 * np.pi * freq * t[drop1_mask])
    
    # Normalize
    track = track / np.max(np.abs(track)) * 0.9
    
    return track.astype(np.float32)


async def test_drop_detection():
    """Test the new drop detection system with synthetic audio."""
    
    print("\n" + "="*80)
    print("TESTING NEW THREE-STAGE DROP DETECTION SYSTEM")
    print("="*80)
    
    # Create synthetic track
    sr = 22050
    y = create_synthetic_edm_track(sr=sr, duration_sec=60, bpm=128)
    
    # Initialize analyzer
    config = AnalysisConfig()
    analyzer = AudioAnalyzer(config)
    
    print("\n" + "-"*80)
    print("RUNNING DROP DETECTION...")
    print("-"*80 + "\n")
    
    # Manual analysis to test drop detection directly
    from src.edm_cue_analyzer.analyses.bpm import analyze_bpm
    from src.edm_cue_analyzer.analyses.energy import analyze_energy
    from src.edm_cue_analyzer.analyses.drops import analyze_drops
    
    # Create context
    context = {
        'y': y,
        'sr': sr,
        'config': config,
        'features': {}
    }
    
    # Run BPM analysis
    print("1. Running BPM analysis...")
    bpm_result = await analyze_bpm(context)
    context['bpm'] = bpm_result
    print(f"   Detected BPM: {bpm_result.bpm:.1f}")
    
    # Extract features for drop detection
    print("\n2. Extracting features...")
    from src.edm_cue_analyzer.analyzer import SpectralFeatureExtractor, OnsetFeatureExtractor
    
    spectral_extractor = SpectralFeatureExtractor()
    onset_extractor = OnsetFeatureExtractor()
    
    context['features'].update(spectral_extractor.extract(y, sr))
    context['features'].update(onset_extractor.extract(y, sr))
    print(f"   Extracted: {list(context['features'].keys())}")
    
    # Run energy analysis
    print("\n3. Running energy analysis...")
    energy_result = await analyze_energy(context)
    context['energy'] = energy_result
    print(f"   Energy curve: {len(energy_result.curve)} points")
    
    # Run drop detection
    print("\n4. Running drop detection...")
    print("-"*80)
    drops = await analyze_drops(context)
    print("-"*80)
    
    # Report results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nExpected drops: 25s")
    print(f"Detected drops: {len(drops)}")
    
    if drops:
        print("\nDetected drop times:")
        for i, drop_time in enumerate(drops, 1):
            print(f"  Drop {i}: {drop_time:.2f}s")
        
        # Check accuracy
        expected = [25.0]
        tolerance = 10.0  # seconds
        
        print(f"\nAccuracy check (tolerance: ±{tolerance}s):")
        for exp_time in expected:
            closest = min(drops, key=lambda d: abs(d - exp_time))
            error = abs(closest - exp_time)
            status = "✓" if error < tolerance else "✗"
            print(f"  {status} Expected {exp_time}s -> Detected {closest:.2f}s (error: {error:.2f}s)")
    else:
        print("\n  ⚠ No drops detected!")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(test_drop_detection())
