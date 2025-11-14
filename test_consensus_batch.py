#!/usr/bin/env python3
"""
Test consensus BPM detection on all 30 sample tracks.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from src.edm_cue_analyzer.consensus import ConsensusBpmDetector

# Setup logging
logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger(__name__)


def test_consensus_bpm(audio_path: Path) -> dict:
    """Test consensus BPM detection on a single file."""
    start_time = time.time()
    
    try:
        # Load audio
        y, sr = sf.read(str(audio_path))
        
        # Convert to mono if needed
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        
        y = y.astype(np.float32)
        
        # Create consensus detector
        detector = ConsensusBpmDetector(
            min_bpm=60.0,
            max_bpm=200.0,
            expected_range=(120.0, 145.0),
            octave_tolerance=0.1,
        )
        
        # Detect BPM
        result = detector.detect(y, sr)
        
        elapsed = time.time() - start_time
        
        return {
            "file": audio_path.name,
            "bpm": round(result.bpm, 1),
            "confidence": round(result.confidence * 100, 1),
            "num_methods": result.metadata.get("num_methods", 0),
            "agreement": result.metadata.get("agreement_count", 0),
            "time_seconds": round(elapsed, 1),
            "success": True,
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error processing {audio_path.name}: {e}")
        return {
            "file": audio_path.name,
            "bpm": None,
            "confidence": None,
            "num_methods": 0,
            "agreement": 0,
            "time_seconds": round(elapsed, 1),
            "success": False,
            "error": str(e),
        }


def main():
    """Run consensus BPM detection on all test tracks."""
    
    # Find audio files
    test_dir = Path("/workspaces/edm-cue-analyzer/tests")
    audio_files = []
    
    for ext in ['*.flac', '*.mp3', '*.wav']:
        audio_files.extend(test_dir.glob(ext))
    
    audio_files = sorted(audio_files)
    
    print("=" * 80)
    print("CONSENSUS BPM DETECTION - BATCH TEST")
    print("=" * 80)
    print(f"Found {len(audio_files)} audio files")
    print()
    
    results = []
    total_start = time.time()
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        
        result = test_consensus_bpm(audio_file)
        results.append(result)
        
        if result["success"]:
            print(f"  ✓ BPM: {result['bpm']} (confidence: {result['confidence']}%, "
                  f"{result['agreement']}/{result['num_methods']} methods, "
                  f"{result['time_seconds']}s)")
        else:
            print(f"  ✗ FAILED: {result.get('error', 'Unknown error')}")
        print()
    
    total_elapsed = time.time() - total_start
    
    # Summary statistics
    successful = [r for r in results if r["success"]]
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(results) - len(successful)}")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Average time per track: {total_elapsed / len(results):.1f}s")
    print()
    
    if successful:
        bpms = [r["bpm"] for r in successful]
        confidences = [r["confidence"] for r in successful]
        times = [r["time_seconds"] for r in successful]
        
        print(f"BPM range: {min(bpms):.1f} - {max(bpms):.1f}")
        print(f"Median BPM: {np.median(bpms):.1f}")
        print(f"Average confidence: {np.mean(confidences):.1f}%")
        print(f"Time range: {min(times):.1f}s - {max(times):.1f}s")
        print()
    
    # Save results
    output_file = test_dir.parent / "consensus_bpm_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Display results table
    print()
    print("=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    print(f"{'Track':<50} {'BPM':>6} {'Conf%':>6} {'Methods':>7} {'Time':>6}")
    print("-" * 80)
    
    for r in results:
        if r["success"]:
            track = r["file"][:47] + "..." if len(r["file"]) > 50 else r["file"]
            print(f"{track:<50} {r['bpm']:>6.1f} {r['confidence']:>6.1f} "
                  f"{r['agreement']}/{r['num_methods']:>1} {r['time_seconds']:>6.1f}s")
        else:
            track = r["file"][:47] + "..." if len(r["file"]) > 50 else r["file"]
            print(f"{track:<50} {'FAIL':>6} {'-':>6} {'-':>7} {r['time_seconds']:>6.1f}s")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
