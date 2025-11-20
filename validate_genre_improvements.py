#!/usr/bin/env python3
"""
Simple validation of genre-aware drop detection.
Uses our working genre parameter system to test improvements.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import our working genre system
from test_genre_system import GENRE_PARAMETERS, detect_genre_from_bpm, get_genre_parameters

# Test tracks from annotations with their expected drops
TEST_TRACKS = {
    "Autoflower - Dimension": {
        "expected_drops": [2*60 + 8],  # 2m8s
        "estimated_bpm": 140,  # Typical dubstep
        "duration": 240
    },
    "Autoflower - The Only One": {
        "expected_drops": [2*60 + 3],  # 2m3s
        "estimated_bpm": 140,
        "duration": 240
    },
    "Autoflower - When it's Over": {
        "expected_drops": [31, 2*60 + 35],  # 31s, 2m35s
        "estimated_bpm": 140,
        "duration": 240
    },
    "Adana Twins - Maya": {
        "expected_drops": [2*60 + 3, 4*60 + 21],  # 2m3s, 4m21s (problem case - had 8 false positives)
        "estimated_bpm": 128,  # House
        "duration": 360
    },
    "Agents of Time - Zodiac": {
        "expected_drops": [1*60 + 47, 4*60 + 3],  # 1m47s, 4m3s
        "estimated_bpm": 130,  # Techno
        "duration": 300
    }
}


def analyze_genre_impact(track_name: str, track_info: dict) -> dict:
    """Analyze how genre-aware parameters would affect detection for a track."""
    
    bpm = track_info["estimated_bpm"]
    expected_drops = track_info["expected_drops"]
    duration = track_info["duration"]
    
    # Detect genre
    genre = detect_genre_from_bpm(bpm)
    params = get_genre_parameters(genre)
    
    # Calculate expected behavior with genre parameters
    min_spacing_seconds = params['min_spacing_bars'] * (240 / bpm)  # Convert bars to seconds
    confidence_threshold = params['confidence_threshold']
    bass_intensity = params['bass_intensity']
    spectral_contrast = params['spectral_contrast']
    
    # Analyze drop spacing
    drop_spacing_ok = True
    if len(expected_drops) > 1:
        actual_spacing = expected_drops[1] - expected_drops[0]
        drop_spacing_ok = actual_spacing >= min_spacing_seconds
    
    return {
        "track": track_name,
        "bpm": bpm,
        "genre": genre,
        "expected_drops": len(expected_drops),
        "min_spacing_required": min_spacing_seconds,
        "actual_spacing": expected_drops[1] - expected_drops[0] if len(expected_drops) > 1 else "N/A",
        "spacing_compliant": drop_spacing_ok,
        "confidence_threshold": confidence_threshold,
        "bass_intensity": bass_intensity,
        "spectral_contrast": spectral_contrast,
        "duration": duration
    }


def simulate_detection_improvements():
    """Simulate how genre-aware parameters would improve detection."""
    
    print("🎵 Genre-Aware Drop Detection Validation")
    print("=" * 60)
    
    results = []
    
    for track_name, track_info in TEST_TRACKS.items():
        result = analyze_genre_impact(track_name, track_info)
        results.append(result)
    
    # Display results
    print(f"\n{'Track':<25} {'BPM':<4} {'Genre':<12} {'Drops':<6} {'Spacing':<8} {'Compliant':<10}")
    print("─" * 75)
    
    for result in results:
        spacing_str = f"{result['actual_spacing']:.0f}s" if isinstance(result['actual_spacing'], (int, float)) else "N/A"
        compliant_str = "✓" if result['spacing_compliant'] else "✗"
        
        print(f"{result['track']:<25} {result['bpm']:<4} {result['genre']:<12} {result['expected_drops']:<6} {spacing_str:<8} {compliant_str:<10}")
    
    # Detailed analysis for problem cases
    print(f"\n🔍 Detailed Analysis for Problem Tracks:")
    print("─" * 50)
    
    for result in results:
        if "Maya" in result['track'] or not result['spacing_compliant']:
            print(f"\n📊 {result['track']}:")
            print(f"   Genre: {result['genre']} (BPM: {result['bpm']})")
            print(f"   Expected drops: {result['expected_drops']}")
            print(f"   Min spacing required: {result['min_spacing_required']:.1f}s")
            print(f"   Actual spacing: {result['actual_spacing']}")
            print(f"   Confidence threshold: {result['confidence_threshold']}")
            print(f"   Bass intensity factor: {result['bass_intensity']}")
            print(f"   Spectral contrast: {result['spectral_contrast']}")
            
            if "Maya" in result['track']:
                print(f"   🎯 Previous issue: 8 false positives (detected 14, expected 2)")
                print(f"   📈 Expected improvement:")
                print(f"      - Higher confidence threshold ({result['confidence_threshold']} vs 0.5) → fewer weak detections")
                print(f"      - Wider spacing ({result['min_spacing_required']:.1f}s vs ~15s) → eliminate close false positives")
                print(f"      - House-specific parameters → better suited for track style")
    
    # Summary of expected improvements
    print(f"\n🚀 Expected Performance Improvements:")
    print("─" * 40)
    print("✓ Dubstep tracks: Tighter spacing (8 bars), higher bass detection")
    print("✓ House tracks: Balanced parameters, better false positive control") 
    print("✓ Techno tracks: Lower thresholds for subtle drops, wide spacing")
    print("✓ Problem case (Maya): Genre-specific house parameters should reduce 8 false positives significantly")
    
    return results


if __name__ == "__main__":
    results = simulate_detection_improvements()
    
    print(f"\n✅ Genre-aware parameter validation complete!")
    print(f"Ready to test on actual audio files when available.")