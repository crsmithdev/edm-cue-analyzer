#!/usr/bin/env python3
"""
Compare automated drop detection results with manual annotations from docs/annotations.md
"""

import os
import sys
from pathlib import Path
import logging
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from edm_cue_analyzer.analyzer import AudioAnalyzer
from edm_cue_analyzer.config import Config

# Manual annotations from docs/annotations.md
MANUAL_ANNOTATIONS = {
    "AUTOFLOWER - When It's Over (Extended Mix).flac": [31, 155],  # 31s, 2m35s
    "Activa - Get On With It (Extended Mix).flac": [27, 243],  # 27s, 4m3s
    "Adana Twins - Maya.flac": [123, 261],  # 2m3s, 4m21s
    "Agents Of Time - Zodiac.flac": [107, 243],  # 1m47s, 4m3s
    "Artbat - Artefact.flac": [30, 90, 167, 215],  # 30s, 1m30s, 2m47s, 3m35s
    "3LAU, Dnmo - Falling.flac": [44, 160],  # 44s, 2m40s
}

async def analyze_track_drops(file_path):
    """Analyze a track and return detected drop times"""
    try:
        config = Config()
        analyzer = AudioAnalyzer(config)
        
        print(f"\nAnalyzing: {file_path.name}")
        structure = await analyzer.analyze_with(file_path, analyses={'drops'})
        
        if hasattr(structure, 'drops') and structure.drops:
            # drops is a list of float time values
            detected_times = [round(drop, 1) for drop in structure.drops]
            print(f"Detected drops: {detected_times}")
            return detected_times
        else:
            print("No drops detected")
            return []
            
    except Exception as e:
        print(f"Error analyzing {file_path.name}: {e}")
        return []

def calculate_accuracy(detected, manual, tolerance=10.0):
    """Calculate detection accuracy with tolerance"""
    if not manual:
        return 0.0, 0.0, 0, 0, 0
    
    true_positives = 0
    matched_manual = set()
    
    # Find true positives (detected drops near manual annotations)
    for det_time in detected:
        for i, man_time in enumerate(manual):
            if abs(det_time - man_time) <= tolerance and i not in matched_manual:
                true_positives += 1
                matched_manual.add(i)
                break
    
    false_positives = len(detected) - true_positives
    false_negatives = len(manual) - true_positives
    
    precision = true_positives / len(detected) if detected else 0
    recall = true_positives / len(manual) if manual else 0
    
    return precision, recall, true_positives, false_positives, false_negatives

async def main():
    music_dir = Path("/music")
    
    if not music_dir.exists():
        print("Music directory /music not found")
        return
    
    print("=== DROP DETECTION COMPARISON REPORT ===")
    print(f"Comparing automated detection vs manual annotations")
    print(f"Tolerance: ±10 seconds\n")
    
    results = []
    total_tp = total_fp = total_fn = 0
    
    for filename, manual_times in MANUAL_ANNOTATIONS.items():
        file_path = music_dir / filename
        
        if not file_path.exists():
            print(f"⚠️  File not found: {filename}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Track: {filename}")
        print(f"Manual annotations: {manual_times}")
        
        detected_times = await analyze_track_drops(file_path)
        
        precision, recall, tp, fp, fn = calculate_accuracy(detected_times, manual_times)
        total_tp += tp
        total_fp += fp  
        total_fn += fn
        
        # Calculate individual drop errors
        drop_errors = []
        for manual_time in manual_times:
            closest_detected = None
            min_error = float('inf')
            
            for det_time in detected_times:
                error = abs(det_time - manual_time)
                if error < min_error:
                    min_error = error
                    closest_detected = det_time
            
            if closest_detected and min_error <= 10:
                drop_errors.append(f"{manual_time}s → {closest_detected}s (±{min_error:.1f}s)")
            else:
                drop_errors.append(f"{manual_time}s → MISSED")
        
        print(f"Drop matching: {drop_errors}")
        print(f"Precision: {precision:.2f} | Recall: {recall:.2f} | TP:{tp} FP:{fp} FN:{fn}")
        
        results.append({
            'filename': filename,
            'manual': manual_times,
            'detected': detected_times,
            'precision': precision,
            'recall': recall,
            'tp': tp, 'fp': fp, 'fn': fn
        })
    
    # Overall statistics
    print(f"\n{'='*60}")
    print("OVERALL RESULTS:")
    print(f"Total tracks analyzed: {len(results)}")
    print(f"Total manual drops: {sum(len(r['manual']) for r in results)}")
    print(f"Total detected drops: {sum(len(r['detected']) for r in results)}")
    print(f"True Positives: {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    
    overall_precision = 0
    overall_recall = 0
    
    if total_tp + total_fp > 0:
        overall_precision = total_tp / (total_tp + total_fp)
        print(f"Overall Precision: {overall_precision:.2f}")
    
    if total_tp + total_fn > 0:
        overall_recall = total_tp / (total_tp + total_fn)
        print(f"Overall Recall: {overall_recall:.2f}")
    
    if overall_precision + overall_recall > 0:
        f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
        print(f"F1 Score: {f1_score:.2f}")
    
    # Track-by-track summary table
    print(f"\n{'='*60}")
    print("TRACK-BY-TRACK SUMMARY:")
    print(f"{'Track':<30} {'Manual':<8} {'Detected':<10} {'P':<5} {'R':<5} {'Status'}")
    print("-" * 70)
    
    for result in results:
        track_name = result['filename'][:25] + "..." if len(result['filename']) > 25 else result['filename']
        manual_count = len(result['manual'])
        detected_count = len(result['detected'])
        precision = result['precision']
        recall = result['recall']
        
        if precision >= 0.8 and recall >= 0.8:
            status = "✅ GOOD"
        elif precision >= 0.5 and recall >= 0.5:
            status = "⚠️  FAIR"
        else:
            status = "❌ POOR"
        
        print(f"{track_name:<30} {manual_count:<8} {detected_count:<10} {precision:<5.2f} {recall:<5.2f} {status}")

if __name__ == "__main__":
    # Set up logging to suppress debug output
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())