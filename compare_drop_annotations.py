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
    "Autoflower - When It's Over.wav": [31, 155],  # 31s, 2m35s
    "Elysian - Come Back Down.wav": [34, 98, 162],  # 34s, 1m38s, 2m42s  
    "Flux Pavilion - I Can't Stop.wav": [27, 91, 155],  # 27s, 1m31s, 2m35s
    "Mat Zo - Superman.wav": [32, 96, 160],  # 32s, 1m36s, 2m40s
    "Modestep - Feel Good.wav": [37, 101, 165],  # 37s, 1m41s, 2m45s
    "MUST DIE! - VIPs.wav": [22, 86],  # 22s, 1m26s
    "Nero - Promises.wav": [38, 102, 166],  # 38s, 1m42s, 2m46s
    "Porter Robinson - Language.wav": [35, 99],  # 35s, 1m39s
    "Skrillex - Bangarang.wav": [21, 64, 107],  # 21s, 1m04s, 1m47s
    "Zomboy - Like A Bitch.wav": [29, 93, 157]  # 29s, 1m33s, 2m37s
}

async def analyze_track_drops(file_path):
    """Analyze a track and return detected drop times"""
    try:
        config = Config()
        analyzer = AudioAnalyzer(config)
        
        print(f"\nAnalyzing: {file_path.name}")
        structure = await analyzer.analyze_file(file_path)
        
        if hasattr(structure, 'drops') and structure.drops:
            detected_times = [round(drop.time, 1) for drop in structure.drops]
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
        
        print(f"{track_name:<30} {manual_count:<8} {detected_count:<10} {precision:.2f:<5} {recall:.2f:<5} {status}")

if __name__ == "__main__":
    # Set up logging to suppress debug output
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())