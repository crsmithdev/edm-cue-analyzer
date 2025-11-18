#!/usr/bin/env python3
"""
Compare detected drops against manual ground truth.
"""

import json
from pathlib import Path

# Manual ground truth (converted to seconds)
GROUND_TRUTH = {
    "AUTOFLOWER - Dimension.flac": [128.0],  # 2m8s
    "AUTOFLOWER - THE ONLY ONE.flac": [123.0],  # 2m3s
    "AUTOFLOWER - When It's Over (Extended Mix).flac": [31.0, 155.0],  # 31s, 2m35s
    "AUTOFLOWER - Wallflower.flac": [62.0, 125.0],  # 1m2s, 2m5s
    "Activa - Get On With It (Extended Mix).flac": [27.0, 243.0],  # 27s, 4m3s
    "Adana Twins - Maya.flac": [123.0, 261.0],  # 2m3s, 4m21s
    "Agents Of Time - Zodiac.flac": [107.0, 243.0],  # 1m47s, 4m3s
    "Artbat - Artefact.flac": [30.0, 90.0, 167.0, 215.0],  # 30s, 1m30s, 2m47s, 3m35s
    "3LAU, Dnmo - Falling.flac": [44.0, 160.0],  # 44s, 2m40s
}

# Load detected results from JSON
def load_detected_results():
    """Load results from validation output."""
    results_file = Path('/workspaces/edm-cue-analyzer/drop_validation_results.json')
    if not results_file.exists():
        print(f"ERROR: {results_file} not found. Run validate_drops.py first.")
        return {}
    
    data = json.load(open(results_file))
    detected = {}
    for result in data['results']:
        if result['success']:
            detected[result['file']] = result['drops']
    return detected

DETECTED = load_detected_results()

def match_drops(detected, ground_truth, tolerance=5.0):
    """
    Match detected drops to ground truth within tolerance.
    Returns (true_positives, false_positives, false_negatives)
    """
    detected = sorted(detected)
    ground_truth = sorted(ground_truth)
    
    matched_gt = set()
    matched_det = set()
    
    # For each detected drop, find closest ground truth
    for i, det in enumerate(detected):
        for j, gt in enumerate(ground_truth):
            if j in matched_gt:
                continue
            if abs(det - gt) <= tolerance:
                matched_gt.add(j)
                matched_det.add(i)
                break
    
    tp = len(matched_det)
    fp = len(detected) - len(matched_det)
    fn = len(ground_truth) - len(matched_gt)
    
    return tp, fp, fn, matched_det, matched_gt

def main():
    print("="*80)
    print("DROP DETECTION ACCURACY ANALYSIS")
    print("="*80)
    print("\nComparing detected drops vs. manual ground truth")
    print("Tolerance: ±5 seconds\n")
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    
    for filename in sorted(GROUND_TRUTH.keys()):
        gt = GROUND_TRUTH[filename]
        det = DETECTED.get(filename, [])
        
        tp, fp, fn, matched_det, matched_gt = match_drops(det, gt)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt += len(gt)
        
        print(f"\n{filename}")
        print(f"  Ground Truth: {len(gt)} drops at {[f'{t:.0f}s' for t in gt]}")
        print(f"  Detected: {len(det)} drops at {[f'{t:.1f}s' for t in det]}")
        print(f"  ✓ True Positives: {tp}")
        print(f"  ✗ False Positives: {fp}")
        print(f"  ✗ False Negatives: {fn}")
        
        if tp > 0:
            for i in matched_det:
                # Find which GT it matched
                for j in matched_gt:
                    if abs(det[i] - gt[j]) <= 5.0:
                        print(f"    • Matched: {det[i]:.1f}s → {gt[j]:.0f}s (Δ={abs(det[i]-gt[j]):.1f}s)")
                        break
        
        if fp > 0:
            unmatched = [det[i] for i in range(len(det)) if i not in matched_det]
            print(f"    • False positives at: {[f'{t:.1f}s' for t in unmatched]}")
        
        if fn > 0:
            unmatched = [gt[j] for j in range(len(gt)) if j not in matched_gt]
            print(f"    • Missed drops at: {[f'{t:.0f}s' for t in unmatched]}")
    
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nTotal Ground Truth Drops: {total_gt}")
    print(f"Total Detected Drops: {total_tp + total_fp}")
    print(f"\nTrue Positives: {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    print(f"\nPrecision: {precision:.1%} ({total_tp}/{total_tp + total_fp})")
    print(f"Recall: {recall:.1%} ({total_tp}/{total_gt})")
    print(f"F1 Score: {f1:.1%}")
    
    print("\n" + "="*80)
    print("CHANGE 7 IMPACT ASSESSMENT")
    print("="*80)
    
    print(f"""
The energy derivative validation (Change 7) shows:
• Precision: {precision:.1%} - Good at avoiding false positives
• Recall: {recall:.1%} - {"Needs improvement" if recall < 0.5 else "Reasonable" if recall < 0.7 else "Good"} at finding drops
• F1: {f1:.1%} - Overall performance metric

Key findings:
- Detected {total_tp} out of {total_gt} actual drops
- {total_fp} false positives (peaks that aren't real drops)
- {total_fn} missed drops (needs investigation)

The derivative check is working but may be:
{"✓ Too strict - missing real drops that have gradual energy changes" if recall < 0.5 else "✓ Well-calibrated - good balance between precision and recall" if f1 > 0.6 else "✓ Moderately effective - room for improvement"}
""")

if __name__ == '__main__':
    main()
