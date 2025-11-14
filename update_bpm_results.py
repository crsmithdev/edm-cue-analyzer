#!/usr/bin/env python3
"""
Update BPM test results with official BPMs and calculate errors.
"""

import csv
import json

# Known official BPMs from Beatport verification
OFFICIAL_BPMS = {
    "3runo Kaufmann - Raised in the 90S (Original Mix)": 135,
    "Adam Beyer - Pilot (Original Mix)": 133,
    "Adana Twins - Maya (Original Mix)": 110,
    "ANDATA Teenage Mutants - Black Milk (Teenage Mutants Remix)": 131,
    # From previous validation
    "Rodg - The Coaster": 128,
    "Audiowerks - Acid Lingue (Original Mix)": 138,
    "AUTOFLOWER - THE ONLY ONE": 126,
    "Digital Mess - Orange Vortex": 120,
}


def update_bpm_results(csv_file='bpm_test_results.csv', output_file='bpm_comparison.csv'):
    """Update CSV with official BPMs and calculate errors."""
    
    results = []
    
    # Read existing results
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Create lookup key
            if row['Title']:
                key = f"{row['Artist']} - {row['Title']}"
            else:
                key = row['Artist']
            
            # Check for official BPM
            official_bpm = OFFICIAL_BPMS.get(key)
            
            # Calculate error if we have official BPM
            if official_bpm:
                detected = float(row['Detected BPM'])
                error = detected - official_bpm
                error_pct = (error / official_bpm) * 100
                
                row['Official BPM'] = official_bpm
                row['Error (BPM)'] = f"{error:+.1f}"
                row['Error %'] = f"{error_pct:+.1f}%"
            
            results.append(row)
    
    # Write updated results
    fieldnames = ['Artist', 'Title', 'Detected BPM', 'Official BPM', 'Error (BPM)', 'Error %', 'Time (s)', 'Filename']
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Updated results saved to: {output_file}")
    
    # Print comparison table
    print("\n" + "=" * 120)
    print("BPM DETECTION ACCURACY COMPARISON")
    print("=" * 120)
    print(f"{'Artist':<30} {'Title':<40} {'Detected':<10} {'Official':<10} {'Error':<10} {'Error %':<10}")
    print("-" * 120)
    
    verified_count = 0
    exact_match = 0
    total_error = 0
    max_error = 0
    max_error_track = ""
    
    for row in results:
        if row.get('Official BPM'):
            verified_count += 1
            artist = row['Artist'][:28] if len(row['Artist']) > 28 else row['Artist']
            title = row['Title'][:38] if len(row['Title']) > 38 else row['Title']
            
            detected = float(row['Detected BPM'])
            official = int(row['Official BPM'])
            error = detected - official
            error_pct = (error / official) * 100
            
            # Highlight errors > 2%
            marker = " ⚠️ " if abs(error_pct) > 2 else " ✓ "
            
            print(f"{artist:<30} {title:<40} {detected:<10.1f} {official:<10} {error:+.1f}{marker:<7} {error_pct:+.1f}%")
            
            if abs(error) < 0.5:
                exact_match += 1
            
            total_error += abs(error)
            if abs(error) > abs(max_error):
                max_error = error
                max_error_track = f"{artist} - {title}"
    
    print("-" * 120)
    print(f"\nSTATISTICS:")
    print(f"  Verified tracks: {verified_count}/{len(results)}")
    print(f"  Exact matches (< 0.5 BPM error): {exact_match}/{verified_count}")
    print(f"  Average absolute error: {total_error/verified_count:.2f} BPM")
    print(f"  Largest error: {max_error:+.1f} BPM on '{max_error_track}'")
    print(f"  Accuracy rate: {(exact_match/verified_count)*100:.1f}%")
    print("=" * 120)


if __name__ == '__main__':
    update_bpm_results()
