#!/usr/bin/env python3
"""
Batch BPM detection test script.
Tests BPM detection on multiple tracks and outputs results for verification.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from edm_cue_analyzer.analyzer import AudioAnalyzer


def test_bpm_batch(music_dir: str, max_tracks: int = 50):
    """Test BPM detection on multiple tracks."""
    music_path = Path(music_dir)
    
    if not music_path.exists():
        print(f"Error: {music_dir} does not exist")
        return
    
    # Get all FLAC files
    flac_files = sorted(music_path.glob("*.flac"))[:max_tracks]
    
    print(f"Testing BPM detection on {len(flac_files)} tracks from {music_dir}\n")
    print("=" * 100)
    
    results = []
    
    for i, flac_file in enumerate(flac_files, 1):
        try:
            print(f"\n[{i}/{len(flac_files)}] Analyzing: {flac_file.name}")

            # Create analyzer and detect BPM
            analyzer = AudioAnalyzer(str(flac_file))
            structure = analyzer.analyze_sync()
            bpm = structure.bpm
            duration = structure.duration            # Extract artist and title from filename
            # Common patterns: "Artist - Title.flac" or "Artist - Title (Mix).flac"
            filename = flac_file.stem
            parts = filename.split(" - ", 1)
            
            if len(parts) == 2:
                artist = parts[0].strip()
                title = parts[1].strip()
            else:
                artist = "Unknown"
                title = filename
            
            result = {
                "file": flac_file.name,
                "artist": artist,
                "title": title,
                "detected_bpm": bpm,
                "duration_min": duration / 60,
                "official_bpm": None,  # To be filled in manually or via API
                "error": None
            }
            
            results.append(result)
            
            print(f"  Artist: {artist}")
            print(f"  Title: {title}")
            print(f"  Detected BPM: {bpm}")
            print(f"  Duration: {duration/60:.2f} min")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "file": flac_file.name,
                "artist": "Error",
                "title": str(e),
                "detected_bpm": None,
                "duration_min": None,
                "official_bpm": None,
                "error": str(e)
            })
    
    # Save results to JSON
    output_file = Path(__file__).parent / "bpm_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 100)
    print(f"\nResults saved to: {output_file}")
    print("\nSummary:")
    print(f"  Total tracks: {len(results)}")
    successful = sum(1 for r in results if r["detected_bpm"] is not None)
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(results) - successful}")
    
    # Generate CSV for easy verification
    csv_file = Path(__file__).parent / "bpm_test_results.csv"
    with open(csv_file, "w") as f:
        f.write("Artist,Title,Detected BPM,Duration (min),Official BPM,Error (%),Filename\n")
        for r in results:
            bpm = r["detected_bpm"] or ""
            duration = f"{r['duration_min']:.2f}" if r["duration_min"] else ""
            f.write(f'"{r["artist"]}","{r["title"]}",{bpm},{duration},,,"{r["file"]}"\n')
    
    print(f"\nCSV saved to: {csv_file}")
    print("\nYou can now:")
    print("1. Open the CSV in a spreadsheet")
    print("2. Manually add official BPMs from Beatport/other sources")
    print("3. Calculate error percentages")
    
    return results


if __name__ == "__main__":
    music_dir = "/music"
    max_tracks = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    
    test_bpm_batch(music_dir, max_tracks)
