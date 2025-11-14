#!/usr/bin/env python3
"""
Fast BPM-only batch testing script.
Skips all heavy analysis and only detects BPM.
"""

import json
import logging
import sys
import time
from pathlib import Path

import soundfile as sf

# Setup minimal logging
logging.basicConfig(level=logging.WARNING)

# Try to import essentia for fast BPM detection
try:
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
    print("Warning: Essentia not available, falling back to aubio")

# Fall back to aubio
if not ESSENTIA_AVAILABLE:
    try:
        import aubio
        AUBIO_AVAILABLE = True
    except ImportError:
        AUBIO_AVAILABLE = False
        print("Warning: Aubio not available, falling back to librosa")
        import librosa


def detect_bpm_fast(audio_path: str) -> float:
    """Detect BPM using the fastest available method."""
    # Load audio at native sample rate (fast)
    audio, sr = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Use Essentia (fastest and best for EDM)
    if ESSENTIA_AVAILABLE:
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, _, _, _ = rhythm_extractor(audio)
        return float(bpm)
    
    # Fall back to aubio
    elif AUBIO_AVAILABLE:
        hop_size = 512
        win_size = 1024
        tempo_tracker = aubio.tempo("default", win_size, hop_size, sr)
        
        # Process audio in chunks
        total_frames = len(audio)
        n_frames = 0
        
        for i in range(0, total_frames - hop_size, hop_size):
            frame = audio[i:i+hop_size]
            if len(frame) < hop_size:
                break
            tempo_tracker(frame.astype('float32'))
            n_frames += 1
        
        bpm = tempo_tracker.get_bpm()
        return float(bpm)
    
    # Fall back to librosa (slowest)
    else:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        return float(tempo[0])


def test_bpm_batch_fast(music_dir: str, max_tracks: int = 50):
    """Fast BPM-only batch test."""
    music_path = Path(music_dir)
    
    if not music_path.exists():
        print(f"Error: {music_dir} does not exist")
        return
    
    # Get all FLAC files
    flac_files = sorted(music_path.glob("*.flac"))[:max_tracks]
    
    print(f"Fast BPM testing on {len(flac_files)} tracks")
    print(f"Method: {'Essentia' if ESSENTIA_AVAILABLE else 'Aubio' if AUBIO_AVAILABLE else 'Librosa'}")
    print("=" * 100)
    
    results = []
    total_time = 0
    
    for i, flac_file in enumerate(flac_files, 1):
        try:
            start_time = time.time()
            
            # Fast BPM detection only
            bpm = detect_bpm_fast(str(flac_file))
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Parse artist/title from filename
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
                "detected_bpm": round(bpm, 1),
                "official_bpm": None,
                "error_pct": None,
                "analysis_time": round(elapsed, 2)
            }
            
            results.append(result)
            
            print(f"[{i:2d}/{len(flac_files)}] {bpm:6.1f} BPM ({elapsed:4.1f}s) - {artist} - {title[:50]}")
            
        except Exception as e:
            print(f"[{i:2d}/{len(flac_files)}] ERROR: {flac_file.name} - {e}")
            results.append({
                "file": flac_file.name,
                "artist": "Error",
                "title": str(e),
                "detected_bpm": None,
                "official_bpm": None,
                "error_pct": None,
                "analysis_time": None
            })
    
    print("=" * 100)
    
    # Save results
    output_file = Path(__file__).parent / "bpm_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Generate CSV
    csv_file = Path(__file__).parent / "bpm_test_results.csv"
    with open(csv_file, "w") as f:
        f.write("Artist,Title,Detected BPM,Official BPM,Error %,Time (s),Filename\n")
        for r in results:
            bpm = r["detected_bpm"] or ""
            time_s = r["analysis_time"] or ""
            artist = r["artist"].replace('"', '""')
            title = r["title"].replace('"', '""')
            f.write(f'"{artist}","{title}",{bpm},,,{time_s},"{r["file"]}"\n')
    
    print(f"✅ CSV saved to: {csv_file}")
    
    # Summary
    successful = sum(1 for r in results if r["detected_bpm"] is not None)
    avg_time = total_time / successful if successful > 0 else 0
    
    print(f"\n📊 Summary:")
    print(f"   Total tracks: {len(results)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {len(results) - successful}")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Avg time/track: {avg_time:.2f}s")
    print(f"   Speed: {60/avg_time:.1f} tracks/min" if avg_time > 0 else "")
    
    print("\n📝 Next steps:")
    print("   1. Open bpm_test_results.csv in a spreadsheet")
    print("   2. Look up official BPMs from Beatport/Discogs/etc")
    print("   3. Add official BPMs to the 'Official BPM' column")
    print("   4. Calculate error: =(C2-D2)/D2*100 in 'Error %' column")
    
    return results


if __name__ == "__main__":
    music_dir = "/music"
    max_tracks = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    
    test_bpm_batch_fast(music_dir, max_tracks)
