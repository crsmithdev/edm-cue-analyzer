#!/usr/bin/env python3
"""
Analyze a specific time point to understand why it was rejected as a drop.
"""

import asyncio
import logging
from pathlib import Path
import numpy as np
import librosa
from scipy.signal import savgol_filter

logging.basicConfig(level=logging.INFO)

async def analyze_specific_time():
    """Analyze the spectrogram around the second expected drop."""
    
    audio_path = Path("/music/AUTOFLOWER - When It's Over (Extended Mix).flac")
    drop_time = 155.0  # Second expected drop
    
    print(f"\nAnalyzing time point: {drop_time}s (2m35s)")
    print("="*80)
    
    # Load audio
    y, sr = librosa.load(str(audio_path), sr=None)
    print(f"Loaded: {len(y)/sr:.1f}s at {sr}Hz")
    
    # Compute spectrogram
    window_size = 2048
    hop_length = 512
    D = librosa.stft(y, n_fft=window_size, hop_length=hop_length)
    magnitude = np.abs(D)
    
    # Define frequency bands
    freqs = librosa.fft_frequencies(sr=sr, n_fft=window_size)
    low_mask = (freqs >= 20) & (freqs <= 150)
    mid_mask = (freqs > 150) & (freqs <= 2000)
    high_mask = freqs > 2000
    
    # Compute band energies
    low_energy = np.mean(magnitude[low_mask, :], axis=0) if np.any(low_mask) else np.zeros(magnitude.shape[1])
    mid_energy = np.mean(magnitude[mid_mask, :], axis=0) if np.any(mid_mask) else np.zeros(magnitude.shape[1])
    high_energy = np.mean(magnitude[high_mask, :], axis=0) if np.any(high_mask) else np.zeros(magnitude.shape[1])
    
    # Smooth
    if len(low_energy) > 51:
        low_smooth = savgol_filter(low_energy, 51, 3)
        mid_smooth = savgol_filter(mid_energy, 51, 3)
        high_smooth = savgol_filter(high_energy, 51, 3)
    else:
        low_smooth = low_energy
        mid_smooth = mid_energy
        high_smooth = high_energy
    
    # Analyze around drop time
    drop_frame = int(drop_time * sr / hop_length)
    window_frames = int(2.0 * sr / hop_length)  # 2 seconds before/after
    
    print(f"\nFrequency band energy analysis:")
    print(f"Window: {drop_time-2:.1f}s to {drop_time+2:.1f}s")
    print("-"*80)
    
    # Compare different time windows
    for offset, label in [(-2, "2s before"), (-1, "1s before"), (0, "at drop"), 
                          (1, "1s after"), (2, "2s after")]:
        frame = drop_frame + int(offset * sr / hop_length)
        if 0 <= frame < len(low_smooth):
            print(f"{label:12s}: Bass={low_smooth[frame]:7.2f}  "
                  f"Mid={mid_smooth[frame]:7.2f}  "
                  f"High={high_smooth[frame]:7.2f}")
    
    # Compare 1s before vs 1s after
    pre_start = max(0, drop_frame - window_frames // 2)
    pre_end = drop_frame
    post_start = drop_frame
    post_end = min(len(low_smooth), drop_frame + window_frames // 2)
    
    pre_bass = np.mean(low_smooth[pre_start:pre_end])
    post_bass = np.mean(low_smooth[post_start:post_end])
    
    pre_mid = np.mean(mid_smooth[pre_start:pre_end])
    post_mid = np.mean(mid_smooth[post_start:post_end])
    
    pre_high = np.mean(high_smooth[pre_start:pre_end])
    post_high = np.mean(high_smooth[post_start:post_end])
    
    print("\n" + "-"*80)
    print("1-second averages (before → after):")
    print(f"  Bass:  {pre_bass:.2f} → {post_bass:.2f}  "
          f"({((post_bass/pre_bass - 1)*100):+.1f}%)")
    print(f"  Mid:   {pre_mid:.2f} → {post_mid:.2f}  "
          f"({((post_mid/pre_mid - 1)*100):+.1f}%)")
    print(f"  High:  {pre_high:.2f} → {post_high:.2f}  "
          f"({((post_high/pre_high - 1)*100):+.1f}%)")
    
    print("\n" + "-"*80)
    print("Drop validation criteria:")
    print(f"  Bass increase required: >20%")
    print(f"  Actual bass increase: {((post_bass/pre_bass - 1)*100):+.1f}%")
    
    if post_bass > pre_bass * 1.2:
        print("  Result: ✓ WOULD BE VALIDATED")
    else:
        print("  Result: ✗ REJECTED (insufficient bass surge)")
    
    # Check if it's more of a breakdown-style drop
    if post_mid < pre_mid * 0.8 or post_high < pre_high * 0.8:
        print("\n  Note: Significant mid/high frequency reduction detected")
        print("        This may be a breakdown-style transition rather than")
        print("        a traditional bass-heavy drop.")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(analyze_specific_time())
