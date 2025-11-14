"""Debug script to analyze drop detection on Rodg - The Coaster."""

import asyncio
from pathlib import Path

import numpy as np

from edm_cue_analyzer.analyzer import AudioAnalyzer
from edm_cue_analyzer.config import get_default_config


async def main():
    config = get_default_config()
    analyzer = AudioAnalyzer(config.analysis)

    # Analyze track
    track_path = Path("/music/Rodg - The Coaster.flac")
    structure = await analyzer.analyze(track_path)

    # Manual drop times
    manual_drops = [17, 137]  # seconds

    # Detected drops
    detected_drops = structure.drops

    print(f"\n{'=' * 80}")
    print(f"Drop Detection Analysis: {track_path.name}")
    print(f"{'=' * 80}\n")

    print(f"Manual drops: {manual_drops}")
    print(f"Detected drops: {[f'{d:.2f}' for d in detected_drops]}\n")

    # Energy curve stats
    energy = structure.energy_curve
    times = structure.energy_times

    print(f"Energy Statistics:")
    print(f"  Mean: {np.mean(energy):.4f}")
    print(f"  Std: {np.std(energy):.4f}")
    print(f"  Max: {np.max(energy):.4f}")
    print(f"  60% of max: {np.max(energy) * 0.6:.4f}")
    print(f"  Mean + 0.7*std: {np.mean(energy) + 0.7 * np.std(energy):.4f}\n")

    # Check energy at manual drop times
    print("Energy at Manual Drop Times:")
    for drop_time in manual_drops:
        idx = np.argmin(np.abs(times - drop_time))
        lookback = int(5.0 / config.analysis.energy_window_seconds)
        recent_avg = np.mean(energy[max(0, idx - lookback) : idx]) if idx > 0 else 0

        print(f"  At {drop_time}s (idx {idx}):")
        print(f"    Energy: {energy[idx]:.4f}")
        print(f"    Recent avg (5s): {recent_avg:.4f}")
        print(f"    Ratio: {energy[idx] / recent_avg if recent_avg > 0 else 0:.2f}x")
        print(f"    > 60% max? {energy[idx] > np.max(energy) * 0.6}")
        print(f"    > mean + 0.7*std? {energy[idx] > np.mean(energy) + 0.7 * np.std(energy)}")
        print()

    # Check energy at detected drop times
    print("Energy at Detected Drop Times:")
    for drop_time in detected_drops:
        idx = np.argmin(np.abs(times - drop_time))
        lookback = int(5.0 / config.analysis.energy_window_seconds)
        recent_avg = np.mean(energy[max(0, idx - lookback) : idx]) if idx > 0 else 0

        print(f"  At {drop_time:.2f}s (idx {idx}):")
        print(f"    Energy: {energy[idx]:.4f}")
        print(f"    Recent avg (5s): {recent_avg:.4f}")
        print(f"    Ratio: {energy[idx] / recent_avg if recent_avg > 0 else 0:.2f}x")
        print(f"    > 60% max? {energy[idx] > np.max(energy) * 0.6}")
        print(f"    > mean + 0.7*std? {energy[idx] > np.mean(energy) + 0.7 * np.std(energy)}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
