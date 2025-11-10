"""Core audio analysis functionality."""

import librosa
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .config import AnalysisConfig


@dataclass
class TrackStructure:
    """Detected structure elements of a track."""
    bpm: float
    duration: float
    beats: np.ndarray
    bar_duration: float
    
    # Energy profile
    energy_curve: np.ndarray
    energy_times: np.ndarray
    
    # Detected sections
    drops: List[float] = None
    breakdowns: List[float] = None
    builds: List[float] = None
    
    # Spectral info
    spectral_centroid: np.ndarray = None
    low_energy: np.ndarray = None
    mid_energy: np.ndarray = None
    high_energy: np.ndarray = None
    
    def __post_init__(self):
        if self.drops is None:
            self.drops = []
        if self.breakdowns is None:
            self.breakdowns = []
        if self.builds is None:
            self.builds = []


class AudioAnalyzer:
    """Analyzes audio files to extract structure and characteristics."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def analyze_file(self, filepath: Path) -> TrackStructure:
        """
        Analyze an audio file and extract structure.
        
        Args:
            filepath: Path to audio file (.mp3, .flac, .wav, etc.)
            
        Returns:
            TrackStructure with detected elements.
        """
        # Load audio
        y, sr = librosa.load(filepath, sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Detect tempo and beats
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # Calculate bar duration (4 beats per bar)
        bar_duration = (60.0 / bpm) * 4
        
        # Calculate energy curve
        energy_times, energy_curve = self._calculate_energy(y, sr)
        
        # Spectral analysis
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Frequency band energy
        low_energy, mid_energy, high_energy = self._calculate_frequency_bands(y, sr)
        
        # Detect structural elements
        drops = self._detect_drops(energy_curve, energy_times, bar_duration)
        breakdowns = self._detect_breakdowns(energy_curve, energy_times, bar_duration)
        builds = self._detect_builds(energy_curve, energy_times)
        
        return TrackStructure(
            bpm=bpm,
            duration=duration,
            beats=beat_times,
            bar_duration=bar_duration,
            energy_curve=energy_curve,
            energy_times=energy_times,
            drops=drops,
            breakdowns=breakdowns,
            builds=builds,
            spectral_centroid=spectral_centroid,
            low_energy=low_energy,
            mid_energy=mid_energy,
            high_energy=high_energy
        )
    
    def _calculate_energy(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate RMS energy over time."""
        hop_length = int(sr * self.config.energy_window_seconds)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        return times, rms
    
    def _calculate_frequency_bands(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate energy in different frequency bands."""
        # Compute STFT
        stft = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Define frequency bands
        low_mask = freqs < self.config.low_freq_max
        mid_mask = (freqs >= self.config.low_freq_max) & (freqs < self.config.mid_freq_max)
        high_mask = freqs >= self.config.mid_freq_max
        
        # Calculate energy in each band
        low_energy = np.mean(stft[low_mask, :], axis=0)
        mid_energy = np.mean(stft[mid_mask, :], axis=0)
        high_energy = np.mean(stft[high_mask, :], axis=0)
        
        return low_energy, mid_energy, high_energy
    
    def _detect_drops(self, energy: np.ndarray, times: np.ndarray, bar_duration: float) -> List[float]:
        """
        Detect drop points (sudden energy increases).
        
        Returns list of timestamps where drops occur.
        """
        drops = []
        
        # Look for significant energy increases
        for i in range(1, len(energy) - 1):
            # Check if energy significantly increases
            if energy[i] > energy[i-1] * self.config.drop_energy_multiplier:
                # Check if this is a local maximum
                if energy[i] > energy[i+1]:
                    # Ensure minimum spacing (at least 16 bars apart)
                    if not drops or (times[i] - drops[-1]) > (bar_duration * 16):
                        drops.append(float(times[i]))
        
        return drops
    
    def _detect_breakdowns(self, energy: np.ndarray, times: np.ndarray, bar_duration: float) -> List[float]:
        """
        Detect breakdown points (energy decreases).
        
        Returns list of timestamps where breakdowns occur.
        """
        breakdowns = []
        max_energy = np.max(energy)
        threshold = max_energy * self.config.breakdown_energy_threshold
        
        # Look for sustained low energy periods
        in_breakdown = False
        breakdown_start = None
        
        for i in range(len(energy)):
            if energy[i] < threshold and not in_breakdown:
                breakdown_start = times[i]
                in_breakdown = True
            elif energy[i] >= threshold and in_breakdown:
                # Check if breakdown was long enough
                if breakdown_start and (times[i] - breakdown_start) > self.config.min_section_duration:
                    breakdowns.append(float(breakdown_start))
                in_breakdown = False
                breakdown_start = None
        
        return breakdowns
    
    def _detect_builds(self, energy: np.ndarray, times: np.ndarray) -> List[float]:
        """
        Detect build-up sections (gradual energy increases).
        
        Returns list of timestamps where builds occur.
        """
        builds = []
        
        # Look for sustained energy increases
        for i in range(len(energy) - 5):
            # Check if energy consistently increases over next 5 samples
            window = energy[i:i+5]
            if np.all(np.diff(window) > 0):
                # Check if total increase is significant
                if window[-1] > window[0] * (1 + self.config.energy_threshold_increase):
                    builds.append(float(times[i]))
        
        return builds


def bars_to_seconds(bars: int, bpm: float) -> float:
    """Convert number of bars to seconds."""
    return (bars * 4 * 60.0) / bpm


def seconds_to_bars(seconds: float, bpm: float) -> int:
    """Convert seconds to number of bars."""
    return int((seconds * bpm) / (4 * 60.0))
