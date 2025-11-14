"""BPM and beat detection analysis."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from ..consensus import ConsensusBpmDetector

logger = logging.getLogger(__name__)

# Import Essentia with proper logging configuration
from ..essentia_config import es, ESSENTIA_AVAILABLE


@dataclass
class BpmResult:
    """Result from BPM analysis."""

    bpm: float
    beats: np.ndarray  # Beat times in seconds
    bar_duration: float  # Duration of one bar (4 beats)
    confidence: float
    num_methods: int


async def analyze_bpm(context: dict) -> BpmResult:
    """
    Detect BPM and beat positions using consensus algorithm.

    Args:
        context: Dictionary containing:
            - y: Audio signal (mono, float32)
            - sr: Sample rate
            - audio_path: Path to audio file (for Essentia)

    Returns:
        BpmResult with tempo, beats, and confidence
    """
    y = context["y"]
    sr = context["sr"]
    audio_path = context.get("audio_path")

    logger.debug("Detecting tempo and beats (consensus model)...")

    # Use consensus BPM detection for improved accuracy
    consensus_detector = ConsensusBpmDetector(
        min_bpm=60.0,
        max_bpm=200.0,
        expected_range=(120.0, 145.0),  # EDM typical range
        octave_tolerance=0.1,
    )

    try:
        # Ensure mono float32 for consensus detector
        y_mono = y if y.ndim == 1 else np.mean(y, axis=1)
        y_mono = y_mono.astype(np.float32)

        bpm_estimate = await consensus_detector.detect_async(y_mono, sr)
        bpm = bpm_estimate.bpm
        beats = bpm_estimate.beats

        logger.info(
            "BPM detected: %.1f (confidence: %.1f%%, consensus from %d methods)",
            bpm,
            bpm_estimate.confidence * 100,
            bpm_estimate.metadata.get("num_methods", 1),
        )

        # Log if low confidence
        if bpm_estimate.confidence < 0.6:
            logger.warning(
                "Low BPM confidence (%.1f%%) - consider manual verification",
                bpm_estimate.confidence * 100,
            )

        # If consensus didn't provide beats, generate them
        if beats is None:
            logger.warning("Consensus didn't provide beat positions, generating from BPM")
            _, beats = librosa.beat.beat_track(y=y_mono, sr=sr, start_bpm=bpm, tightness=100)

    except Exception as e:
        logger.error("Consensus BPM detection failed: %s", e)
        # Ultimate fallback to single method
        logger.warning("Falling back to single-method BPM detection")
        if ESSENTIA_AVAILABLE and audio_path:
            try:
                bpm, beats = _detect_bpm_essentia(audio_path, y, sr)
            except Exception as e2:
                logger.warning("Essentia fallback failed: %s", e2)
                bpm, beats = _detect_bpm_librosa(y, sr)
        else:
            bpm, beats = _detect_bpm_librosa(y, sr)

        # Create fallback estimate
        bpm_estimate = type("BpmEstimate", (), {
            "bpm": bpm,
            "beats": beats,
            "confidence": 0.5,
            "metadata": {"num_methods": 1}
        })()

    beat_times = librosa.frames_to_time(beats, sr=sr)
    bar_duration = (60.0 / bpm) * 4

    return BpmResult(
        bpm=bpm,
        beats=beat_times,
        bar_duration=bar_duration,
        confidence=bpm_estimate.confidence,
        num_methods=bpm_estimate.metadata.get("num_methods", 1),
    )


def _detect_bpm_librosa(y: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    """Fallback BPM detection using librosa."""
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo), beats


def _detect_bpm_essentia(audio_path: Path, y: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    """Fallback BPM detection using Essentia."""
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, beat_times, _, _, _ = rhythm_extractor(y)
    beats = librosa.time_to_frames(beat_times, sr=sr)
    return float(bpm), beats
