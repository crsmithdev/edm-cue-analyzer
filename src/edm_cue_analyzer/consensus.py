"""Consensus-based analysis algorithms for improved accuracy."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Import Essentia with proper logging configuration
from .essentia_config import es, ESSENTIA_AVAILABLE

# Try to import aubio
try:
    import aubio

    AUBIO_AVAILABLE = True
except ImportError:
    AUBIO_AVAILABLE = False


@dataclass
class BpmEstimate:
    """BPM estimate from a single detection method."""

    bpm: float
    confidence: float
    method: str
    beats: np.ndarray = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ConsensusBpmDetector:
    """
    Multi-method BPM detection with consensus voting.

    Combines multiple BPM detection algorithms and uses statistical
    methods to determine the most reliable estimate. Detects and corrects
    octave errors (half-time/double-time confusion).
    """

    def __init__(
        self,
        min_bpm: float = 60.0,
        max_bpm: float = 200.0,
        expected_range: tuple[float, float] = (120.0, 145.0),
        octave_tolerance: float = 0.1,
    ):
        """
        Initialize consensus BPM detector.

        Args:
            min_bpm: Minimum valid BPM
            max_bpm: Maximum valid BPM
            expected_range: Expected BPM range for genre (helps resolve octave errors)
            octave_tolerance: Tolerance for octave relationship detection (0-1)
        """
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.expected_range = expected_range
        self.octave_tolerance = octave_tolerance

    async def detect_async(self, y: np.ndarray, sr: int) -> BpmEstimate:
        """
        Detect BPM using consensus of multiple methods (async, parallelized).

        Args:
            y: Audio time series (mono, float32)
            sr: Sample rate

        Returns:
            BpmEstimate with consensus BPM and confidence
        """
        # Define all detection tasks
        tasks = []
        
        if ESSENTIA_AVAILABLE:
            tasks.extend([
                ("essentia_rhythm", self._detect_essentia_rhythm_descriptors),
                ("essentia_multifeature", self._detect_essentia_multifeature),
                ("essentia_degara", self._detect_essentia_degara),
                ("essentia_percival", self._detect_essentia_percival),
            ])
        
        if AUBIO_AVAILABLE:
            tasks.append(("aubio", self._detect_aubio))
        
        tasks.append(("librosa", self._detect_librosa))
        
        # Run all methods in parallel
        logger.debug(f"Running {len(tasks)} BPM detection methods in parallel...")
        
        import time
        
        async def run_method(name: str, method):
            """Run a single detection method in a thread."""
            start = time.perf_counter()
            try:
                result = await asyncio.to_thread(method, y, sr)
                elapsed = time.perf_counter() - start
                logger.debug(f"{name} completed in {elapsed:.2f}s")
                # Handle methods that return lists
                if isinstance(result, list):
                    return result
                return [result]
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.debug(f"{name} failed after {elapsed:.2f}s: {e}")
                return []
        
        # Execute all methods concurrently
        start_parallel = time.perf_counter()
        results = await asyncio.gather(*[run_method(name, method) for name, method in tasks])
        parallel_time = time.perf_counter() - start_parallel
        logger.debug(f"Parallel execution completed in {parallel_time:.2f}s")
        
        # Flatten results (some methods return lists)
        estimates = []
        for result_list in results:
            estimates.extend(result_list)
        
        if not estimates:
            raise RuntimeError("All BPM detection methods failed")
        
        logger.debug(f"Collected {len(estimates)} BPM estimates")
        
        # Build consensus from all available estimates
        consensus = self._build_consensus(estimates)
        
        logger.info(
            "BPM Consensus: %.1f BPM (confidence: %.2f%%, %d methods agreed)",
            consensus.bpm,
            consensus.confidence * 100,
            len([e for e in estimates if abs(e.bpm - consensus.bpm) < 2]),
        )
        
        return consensus

    def detect(self, y: np.ndarray, sr: int) -> BpmEstimate:
        """
        Detect BPM using consensus of multiple methods (synchronous wrapper).

        Args:
            y: Audio time series (mono, float32)
            sr: Sample rate

        Returns:
            BpmEstimate with consensus BPM and confidence
        """
        # Run async version in event loop
        try:
            # If there's a running loop this will succeed and we instruct the
            # caller to use the async API. We don't need to keep the returned
            # loop object, so avoid assigning to an unused variable.
            asyncio.get_running_loop()
            # We're already in an async context, use the async version directly
            raise RuntimeError("Use detect_async() when in async context")
        except RuntimeError:
            # No running loop, create one
            return asyncio.run(self.detect_async(y, sr))

    def _detect_essentia_rhythm_descriptors(self, y: np.ndarray, sr: int) -> list[BpmEstimate]:
        """
        Use Essentia RhythmDescriptors for comprehensive analysis.
        
        Returns a list of estimates (may include both first and second peak if significant).
        """
        rhythm_desc = es.RhythmDescriptors()
        (
            beats_pos,
            confidence,
            bpm,
            _estimates_list,
            _intervals,
            first_peak_bpm,
            _first_peak_spread,
            first_peak_weight,
            second_peak_bpm,
            _second_peak_spread,
            second_peak_weight,
            _histogram,
        ) = rhythm_desc(y)

        import librosa
        beat_frames = librosa.time_to_frames(beats_pos, sr=sr)

        estimates = []
        
        # Always add the primary detection
        estimates.append(BpmEstimate(
            bpm=float(bpm),
            confidence=float(confidence / 5.32),  # Normalize to 0-1
            method="essentia_rhythm_descriptors",
            beats=beat_frames,
            metadata={
                "first_peak_bpm": float(first_peak_bpm),
                "first_peak_weight": float(first_peak_weight),
                "second_peak_bpm": float(second_peak_bpm),
                "second_peak_weight": float(second_peak_weight),
            },
        ))
        
        # If second peak is significant AND in expected range, add it as alternative
        if second_peak_weight > 0.25 and second_peak_bpm > 0:
            if self.expected_range[0] <= second_peak_bpm <= self.expected_range[1]:
                # Check if it's different enough from first peak
                if abs(second_peak_bpm - bpm) > 5:
                    logger.debug(
                        "RhythmDescriptors: Adding second peak %.1f BPM (weight: %.2f) as alternative",
                        second_peak_bpm,
                        second_peak_weight,
                    )
                    estimates.append(BpmEstimate(
                        bpm=float(second_peak_bpm),
                        confidence=float(second_peak_weight),  # Use peak weight as confidence
                        method="essentia_rhythm_descriptors_alt",
                        beats=beat_frames,
                        metadata={
                            "peak": "second",
                            "first_peak_bpm": float(first_peak_bpm),
                            "second_peak_bpm": float(second_peak_bpm),
                        },
                    ))

        return estimates

    def _detect_essentia_multifeature(self, y: np.ndarray, sr: int) -> BpmEstimate:
        """Use Essentia RhythmExtractor2013 multifeature method."""
        extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beat_times, confidence, estimates, intervals = extractor(y)

        import librosa

        beat_frames = librosa.time_to_frames(beat_times, sr=sr)

        return BpmEstimate(
            bpm=float(bpm),
            confidence=float(np.mean(confidence) / 5.32) if len(confidence) > 0 else 0.5,
            method="essentia_multifeature",
            beats=beat_frames,
        )

    def _detect_essentia_degara(self, y: np.ndarray, sr: int) -> BpmEstimate:
        """Use Essentia RhythmExtractor2013 degara method (fast)."""
        extractor = es.RhythmExtractor2013(method="degara")
        bpm, beat_times, _, estimates, intervals = extractor(y)

        import librosa

        beat_frames = librosa.time_to_frames(beat_times, sr=sr)

        return BpmEstimate(
            bpm=float(bpm),
            confidence=0.6,  # degara doesn't provide confidence, use moderate default
            method="essentia_degara",
            beats=beat_frames,
        )

    def _detect_essentia_percival(self, y: np.ndarray, sr: int) -> BpmEstimate:
        """Use Essentia PercivalBpmEstimator."""
        estimator = es.PercivalBpmEstimator()
        bpm = estimator(y)

        return BpmEstimate(
            bpm=float(bpm),
            confidence=0.7,  # No confidence score, use moderate default
            method="essentia_percival",
        )

    def _detect_aubio(self, y: np.ndarray, sr: int) -> BpmEstimate:
        """Use Aubio tempo detector."""
        hop_size = 512
        win_s = 1024

        tempo_detector = aubio.tempo("default", win_s, hop_size, sr)

        beat_times = []
        n_samples = len(y)
        for i in range(0, n_samples, hop_size):
            chunk = y[i : i + hop_size]
            if len(chunk) < hop_size:
                chunk = np.pad(chunk, (0, hop_size - len(chunk)))

            is_beat = tempo_detector(chunk.astype(np.float32))
            if is_beat:
                beat_times.append(tempo_detector.get_last_s())

        if len(beat_times) < 2:
            raise ValueError("Aubio detected fewer than 2 beats")

        intervals = np.diff(beat_times)
        median_interval = np.median(intervals)
        bpm = 60.0 / median_interval

        import librosa

        beat_frames = librosa.time_to_frames(np.array(beat_times), sr=sr)

        return BpmEstimate(
            bpm=float(bpm),
            confidence=0.65,  # No confidence score, use moderate default
            method="aubio",
            beats=beat_frames,
        )

    def _detect_librosa(self, y: np.ndarray, sr: int) -> BpmEstimate:
        """Use Librosa beat tracker (fallback)."""
        import librosa

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

        return BpmEstimate(
            bpm=float(tempo),
            confidence=0.5,  # No confidence score, use default
            method="librosa",
            beats=beats,
        )

    def _is_octave_relationship(self, ratio: float) -> bool:
        """Check if two BPMs are in octave or simple harmonic relationship."""
        # Common tempo relationships to check
        relationships = [
            2.0,    # 2:1 (double-time)
            0.5,    # 1:2 (half-time)
            1.5,    # 3:2
            0.667,  # 2:3
            4.0,    # 4:1
            0.25,   # 1:4
            3.0,    # 3:1
            0.333,  # 1:3
        ]
        
        for target in relationships:
            if abs(ratio - target) < self.octave_tolerance:
                return True
        
        return False

    def _build_consensus(self, estimates: list[BpmEstimate]) -> BpmEstimate:
        """
        Build consensus from multiple BPM estimates.

        Uses weighted voting with octave error detection and correction.
        """
        if not estimates:
            raise ValueError("No estimates provided")

        if len(estimates) == 1:
            return estimates[0]

        # Log all estimates
        logger.debug("BPM estimates from %d methods:", len(estimates))
        for est in estimates:
            logger.debug("  %s: %.1f BPM (confidence: %.2f)", est.method, est.bpm, est.confidence)

        # Detect and resolve octave errors
        estimates = self._resolve_octave_errors(estimates)

        # Calculate weighted median BPM
        bpms = np.array([e.bpm for e in estimates])
        weights = np.array([e.confidence for e in estimates])

        # Normalize weights
        weights = weights / np.sum(weights)

        # Weighted median
        sorted_indices = np.argsort(bpms)
        sorted_bpms = bpms[sorted_indices]
        sorted_weights = weights[sorted_indices]

        cumsum = np.cumsum(sorted_weights)
        median_idx = np.searchsorted(cumsum, 0.5)
        consensus_bpm = float(sorted_bpms[median_idx])

        # Calculate consensus confidence based on agreement
        # Higher confidence if methods agree within ±2 BPM
        agreement_count = np.sum(np.abs(bpms - consensus_bpm) < 2)
        consensus_confidence = agreement_count / len(estimates)

        # Boost confidence if high-confidence methods agree
        high_conf_estimates = [e for e in estimates if e.confidence > 0.7]
        if high_conf_estimates:
            high_conf_bpms = [e.bpm for e in high_conf_estimates]
            high_conf_agreement = np.sum(np.abs(np.array(high_conf_bpms) - consensus_bpm) < 2)
            consensus_confidence = max(
                consensus_confidence, high_conf_agreement / len(high_conf_estimates) * 0.9
            )

        # Get beats from highest confidence estimate that has beats
        beats = None
        for estimate in sorted(estimates, key=lambda e: e.confidence, reverse=True):
            if estimate.beats is not None:
                beats = estimate.beats
                break
        
        # If no method provided beats, we'll need to generate them later
        # (this should rarely happen as most methods provide beats)
        if beats is None:
            logger.warning("No consensus method provided beat positions")

        # Round to nearest 0.1 BPM
        consensus_bpm = round(consensus_bpm, 1)

        return BpmEstimate(
            bpm=consensus_bpm,
            confidence=float(consensus_confidence),
            method="consensus",
            beats=beats,
            metadata={
                "num_methods": len(estimates),
                "agreement_count": int(agreement_count),
                "methods_used": [e.method for e in estimates],
            },
        )

    def _resolve_octave_errors(self, estimates: list[BpmEstimate]) -> list[BpmEstimate]:
        """
        Detect and resolve octave errors in estimates.

        If estimates cluster around octave-related values (e.g., 120 and 240),
        choose the cluster in the expected range.
        """
        if len(estimates) < 2:
            return estimates

        bpms = np.array([e.bpm for e in estimates])

        # Check for octave clustering
        # Group BPMs by octave relationships
        clusters = []
        used = set()

        for i, bpm1 in enumerate(bpms):
            if i in used:
                continue

            cluster = [i]
            used.add(i)

            for j, bpm2 in enumerate(bpms):
                if j in used:
                    continue

                ratio = bpm1 / bpm2 if bpm2 > 0 else 0

                if self._is_octave_relationship(ratio):
                    cluster.append(j)
                    used.add(j)

            if len(cluster) > 0:
                clusters.append(cluster)

        # If we found octave clusters, resolve them
        if len(clusters) > 1:
            logger.debug("Found %d octave-related clusters", len(clusters))

            corrected_estimates = []

            for cluster_indices in clusters:
                cluster_bpms = [bpms[i] for i in cluster_indices]
                cluster_estimates = [estimates[i] for i in cluster_indices]

                # Choose BPM from cluster that's in expected range
                in_range = [
                    bpm
                    for bpm in cluster_bpms
                    if self.expected_range[0] <= bpm <= self.expected_range[1]
                ]

                if in_range:
                    # Use the one in expected range
                    target_bpm = in_range[0]
                else:
                    # Use median of cluster
                    target_bpm = float(np.median(cluster_bpms))

                # Correct all estimates in cluster to target BPM
                for est in cluster_estimates:
                    if abs(est.bpm - target_bpm) > 5:
                        logger.debug(
                            "Correcting octave error: %s %.1f -> %.1f",
                            est.method,
                            est.bpm,
                            target_bpm,
                        )
                        est.bpm = target_bpm

                corrected_estimates.extend(cluster_estimates)

            return corrected_estimates

        return estimates
