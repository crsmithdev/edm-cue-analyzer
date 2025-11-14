"""Core audio analysis functionality."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .config import AnalysisConfig
from .consensus import ConsensusBpmDetector

logger = logging.getLogger(__name__)


def timed(operation_name: str = None):
    """Decorator to time functions and log performance."""

    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info(f"⏱️  {name}: {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(f"⏱️  {name}: {elapsed:.2f}s (failed: {e})")
                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info(f"⏱️  {name}: {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(f"⏱️  {name}: {elapsed:.2f}s (failed: {e})")
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# Try to import essentia for best beat tracking (especially for EDM)
try:
    import essentia.standard as es

    ESSENTIA_AVAILABLE = True
    logger.debug("Essentia available - will use for BPM detection")
except ImportError:
    ESSENTIA_AVAILABLE = False
    logger.debug("Essentia not available - trying aubio")

# Fall back to aubio if essentia not available
if not ESSENTIA_AVAILABLE:
    try:
        import aubio

        AUBIO_AVAILABLE = True
        logger.debug("Aubio available - will use for BPM detection")
    except ImportError:
        AUBIO_AVAILABLE = False
        logger.debug("Aubio not available - using librosa for BPM detection")


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
    drops: list[float] = None
    breakdowns: list[float] = None
    builds: list[float] = None

    # Feature storage - extensible dictionary for all features
    features: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        if self.drops is None:
            self.drops = []
        if self.breakdowns is None:
            self.breakdowns = []
        if self.builds is None:
            self.builds = []


class FeatureExtractor(ABC):
    """Base class for feature extraction plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this feature extractor."""
        pass

    @abstractmethod
    def extract(self, y: np.ndarray, sr: int, **kwargs) -> dict[str, np.ndarray]:
        """
        Extract features from audio.

        Args:
            y: Audio time series
            sr: Sample rate
            **kwargs: Additional arguments (e.g., y_harmonic, y_percussive)

        Returns:
            Dictionary of feature name -> feature array
        """
        pass


class HPSSFeatureExtractor(FeatureExtractor):
    """Harmonic-Percussive Source Separation feature extractor."""

    @property
    def name(self) -> str:
        return "hpss"

    @timed("HPSS feature extraction")
    def extract(self, y: np.ndarray, sr: int, **kwargs) -> dict[str, np.ndarray]:
        """Extract harmonic and percussive components."""
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # Calculate energy for each component
        hop_length = int(sr * kwargs.get("energy_window", 1.0))
        harmonic_rms = librosa.feature.rms(y=y_harmonic, hop_length=hop_length)[0]
        percussive_rms = librosa.feature.rms(y=y_percussive, hop_length=hop_length)[0]

        return {
            "y_harmonic": y_harmonic,
            "y_percussive": y_percussive,
            "harmonic_energy": harmonic_rms,
            "percussive_energy": percussive_rms,
        }


class SpectralFeatureExtractor(FeatureExtractor):
    """Spectral features extractor."""

    @property
    def name(self) -> str:
        return "spectral"

    @timed("Spectral features (librosa)")
    def extract(self, y: np.ndarray, sr: int, **kwargs) -> dict[str, np.ndarray]:
        """Extract spectral features."""
        features = {}

        # Spectral centroid
        features["spectral_centroid"] = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

        # Frequency band energy
        stft = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)

        low_freq_max = kwargs.get("low_freq_max", 250)
        mid_freq_max = kwargs.get("mid_freq_max", 4000)

        low_mask = freqs < low_freq_max
        mid_mask = (freqs >= low_freq_max) & (freqs < mid_freq_max)
        high_mask = freqs >= mid_freq_max

        features["low_energy"] = np.mean(stft[low_mask, :], axis=0)
        features["mid_energy"] = np.mean(stft[mid_mask, :], axis=0)
        features["high_energy"] = np.mean(stft[high_mask, :], axis=0)

        return features


class OnsetFeatureExtractor(FeatureExtractor):
    """Onset detection feature extractor for precise event timing (librosa-based)."""

    @property
    def name(self) -> str:
        return "onset"

    @timed("Onset detection (librosa)")
    def extract(self, y: np.ndarray, sr: int, **kwargs) -> dict[str, np.ndarray]:
        """Extract onset strength envelope and onset times."""
        # Onset strength envelope - shows likelihood of new events
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        # Detect actual onset times
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            backtrack=True,  # Refine onset times to local minimum
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        return {
            "onset_strength": onset_env,
            "onset_times": onset_times,
        }


class EssentiaOnsetFeatureExtractor(FeatureExtractor):
    """Essentia-based onset detection (superior for EDM transients)."""

    @property
    def name(self) -> str:
        return "onset"

    @timed("Onset detection (Essentia)")
    def extract(self, y: np.ndarray, sr: int, **kwargs) -> dict[str, np.ndarray]:
        """Extract onset features using Essentia's onset detection."""
        if not ESSENTIA_AVAILABLE:
            logger.warning("Essentia not available, falling back to librosa onset detection")
            return OnsetFeatureExtractor().extract(y, sr, **kwargs)

        framesize = 2048
        hopsize = 512

        # Setup Essentia algorithms
        w = es.Windowing(type="hann")
        fft = es.FFT()
        c2p = es.CartesianToPolar()

        # Use 'complex' method - best for percussive EDM drops
        onset_detection = es.OnsetDetection(method="complex")

        # Process frames
        onset_values = []
        for frame in es.FrameGenerator(y, frameSize=framesize, hopSize=hopsize):
            mag, phase = c2p(fft(w(frame)))
            onset_values.append(onset_detection(mag, phase))

        onset_strength = np.array(onset_values)

        # Detect onset peaks using simple peak detection
        # Essentia's Onsets() expects different format, so we'll use a simpler approach
        threshold = np.mean(onset_strength) + np.std(onset_strength)
        onset_indices = []

        # Find peaks above threshold with minimum spacing
        min_spacing = int(0.07 * sr / hopsize)  # 70ms minimum between onsets
        last_onset = -min_spacing

        for i in range(1, len(onset_strength) - 1):
            if (
                onset_strength[i] > threshold
                and onset_strength[i] > onset_strength[i - 1]
                and onset_strength[i] > onset_strength[i + 1]
                and (i - last_onset) > min_spacing
            ):
                onset_indices.append(i)
                last_onset = i

        onset_times = np.array(onset_indices) * hopsize / sr  # Convert to seconds

        logger.debug(f"Essentia detected {len(onset_times)} onsets")

        return {
            "onset_strength": onset_strength,
            "onset_times": onset_times,
        }


class EssentiaSpectralFeatureExtractor(FeatureExtractor):
    """Essentia-based spectral features (better for texture analysis)."""

    @property
    def name(self) -> str:
        return "spectral"

    @timed("Spectral features (Essentia)")
    def extract(self, y: np.ndarray, sr: int, **kwargs) -> dict[str, np.ndarray]:
        """Extract spectral features using Essentia."""
        if not ESSENTIA_AVAILABLE:
            logger.warning("Essentia not available, falling back to librosa spectral features")
            return SpectralFeatureExtractor().extract(y, sr, **kwargs)

        framesize = 2048
        hopsize = 512

        # Setup Essentia algorithms
        w = es.Windowing(type="hann")
        spectrum = es.Spectrum()

        # Key algorithms for EDM analysis
        complexity_algo = es.SpectralComplexity()
        contrast_algo = es.SpectralContrast()
        centroid_algo = es.Centroid()
        hfc_algo = es.HFC()  # High Frequency Content

        # Process frames
        complexities = []
        contrasts = []
        centroids = []
        hfcs = []

        for frame in es.FrameGenerator(y, frameSize=framesize, hopSize=hopsize):
            spec = spectrum(w(frame))
            complexities.append(complexity_algo(spec))
            contrasts.append(contrast_algo(spec))
            centroids.append(centroid_algo(spec))
            hfcs.append(hfc_algo(spec))

        features = {
            "spectral_complexity": np.array(complexities),
            "spectral_contrast": np.array(contrasts),
            "spectral_centroid": np.array(centroids),
            "hfc": np.array(hfcs),  # High frequency content
        }

        # Also calculate frequency band energy for compatibility
        stft = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)

        low_freq_max = kwargs.get("low_freq_max", 250)
        mid_freq_max = kwargs.get("mid_freq_max", 4000)

        low_mask = freqs < low_freq_max
        mid_mask = (freqs >= low_freq_max) & (freqs < mid_freq_max)
        high_mask = freqs >= mid_freq_max

        features["low_energy"] = np.mean(stft[low_mask, :], axis=0)
        features["mid_energy"] = np.mean(stft[mid_mask, :], axis=0)
        features["high_energy"] = np.mean(stft[high_mask, :], axis=0)

        logger.debug("Essentia extracted spectral features: complexity, contrast, centroid, HFC")

        return features


class AudioAnalyzer:
    """Analyzes audio files to extract structure and characteristics."""

    def __init__(
        self, config: AnalysisConfig, feature_extractors: list[FeatureExtractor] | None = None
    ):
        """
        Initialize analyzer with configuration and optional feature extractors.

        Args:
            config: Analysis configuration
            feature_extractors: List of feature extractor plugins (default: HPSS + Spectral)
        """
        self.config = config

        # Default feature extractors - use Essentia when available for better EDM analysis
        if feature_extractors is None:
            self.feature_extractors = []

            # Add HPSS for accurate drop/breakdown detection (percussive energy tracking)
            self.feature_extractors.append(HPSSFeatureExtractor())

            # Add spectral feature extractor (Essentia preferred, librosa fallback)
            if ESSENTIA_AVAILABLE:
                self.feature_extractors.append(EssentiaSpectralFeatureExtractor())
                logger.debug("Using Essentia spectral features + HPSS for drop detection")
            else:
                self.feature_extractors.append(SpectralFeatureExtractor())
                logger.debug("Using librosa-based feature extractors with HPSS")

            # Add onset detector (Essentia preferred, librosa fallback)
            if ESSENTIA_AVAILABLE:
                self.feature_extractors.append(EssentiaOnsetFeatureExtractor())
            else:
                self.feature_extractors.append(OnsetFeatureExtractor())
        else:
            self.feature_extractors = feature_extractors

        # Create lookup by name
        self.extractors_by_name = {ext.name: ext for ext in self.feature_extractors}

    def add_feature_extractor(self, extractor: FeatureExtractor):
        """Add a new feature extractor plugin."""
        self.feature_extractors.append(extractor)
        self.extractors_by_name[extractor.name] = extractor

    def remove_feature_extractor(self, name: str):
        """Remove a feature extractor by name."""
        if name in self.extractors_by_name:
            extractor = self.extractors_by_name[name]
            self.feature_extractors.remove(extractor)
            del self.extractors_by_name[name]

    @timed("Total track analysis")
    async def analyze(self, audio_path: Path) -> TrackStructure:
        """
        Analyze an audio file to extract structure.

        Args:
            audio_path: Path to audio file

        Returns:
            TrackStructure with detected elements

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio file is invalid or too short
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.debug("Loading audio file: %s", audio_path)
        # Load audio file using soundfile (faster, no resampling)
        # Then resample only if needed for librosa compatibility
        load_start = time.perf_counter()

        # Load with native sample rate (much faster than librosa.load)
        y, sr = await asyncio.to_thread(sf.read, str(audio_path), dtype="float32")

        # Convert to mono if stereo
        if y.ndim > 1:
            y = y.mean(axis=1)

        duration = len(y) / sr
        load_time = time.perf_counter() - load_start
        logger.info(f"⏱️  Audio loading: {load_time:.2f}s (native {sr} Hz, no resampling)")
        logger.debug("Audio loaded: duration=%.2fs, sample_rate=%d", duration, sr)

        # Validate minimum duration
        if duration < 10.0:
            raise ValueError(f"Track too short for analysis: {duration:.1f}s (minimum 10s)")

        # Detect tempo and beats using consensus model
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

            bpm_estimate = consensus_detector.detect(y_mono, sr)
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
                import librosa
                _, beats = librosa.beat.beat_track(y=y_mono, sr=sr, start_bpm=bpm, tightness=100)

        except Exception as e:
            logger.error("Consensus BPM detection failed: %s", e)
            # Ultimate fallback to single method
            logger.warning("Falling back to single-method BPM detection")
            if ESSENTIA_AVAILABLE:
                try:
                    bpm, beats = self._detect_bpm_essentia(audio_path, y, sr)
                except Exception as e2:
                    logger.warning("Essentia fallback failed: %s", e2)
                    bpm, beats = self._detect_bpm_librosa(y, sr)
            else:
                bpm, beats = self._detect_bpm_librosa(y, sr)

        beat_times = librosa.frames_to_time(beats, sr=sr)

        # Calculate bar duration (4 beats per bar)
        bar_duration = (60.0 / bpm) * 4

        # Calculate base energy curve
        energy_times, energy_curve = self._calculate_energy(y, sr)

        # Run all feature extractors
        logger.debug("Running feature extractors...")
        all_features = {}
        extractor_kwargs = {
            "energy_window": self.config.energy_window_seconds,
            "low_freq_max": self.config.low_freq_max,
            "mid_freq_max": self.config.mid_freq_max,
        }

        for extractor in self.feature_extractors:
            logger.debug("Extracting features: %s", extractor.name)
            try:
                features = extractor.extract(y, sr, **extractor_kwargs)
                if not isinstance(features, dict):
                    logger.warning(
                        "Feature extractor '%s' returned non-dict result, skipping", extractor.name
                    )
                    continue
                all_features.update(features)
            except Exception as e:
                logger.error("Feature extractor '%s' failed: %s", extractor.name, e, exc_info=True)
                # Continue with other extractors rather than failing completely

        # Detect structural elements
        logger.debug("Detecting structural elements...")
        drops = self._detect_drops(energy_curve, energy_times, bar_duration, all_features)
        logger.debug("Found %d drops", len(drops))

        breakdowns = self._detect_breakdowns(energy_curve, energy_times, bar_duration, all_features)
        logger.debug("Found %d breakdowns", len(breakdowns))

        builds = self._detect_builds(energy_curve, energy_times, all_features)
        logger.debug("Found %d builds", len(builds))

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
            features=all_features,
        )

    async def analyze_file(self, audio_path: Path) -> TrackStructure:
        """Alias for analyze() for backward compatibility."""
        return await self.analyze(audio_path)

    def _calculate_energy(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """Calculate RMS energy over time."""
        logger.debug(
            "Calculating energy curve with window=%.2fs", self.config.energy_window_seconds
        )
        hop_length = int(sr * self.config.energy_window_seconds)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        logger.debug("Energy curve calculated: %d points", len(rms))
        return times, rms

    @timed("Drop detection")
    def _detect_drops(
        self,
        energy: np.ndarray,
        times: np.ndarray,
        bar_duration: float,
        features: dict[str, np.ndarray],
    ) -> list[float]:
        """
        Detect drop points (beat/bass returns after breakdowns or track start).

        In EDM, a "drop" is when the beat/bass returns in full force, either:
        1. Initial drop: First strong beat establishment
        2. Post-breakdown drop: Beat returns after being stripped away

        Args:
            energy: Overall energy curve
            times: Time points
            bar_duration: Duration of one bar
            features: Dictionary of extracted features

        Returns:
            List of timestamps where drops occur.
        """
        drops = []
        min_spacing = bar_duration * self.config.drop_min_spacing_bars

        # Check available features
        has_onsets = "onset_times" in features and "onset_strength" in features

        if not has_onsets:
            logger.warning("No onset detection available, drop detection may be inaccurate")
            return drops

        # Get features
        onset_times = features["onset_times"]
        onset_strength = features["onset_strength"]
        low_energy = features.get("low_energy", energy)  # Fallback to overall energy
        percussive_energy = features.get("percussive_energy", energy)

        # Align low_energy and percussive_energy to times array
        from scipy import interpolate

        if len(low_energy) != len(times):
            low_energy_interp = interpolate.interp1d(
                np.linspace(0, times[-1], len(low_energy)),
                low_energy,
                kind="linear",
                fill_value="extrapolate",
            )
            low_energy = low_energy_interp(times)

        if len(percussive_energy) != len(times):
            perc_energy_interp = interpolate.interp1d(
                np.linspace(0, times[-1], len(percussive_energy)),
                percussive_energy,
                kind="linear",
                fill_value="extrapolate",
            )
            percussive_energy = perc_energy_interp(times)

        # Calculate statistics
        mean_low = np.mean(low_energy)
        std_low = np.std(low_energy)
        mean_perc = np.mean(percussive_energy)
        std_perc = np.std(percussive_energy)
        mean_onset = np.mean(onset_strength)
        std_onset = np.std(onset_strength)

        # Thresholds for "significant" bass/beat presence
        # Lower threshold - bass doesn't need to be super loud
        low_threshold = mean_low + 0.3 * std_low
        perc_threshold = mean_perc + 0.3 * std_perc
        onset_threshold = mean_onset + 0.8 * std_onset  # Strong onset required

        logger.debug(
            "Drop detection: Low threshold=%.4f, Perc threshold=%.4f, Onset threshold=%.4f",
            low_threshold,
            perc_threshold,
            onset_threshold,
        )

        # Strategy: Find strong onsets where bass/percussion RETURNS after being low
        lookback_seconds = 8.0  # Look back 8 seconds to see if bass was low
        lookback_frames = int(lookback_seconds / self.config.energy_window_seconds)

        for onset_time in onset_times:
            # Find closest time index
            idx = np.argmin(np.abs(times - onset_time))

            if idx < lookback_frames or idx >= len(times) - 2:
                continue

            # Get onset strength
            onset_idx = int(onset_time * len(onset_strength) / times[-1])
            if onset_idx >= len(onset_strength):
                continue

            # Check if this is a strong onset
            if onset_strength[onset_idx] < onset_threshold:
                continue

            # Check if low-frequency (bass) and percussion are present at drop
            has_bass = low_energy[idx] > low_threshold
            has_beat = percussive_energy[idx] > perc_threshold

            if not (has_bass and has_beat):
                continue

            # KEY: Check if bass/percussion were LOW recently (breakdown before drop)
            # or if this is the first strong bass/beat (initial drop)
            lookback_start = max(0, idx - lookback_frames)
            recent_low_avg = np.mean(low_energy[lookback_start:idx])
            recent_perc_avg = np.mean(percussive_energy[lookback_start:idx])

            # Drop detected if:
            # 1. Bass/beat are NOW strong, AND
            # 2. Bass/beat were recently weak (indicating a return/drop moment)
            bass_increase = low_energy[idx] > recent_low_avg * 1.4
            beat_increase = percussive_energy[idx] > recent_perc_avg * 1.3

            # Also allow early track drops where bass simply starts strong
            is_early_drop = onset_time < 60  # First minute
            bass_strong = low_energy[idx] > mean_low + std_low

            # Check minimum spacing and add drop
            is_valid_drop = (bass_increase and beat_increase) or (
                is_early_drop and bass_strong and has_beat
            )
            if is_valid_drop and (not drops or (onset_time - drops[-1]) > min_spacing):
                drops.append(float(onset_time))
                logger.debug(
                    "Drop detected at %.2fs (low: %.4f, perc: %.4f, onset: %.4f)",
                    onset_time,
                    low_energy[idx],
                    percussive_energy[idx],
                    onset_strength[onset_idx],
                )

        return drops

    @timed("Breakdown detection")
    def _detect_breakdowns(
        self,
        energy: np.ndarray,
        times: np.ndarray,
        bar_duration: float,
        features: dict[str, np.ndarray],
    ) -> list[float]:
        """
        Detect breakdown points using combined energy and available features.

        Uses spectral complexity and HFC from Essentia if available for better accuracy.

        Args:
            energy: Overall energy curve
            times: Time points
            bar_duration: Duration of one bar
            features: Dictionary of extracted features

        Returns:
            List of timestamps where breakdowns occur.
        """
        breakdowns = []

        max_energy = np.max(energy)
        mean_energy = np.mean(energy)

        # Check for Essentia spectral features (best for breakdown detection)
        use_spectral = "spectral_complexity" in features and "hfc" in features
        use_hpss = "harmonic_energy" in features and "percussive_energy" in features

        if use_spectral:
            logger.debug("Breakdown detection mode: Essentia spectral-based")
        elif use_hpss:
            logger.debug("Breakdown detection mode: HPSS-based")
        else:
            logger.debug("Breakdown detection mode: energy-based")

        # Setup thresholds based on available features
        if use_spectral:
            spectral_complexity = features["spectral_complexity"]
            hfc = features["hfc"]

            # Align spectral features with energy curve
            # Spectral features may have different frame count, interpolate if needed
            if len(spectral_complexity) != len(energy):
                from scipy import interpolate

                old_indices = np.linspace(0, len(times) - 1, len(spectral_complexity))
                new_indices = np.arange(len(times))
                f_complexity = interpolate.interp1d(
                    old_indices, spectral_complexity, kind="linear", fill_value="extrapolate"
                )
                f_hfc = interpolate.interp1d(
                    old_indices, hfc, kind="linear", fill_value="extrapolate"
                )
                spectral_complexity = f_complexity(new_indices)
                hfc = f_hfc(new_indices)

            mean_complexity = np.mean(spectral_complexity)
            mean_hfc = np.mean(hfc)

            # Breakdowns have low complexity and low HFC
            complexity_threshold = mean_complexity * 0.7
            hfc_threshold = mean_hfc * 0.6
            energy_threshold = min(
                max_energy * self.config.breakdown_energy_threshold, mean_energy * 0.85
            )

        elif use_hpss:
            harmonic_energy = features["harmonic_energy"]
            percussive_energy = features["percussive_energy"]
            mean_perc = np.mean(percussive_energy)

            # Calculate ratio of harmonic to percussive
            epsilon = 1e-10
            harmonic_ratio = harmonic_energy / (percussive_energy + epsilon)
            mean_ratio = np.mean(harmonic_ratio)

            # Breakdown thresholds (using config values)
            energy_threshold = min(
                max_energy * self.config.breakdown_energy_threshold, mean_energy * 0.85
            )
            perc_threshold = mean_perc * self.config.breakdown_perc_threshold
            ratio_threshold = mean_ratio * self.config.breakdown_ratio_threshold
        else:
            # Fallback to simple energy-based detection
            energy_threshold = min(
                max_energy * self.config.breakdown_energy_threshold, mean_energy * 0.85
            )

        in_breakdown = False
        breakdown_start = None
        breakdown_start_idx = None

        for i in range(len(energy)):
            # Determine if this is breakdown-like based on available features
            if use_spectral:
                # Essentia-based: low energy + low complexity + low HFC = breakdown
                is_breakdown_like = (
                    energy[i] < energy_threshold
                    and spectral_complexity[i] < complexity_threshold
                    and hfc[i] < hfc_threshold
                )
            elif use_hpss:
                is_breakdown_like = energy[i] < energy_threshold and (
                    percussive_energy[i] < perc_threshold or harmonic_ratio[i] > ratio_threshold
                )
            else:
                is_breakdown_like = energy[i] < energy_threshold

            if is_breakdown_like and not in_breakdown:
                breakdown_start = times[i]
                breakdown_start_idx = i
                in_breakdown = True
            elif not is_breakdown_like and in_breakdown:
                # Check if breakdown was long enough (using config)
                min_duration = bar_duration * self.config.breakdown_min_duration_bars
                if breakdown_start and (times[i] - breakdown_start) >= min_duration:
                    # Verify it's a significant breakdown
                    breakdown_avg = np.mean(energy[breakdown_start_idx:i])

                    valid_breakdown = breakdown_avg < mean_energy * 0.9

                    if use_spectral:
                        # Essentia validation: breakdown region should have low complexity
                        breakdown_complexity_avg = np.mean(
                            spectral_complexity[breakdown_start_idx:i]
                        )
                        valid_breakdown = valid_breakdown and (
                            breakdown_complexity_avg < mean_complexity * 0.8
                        )
                    elif use_hpss:
                        breakdown_perc_avg = np.mean(percussive_energy[breakdown_start_idx:i])
                        valid_breakdown = valid_breakdown and (breakdown_perc_avg < mean_perc * 0.8)

                    if valid_breakdown:
                        # Ensure minimum spacing between breakdowns (using config)
                        min_spacing = bar_duration * self.config.breakdown_min_spacing_bars
                        if not breakdowns or (breakdown_start - breakdowns[-1]) > min_spacing:
                            breakdowns.append(float(breakdown_start))
                in_breakdown = False
                breakdown_start = None
                breakdown_start_idx = None

        return breakdowns

    @timed("Build detection")
    def _detect_builds(
        self, energy: np.ndarray, times: np.ndarray, features: dict[str, np.ndarray]
    ) -> list[float]:
        """
        Detect build-up sections (gradual energy increases).

        Args:
            energy: Overall energy curve
            times: Time points
            features: Dictionary of extracted features

        Returns:
            List of timestamps where builds occur.
        """
        builds = []

        # Look for sustained energy increases (using config window size)
        window_size = self.config.build_window_size
        logger.debug(
            "Build detection: window_size=%d, threshold=%.2f%%",
            window_size,
            self.config.energy_threshold_increase * 100,
        )

        for i in range(len(energy) - window_size):
            # Check if energy consistently increases over window
            window = energy[i : i + window_size]
            if np.all(np.diff(window) > 0) and window[-1] > window[0] * (
                1 + self.config.energy_threshold_increase
            ):
                builds.append(float(times[i]))

        return builds

    @timed("BPM detection (librosa)")
    def _detect_bpm_librosa(self, y: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
        """
        Detect BPM using librosa's beat tracker.

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Tuple of (bpm, beat_frames)
        """
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo)

        # Refine BPM - check for common rounding issues and half/double time
        # Round to nearest 0.5 BPM for common tempo grid alignment
        bpm_rounded = round(bpm * 2) / 2

        # Check if we're detecting half-time or double-time
        # For EDM, common range is 120-150 BPM
        if bpm_rounded < 100 and bpm_rounded * 2 >= 120 and bpm_rounded * 2 <= 150:
            logger.debug("Detected half-time tempo, doubling: %.1f -> %.1f", bpm, bpm_rounded * 2)
            bpm = bpm_rounded * 2
        elif bpm_rounded > 160 and bpm_rounded / 2 >= 120 and bpm_rounded / 2 <= 150:
            logger.debug("Detected double-time tempo, halving: %.1f -> %.1f", bpm, bpm_rounded / 2)
            bpm = bpm_rounded / 2
        else:
            bpm = bpm_rounded

        if bpm < 60 or bpm > 200:
            raise ValueError(f"Detected BPM {bpm:.1f} outside valid range (60-200)")

        logger.debug("Detected BPM (librosa): %.1f", bpm)
        return bpm, beats

    @timed("BPM detection (Essentia)")
    def _detect_bpm_essentia(
        self, audio_path: Path, y: np.ndarray, sr: int
    ) -> tuple[float, np.ndarray]:
        """
        Detect BPM using essentia's RhythmExtractor2013 (best for EDM).

        This uses the winning algorithm from MIREX 2013 beat tracking competition,
        optimized for electronic dance music with steady 4/4 rhythms.

        Args:
            audio_path: Path to audio file
            y: Audio time series
            sr: Sample rate

        Returns:
            Tuple of (bpm, beat_frames)
        """
        # Essentia expects mono float32 audio
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        y = y.astype(np.float32)

        # Use RhythmExtractor2013 - multifeature method (best for EDM)
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm_raw, beat_times, beat_confidence, _, beat_intervals = rhythm_extractor(y)

        if len(beat_times) < 2:
            raise ValueError("Essentia detected fewer than 2 beats")

        # Refine BPM - check for common rounding issues and half/double time
        bpm_rounded = round(bpm_raw * 2) / 2

        # Check if we're detecting half-time or double-time
        # For EDM, common range is 120-150 BPM
        if bpm_rounded < 100 and bpm_rounded * 2 >= 120 and bpm_rounded * 2 <= 150:
            logger.debug(
                "Detected half-time tempo, doubling: %.1f -> %.1f", bpm_raw, bpm_rounded * 2
            )
            bpm = bpm_rounded * 2
        elif bpm_rounded > 160 and bpm_rounded / 2 >= 120 and bpm_rounded / 2 <= 150:
            logger.debug(
                "Detected double-time tempo, halving: %.1f -> %.1f", bpm_raw, bpm_rounded / 2
            )
            bpm = bpm_rounded / 2
        else:
            bpm = bpm_rounded

        if bpm < 60 or bpm > 200:
            raise ValueError(f"Detected BPM {bpm:.1f} outside valid range (60-200)")

        logger.debug(
            "Detected BPM (essentia): %.1f (confidence: %.2f)", bpm, np.mean(beat_confidence)
        )

        # Convert beat times to frames for consistency with librosa
        beat_frames = librosa.time_to_frames(beat_times, sr=sr)

        return bpm, beat_frames

    @timed("BPM detection (Aubio)")
    def _detect_bpm_aubio(
        self, audio_path: Path, y: np.ndarray, sr: int
    ) -> tuple[float, np.ndarray]:
        """
        Detect BPM using aubio's beat tracking (better than librosa).

        Args:
            audio_path: Path to audio file
            y: Audio time series
            sr: Sample rate

        Returns:
            Tuple of (bpm, beat_frames)
        """
        # Aubio parameters
        hop_size = 512
        win_s = 1024

        # Create tempo detection object
        tempo = aubio.tempo("default", win_s, hop_size, sr)

        # Process audio in chunks
        beat_times = []
        n_samples = len(y)
        for i in range(0, n_samples, hop_size):
            # Get audio chunk
            chunk = y[i : i + hop_size]
            if len(chunk) < hop_size:
                chunk = np.pad(chunk, (0, hop_size - len(chunk)))

            # Detect beat
            is_beat = tempo(chunk.astype(np.float32))
            if is_beat:
                beat_time = tempo.get_last_s()
                beat_times.append(beat_time)

        if len(beat_times) < 2:
            raise ValueError("Aubio detected fewer than 2 beats")

        # Calculate BPM from beat intervals
        beat_times = np.array(beat_times)
        intervals = np.diff(beat_times)
        median_interval = np.median(intervals)
        bpm = 60.0 / median_interval

        # Refine BPM - check for common rounding issues and half/double time
        bpm_rounded = round(bpm * 2) / 2

        # Check if we're detecting half-time or double-time
        # For EDM, common range is 120-150 BPM
        if bpm_rounded < 100 and bpm_rounded * 2 >= 120 and bpm_rounded * 2 <= 150:
            logger.debug("Detected half-time tempo, doubling: %.1f -> %.1f", bpm, bpm_rounded * 2)
            bpm = bpm_rounded * 2
        elif bpm_rounded > 160 and bpm_rounded / 2 >= 120 and bpm_rounded / 2 <= 150:
            logger.debug("Detected double-time tempo, halving: %.1f -> %.1f", bpm, bpm_rounded / 2)
            bpm = bpm_rounded / 2
        else:
            bpm = bpm_rounded

        if bpm < 60 or bpm > 200:
            raise ValueError(f"Detected BPM {bpm:.1f} outside valid range (60-200)")

        logger.debug("Detected BPM (aubio): %.1f", bpm)

        # Convert beat times to frames for consistency with librosa
        beat_frames = librosa.time_to_frames(beat_times, sr=sr)

        return bpm, beat_frames


def bars_to_seconds(bars: int, bpm: float) -> float:
    """
    Convert number of bars to seconds.

    Args:
        bars: Number of 4/4 bars
        bpm: Beats per minute

    Returns:
        Duration in seconds
    """
    return (bars * 4 * 60.0) / bpm


def seconds_to_bars(seconds: float, bpm: float) -> int:
    """
    Convert seconds to number of bars.

    Args:
        seconds: Duration in seconds
        bpm: Beats per minute

    Returns:
        Number of 4/4 bars (rounded down)
    """
    return int((seconds * bpm) / (4 * 60.0))


def calculate_energy_stats(energy: np.ndarray) -> dict[str, float]:
    """
    Calculate statistical measures of energy curve.

    Args:
        energy: Energy curve array

    Returns:
        Dictionary with mean, std, max, min, and median energy
    """
    return {
        "mean": float(np.mean(energy)),
        "std": float(np.std(energy)),
        "max": float(np.max(energy)),
        "min": float(np.min(energy)),
        "median": float(np.median(energy)),
    }
