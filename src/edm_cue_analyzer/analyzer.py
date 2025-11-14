"""Core audio analysis functionality."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

import librosa
import numpy as np
import soundfile as sf

from .config import AnalysisConfig
from .consensus import ConsensusBpmDetector

if TYPE_CHECKING:
    from .config import Config

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
                logger.info(f"{name}: {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(f"{name}: {elapsed:.2f}s (failed: {e})")
                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info(f"{name}: {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(f"{name}: {elapsed:.2f}s (failed: {e})")
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
    """
    Analyzes audio files to extract structure and characteristics.
    
    This is a single-file analysis library. It provides async primitives for
    analyzing individual tracks. Batch processing and parallelization across
    multiple files should be handled by the calling code (e.g., CLI, web API).
    
    Internal operations (like BPM consensus detection) may use parallelization
    as an implementation detail, but this is transparent to the caller.
    """

    def __init__(
        self, config: "AnalysisConfig | Config", feature_extractors: list[FeatureExtractor] | None = None
    ):
        """
        Initialize analyzer with configuration and optional feature extractors.

        Args:
            config: Analysis configuration (AnalysisConfig) or full Config object
            feature_extractors: List of feature extractor plugins (default: HPSS + Spectral)
        """
        # Handle both AnalysisConfig and full Config objects
        from .config import Config as ConfigClass
        if isinstance(config, ConfigClass):
            self.full_config = config
            self.config = config.analysis
        else:
            self.full_config = None
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

    @timed("BPM detection")
    async def analyze_with(
        self, audio_path: Path, analyses: set[str] | str = "full"
    ) -> TrackStructure:
        """
        Analyze an audio file running only requested analyses with automatic dependency resolution.

        Args:
            audio_path: Path to audio file
            analyses: Set of analysis names ('bpm', 'energy', 'drops', 'breakdowns', 'builds'),
                     or preset name ('bpm-only', 'structure', 'full')

        Returns:
            TrackStructure with requested analyses populated

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If unknown analysis requested or circular dependency detected
        """
        from .analyses import expand_preset, resolve_dependencies, ANALYSES

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Expand preset to set of analyses
        requested = expand_preset(analyses)

        # Resolve dependencies and get execution order
        execution_order = resolve_dependencies(requested)

        logger.debug("Requested analyses: %s", requested)
        logger.debug("Execution order: %s", execution_order)

        # Load audio once
        logger.debug("Loading audio file: %s", audio_path)
        load_start = time.perf_counter()

        y, sr = await asyncio.to_thread(sf.read, str(audio_path), dtype="float32")

        # Convert to mono if stereo
        if y.ndim > 1:
            y = y.mean(axis=1)

        duration = len(y) / sr
        load_time = time.perf_counter() - load_start
        logger.info(f"Audio loading: {load_time:.2f}s (native {sr} Hz, no resampling)")

        # Validate minimum duration
        if duration < 10.0:
            raise ValueError(f"Track too short for analysis: {duration:.1f}s (minimum 10s)")

        # Prepare context for analyses
        context = {
            "y": y,
            "sr": sr,
            "audio_path": audio_path,
            "config": self.config,
            "features": {},  # Will be populated by feature extractors if needed
        }

        # Run analyses in dependency order
        results = {}
        for analysis_name in execution_order:
            analysis = ANALYSES[analysis_name]
            logger.debug(f"Running analysis: {analysis_name}")

            start = time.perf_counter()
            try:
                result = await analysis.func(context)
                elapsed = time.perf_counter() - start
                logger.info(f"{analysis.description}: {elapsed:.2f}s")

                # Store result in context for dependent analyses
                context[analysis_name] = result
                results[analysis_name] = result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(
                    f"{analysis.description}: {elapsed:.2f}s (failed: {e})", exc_info=True
                )
                raise

        # Build TrackStructure from results
        bpm_result = results.get("bpm")
        energy_result = results.get("energy")

        # Apply BPM precision rounding
        bpm = bpm_result.bpm if bpm_result else 0.0
        if bpm > 0 and self.full_config:
            bpm = round(bpm, self.full_config.bpm_precision)

        return TrackStructure(
            bpm=bpm,
            duration=duration,
            beats=bpm_result.beats if bpm_result else np.array([]),
            bar_duration=bpm_result.bar_duration if bpm_result else 0.0,
            energy_curve=energy_result.curve if energy_result else np.array([]),
            energy_times=energy_result.times if energy_result else np.array([]),
            drops=results.get("drops", []),
            breakdowns=results.get("breakdowns", []),
            builds=results.get("builds", []),
            features=context.get("features", {}),
        )

    @timed("Total track analysis")
    async def analyze_file(self, audio_path: Path) -> TrackStructure:
        """Analyze an audio file with full analysis.
        
        This is the main entry point for backward compatibility.
        Runs all available analyses.
        """
        return await self.analyze_with(audio_path, analyses=["full"])

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
