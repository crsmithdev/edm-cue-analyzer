"""
Example: Creating and using custom feature extractors

This demonstrates the modular plugin architecture for adding new analysis features.
"""

import librosa
import numpy as np

from edm_cue_analyzer import (
    AudioAnalyzer,
    FeatureExtractor,
    HPSSFeatureExtractor,
    load_config,
)


# Example 1: Create a custom feature extractor
class OnsetFeatureExtractor(FeatureExtractor):
    """Extract onset strength for precise drop detection."""

    @property
    def name(self) -> str:
        return "onset"

    def extract(self, y: np.ndarray, sr: int, **kwargs) -> dict:
        """Extract onset strength envelope."""
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_times = librosa.times_like(onset_env, sr=sr)

        return {
            'onset_strength': onset_env,
            'onset_times': onset_times
        }


class SpectralContrastExtractor(FeatureExtractor):
    """Extract spectral contrast for better section differentiation."""

    @property
    def name(self) -> str:
        return "spectral_contrast"

    def extract(self, y: np.ndarray, sr: int, **kwargs) -> dict:
        """Extract spectral contrast."""
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        return {
            'spectral_contrast': contrast,
            'spectral_contrast_mean': np.mean(contrast, axis=0)
        }


def example_default_features():
    """Example 1: Use default feature extractors (HPSS + Spectral)"""
    print("=" * 80)
    print("Example 1: Default Features (HPSS + Spectral)")
    print("=" * 80)

    config = load_config()
    analyzer = AudioAnalyzer(config.analysis)

    # Show which extractors are active
    print(f"Active extractors: {[e.name for e in analyzer.feature_extractors]}")

    # Analyze a track
    # structure = analyzer.analyze_file(Path("your_track.mp3"))
    # print(f"Available features: {list(structure.features.keys())}")


def example_custom_features():
    """Example 2: Add custom feature extractors"""
    print("\n" + "=" * 80)
    print("Example 2: Adding Custom Features")
    print("=" * 80)

    config = load_config()
    analyzer = AudioAnalyzer(config.analysis)

    # Add custom extractors
    analyzer.add_feature_extractor(OnsetFeatureExtractor())
    analyzer.add_feature_extractor(SpectralContrastExtractor())

    print(f"Active extractors: {[e.name for e in analyzer.feature_extractors]}")

    # Analyze a track
    # structure = analyzer.analyze_file(Path("your_track.mp3"))
    # print(f"Available features: {list(structure.features.keys())}")


def example_selective_features():
    """Example 3: Use only specific feature extractors"""
    print("\n" + "=" * 80)
    print("Example 3: Selective Features (Only HPSS)")
    print("=" * 80)

    config = load_config()

    # Create analyzer with only HPSS
    analyzer = AudioAnalyzer(
        config.analysis,
        feature_extractors=[HPSSFeatureExtractor()]
    )

    print(f"Active extractors: {[e.name for e in analyzer.feature_extractors]}")


def example_remove_features():
    """Example 4: Remove unwanted feature extractors"""
    print("\n" + "=" * 80)
    print("Example 4: Removing Features")
    print("=" * 80)

    config = load_config()
    analyzer = AudioAnalyzer(config.analysis)

    print(f"Initial extractors: {[e.name for e in analyzer.feature_extractors]}")

    # Remove spectral analysis to speed up processing
    analyzer.remove_feature_extractor("spectral")

    print(f"After removal: {[e.name for e in analyzer.feature_extractors]}")


def example_no_features():
    """Example 5: Energy-only analysis (no extra features)"""
    print("\n" + "=" * 80)
    print("Example 5: Minimal Analysis (Energy Only)")
    print("=" * 80)

    config = load_config()

    # Create analyzer with no feature extractors
    analyzer = AudioAnalyzer(
        config.analysis,
        feature_extractors=[]
    )

    print(f"Active extractors: {[e.name for e in analyzer.feature_extractors]}")
    print("Will use basic energy analysis only - fastest mode!")


if __name__ == "__main__":
    example_default_features()
    example_custom_features()
    example_selective_features()
    example_remove_features()
    example_no_features()

    print("\n" + "=" * 80)
    print("Benefits of Modular Architecture:")
    print("=" * 80)
    print("✓ Easy to add new features without changing core code")
    print("✓ Enable/disable features based on performance needs")
    print("✓ Test different feature combinations")
    print("✓ Extend functionality via plugins")
    print("✓ No need to refactor when adding new analysis methods")
