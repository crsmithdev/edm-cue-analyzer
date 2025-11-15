# Removed

This document has been removed. See `docs/ARCHITECTURE.md` for the
architectural overview and notes on the modular feature extractors.

```python
from edm_cue_analyzer import (
    AudioAnalyzer,
    HPSSFeatureExtractor,
    load_config
)

config = load_config()

# Only use HPSS
analyzer = AudioAnalyzer(
    config.analysis,
    feature_extractors=[HPSSFeatureExtractor()]
)
```

### Adding Custom Features

```python
# Add features at runtime
analyzer.add_feature_extractor(MyCustomExtractor())

# Remove features
analyzer.remove_feature_extractor("spectral")
```

### Energy-Only Mode (Fastest)

```python
# No feature extractors = energy analysis only
analyzer = AudioAnalyzer(
    config.analysis,
    feature_extractors=[]
)
```

## Creating Custom Feature Extractors

### Step 1: Inherit from FeatureExtractor

```python
from edm_cue_analyzer import FeatureExtractor
import librosa
import numpy as np

class OnsetFeatureExtractor(FeatureExtractor):
    @property
    def name(self) -> str:
        return "onset"
    
    def extract(self, y: np.ndarray, sr: int, **kwargs) -> dict:
        """Extract onset strength."""
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        return {
            'onset_strength': onset_env,
        }
```

### Step 2: Use the Custom Extractor

```python
analyzer = AudioAnalyzer(config.analysis)
analyzer.add_feature_extractor(OnsetFeatureExtractor())

structure = analyzer.analyze_file(Path("track.mp3"))

# Access extracted features
onset_strength = structure.features['onset_strength']
```

## Available kwargs in extract()

Feature extractors receive these parameters:

- `y` - Audio time series
- `sr` - Sample rate
- `energy_window` - Energy window size in seconds
- `low_freq_max` - Low frequency band maximum (Hz)
- `mid_freq_max` - Mid frequency band maximum (Hz)

## Feature Storage

All extracted features are stored in `TrackStructure.features` dictionary:

```python
structure = analyzer.analyze_file(Path("track.mp3"))

# Default features
harmonic_energy = structure.features['harmonic_energy']
percussive_energy = structure.features['percussive_energy']
spectral_centroid = structure.features['spectral_centroid']
low_energy = structure.features['low_energy']

# Custom features (if added)
onset_strength = structure.features.get('onset_strength')
```

## Detection Methods

Detection methods (`_detect_drops`, `_detect_breakdowns`, `_detect_builds`) receive the features dictionary and can use any available features:

```python
def _detect_drops(self, energy, times, bar_duration, features):
    # Check if HPSS features are available
    if 'percussive_energy' in features:
        # Use percussive energy for better drop detection
        perc_energy = features['percussive_energy']
        # ... enhanced detection logic
    else:
        # Fallback to energy-only detection
        # ... basic detection logic
```

## Example Feature Extractors

### Onset Detection

```python
class OnsetFeatureExtractor(FeatureExtractor):
    @property
    def name(self) -> str:
        return "onset"
    
    def extract(self, y, sr, **kwargs):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        return {'onset_strength': onset_env}
```

### Spectral Contrast

```python
class SpectralContrastExtractor(FeatureExtractor):
    @property
    def name(self) -> str:
        return "spectral_contrast"
    
    def extract(self, y, sr, **kwargs):
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        return {
            'spectral_contrast': contrast,
            'spectral_contrast_mean': np.mean(contrast, axis=0)
        }
```

### Tempogram

```python
class TempogramExtractor(FeatureExtractor):
    @property
    def name(self) -> str:
        return "tempogram"
    
    def extract(self, y, sr, **kwargs):
        tempogram = librosa.feature.tempogram(y=y, sr=sr)
        return {'tempogram': tempogram}
```

### Chroma Features

```python
class ChromaExtractor(FeatureExtractor):
    @property
    def name(self) -> str:
        return "chroma"
    
    def extract(self, y, sr, **kwargs):
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        return {'chroma': chroma}
```

## Performance Considerations

Feature extraction order of computational cost (fastest to slowest):

1. **Energy-only** (no extractors) - ~1x baseline
2. **Spectral** - ~2-3x baseline
3. **HPSS** - ~3-4x baseline (runs STFT separation)
4. **HPSS + Spectral** (default) - ~4-5x baseline
5. **All features** - ~6-8x baseline

**Tip:** For real-time or batch processing, disable features you don't need.

## Migration from Old Code

The new architecture is **backward compatible**. Old code continues to work:

```python
# Old way (still works)
analyzer = AudioAnalyzer(config.analysis)
structure = analyzer.analyze_file(Path("track.mp3"))

# Access features via dictionary instead of attributes
harmonic_energy = structure.features['harmonic_energy']
```

## Future Extensions

Easy to add:

- **Onset detection** for precise drop timing
- **Tempogram** for build/tempo change detection  
- **Chroma features** for harmonic/key analysis
- **Self-similarity** for repeated section detection
- **Deep learning** embeddings
- **Custom domain-specific** features

No core code changes needed - just create a new `FeatureExtractor` subclass!
