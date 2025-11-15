# Removed

This document has been removed. See `docs/ARCHITECTURE.md` for the
consolidated architecture summary and any notes about Essentia usage.
   - Emphasizes beat-related frequency bands
   - Focuses on typical percussion ranges

5. **Histogram Information Gain** (2048/512)
   - Measures changes in spectral distribution
   - Good for harmonic changes

### Algorithm Pipeline

```
Audio Input (44.1kHz)
    ↓
[Compute 5 Onset Detection Functions in parallel]
    ↓
[TempoTapDegara] ← Extract beat candidates from each function
    ↓
[TempoTapMaxAgreement] ← Select best candidates via voting
    ↓
Final Beats + BPM + Confidence Score
```

### Confidence Score Interpretation

The algorithm outputs a confidence value (0-5.32) indicating reliability:

| Range | Quality | Expected Accuracy |
|-------|---------|-------------------|
| **0.0 - 0.99** | ⚠️ Very Low | Hard signal for trackers |
| **1.0 - 1.5** | ⚠️ Low | Uncertain detection |
| **1.5 - 3.5** | ✅ Good | ~80% accuracy (AMLt measure) |
| **3.5 - 5.32** | ✅✅ Excellent | Very reliable |

### When to Use

✅ **Use multifeature when:**
- Accuracy is more important than speed
- Working with complex rhythms (polyrhythms, syncopation)
- Need confidence estimation for quality control
- Processing EDM with layered percussion
- Analyzing tracks with tempo changes

❌ **Don't use multifeature when:**
- Processing thousands of files (too slow)
- Working with simple 4/4 beats only
- Speed is critical

### Performance

- **Processing time**: ~15-25 seconds per 5-minute track
- **Memory**: Moderate (5 parallel onset detections)
- **Accuracy**: **Best available** in Essentia

---

## Method 2: `degara`

### How It Works

Uses **single onset detection function** with optimized beat tracking:

1. **Complex Spectral Difference Only** (2048/1024 frame/hop, x2 upsampled)
   - Fast computation
   - Good for percussive music

### Algorithm Pipeline

```
Audio Input (44.1kHz)
    ↓
[Complex Spectral Difference Detection]
    ↓
[TempoTapDegara] ← Reliability-informed beat tracking
    ↓
Final Beats + BPM (no confidence score)
```

### Key Difference: Reliability-Informed Tracking

The **TempoTapDegara** algorithm uses a reliability measure internally to weight beat candidates, even though it doesn't output a confidence score. This makes it more robust than naive single-function tracking.

### When to Use

✅ **Use degara when:**
- Processing large batches (1000+ files)
- Speed is more important than accuracy
- Working with simple 4/4 electronic music
- Confidence scores not needed
- Initial rapid analysis before detailed processing

❌ **Don't use degara when:**
- Need confidence estimation
- Working with complex rhythms
- Analyzing jazz, breakbeats, or irregular timing
- Maximum accuracy required

### Performance

- **Processing time**: ~8-15 seconds per 5-minute track (**2-3x faster** than multifeature)
- **Memory**: Low (single onset detection)
- **Accuracy**: Good, but **lower than multifeature**

---

## Usage Examples

### Basic Usage - multifeature (default)

```python
import essentia.standard as es

# Load audio
audio = es.MonoLoader(filename='track.mp3', sampleRate=44100)()

# Extract BPM with multifeature method
rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
bpm, beats, confidence, estimates, intervals = rhythm_extractor(audio)

print(f"BPM: {bpm:.1f}")
print(f"Confidence: {confidence:.2f}")

# Assess quality
if confidence > 3.5:
    print("✅ Excellent detection")
elif confidence > 1.5:
    print("✅ Good detection (~80% accuracy)")
elif confidence > 1.0:
    print("⚠️ Low confidence")
else:
    print("❌ Very low confidence - manual review needed")
```

### Fast Batch Processing - degara

```python
import essentia.standard as es

# Use degara for speed
rhythm_extractor = es.RhythmExtractor2013(method="degara")
bpm, beats, confidence, estimates, intervals = rhythm_extractor(audio)

print(f"BPM: {bpm:.1f}")
# Note: confidence will always be 0 with degara
```

### Custom Tempo Range

```python
# Limit BPM detection range (useful for specific genres)
rhythm_extractor = es.RhythmExtractor2013(
    method="multifeature",
    minTempo=120,  # Techno/House minimum
    maxTempo=140   # Techno/House maximum
)
bpm, beats, confidence, estimates, intervals = rhythm_extractor(audio)
```

---

## Performance Benchmarks

Based on testing with EDM tracks (5-minute average length):

| Method | Avg Time | Min Time | Max Time | Accuracy* |
|--------|----------|----------|----------|-----------|
| **multifeature** | 19.5s | 15s | 25s | 87.5% (< 2 BPM error) |
| **degara** | 8.2s | 6s | 12s | ~75-80% (estimated) |

*Based on 8 verified tracks from our test dataset

---

## Which Method Should You Use?

### Decision Tree

```
Do you need confidence scores?
├─ YES → Use multifeature
└─ NO
   ├─ Processing < 100 files?
   │  └─ YES → Use multifeature (better accuracy)
   └─ Processing 100+ files?
      └─ YES → Use degara (faster)

Is your music complex (jazz, polyrhythms, tempo changes)?
├─ YES → Use multifeature
└─ NO (simple 4/4 EDM/House)
   └─ Either works, degara is faster
```

### Recommended Strategy: Hybrid Approach

```python
def smart_bpm_detection(audio):
    """Use degara for speed, fall back to multifeature if uncertain"""
    
    # First pass: fast detection with degara
    extractor_fast = es.RhythmExtractor2013(method="degara")
    bpm_fast, beats_fast, _, _, _ = extractor_fast(audio)
    
    # Check if BPM is in expected range
    if 115 < bpm_fast < 145:  # Normal EDM range
        return bpm_fast, "degara"
    
    # Second pass: use multifeature for difficult cases
    extractor_accurate = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, confidence, _, _ = extractor_accurate(audio)
    
    if confidence < 1.5:
        print(f"⚠️ Low confidence ({confidence:.2f}), manual review recommended")
    
    return bpm, "multifeature"
```

---

## Common Issues & Solutions

### Issue 1: Half-time/Double-time Detection

Both methods can lock onto half-time or double-time grooves.

**Solution:**
```python
def validate_bpm_octave(bpm, expected_range=(120, 140)):
    """Check for octave errors"""
    candidates = [bpm/2, bpm, bpm*2]
    
    for candidate in candidates:
        if expected_range[0] <= candidate <= expected_range[1]:
            return candidate
    
    return bpm  # Keep original if none fit
```

### Issue 2: Tempo Changes

Neither method handles tempo changes well (will return average or dominant tempo).

**Solution:**
- Split track into sections
- Analyze each section separately
- Or use specialized tempo curve extraction

### Issue 3: Low Confidence with multifeature

If confidence < 1.5, the beat tracking may be unreliable.

**Solution:**
```python
if confidence < 1.5:
    # Try with different tempo range
    extractor = es.RhythmExtractor2013(
        method="multifeature",
        minTempo=max(40, bpm - 20),
        maxTempo=min(250, bpm + 20)
    )
    # Re-analyze with focused range
```

---

## Scientific References

1. **multifeature method**:
   - Zapata, J., Davies, M., & Gómez, E. (2014). "Multi-feature beat tracker," IEEE/ACM Transactions on Audio, Speech and Language Processing, 22(4), 816-825.

2. **degara method**:
   - Degara, N., et al. (2012). "Reliability-informed beat tracking of musical signals," IEEE Transactions on Audio, Speech, and Language Processing, 20(1), 290-301.

3. **Confidence assessment**:
   - Zapata, J.R., et al. (2012). "Assigning a confidence threshold on automatic beat annotation in large datasets," ISMIR, pp. 157-162.

---

## Summary

| Method | Best Use Case | Speed | Accuracy | Confidence Score |
|--------|--------------|-------|----------|------------------|
| **multifeature** | High-quality analysis, complex rhythms, EDM | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Yes (0-5.32) |
| **degara** | Batch processing, simple beats, speed-critical | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ No |

**Recommendation for EDM Cue Analyzer:**
- **Default**: `multifeature` for best accuracy and confidence scores
- **Batch mode**: `degara` with octave validation for speed
- **Hybrid**: Use degara first, fall back to multifeature for outliers

---

**Last Updated**: November 11, 2025
