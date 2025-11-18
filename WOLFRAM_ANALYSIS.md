# Wolfram Notebook Analysis - EDM Drop Detection

## Executive Summary

I've analyzed the Wolfram Community notebook "[WSC22] Analyze 'the drop' in EDM songs" by Suhaan Mobhani. The notebook presents a Short-Time Energy (STE) based approach to drop detection that complements our current onset-based method.

## Key Findings

### 1. Short-Time Energy as Primary Feature

**What They Do:**
```mathematica
energy = Map[Total[Abs[#]^2], SpectrogramArray[audio, window_size]]
```

**Key Insight**: "The quantity that we'll be using as a function of time will be the short time energy of the track, which will combine all the acoustics and give us an accurate idea of the 'power' that the music emits throughout"

**Our Equivalent**:
We already calculate `low_energy` and `percussive_energy` from HPSS, which are similar but more targeted signals.

### 2. Drop Characteristics

From the notebook's description:
- **"Loudest"**: "drops are typically the loudest and most unique song of the music"
- **Rapid fall**: "where the track has built up to its highest energy, which then rapidly falls"
- **Build-up pattern**: "most of them following a buildup"
- **Darker on spectrogram**: "could be seen as the darker outlines on a spectrogram"

### 3. Implementation Details

**Window Sizes**:
- Full track analysis: `SpectrogramArray[audio, 131072]`
- Detailed drop analysis: `SpectrogramArray[audio, 65536]`

**Novel Feature - Drop Region Analysis**:
- Extracts 5-second windows before/after each drop
- Pads to uniform 10-second duration
- Compares energy profiles across all drops for similarity

## Comparison with Our Approach

| Aspect | Our Onset-Based | Wolfram STE-Based |
|--------|----------------|-------------------|
| Primary Signal | onset_strength | Short-Time Energy |
| Detection Method | Peak finding | Energy buildup → peak → rapid fall |
| Thresholds | mean + 1.2*std | Not explicitly stated |
| Validation | Energy contrast + sustained breakdown | Drop similarity analysis |
| Post-Processing | Spacing constraints (16 bars) | Pattern matching across drops |

## Recommended Improvements

### Change 7: Energy Derivative Analysis (Highest Priority)

**Implementation**:
```python
# In drops.py, after calculating energy arrays
low_derivative = np.gradient(low_energy)
perc_derivative = np.gradient(percussive_energy)

# After identifying onset peak, verify energy drop
lookforward_frames = int((4 * bar_duration) / energy_window_seconds)
post_drop_low = low_derivative[idx:idx+lookforward_frames]
post_drop_perc = perc_derivative[idx:idx+lookforward_frames]

# Drops should have predominantly negative derivatives after peak
rapid_fall = (np.sum(post_drop_low < 0) / len(post_drop_low) > 0.6 and
              np.sum(post_drop_perc < 0) / len(post_drop_perc) > 0.6)
```

**Why**: The Wolfram notebook emphasizes that drops "rapidly fall" after the peak. This derivative analysis directly captures that characteristic.

**Expected Impact**: Reduce false positives where onset peaks don't lead to energy drops.

### Change 8: Build-Up Detection (Medium Priority)

**Implementation**:
```python
# Before onset peak, look for increasing energy trend
pre_drop_frames = int((8 * bar_duration) / energy_window_seconds)
pre_drop_start = max(min_track_time_frames, idx - pre_drop_frames)
buildup_low = low_energy[pre_drop_start:idx]
buildup_perc = percussive_energy[pre_drop_start:idx]

# Calculate slope of energy increase
buildup_slope_low = np.polyfit(range(len(buildup_low)), buildup_low, 1)[0]
buildup_slope_perc = np.polyfit(range(len(buildup_perc)), buildup_perc, 1)[0]

# Positive slope indicates build-up
has_buildup = buildup_slope_low > 0 and buildup_slope_perc > 0
```

**Why**: "most of them following a buildup" - this distinguishes real drops from random energy spikes.

**Expected Impact**: Increase precision by filtering peaks without proper build-up.

### Change 9: Drop Pattern Matching (Lower Priority)

**Implementation**:
```python
# After initial detection, extract drop profiles
def extract_drop_profile(energy, drop_time, sr, window_seconds=5):
    frames = int((window_seconds * sr) / hop_size)
    start = max(0, drop_time - frames)
    end = min(len(energy), drop_time + frames)
    return energy[start:end]

# Compare all detected drops
drop_profiles = [extract_drop_profile(low_energy, d, sr) for d in detected_drops]

# Calculate pairwise correlations
from scipy.stats import pearsonr
correlations = []
for i, profile_i in enumerate(drop_profiles):
    for j, profile_j in enumerate(drop_profiles[i+1:]):
        # Normalize profiles to same length
        min_len = min(len(profile_i), len(profile_j))
        corr, _ = pearsonr(profile_i[:min_len], profile_j[:min_len])
        correlations.append(corr)

# Filter drops with low average correlation to others
median_correlation = np.median(correlations)
# Keep drops that are similar to the typical pattern
```

**Why**: The Wolfram notebook analyzes "similarity between drops" to validate detections.

**Expected Impact**: Secondary validation to ensure consistency.

## Implementation Strategy

1. **Start with Change 7** (derivatives) - direct from paper, easy to implement
2. **Add Change 8** (build-up) if Change 7 shows promise
3. **Consider Change 9** (pattern matching) as advanced feature

## Test with Current Metrics

**Baseline (6 improvements applied)**:
- F1: 42%
- Precision: 45%
- Recall: 38%
- False Positives: 6 (down from 17)

**Target After Wolfram Techniques**:
- F1: >55%
- Precision: >60%
- Recall: >50%

## Key Insight from Wolfram

The emphasis on **"rapidly falls"** after the peak is the most actionable insight. Our current approach focuses on onset peaks (sudden increases) but doesn't verify the subsequent energy drop pattern. Adding derivative analysis addresses this gap directly.

That's another antipattern I could do without too: 'minor physical movement is immediately interpreted as total rejection', as just happened when I shifted around to be more comfy (lower back, tried to explain).  Generates a lot of friction over momentary disconnect, FYI.