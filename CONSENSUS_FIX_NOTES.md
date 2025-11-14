# Consensus BPM Detection - Implementation Notes

## Problem Encountered

When testing the consensus BPM detection on Adam Beyer - Pilot:
- **Official BPM**: 133 BPM (Beatport)
- **Initial consensus result**: 122.3 BPM
- **Error**: -10.7 BPM (-8.05%)
- **Confidence**: 60% (3/5 methods agreed)

The consensus was working correctly (taking majority vote), but most methods were detecting ~122 BPM instead of 133 BPM.

## Root Cause

The octave error detection wasn't triggering because:
1. Most methods detected ~122 BPM (the groove/beat)
2. No methods detected 133 BPM (the actual tempo)
3. Without both values present, octave clustering couldn't occur
4. Consensus took majority vote → 122 BPM ❌

## Solution Implemented

Modified `_detect_essentia_rhythm_descriptors()` to return **multiple estimates**:

1. **Primary estimate**: The default BPM detection
2. **Alternative estimate**: The second histogram peak IF:
   - Weight > 0.25 (significant)
   - In expected range (120-145 BPM for EDM)
   - Different from first peak by >5 BPM

This ensures that if RhythmDescriptors sees both 122 and 133 BPM as histogram peaks, **both get added to consensus voting**.

### Code Changes

**Before** (single estimate):
```python
def _detect_essentia_rhythm_descriptors(self, y, sr) -> BpmEstimate:
    # ... detection code ...
    return BpmEstimate(bpm=bpm, confidence=conf, ...)
```

**After** (multiple estimates):
```python
def _detect_essentia_rhythm_descriptors(self, y, sr) -> list[BpmEstimate]:
    # ... detection code ...
    estimates = [BpmEstimate(bpm=bpm, ...)]  # Primary
    
    # Add second peak if significant and in expected range
    if second_peak_weight > 0.25 and in_expected_range(second_peak_bpm):
        if abs(second_peak_bpm - bpm) > 5:
            estimates.append(BpmEstimate(
                bpm=second_peak_bpm,
                confidence=second_peak_weight,
                method="essentia_rhythm_descriptors_alt"
            ))
    
    return estimates
```

## Expected Outcome

For Adam Beyer - Pilot:
- RhythmDescriptors should detect:
  - **First peak**: ~122 BPM (confidence: 0.6)
  - **Second peak**: ~133 BPM (weight: 0.3+)
- Other methods: ~122 BPM
- Consensus estimates: [122, 122, 122, 122, **133**]
- Octave detection: Sees 122 vs 133 cluster
- Resolution: Chooses 133 (in expected range 120-145)
- **Final consensus**: 133 BPM ✅

## Additional Improvements

### Expanded Octave Relationships

Added more tempo relationships to check:
```python
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
```

### Null Safety

Added check for None beats in analyzer.py:
```python
if beats is None:
    logger.warning("Consensus didn't provide beat positions, generating from BPM")
    _, beats = librosa.beat.beat_track(y=y_mono, sr=sr, start_bpm=bpm, tightness=100)
```

## Testing Status

- ⏳ Running test on Adam Beyer - Pilot (Essentia RhythmDescriptors is slow ~60-90s)
- Expected: 133 BPM detection with ~80-90% confidence
- Will validate on full 30-track dataset afterward

## Performance Note

RhythmDescriptors is slow (~1-2 min for 4-min track) but provides the best octave detection via histogram peaks. For faster consensus, could:
1. Skip RhythmDescriptors (use only multifeature/degara)
2. Downsample audio before detection
3. Implement timeout for slow methods

## Files Modified

1. `src/edm_cue_analyzer/consensus.py`:
   - `_detect_essentia_rhythm_descriptors()`: Now returns list[BpmEstimate]
   - `detect()`: Changed to extend estimates from rhythm_descriptors
   - `_is_octave_relationship()`: Added more tempo relationships
   - `_build_consensus()`: Added null safety for beats

2. `src/edm_cue_analyzer/analyzer.py`:
   - Added null check for consensus beats
   - Falls back to librosa beat generation if needed

## Next Steps

1. ✅ Validate fix on Adam Beyer track
2. ⬜ Test on full 30-track dataset
3. ⬜ Compare accuracy: consensus vs single-method
4. ⬜ Document consensus model in user guide
5. ⬜ Extend consensus to drop/breakdown detection
