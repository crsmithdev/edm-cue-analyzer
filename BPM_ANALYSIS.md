# BPM Detection Analysis

Comparison of detected BPM vs actual BPM (from Rekordbox) for test tracks.

## Results

| Track | Artist | Detected BPM | Actual BPM | Difference | % Error |
|-------|--------|--------------|------------|------------|---------|
| Pilot | Adam Beyer | 136.0 | 133.00 | +3.0 | +2.26% |
| Zodiac | Agents Of Time | 123.0 | 126.00 | -3.0 | -2.38% |
| Sound in Motion (Extended Mix) | Attican | 123.0 | 124.00 | -1.0 | -0.81% |
| Mr. Frast (Original Mix) | Ellez Ria | 143.5 | 142.00 | +1.5 | +1.06% |
| Future Memories | Estiva | 123.0 | 123.00 | 0.0 | 0.00% ✓ |
| Mondfinsternis | Innellea, Kevin de Vries | 123.0 | 125.00 | -2.0 | -1.60% |
| Higher Resolution | Kasablanca | 123.0 | 124.00 | -1.0 | -0.81% |
| Always You (Deeparture Extended Remix) | Matt Fax | 123.0 | 122.00 | +1.0 | +0.82% |
| Origins (Extended Mix) | Matt Fax | 123.0 | 124.00 | -1.0 | -0.81% |
| Hide | Paraleven, Mont Blvck | 123.0 | 121.00 | +2.0 | +1.65% |
| 9th Ave (Extended Mix) | Rodg | 129.0 | 128.00 | +1.0 | +0.78% |
| Insane | Saekone | 136.0 | 135.00 | +1.0 | +0.74% |
| Heart | Sainte Vie | 123.0 | 122.00 | +1.0 | +0.82% |
| We Call This Acid | Silver Panda | 123.0 | 124.00 | -1.0 | -0.81% |
| Summer Rain | Sound Quelle, Diana Miro | 117.5 | 116.00 | +1.5 | +1.29% |

## Observations

### Accuracy Summary
- **Perfect match (0% error)**: 1/15 tracks (6.7%)
  - Estiva - Future Memories
- **±1 BPM error**: 10/15 tracks (66.7%)
  - Within acceptable range for beat tracking
- **±2 BPM error**: 2/15 tracks (13.3%)
- **±3 BPM error**: 2/15 tracks (13.3%)
  - Adam Beyer - Pilot: +3 BPM (2.26% error)
  - Agents Of Time - Zodiac: -3 BPM (-2.38% error)

### Key Findings

1. **Overall Performance**:
   - Average absolute error: **1.4 BPM**
   - Average percentage error: **1.15%**
   - Most errors are within ±1-2 BPM, which is acceptable for librosa's beat tracking
   
2. **Systematic Issues**:
   - Many 122-124 BPM tracks are detected as 123.0 BPM
   - Suggests librosa tends to converge on similar BPMs for progressive house/melodic techno
   
3. **Notable Cases**:
   - **Adam Beyer - Pilot**: Largest error (+3 BPM, detected 136 vs actual 133)
   - **Agents Of Time - Zodiac**: Second largest error (-3 BPM, detected 123 vs actual 126)
   - **Rodg - 9th Ave**: Critical for drop detection (+1 BPM affects bar timing)

4. **Genre Patterns**:
   - Progressive House/Melodic Techno (120-126 BPM range): Often detected as 123 BPM
   - Techno (130-140 BPM range): Better accuracy (±1 BPM)
   - Slower tracks (116-118 BPM): Reasonable accuracy (±1.5 BPM)

### Impact on Drop Detection

For **Rodg - 9th Ave**:
- Actual BPM: 128 BPM → Bar duration: 1.875s
- Detected BPM: 129 BPM → Bar duration: 1.860s
- Error per bar: 0.015s (15ms)
- **At bar 33 (first drop at 1:00)**: 0.5s cumulative error
- **At bar 106 (second drop at 3:17)**: 1.6s cumulative error

This explains why drops are detected late!

## Recommendations

1. **Implement BPM override flag**: `--bpm <value>` for manual correction
2. **Add BPM snapping logic**: Round to common EDM BPMs (120, 122, 124, 126, 128, 130, 133, 135, 138, 140)
3. **Improve BPM refinement**: Use multiple detection methods and average results
4. **Consider tempo tolerance**: Allow ±0.5 BPM adjustment based on genre detection
