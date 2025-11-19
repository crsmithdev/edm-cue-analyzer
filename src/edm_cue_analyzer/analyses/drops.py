"""Advanced drop detection using two-stage segmentation and classification.

Based on state-of-the-art techniques from the EDM Structure Detection Guide:
- Stage 1: Self-similarity matrix for structural boundary detection
- Stage 2: Multi-feature classification of boundaries as drops
- Spectrogram-based pattern detection (buildup → silence/filter → bass return)
- Multi-band frequency analysis for bass drop detection
- MFCC analysis for timbral change detection
- Rhythm and tempo consistency checks

Implementation follows the guide's recommended approach combining:
- librosa for core audio processing
- scipy for signal processing and filtering
- sklearn for feature standardization (if needed in future)
"""

import logging

import librosa
import numpy as np
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


async def analyze_drops(context: dict) -> list[float]:
    """
    Detect drop points using advanced two-stage approach from the guide.

    This implements the state-of-the-art method recommended in the EDM Structure
    Detection Guide, combining structural segmentation with feature-based classification.

    Stage 1: Structural segmentation using self-similarity matrix
    - Compute chroma features for harmonic structure analysis
    - Build recurrence/self-similarity matrix to find where music changes
    - Extract novelty curve and detect peaks as structural boundaries

    Stage 2: Multi-feature classification of boundaries as drops
    - Extract comprehensive features around each boundary:
      * Energy buildup detection (pre-drop increasing energy)
      * Bass drop magnitude (low frequency surge post-drop)
      * Onset strength (strong percussive events)
      * Spectral contrast (dynamic range changes)
      * Filter sweep detection (spectral centroid movement)
      * MFCC analysis (timbral changes)
      * Rhythm features (beat density changes)
    - Compute confidence score based on feature combination
    - Apply threshold to classify as drop

    Stage 3: Spectrogram-based validation
    - Analyze frequency band energies over time
    - Validate characteristic drop pattern:
      * High/mid frequencies decrease (filter sweep or breakdown)
      * Low frequencies surge (bass drop)
    - Remove false positives

    Args:
        context: Dictionary containing:
            - y: Audio signal (mono, float32)
            - sr: Sample rate
            - bpm: BpmResult from BPM analysis
            - energy: EnergyResult from energy analysis
            - features: Dict with onset_strength, spectral features, etc.
            - config: Optional analysis config
            - metadata: Optional track metadata

    Returns:
        List of timestamps where drops occur, sorted chronologically
    """
    y = context["y"]
    sr = context["sr"]
    bpm_result = context["bpm"]
    energy_result = context["energy"]
    features = context.get("features", {})
    config = context.get("config")

    bar_duration = bpm_result.bar_duration

    # Get config values
    drop_min_spacing_bars = 16
    if config:
        drop_min_spacing_bars = getattr(config, "drop_min_spacing_bars", 16)

    min_spacing = bar_duration * drop_min_spacing_bars

    logger.debug(
        "Starting advanced drop detection (two-stage approach from guide)"
    )
    logger.debug(f"Track: {len(y)/sr:.1f}s at {sr}Hz, BPM: {bpm_result.bpm:.1f}")

    # Stage 1: Find structural boundaries using self-similarity
    logger.debug("Stage 1: Detecting structural boundaries with self-similarity matrix...")
    boundaries = _detect_structural_boundaries(y, sr, bpm_result)
    logger.debug(f"Found {len(boundaries)} structural boundaries")

    if len(boundaries) == 0:
        logger.warning("No structural boundaries found - track may be too short or uniform")
        return []

    # Stage 2: Classify boundaries as drops using comprehensive feature analysis
    logger.debug("Stage 2: Classifying boundaries with multi-feature analysis...")
    drop_candidates = []

    for boundary_time in boundaries:
        features_dict = _extract_drop_features(
            y, sr, boundary_time, energy_result, features, bpm_result
        )

        confidence = _compute_drop_confidence(features_dict)

        if confidence > 0.5:  # Threshold for drop classification
            drop_candidates.append({"time": boundary_time, "confidence": confidence})
            logger.debug(
                f"Drop candidate at {boundary_time:.2f}s (confidence: {confidence:.3f})"
            )

    logger.debug(f"Found {len(drop_candidates)} drop candidates after feature classification")

    if len(drop_candidates) == 0:
        logger.info("No drops detected after feature classification")
        return []

    # Stage 3: Validate with spectrogram-based pattern detection
    logger.debug("Stage 3: Validating with spectrogram pattern analysis...")
    validated_drops = _validate_drops_with_spectrogram(y, sr, drop_candidates)

    logger.debug(f"Validated {len(validated_drops)} drops with spectrogram analysis")

    # Sort by time
    validated_drops.sort()

    # Apply minimum spacing constraint
    final_drops = []
    for drop_time in validated_drops:
        if not final_drops or (drop_time - final_drops[-1]) > min_spacing:
            final_drops.append(float(drop_time))
        else:
            logger.debug(
                f"Removed drop at {drop_time:.2f}s (too close to previous: {drop_time - final_drops[-1]:.1f}s < {min_spacing:.1f}s)"
            )

    logger.info(
        f"Drop detection complete: {len(final_drops)} drops found (min spacing: {min_spacing:.1f}s)"
    )

    return final_drops


def _detect_structural_boundaries(
    y: np.ndarray, sr: int, bpm_result, min_duration_bars: int = 8
) -> list[float]:
    """
    Stage 1: Detect structural boundaries using self-similarity matrix.

    Implements the guide's recommendation for structure segmentation using:
    - Chroma features for harmonic content analysis
    - Recurrence matrix for self-similarity computation
    - Novelty curve extraction for boundary detection
    - Peak picking with musically-informed spacing

    This finds major transitions in the track structure where the harmonic
    content changes significantly, which often corresponds to drops, breakdowns,
    or other structural elements.

    Args:
        y: Audio signal
        sr: Sample rate
        bpm_result: BPM analysis result for musical timing
        min_duration_bars: Minimum duration between boundaries (in bars)

    Returns:
        List of boundary times in seconds
    """
    # Compute chroma features (12-dimensional pitch class profiles)
    # Chroma is ideal for finding structural boundaries as it captures
    # harmonic content while being invariant to timbre
    hop_length = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

    # Build self-similarity matrix using recurrence matrix
    # This creates a matrix where ssm[i,j] indicates how similar
    # frame i is to frame j in terms of harmonic content
    try:
        ssm = librosa.segment.recurrence_matrix(
            chroma,
            k=5,  # Number of nearest neighbors
            mode="affinity",  # Use affinity (similarity) rather than distance
            metric="cosine",  # Cosine similarity for chroma
            sparse=False,  # Dense matrix for full analysis
        )
    except Exception as e:
        logger.warning(f"Failed to compute recurrence matrix: {e}")
        # Fallback to simpler approach
        ssm = np.dot(chroma.T, chroma)

    # Compute novelty curve from self-similarity matrix
    # High novelty indicates a structural boundary
    # We look at the change in similarity patterns over time
    if ssm.ndim == 2 and ssm.shape[0] > 1:
        # Compute the derivative along the diagonal
        novelty = np.sum(np.abs(np.diff(ssm, axis=1)), axis=0)
    else:
        logger.warning("Invalid SSM shape, using fallback boundary detection")
        return []

    # Smooth the novelty curve to reduce noise
    if len(novelty) > 51:
        window_length = 51
        if window_length % 2 == 0:
            window_length += 1
        novelty = savgol_filter(novelty, window_length, 3)

    # Calculate minimum spacing in frames based on musical bars
    bar_duration = bpm_result.bar_duration
    min_duration_sec = min_duration_bars * bar_duration
    min_duration_frames = int(min_duration_sec * sr / hop_length)

    # Find peaks in novelty curve using librosa's peak picker
    # These peaks correspond to structural boundaries
    try:
        boundaries = librosa.util.peak_pick(
            novelty,
            pre_max=5,  # Frames before peak that must be lower
            post_max=5,  # Frames after peak that must be lower
            pre_avg=5,  # Frames before to compare average
            post_avg=5,  # Frames after to compare average
            delta=0.1,  # Minimum difference from average
            wait=max(min_duration_frames, 30),  # Minimum frames between peaks
        )
    except Exception as e:
        logger.warning(f"Peak picking failed: {e}")
        return []

    # Convert frame indices to time
    boundary_times = librosa.frames_to_time(boundaries, sr=sr, hop_length=hop_length)

    logger.debug(f"Detected {len(boundary_times)} boundaries at: {boundary_times}")

    return boundary_times.tolist()


def _extract_drop_features(
    y: np.ndarray,
    sr: int,
    boundary_time: float,
    energy_result,
    features: dict,
    bpm_result,
) -> dict:
    """
    Stage 2: Extract comprehensive features around a boundary for drop classification.

    Implements the guide's comprehensive feature extraction approach including:
    1. Energy buildup detection (pre-drop increasing energy)
    2. Bass drop magnitude (low frequency surge)
    3. Onset strength (percussive event detection)
    4. Spectral contrast (dynamic range)
    5. Filter sweep detection (spectral centroid movement)
    6. MFCCs for timbral change detection
    7. Rhythm features (beat density)

    Args:
        y: Audio signal
        sr: Sample rate
        boundary_time: Time of the boundary to analyze
        energy_result: Energy analysis results
        features: Dictionary of pre-extracted features
        bpm_result: BPM analysis results

    Returns:
        Dictionary of feature values for classification
    """
    window_sec = 5.0  # Analyze 5 seconds before and after (as per guide)
    feature_dict = {}

    # Get audio segment around boundary
    center_sample = int(boundary_time * sr)
    window_samples = int(window_sec * sr)

    start = max(0, center_sample - window_samples)
    end = min(len(y), center_sample + window_samples)

    if end - start < sr:  # Need at least 1 second
        logger.debug(f"Boundary at {boundary_time:.2f}s too close to edge")
        return feature_dict

    # === Feature 1: Energy buildup detection ===
    pre_drop = y[max(0, center_sample - window_samples) : center_sample]
    if len(pre_drop) > sr:
        energy_rms = librosa.feature.rms(y=pre_drop, hop_length=512)[0]
        if len(energy_rms) > 2:
            # Fit linear trend to energy
            x = np.arange(len(energy_rms))
            slope = np.polyfit(x, energy_rms, 1)[0]
            # Positive slope indicates buildup
            feature_dict["energy_buildup"] = float(slope)

    # === Feature 2: Bass drop magnitude ===
    post_drop = y[center_sample : min(len(y), center_sample + sr)]  # 1 second after
    if len(post_drop) > 0:
        stft = librosa.stft(post_drop, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)

        # Define frequency bands (as per guide)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        bass_mask = (freqs >= 20) & (freqs <= 250)  # Bass: 20-250 Hz
        mid_mask = (freqs > 250) & (freqs <= 4000)  # Mid: 250-4000 Hz
        high_mask = freqs > 4000  # High: 4000+ Hz

        # Compute band energies
        bass_energy = np.mean(magnitude[bass_mask, :])
        mid_energy = np.mean(magnitude[mid_mask, :])
        high_energy = np.mean(magnitude[high_mask, :])

        feature_dict["bass_energy"] = float(bass_energy)
        feature_dict["mid_energy"] = float(mid_energy)
        feature_dict["high_energy"] = float(high_energy)

        # Bass-to-high ratio (drops have strong bass)
        if high_energy > 0:
            feature_dict["bass_to_high_ratio"] = float(bass_energy / (high_energy + 1e-6))

    # === Feature 3: Onset strength ===
    onset_strength = features.get("onset_strength")
    if onset_strength is not None and len(onset_strength) > 0:
        hop_length = 512
        onset_times = np.arange(len(onset_strength)) * hop_length / sr

        # Find onset strength near boundary
        time_idx = np.searchsorted(onset_times, boundary_time)
        if 0 <= time_idx < len(onset_strength):
            # Get max onset in ±0.5s window
            window_frames = int(0.5 * sr / hop_length)
            start_idx = max(0, time_idx - window_frames)
            end_idx = min(len(onset_strength), time_idx + window_frames)

            max_onset = np.max(onset_strength[start_idx:end_idx])
            mean_onset = np.mean(onset_strength)
            std_onset = np.std(onset_strength)

            feature_dict["onset_strength_max"] = float(max_onset)
            feature_dict["onset_strength_normalized"] = float(
                (max_onset - mean_onset) / (std_onset + 1e-6)
            )

    # === Feature 4: Spectral contrast ===
    spectral_contrast = features.get("spectral_contrast")
    if spectral_contrast is not None:
        hop_length = 512
        if spectral_contrast.ndim == 1:
            # Essentia returns 1D array
            contrast_times = np.arange(len(spectral_contrast)) * hop_length / sr
            time_idx = np.searchsorted(contrast_times, boundary_time)
            if 0 <= time_idx < len(spectral_contrast):
                window_frames = int(1.0 * sr / hop_length)
                start_idx = max(0, time_idx - window_frames)
                end_idx = min(len(spectral_contrast), time_idx + window_frames)
                feature_dict["spectral_contrast"] = float(
                    np.mean(spectral_contrast[start_idx:end_idx])
                )
        elif spectral_contrast.ndim == 2:
            # Librosa returns 2D array (bands x time)
            contrast_times = np.arange(spectral_contrast.shape[1]) * hop_length / sr
            time_idx = np.searchsorted(contrast_times, boundary_time)
            if 0 <= time_idx < spectral_contrast.shape[1]:
                window_frames = int(1.0 * sr / hop_length)
                start_idx = max(0, time_idx - window_frames)
                end_idx = min(spectral_contrast.shape[1], time_idx + window_frames)
                # Average across bands and time
                feature_dict["spectral_contrast"] = float(
                    np.mean(spectral_contrast[:, start_idx:end_idx])
                )

    # === Feature 5: Filter sweep detection (spectral centroid movement) ===
    spectral_centroid = features.get("spectral_centroid")
    if spectral_centroid is not None and len(spectral_centroid) > 0:
        hop_length = 512
        centroid_times = np.arange(len(spectral_centroid)) * hop_length / sr

        # Look for rising centroid before drop (high-pass filter sweep)
        pre_drop_frames = int(2.0 * sr / hop_length)  # 2 seconds before
        time_idx = np.searchsorted(centroid_times, boundary_time)
        start_idx = max(0, time_idx - pre_drop_frames)

        if start_idx < time_idx and time_idx > 0:
            pre_centroid = spectral_centroid[start_idx:time_idx]
            if len(pre_centroid) > 2:
                # Check for upward trend (filter sweep)
                x = np.arange(len(pre_centroid))
                slope = np.polyfit(x, pre_centroid, 1)[0]
                feature_dict["filter_sweep"] = float(slope)

    # === Feature 6: MFCC analysis for timbral change ===
    segment = y[start:end]
    if len(segment) > sr:
        # Compute MFCCs for timbral analysis (as per guide)
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)

        # Compute mean and std of MFCCs
        feature_dict["mfcc_mean"] = float(np.mean(mfccs))
        feature_dict["mfcc_std"] = float(np.std(mfccs))

        # Compute delta (change over time)
        mfcc_delta = librosa.feature.delta(mfccs)
        feature_dict["mfcc_delta_mean"] = float(np.mean(np.abs(mfcc_delta)))

    # === Feature 7: Rhythm features (beat density) ===
    if len(segment) > sr:
        try:
            tempo, beats = librosa.beat.beat_track(y=segment, sr=sr)
            beat_density = len(beats) / (len(segment) / sr)
            feature_dict["beat_density"] = float(beat_density)
        except Exception as e:
            logger.debug(f"Beat tracking failed for segment: {e}")

    return feature_dict


def _compute_drop_confidence(features_dict: dict) -> float:
    """
    Compute confidence score that a boundary is a drop based on extracted features.

    Combines multiple features with appropriate weighting based on importance
    for drop detection. Implements the guide's multi-feature approach.

    Args:
        features_dict: Dictionary of extracted features

    Returns:
        Confidence score from 0.0 to 1.0
    """
    if not features_dict:
        return 0.0

    confidence_factors = []

    # Factor 1: Energy buildup (positive slope before drop)
    if "energy_buildup" in features_dict:
        slope = features_dict["energy_buildup"]
        buildup_score = min(1.0, max(0.0, slope * 1000))  # Scale to 0-1
        confidence_factors.append(buildup_score)

    # Factor 2: Bass surge (strong bass energy after drop)
    if "bass_to_high_ratio" in features_dict:
        ratio = features_dict["bass_to_high_ratio"]
        bass_score = min(1.0, ratio / 5.0)  # Normalize (typical drops have ratio > 2)
        confidence_factors.append(bass_score)

    # Factor 3: Strong onset at drop point
    if "onset_strength_normalized" in features_dict:
        normalized_onset = features_dict["onset_strength_normalized"]
        onset_score = min(1.0, max(0.0, normalized_onset / 3.0))
        confidence_factors.append(onset_score)

    # Factor 4: Spectral contrast (drops have high dynamic range)
    if "spectral_contrast" in features_dict:
        contrast = features_dict["spectral_contrast"]
        # Normalize to reasonable range
        contrast_score = min(1.0, max(0.0, contrast / 30.0))
        confidence_factors.append(contrast_score)

    # Factor 5: Filter sweep (rising centroid before drop)
    if "filter_sweep" in features_dict:
        sweep = features_dict["filter_sweep"]
        sweep_score = min(1.0, max(0.0, sweep / 100))
        confidence_factors.append(sweep_score)

    # Factor 6: Timbral change (MFCC delta)
    if "mfcc_delta_mean" in features_dict:
        delta = features_dict["mfcc_delta_mean"]
        timbral_score = min(1.0, max(0.0, delta / 5.0))
        confidence_factors.append(timbral_score)

    # Combine factors with weighted average
    # Give more weight to bass surge and onset strength (most reliable indicators)
    if len(confidence_factors) > 0:
        # Weight: bass and onset are more important
        weights = []
        if "bass_to_high_ratio" in features_dict:
            weights.append(2.0)  # Double weight for bass
        if "onset_strength_normalized" in features_dict:
            weights.append(2.0)  # Double weight for onset
        # All other factors get weight 1.0
        while len(weights) < len(confidence_factors):
            weights.append(1.0)

        weighted_sum = sum(f * w for f, w in zip(confidence_factors, weights, strict=True))
        total_weight = sum(weights)
        confidence = weighted_sum / total_weight
    else:
        confidence = 0.0

    return confidence


def _validate_drops_with_spectrogram(
    y: np.ndarray, sr: int, drop_candidates: list[dict]
) -> list[float]:
    """
    Stage 3: Validate drop candidates using spectrogram pattern analysis.

    Implements the guide's spectrogram-based validation approach:
    - Analyze frequency band energies over time
    - Look for characteristic drop pattern:
      * High/mid frequencies decrease (filter sweep or breakdown)
      * Low frequencies surge (bass drop)
    - Remove false positives that don't match the pattern

    Args:
        y: Audio signal
        sr: Sample rate
        drop_candidates: List of candidate drops with time and confidence

    Returns:
        List of validated drop times
    """
    # Compute spectrogram with appropriate window size
    window_size = 2048
    hop_length = 512

    stft_result = librosa.stft(y, n_fft=window_size, hop_length=hop_length)
    magnitude = np.abs(stft_result)

    # Define frequency bands as per guide
    freqs = librosa.fft_frequencies(sr=sr, n_fft=window_size)

    # Bass: 20-150 Hz, Mid: 150-2000 Hz, High: 2000+ Hz
    low_mask = (freqs >= 20) & (freqs <= 150)
    # mid_mask = (freqs > 150) & (freqs <= 2000)  # Not used in current validation
    # high_mask = freqs > 2000  # Not used in current validation

    # Compute band energies over time
    low_energy = np.mean(magnitude[low_mask, :], axis=0) if np.any(low_mask) else np.zeros(magnitude.shape[1])
    # mid_energy = np.mean(magnitude[mid_mask, :], axis=0) if np.any(mid_mask) else np.zeros(magnitude.shape[1])  # Not used in current validation
    # high_energy = np.mean(magnitude[high_mask, :], axis=0) if np.any(high_mask) else np.zeros(magnitude.shape[1])  # Not used in current validation

    # Smooth energy curves using Savitzky-Golay filter (as per guide)
    low_smooth = savgol_filter(low_energy, 51, 3) if len(low_energy) > 51 else low_energy

    # Validate each candidate
    validated_drops = []

    for candidate in drop_candidates:
        drop_time = candidate["time"]
        # confidence = candidate["confidence"]

        # Convert time to frame
        drop_frame = int(drop_time * sr / hop_length)

        # Skip if too close to edges
        if drop_frame < 50 or drop_frame >= len(low_smooth) - 50:
            logger.debug(f"Drop at {drop_time:.2f}s too close to edge, skipping validation")
            continue

        # Analyze pattern around drop point
        # Compare 1 second before vs 1 second after
        # Note: The drop might occur slightly before or after the structural boundary,
        # so we check multiple windows around the boundary point
        window_frames = int(1.0 * sr / hop_length)

        pre_start = max(0, drop_frame - window_frames)

        # Check multiple drop positions: 1s before boundary, at boundary, +0.5s, +1.0s after
        # This handles cases where the structural change is detected slightly before/after
        # the actual bass drop
        best_bass_increase = False
        best_pre_bass = 0
        best_post_bass = 0
        best_offset = 0

        for offset_frames in [-window_frames, -int(0.5 * sr / hop_length), 0,
                               int(0.5 * sr / hop_length), window_frames]:
            test_drop_frame = drop_frame + offset_frames

            if test_drop_frame < window_frames or test_drop_frame >= len(low_smooth) - window_frames:
                continue

            test_pre_start = max(0, test_drop_frame - window_frames)
            test_post_end = min(len(low_smooth), test_drop_frame + window_frames)

            pre_bass = np.mean(low_smooth[test_pre_start:test_drop_frame])
            post_bass = np.mean(low_smooth[test_drop_frame:test_post_end])

            # Check if this window shows a strong bass increase
            # Use the window with the strongest bass surge
            if post_bass > pre_bass * 1.2 and post_bass > best_post_bass:
                best_bass_increase = True
                best_pre_bass = pre_bass
                best_post_bass = post_bass
                best_offset = offset_frames * hop_length / sr

        pre_bass = best_pre_bass if best_bass_increase else np.mean(low_smooth[pre_start:drop_frame])
        post_end = min(len(low_smooth), drop_frame + window_frames)
        post_bass = best_post_bass if best_bass_increase else np.mean(low_smooth[drop_frame:post_end])

        # Drop pattern validation (as per guide):
        # 1. Bass increases significantly (>20% as per guide)
        bass_increase = best_bass_increase

        # 2. Optional: High frequencies may decrease (filter sweep) or stay similar
        # We don't require high decrease, as some drops maintain highs

        # Validation decision
        if bass_increase:
            # Adjust drop time if bass surge was detected with an offset
            adjusted_drop_time = drop_time + best_offset
            validated_drops.append(adjusted_drop_time)
            offset_msg = f" (adjusted +{best_offset:.1f}s)" if best_offset > 0 else ""
            logger.debug(
                f"✓ Validated drop at {drop_time:.2f}s{offset_msg}: "
                f"bass {pre_bass:.3f}→{post_bass:.3f} "
                f"(+{((post_bass/pre_bass - 1)*100):.0f}%)"
            )
        else:
            logger.debug(
                f"✗ Rejected drop at {drop_time:.2f}s: "
                f"bass {pre_bass:.3f}→{post_bass:.3f} "
                f"(+{((post_bass/pre_bass - 1)*100):.0f}%) - insufficient surge"
            )

    return validated_drops
