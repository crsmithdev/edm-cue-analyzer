# Drop Detection Parameter Tuning Guide

## Understanding Drop Characteristics by Genre

### Genre-Specific Drop Profiles
```python
# Base parameters that vary by genre
GENRE_PARAMETERS = {
    'dubstep': {
        'bpm_range': (138, 142),
        'drop_duration': 16,  # bars
        'buildup_duration': 8,  # bars
        'bass_frequency_cutoff': 100,  # Hz
        'energy_increase_ratio': 2.0,  # Post-drop vs pre-drop
        'silence_gap': True,  # Often has silence before drop
        'filter_sweep_common': True
    },
    'house': {
        'bpm_range': (120, 130),
        'drop_duration': 32,  # bars
        'buildup_duration': 16,
        'bass_frequency_cutoff': 150,
        'energy_increase_ratio': 1.5,
        'silence_gap': False,
        'filter_sweep_common': True
    },
    'drum_and_bass': {
        'bpm_range': (170, 180),
        'drop_duration': 32,
        'buildup_duration': 8,
        'bass_frequency_cutoff': 80,
        'energy_increase_ratio': 1.8,
        'silence_gap': True,
        'filter_sweep_common': False
    },
    'future_bass': {
        'bpm_range': (130, 160),
        'drop_duration': 16,
        'buildup_duration': 8,
        'bass_frequency_cutoff': 120,
        'energy_increase_ratio': 1.6,
        'silence_gap': False,
        'filter_sweep_common': True
    },
    'techno': {
        'bpm_range': (125, 135),
        'drop_duration': 32,
        'buildup_duration': 16,
        'bass_frequency_cutoff': 100,
        'energy_increase_ratio': 1.3,  # More subtle
        'silence_gap': False,
        'filter_sweep_common': True
    }
}
```

## Two-Stage Method Parameter Tuning

### Stage 1: Segmentation Parameters

```python
class OptimizedSegmentation:
    """
    Segmentation with genre-aware parameter tuning.
    """
    
    def __init__(self, genre='house'):
        self.genre_params = GENRE_PARAMETERS.get(genre, GENRE_PARAMETERS['house'])
        
    def segment_with_tuned_parameters(self, audio_path):
        """
        Genre-specific segmentation parameters.
        """
        y, sr = librosa.load(audio_path)
        tempo = self.estimate_tempo(y, sr)
        
        # Adjust hop length based on tempo
        # Faster tempos need smaller hop lengths for precision
        if tempo > 160:  # DnB
            hop_length = 256
        elif tempo > 130:  # Dubstep
            hop_length = 512
        else:  # House/Techno
            hop_length = 1024
            
        # Compute features with adjusted parameters
        chroma = librosa.feature.chroma_cqt(
            y=y, 
            sr=sr,
            hop_length=hop_length,
            n_chroma=12,
            bins_per_octave=36  # Higher for better harmonic resolution
        )
        
        # Self-similarity with genre-specific parameters
        # Longer segments for house/techno, shorter for dubstep/DnB
        if self.genre_params['drop_duration'] > 16:
            k = 7  # Larger neighborhood for longer sections
            width = 9
        else:
            k = 5  # Smaller for shorter sections
            width = 7
            
        ssm = librosa.segment.recurrence_matrix(
            chroma,
            k=k,
            width=width,
            metric='cosine',
            mode='affinity'
        )
        
        # Novelty curve computation
        novelty = self.compute_novelty(ssm, kernel_size=self._get_kernel_size(tempo))
        
        # Peak picking with genre-specific thresholds
        boundaries = self.pick_peaks_tuned(novelty, tempo)
        
        return librosa.frames_to_time(boundaries, sr=sr, hop_length=hop_length)
    
    def _get_kernel_size(self, tempo):
        """
        Kernel size for novelty detection based on tempo.
        """
        # Approximate bars to frames
        seconds_per_bar = (60.0 / tempo) * 4
        frames_per_bar = seconds_per_bar * 22050 / 512  # Assuming standard settings
        
        # Kernel should span about 1-2 bars
        kernel_size = int(frames_per_bar * 1.5)
        
        # Ensure odd number
        return kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    def compute_novelty(self, ssm, kernel_size=65):
        """
        Compute novelty curve with Gaussian-tapered checkerboard kernel.
        """
        # Create checkerboard kernel
        kernel = np.zeros((kernel_size, kernel_size))
        for i in range(kernel_size):
            for j in range(kernel_size):
                if (i < kernel_size//2) != (j < kernel_size//2):
                    kernel[i, j] = 1
                else:
                    kernel[i, j] = -1
        
        # Apply Gaussian taper
        from scipy.signal.windows import gaussian
        g = gaussian(kernel_size, std=kernel_size/4)
        kernel = kernel * np.outer(g, g)
        
        # Convolve with SSM
        from scipy.signal import convolve2d
        novelty = convolve2d(ssm, kernel, mode='same')
        
        # Sum along diagonal
        return np.sum(novelty, axis=0)
    
    def pick_peaks_tuned(self, novelty, tempo):
        """
        Peak picking with tempo-aware parameters.
        """
        # Minimum distance between boundaries (in frames)
        # Should be at least 4 bars
        seconds_per_bar = (60.0 / tempo) * 4
        min_distance = int(4 * seconds_per_bar * 22050 / 512)
        
        # Dynamic thresholding based on genre
        if self.genre_params['silence_gap']:
            # More aggressive for genres with clear breaks
            delta = 0.15
        else:
            # More conservative for continuous genres
            delta = 0.25
            
        peaks = librosa.util.peak_pick(
            novelty,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=delta,
            wait=min_distance
        )
        
        return peaks
    
    def estimate_tempo(self, y, sr):
        """Quick tempo estimation for parameter adjustment."""
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Snap to expected range
        expected_min, expected_max = self.genre_params['bpm_range']
        
        # Handle half/double tempo detection
        if tempo < expected_min / 1.5:
            tempo *= 2
        elif tempo > expected_max * 1.5:
            tempo /= 2
            
        return tempo
```

### Stage 2: Classification Parameters

```python
class TunedDropClassifier:
    """
    Drop classification with detailed parameter control.
    """
    
    def __init__(self, genre='house'):
        self.genre = genre
        self.params = GENRE_PARAMETERS[genre]
        
        # Feature extraction windows (in seconds)
        self.window_sizes = {
            'pre_drop': 8.0,   # Before boundary
            'at_drop': 2.0,    # Around boundary
            'post_drop': 4.0   # After boundary
        }
        
        # Thresholds
        self.thresholds = self.get_genre_thresholds()
    
    def get_genre_thresholds(self):
        """
        Genre-specific detection thresholds.
        """
        thresholds = {
            'dubstep': {
                'energy_ratio': 2.0,      # High contrast
                'bass_surge': 0.7,        # Strong bass return
                'high_freq_drop': 0.5,     # Significant high cut
                'silence_threshold': 0.01,  # Clear silence
                'buildup_slope': 0.002,    # Steep buildup
                'confidence_threshold': 0.6
            },
            'house': {
                'energy_ratio': 1.5,
                'bass_surge': 0.5,
                'high_freq_drop': 0.3,
                'silence_threshold': 0.05,  # No full silence
                'buildup_slope': 0.001,
                'confidence_threshold': 0.5
            },
            'drum_and_bass': {
                'energy_ratio': 1.8,
                'bass_surge': 0.8,        # Very strong bass
                'high_freq_drop': 0.4,
                'silence_threshold': 0.02,
                'buildup_slope': 0.003,    # Fast buildup
                'confidence_threshold': 0.55
            },
            'techno': {
                'energy_ratio': 1.3,       # Subtle changes
                'bass_surge': 0.4,
                'high_freq_drop': 0.2,
                'silence_threshold': 0.1,   # No silence
                'buildup_slope': 0.0005,   # Gradual
                'confidence_threshold': 0.45
            }
        }
        
        return thresholds.get(self.genre, thresholds['house'])
    
    def extract_drop_features_detailed(self, y, sr, boundary_time):
        """
        Extract features with multiple time windows for better accuracy.
        """
        features = {}
        
        # Get different time windows
        windows = self.get_analysis_windows(y, sr, boundary_time)
        
        # 1. Energy progression features
        features['energy'] = self.compute_energy_features(windows, sr)
        
        # 2. Spectral features
        features['spectral'] = self.compute_spectral_features(windows, sr)
        
        # 3. Rhythm features
        features['rhythm'] = self.compute_rhythm_features(windows, sr)
        
        # 4. Silence/gap detection
        features['silence'] = self.detect_silence_gap(windows['at_drop'], sr)
        
        return features
    
    def get_analysis_windows(self, y, sr, boundary_time):
        """
        Extract multiple analysis windows around boundary.
        """
        windows = {}
        boundary_sample = int(boundary_time * sr)
        
        for name, duration in self.window_sizes.items():
            window_samples = int(duration * sr)
            
            if name == 'pre_drop':
                start = max(0, boundary_sample - window_samples)
                end = boundary_sample
            elif name == 'at_drop':
                half_window = window_samples // 2
                start = max(0, boundary_sample - half_window)
                end = min(len(y), boundary_sample + half_window)
            else:  # post_drop
                start = boundary_sample
                end = min(len(y), boundary_sample + window_samples)
            
            windows[name] = y[start:end]
        
        return windows
    
    def compute_energy_features(self, windows, sr):
        """
        Detailed energy analysis across time windows.
        """
        features = {}
        
        # RMS energy for each window
        for name, audio in windows.items():
            if len(audio) > 0:
                rms = librosa.feature.rms(y=audio)[0]
                features[f'{name}_rms_mean'] = np.mean(rms)
                features[f'{name}_rms_std'] = np.std(rms)
                features[f'{name}_rms_max'] = np.max(rms)
        
        # Energy ratios (key indicators)
        if 'pre_drop_rms_mean' in features and 'post_drop_rms_mean' in features:
            features['energy_ratio'] = (
                features['post_drop_rms_mean'] / 
                (features['pre_drop_rms_mean'] + 1e-6)
            )
        
        # Buildup detection (energy slope in pre_drop)
        if len(windows['pre_drop']) > sr:
            rms = librosa.feature.rms(y=windows['pre_drop'], hop_length=512)[0]
            x = np.arange(len(rms))
            slope, _ = np.polyfit(x, rms, 1)
            features['buildup_slope'] = slope
        
        return features
    
    def compute_spectral_features(self, windows, sr):
        """
        Frequency-based drop indicators.
        """
        features = {}
        
        # Focus on the transition moment
        if len(windows['at_drop']) > 0:
            stft = librosa.stft(windows['at_drop'])
            magnitude = np.abs(stft)
            
            # Frequency bins for different ranges
            freq_bins = stft.shape[0]
            bass_cutoff = int(self.params['bass_frequency_cutoff'] * freq_bins / (sr/2))
            mid_cutoff = int(1000 * freq_bins / (sr/2))
            
            # Split spectrum into bands
            bass = magnitude[:bass_cutoff, :]
            mids = magnitude[bass_cutoff:mid_cutoff, :]
            highs = magnitude[mid_cutoff:, :]
            
            # Compute energy in each band over time
            bass_energy = np.mean(bass, axis=0)
            mid_energy = np.mean(mids, axis=0)
            high_energy = np.mean(highs, axis=0)
            
            # Look for characteristic drop pattern
            mid_point = len(bass_energy) // 2
            
            # Bass surge: compare before and after midpoint
            features['bass_surge'] = (
                np.mean(bass_energy[mid_point:]) / 
                (np.mean(bass_energy[:mid_point]) + 1e-6)
            )
            
            # High frequency drop
            features['high_freq_drop'] = (
                np.mean(high_energy[:mid_point]) / 
                (np.mean(high_energy[mid_point:]) + 1e-6)
            )
            
            # Spectral contrast (increases at drop)
            contrast = librosa.feature.spectral_contrast(
                y=windows['at_drop'], sr=sr
            )
            features['contrast_change'] = np.max(contrast) - np.min(contrast)
        
        return features
    
    def compute_rhythm_features(self, windows, sr):
        """
        Rhythm and onset-based features.
        """
        features = {}
        
        # Onset density change
        for name in ['pre_drop', 'post_drop']:
            if name in windows and len(windows[name]) > sr:
                onset_env = librosa.onset.onset_strength(
                    y=windows[name], sr=sr
                )
                features[f'{name}_onset_density'] = np.mean(onset_env)
        
        # Onset density ratio
        if 'pre_drop_onset_density' in features and 'post_drop_onset_density' in features:
            features['onset_ratio'] = (
                features['post_drop_onset_density'] / 
                (features['pre_drop_onset_density'] + 1e-6)
            )
        
        # Beat strength at boundary
        if len(windows['at_drop']) > 0:
            tempo, beats = librosa.beat.beat_track(
                y=windows['at_drop'], sr=sr
            )
            features['beat_strength'] = len(beats) / (len(windows['at_drop']) / sr)
        
        return features
    
    def detect_silence_gap(self, audio, sr):
        """
        Detect characteristic silence/filter before drop.
        """
        if len(audio) == 0:
            return {'has_silence': False, 'silence_duration': 0}
        
        # Compute frame-wise energy
        hop_length = 512
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        
        # Find low-energy frames
        silence_threshold = self.thresholds['silence_threshold']
        silent_frames = rms < silence_threshold
        
        # Find longest consecutive silence
        max_silence_frames = 0
        current_silence = 0
        
        for is_silent in silent_frames:
            if is_silent:
                current_silence += 1
                max_silence_frames = max(max_silence_frames, current_silence)
            else:
                current_silence = 0
        
        silence_duration = max_silence_frames * hop_length / sr
        
        return {
            'has_silence': silence_duration > 0.1,  # At least 100ms
            'silence_duration': silence_duration,
            'silence_ratio': np.mean(silent_frames)
        }
    
    def classify_drop(self, features):
        """
        Rule-based classification with genre-specific thresholds.
        """
        score = 0.0
        max_score = 0.0
        
        # Energy ratio check (most important)
        if features['energy'].get('energy_ratio', 0) > self.thresholds['energy_ratio']:
            score += 0.3
        max_score += 0.3
        
        # Bass surge check
        if features['spectral'].get('bass_surge', 0) > self.thresholds['bass_surge']:
            score += 0.25
        max_score += 0.25
        
        # High frequency drop check
        if features['spectral'].get('high_freq_drop', 0) > self.thresholds['high_freq_drop']:
            score += 0.15
        max_score += 0.15
        
        # Buildup check
        if features['energy'].get('buildup_slope', 0) > self.thresholds['buildup_slope']:
            score += 0.15
        max_score += 0.15
        
        # Silence gap (genre-dependent importance)
        if self.params['silence_gap']:
            if features['silence']['has_silence']:
                score += 0.15
            max_score += 0.15
        else:
            # Penalize silence for continuous genres
            if not features['silence']['has_silence']:
                score += 0.05
            max_score += 0.05
        
        # Normalize score
        confidence = score / max_score if max_score > 0 else 0
        
        return {
            'is_drop': confidence > self.thresholds['confidence_threshold'],
            'confidence': confidence,
            'score_breakdown': {
                'energy': features['energy'].get('energy_ratio', 0),
                'bass': features['spectral'].get('bass_surge', 0),
                'highs': features['spectral'].get('high_freq_drop', 0),
                'buildup': features['energy'].get('buildup_slope', 0),
                'silence': features['silence']['has_silence']
            }
        }
```

## Spectrogram Method Parameter Tuning

```python
class SpectrogramDropDetector:
    """
    Spectrogram-based detection with visual pattern matching.
    """
    
    def __init__(self, genre='house'):
        self.genre = genre
        self.params = GENRE_PARAMETERS[genre]
        
        # Spectrogram parameters
        self.spec_params = self.get_spectrogram_params()
        
    def get_spectrogram_params(self):
        """
        Genre-optimized spectrogram parameters.
        """
        params = {
            'dubstep': {
                'n_fft': 4096,      # Larger for better bass resolution
                'hop_length': 512,
                'n_mels': 128,
                'fmin': 20,         # Include sub-bass
                'fmax': 8000        # Less important highs
            },
            'drum_and_bass': {
                'n_fft': 2048,      # Smaller for time resolution
                'hop_length': 256,  # Better for fast transients
                'n_mels': 128,
                'fmin': 30,
                'fmax': 16000       # Important highs (breaks)
            },
            'house': {
                'n_fft': 2048,
                'hop_length': 512,
                'n_mels': 96,
                'fmin': 30,
                'fmax': 12000
            },
            'techno': {
                'n_fft': 2048,
                'hop_length': 512,
                'n_mels': 96,
                'fmin': 25,
                'fmax': 10000
            }
        }
        
        return params.get(self.genre, params['house'])
    
    def detect_drops_visual(self, audio_path, visualize=True):
        """
        Detect drops using spectrogram pattern matching.
        """
        y, sr = librosa.load(audio_path)
        
        # Compute mel spectrogram with genre-specific params
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr,
            **self.spec_params
        )
        
        # Convert to dB scale
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Detect patterns
        drops = self.find_drop_patterns(mel_db, sr)
        
        if visualize:
            self.visualize_detection(mel_db, sr, drops)
        
        return drops
    
    def find_drop_patterns(self, mel_db, sr):
        """
        Find characteristic drop patterns in spectrogram.
        """
        drops = []
        
        # Time resolution
        hop_length = self.spec_params['hop_length']
        time_resolution = hop_length / sr
        
        # Frequency band indices
        n_mels = mel_db.shape[0]
        bass_bands = slice(0, n_mels // 4)
        mid_bands = slice(n_mels // 4, 3 * n_mels // 4)
        high_bands = slice(3 * n_mels // 4, n_mels)
        
        # Compute band energies
        bass_energy = np.mean(mel_db[bass_bands, :], axis=0)
        mid_energy = np.mean(mel_db[mid_bands, :], axis=0)
        high_energy = np.mean(mel_db[high_bands, :], axis=0)
        
        # Smooth for stability
        from scipy.ndimage import gaussian_filter1d
        sigma = 10  # Smoothing factor
        
        bass_smooth = gaussian_filter1d(bass_energy, sigma)
        mid_smooth = gaussian_filter1d(mid_energy, sigma)
        high_smooth = gaussian_filter1d(high_energy, sigma)
        
        # Pattern detection with sliding window
        window_size = int(4.0 / time_resolution)  # 4-second window
        
        for i in range(window_size, len(bass_smooth) - window_size):
            # Check for buildup pattern (increasing highs/mids)
            pre_window = slice(i - window_size, i)
            post_window = slice(i, i + window_size)
            
            # Buildup indicators
            high_buildup = np.mean(np.diff(high_smooth[pre_window])) > 0.05
            mid_buildup = np.mean(np.diff(mid_smooth[pre_window])) > 0.03
            
            # Drop indicators
            bass_jump = (
                np.mean(bass_smooth[post_window]) - 
                np.mean(bass_smooth[pre_window])
            ) > 5  # dB increase
            
            high_drop = (
                np.mean(high_smooth[pre_window]) - 
                np.mean(high_smooth[post_window])
            ) > 3  # dB decrease
            
            # Combined criteria
            if (high_buildup or mid_buildup) and bass_jump and high_drop:
                drop_time = i * time_resolution
                
                # Calculate confidence based on pattern strength
                confidence = self.calculate_pattern_confidence(
                    mel_db, i, window_size
                )
                
                drops.append({
                    'time': drop_time,
                    'confidence': confidence,
                    'bass_jump_db': bass_jump,
                    'high_drop_db': high_drop
                })
        
        # Remove duplicates within 5 seconds
        drops = self.filter_duplicate_drops(drops, min_distance=5.0)
        
        return drops
    
    def calculate_pattern_confidence(self, mel_db, center_frame, window_size):
        """
        Calculate confidence based on pattern clarity.
        """
        # Extract region around potential drop
        start = max(0, center_frame - window_size)
        end = min(mel_db.shape[1], center_frame + window_size)
        region = mel_db[:, start:end]
        
        if region.shape[1] < window_size:
            return 0.0
        
        # Compute various indicators
        scores = []
        
        # 1. Energy contrast
        pre_energy = np.mean(region[:, :window_size//2])
        post_energy = np.mean(region[:, window_size//2:])
        energy_contrast = abs(post_energy - pre_energy) / 40  # Normalize
        scores.append(min(energy_contrast, 1.0))
        
        # 2. Vertical edge strength (sudden change)
        vertical_gradient = np.abs(np.diff(region, axis=1))
        max_gradient = np.max(np.mean(vertical_gradient, axis=0))
        edge_strength = max_gradient / 20  # Normalize
        scores.append(min(edge_strength, 1.0))
        
        # 3. Pattern consistency (low variance in similar regions)
        pre_variance = np.var(region[:, :window_size//4])
        post_variance = np.var(region[:, -window_size//4:])
        consistency = 1.0 - (min(pre_variance, post_variance) / 100)
        scores.append(max(consistency, 0))
        
        return np.mean(scores)
    
    def filter_duplicate_drops(self, drops, min_distance=5.0):
        """
        Remove duplicate detections, keeping highest confidence.
        """
        if len(drops) <= 1:
            return drops
        
        # Sort by confidence
        drops_sorted = sorted(drops, key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for drop in drops_sorted:
            # Check if too close to existing drops
            too_close = False
            for existing in filtered:
                if abs(drop['time'] - existing['time']) < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                filtered.append(drop)
        
        # Sort by time
        filtered.sort(key=lambda x: x['time'])
        
        return filtered
    
    def visualize_detection(self, mel_db, sr, drops):
        """
        Visualize spectrogram with detected drops.
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))
        
        # Time axis
        hop_length = self.spec_params['hop_length']
        time_frames = range(mel_db.shape[1])
        time_seconds = librosa.frames_to_time(
            time_frames, sr=sr, hop_length=hop_length
        )
        
        # Full spectrogram
        img = librosa.display.specshow(
            mel_db, sr=sr, hop_length=hop_length,
            x_axis='time', y_axis='mel', ax=ax1
        )
        ax1.set_title('Mel Spectrogram with Detected Drops')
        
        # Mark drops
        for drop in drops:
            ax1.axvline(x=drop['time'], color='red', linestyle='--', 
                       alpha=0.7, linewidth=2)
            ax1.text(drop['time'], mel_db.shape[0] * 0.9,
                    f"{drop['confidence']:.2f}", 
                    color='white', fontsize=8)
        
        # Band energies
        n_mels = mel_db.shape[0]
        bass_energy = np.mean(mel_db[:n_mels//3, :], axis=0)
        mid_energy = np.mean(mel_db[n_mels//3:2*n_mels//3, :], axis=0)
        high_energy = np.mean(mel_db[2*n_mels//3:, :], axis=0)
        
        ax2.plot(time_seconds, bass_energy, label='Bass', linewidth=2)
        ax2.plot(time_seconds, mid_energy, label='Mids', linewidth=1.5, alpha=0.7)
        ax2.plot(time_seconds, high_energy, label='Highs', linewidth=1, alpha=0.7)
        ax2.set_ylabel('Energy (dB)')
        ax2.set_title('Frequency Band Energies')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Mark drops on energy plot
        for drop in drops:
            ax2.axvline(x=drop['time'], color='red', linestyle='--', alpha=0.5)
        
        # Confidence timeline
        if drops:
            drop_times = [d['time'] for d in drops]
            confidences = [d['confidence'] for d in drops]
            
            ax3.stem(drop_times, confidences, basefmt=' ')
            ax3.set_ylabel('Confidence')
            ax3.set_xlabel('Time (s)')
            ax3.set_title('Drop Detection Confidence')
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)
            
            # Threshold line
            ax3.axhline(y=0.5, color='orange', linestyle=':', 
                       label='Threshold')
            ax3.legend()
        
        plt.tight_layout()
        plt.show()
```

## Adaptive Threshold Learning

```python
class AdaptiveDropDetector:
    """
    Self-tuning drop detector that learns from user feedback.
    """
    
    def __init__(self, initial_genre='house'):
        self.genre = initial_genre
        self.base_thresholds = self.get_initial_thresholds()
        self.adaptive_thresholds = self.base_thresholds.copy()
        self.feedback_history = []
        
    def get_initial_thresholds(self):
        """
        Conservative initial thresholds.
        """
        return {
            'energy_ratio': 1.4,
            'bass_surge': 0.5,
            'high_freq_drop': 0.3,
            'silence_threshold': 0.05,
            'buildup_slope': 0.001,
            'confidence_threshold': 0.5,
            'min_drop_spacing': 16  # bars
        }
    
    def detect_with_learning(self, audio_path):
        """
        Detect drops and return with confidence scores for user validation.
        """
        # Initial detection with current thresholds
        candidates = self.detect_drop_candidates(audio_path)
        
        # Apply learned adjustments
        filtered = self.apply_learned_filters(candidates)
        
        return filtered
    
    def detect_drop_candidates(self, audio_path):
        """
        Get all potential drops with varying confidence.
        """
        y, sr = librosa.load(audio_path)
        tempo = librosa.beat.tempo(y=y, sr=sr)[0]
        
        # Get boundaries
        boundaries = self.get_boundaries(y, sr)
        
        candidates = []
        for boundary_time in boundaries:
            features = self.extract_comprehensive_features(y, sr, boundary_time, tempo)
            score = self.score_candidate(features)
            
            candidates.append({
                'time': boundary_time,
                'confidence': score,
                'features': features,
                'tempo': tempo
            })
        
        return candidates
    
    def score_candidate(self, features):
        """
        Score based on current adaptive thresholds.
        """
        scores = []
        weights = []
        
        # Energy ratio
        if features.get('energy_ratio', 0) > self.adaptive_thresholds['energy_ratio']:
            score = min(features['energy_ratio'] / 3.0, 1.0)
            scores.append(score)
            weights.append(0.3)
        else:
            scores.append(0)
            weights.append(0.3)
        
        # Bass surge
        if features.get('bass_surge', 0) > self.adaptive_thresholds['bass_surge']:
            score = min(features['bass_surge'], 1.0)
            scores.append(score)
            weights.append(0.25)
        else:
            scores.append(0)
            weights.append(0.25)
        
        # Calculate weighted score
        total_score = sum(s * w for s, w in zip(scores, weights))
        
        return total_score / sum(weights)
    
    def update_thresholds(self, feedback_batch):
        """
        Update thresholds based on user feedback.
        
        feedback_batch: List of {'time': float, 'is_correct': bool, 'features': dict}
        """
        self.feedback_history.extend(feedback_batch)
        
        # Separate true positives, false positives, false negatives
        true_positives = [f for f in feedback_batch if f['is_correct']]
        false_positives = [f for f in feedback_batch if not f['is_correct']]
        
        # Adjust thresholds
        if len(true_positives) > 0 and len(false_positives) > 0:
            # Find feature ranges that separate TP from FP
            for feature_name in ['energy_ratio', 'bass_surge', 'high_freq_drop']:
                tp_values = [f['features'].get(feature_name, 0) 
                           for f in true_positives]
                fp_values = [f['features'].get(feature_name, 0) 
                           for f in false_positives]
                
                if tp_values and fp_values:
                    # Find optimal threshold
                    tp_min = min(tp_values)
                    fp_max = max(fp_values)
                    
                    if tp_min > fp_max:
                        # Clear separation - use midpoint
                        new_threshold = (tp_min + fp_max) / 2
                    else:
                        # Overlap - use weighted average
                        tp_mean = np.mean(tp_values)
                        fp_mean = np.mean(fp_values)
                        new_threshold = (tp_mean * 0.7 + fp_mean * 0.3)
                    
                    # Gradual adjustment (momentum)
                    old_threshold = self.adaptive_thresholds[feature_name]
                    self.adaptive_thresholds[feature_name] = (
                        0.7 * old_threshold + 0.3 * new_threshold
                    )
        
        # Adjust confidence threshold
        if len(self.feedback_history) > 20:
            recent_accuracy = sum(
                1 for f in self.feedback_history[-20:] if f['is_correct']
            ) / 20
            
            if recent_accuracy < 0.7:
                # Too many false positives - increase threshold
                self.adaptive_thresholds['confidence_threshold'] *= 1.05
            elif recent_accuracy > 0.9:
                # Could be missing drops - decrease threshold
                self.adaptive_thresholds['confidence_threshold'] *= 0.95
        
        print(f"Updated thresholds: {self.adaptive_thresholds}")
    
    def extract_comprehensive_features(self, y, sr, boundary_time, tempo):
        """
        Extract all relevant features for learning.
        """
        features = {}
        
        # Time windows
        boundary_sample = int(boundary_time * sr)
        window_samples = int(4.0 * sr)  # 4-second window
        
        pre_start = max(0, boundary_sample - window_samples)
        pre_end = boundary_sample
        post_start = boundary_sample
        post_end = min(len(y), boundary_sample + window_samples)
        
        pre_segment = y[pre_start:pre_end]
        post_segment = y[post_start:post_end]
        
        # Energy features
        if len(pre_segment) > 0 and len(post_segment) > 0:
            pre_rms = np.mean(librosa.feature.rms(y=pre_segment)[0])
            post_rms = np.mean(librosa.feature.rms(y=post_segment)[0])
            features['energy_ratio'] = post_rms / (pre_rms + 1e-6)
        
        # Spectral features
        if len(post_segment) > 1024:
            stft = librosa.stft(post_segment[:sr])  # 1 second
            magnitude = np.abs(stft)
            
            bass_bins = int(150 * len(magnitude) / (sr/2))
            features['bass_surge'] = np.mean(magnitude[:bass_bins, :])
        
        # More features for learning...
        features['tempo'] = tempo
        features['position_in_track'] = boundary_time / (len(y) / sr)
        
        return features
```

## Testing and Validation

```python
def test_drop_detection_accuracy(detector, test_dataset):
    """
    Evaluate detector performance on annotated dataset.
    """
    results = {
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'timing_errors': []
    }
    
    for track in test_dataset:
        audio_path = track['path']
        ground_truth_drops = track['drops']  # List of times
        
        # Detect drops
        detected_drops = detector.detect_drops(audio_path)
        
        # Match detections to ground truth
        matched = []
        for gt_time in ground_truth_drops:
            # Find closest detection within tolerance window
            closest_detection = None
            min_distance = float('inf')
            
            for det in detected_drops:
                distance = abs(det['time'] - gt_time)
                if distance < min_distance and distance < 3.0:  # 3-second tolerance
                    min_distance = distance
                    closest_detection = det
            
            if closest_detection:
                results['true_positives'] += 1
                results['timing_errors'].append(min_distance)
                matched.append(closest_detection)
            else:
                results['false_negatives'] += 1
        
        # Count unmatched detections as false positives
        for det in detected_drops:
            if det not in matched:
                results['false_positives'] += 1
    
    # Calculate metrics
    precision = results['true_positives'] / (
        results['true_positives'] + results['false_positives'] + 1e-6
    )
    recall = results['true_positives'] / (
        results['true_positives'] + results['false_negatives'] + 1e-6
    )
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    
    avg_timing_error = np.mean(results['timing_errors']) if results['timing_errors'] else 0
    
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    print(f"Average Timing Error: {avg_timing_error:.2f} seconds")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_timing_error': avg_timing_error
    }
```

## Quick Reference: Key Tuning Parameters

### Critical Parameters by Priority:

1. **Energy Ratio Threshold** (Most Important)
   - Dubstep: 1.8-2.5
   - House: 1.3-1.7
   - DnB: 1.6-2.0
   - Techno: 1.2-1.5

2. **Bass Frequency Cutoff**
   - Dubstep: 80-100 Hz
   - House: 100-150 Hz
   - DnB: 60-80 Hz
   - Techno: 80-120 Hz

3. **Window Sizes**
   - Pre-drop analysis: 4-8 seconds
   - At-drop window: 1-2 seconds
   - Post-drop verification: 2-4 seconds

4. **Confidence Thresholds**
   - Conservative (fewer false positives): 0.6-0.7
   - Balanced: 0.5
   - Aggressive (catch more drops): 0.4-0.45

5. **Minimum Drop Spacing**
   - Fast genres (DnB): 8 bars
   - Standard (House/Techno): 16 bars
   - Slow builds (Progressive): 32 bars

### Troubleshooting Common Issues:

- **Too many false positives**: Increase energy_ratio and confidence thresholds
- **Missing subtle drops**: Decrease thresholds, increase window sizes
- **Detecting breaks as drops**: Check silence_threshold, require bass_surge
- **Poor timing accuracy**: Reduce hop_length, use beat-aligned boundaries
- **Genre confusion**: Use adaptive learning or genre detection preprocessing
