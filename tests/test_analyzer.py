"""Basic tests for EDM Cue Analyzer."""

import pytest
import numpy as np
from pathlib import Path

from edm_cue_analyzer.config import load_config, get_default_config
from edm_cue_analyzer.analyzer import AudioAnalyzer, TrackStructure, bars_to_seconds, seconds_to_bars
from edm_cue_analyzer.cue_generator import CueGenerator, CuePoint


class TestConfig:
    """Test configuration loading."""
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        config = get_default_config()
        
        assert config is not None
        assert len(config.hot_cues) == 8  # A-H
        assert 'A' in config.hot_cues
        assert 'H' in config.hot_cues
        assert config.analysis.energy_window_seconds > 0
    
    def test_hot_cue_structure(self):
        """Test hot cue configuration structure."""
        config = get_default_config()
        
        cue_a = config.hot_cues['A']
        assert cue_a.name == "Early Intro"
        assert cue_a.position_percent == 0.08
        assert cue_a.loop_bars == 8
        assert cue_a.color == "BLUE"


class TestAnalyzer:
    """Test audio analyzer functions."""
    
    def test_bars_to_seconds_conversion(self):
        """Test converting bars to seconds."""
        # At 128 BPM, 1 bar = 1.875 seconds
        seconds = bars_to_seconds(8, 128.0)
        assert abs(seconds - 15.0) < 0.1
        
        # At 140 BPM, 1 bar ≈ 1.714 seconds
        seconds = bars_to_seconds(16, 140.0)
        assert abs(seconds - 27.43) < 0.1
    
    def test_seconds_to_bars_conversion(self):
        """Test converting seconds to bars."""
        # At 128 BPM, 15 seconds = 8 bars
        bars = seconds_to_bars(15.0, 128.0)
        assert bars == 8
        
        # At 140 BPM, 27.43 seconds ≈ 16 bars
        bars = seconds_to_bars(27.43, 140.0)
        assert bars == 16


class TestCueGenerator:
    """Test cue point generation."""
    
    @pytest.fixture
    def mock_structure(self):
        """Create a mock track structure for testing."""
        return TrackStructure(
            bpm=128.0,
            duration=300.0,  # 5 minutes
            beats=np.array([0.0, 0.46875, 0.9375]),  # Mock beat times
            bar_duration=1.875,
            energy_curve=np.linspace(0.3, 0.8, 100),
            energy_times=np.linspace(0, 300, 100),
            drops=[60.0, 180.0],  # Two drops
            breakdowns=[120.0],  # One breakdown
            builds=[55.0, 175.0],  # Two builds
        )
    
    def test_cue_generation_count(self, mock_structure):
        """Test that cues are generated."""
        config = get_default_config()
        generator = CueGenerator(config)
        
        cues = generator.generate_cues(mock_structure)
        
        # Should have hot cues + memory cues
        assert len(cues) > 0
        
        hot_cues = [c for c in cues if c.cue_type == 'hot']
        memory_cues = [c for c in cues if c.cue_type == 'memory']
        
        assert len(hot_cues) == 8  # A-H
        assert len(memory_cues) == 5  # Default memory cues
    
    def test_position_based_cues(self, mock_structure):
        """Test percentage-based cue positioning."""
        config = get_default_config()
        generator = CueGenerator(config)
        
        cues = generator.generate_cues(mock_structure)
        
        # Hot Cue A should be at 8% (24 seconds into 300s track)
        cue_a = next(c for c in cues if c.hot_cue_number == 0)
        expected_position = 300.0 * 0.08
        assert abs(cue_a.position - expected_position) < 1.0
    
    def test_structure_based_cues(self, mock_structure):
        """Test structure-detection based cue positioning."""
        config = get_default_config()
        generator = CueGenerator(config)
        
        cues = generator.generate_cues(mock_structure)
        
        # Should have cues near detected drops
        drop_cues = [c for c in cues if 'drop' in c.label.lower()]
        assert len(drop_cues) > 0
    
    def test_cue_colors(self, mock_structure):
        """Test that cue colors are assigned."""
        config = get_default_config()
        generator = CueGenerator(config)
        
        cues = generator.generate_cues(mock_structure)
        hot_cues = [c for c in cues if c.cue_type == 'hot']
        
        for cue in hot_cues:
            assert cue.color is not None
            assert cue.color in ['BLUE', 'GREEN', 'TEAL', 'YELLOW', 'ORANGE', 'PURPLE', 'RED']


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_analysis_synthetic(self):
        """Test complete analysis pipeline with synthetic audio."""
        # This test would require actual audio file
        # For now, just test that the components work together
        config = get_default_config()
        
        # Create mock structure
        structure = TrackStructure(
            bpm=140.0,
            duration=360.0,
            beats=np.array([]),
            bar_duration=1.714,
            energy_curve=np.array([]),
            energy_times=np.array([]),
            drops=[70.0, 210.0],
            breakdowns=[140.0],
            builds=[65.0, 205.0],
        )
        
        # Generate cues
        generator = CueGenerator(config)
        cues = generator.generate_cues(structure)
        
        # Verify we got cues
        assert len(cues) > 0
        
        # Verify all cues are within track duration
        for cue in cues:
            assert 0.0 <= cue.position <= structure.duration


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
