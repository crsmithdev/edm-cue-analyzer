import math

import numpy as np
import pytest
import soundfile as sf

from edm_cue_analyzer.analyzer import AudioAnalyzer
from edm_cue_analyzer.config import get_default_config
from edm_cue_analyzer.metadata.base import MetadataSource, TrackMetadata


@pytest.mark.asyncio
async def test_analyze_with_online_metadata_skips_bpm(tmp_path, monkeypatch):
    """Verify analyzer uses online metadata BPM (mocked) and skips heavy BPM detection."""

    # Create a short synthetic audio file (12s sine wave) - long enough for analyzer (>10s)
    sr = 22050
    duration = 12.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.05 * np.sin(2.0 * math.pi * 440.0 * t)

    # Name file with 'Artist - Title' so analyzer can derive artist/title for online lookup
    audio_path = tmp_path / "MockArtist - MockTitle.wav"
    sf.write(str(audio_path), y.astype(np.float32), sr)

    # Mock MetadataAggregator.get_metadata to return a TrackMetadata with BPM
    async def fake_get_metadata(self, artist: str, title: str, **kwargs):
        return TrackMetadata(
            artist=artist,
            title=title,
            bpm=128.0,
            source=MetadataSource.GETSONGBPM,
            confidence=0.9,
        )

    monkeypatch.setattr(
        "edm_cue_analyzer.metadata.aggregator.MetadataAggregator.get_metadata",
        fake_get_metadata,
    )

    config = get_default_config()
    analyzer = AudioAnalyzer(config)

    # Request only bpm analysis; since online metadata provides BPM, analyzer should skip heavy detection
    structure = await analyzer.analyze_with(audio_path, analyses={"bpm"})

    # Detected BPM should equal the metadata BPM provided by the mocked aggregator
    assert structure.detected_bpm == pytest.approx(128.0, rel=1e-3)

    # Bar duration should be consistent with 128 BPM (4 beats/bar)
    expected_bar = (4.0 * 60.0) / 128.0
    assert structure.bar_duration == pytest.approx(expected_bar, rel=1e-3)
