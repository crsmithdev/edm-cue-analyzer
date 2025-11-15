"""Tests for metadata provider system."""

import asyncio
from pathlib import Path

import pytest

from edm_cue_analyzer.metadata import (
    MetadataAggregator,
    MetadataProvider,
    MetadataSource,
    TrackMetadata,
)


class MockProvider(MetadataProvider):
    """Mock provider for testing."""

    def __init__(self, source: MetadataSource, return_value: TrackMetadata):
        self._source = source
        self._return_value = return_value

    @property
    def source(self) -> MetadataSource:
        return self._source

    async def get_metadata(self, **kwargs):
        return self._return_value


class TestTrackMetadata:
    """Test TrackMetadata dataclass."""

    def test_creation(self):
        """Test basic metadata creation."""
        metadata = TrackMetadata(
            artist="Test Artist",
            title="Test Title",
            bpm=128.0,
            source=MetadataSource.LOCAL_FILE,
        )
        assert metadata.artist == "Test Artist"
        assert metadata.title == "Test Title"
        assert metadata.bpm == 128.0
        assert metadata.source == MetadataSource.LOCAL_FILE

    def test_merge_prefer_online_bpm(self):
        """Test merging prefers online BPM over local."""
        local = TrackMetadata(
            artist="Artist",
            title="Title",
            bpm=125.0,
            duration=300.0,
            source=MetadataSource.LOCAL_FILE,
            confidence=0.5,
        )
        online = TrackMetadata(
            artist="Artist",
            title="Title",
            bpm=128.0,
            key="Am",
            source=MetadataSource.GETSONGBPM,
            confidence=0.9,
        )

        merged = local.merge(online)

        # Online BPM and key win
        assert merged.bpm == 128.0
        assert merged.key == "Am"
        # Local duration is preserved
        assert merged.duration == 300.0
        # Confidence is from online
        assert merged.confidence == 0.9

    def test_merge_keeps_local_when_online_none(self):
        """Test merging keeps local values when online is None."""
        local = TrackMetadata(
            artist="Artist",
            title="Title",
            bpm=125.0,
            duration=300.0,
            source=MetadataSource.LOCAL_FILE,
        )
        online = TrackMetadata(
            artist="Artist",
            title="Title",
            bpm=None,  # No BPM found online
            source=MetadataSource.GETSONGBPM,
        )

        merged = local.merge(online)

        # Local BPM is kept
        assert merged.bpm == 125.0
        assert merged.duration == 300.0

    def test_merge_symmetric(self):
        """Test merge order doesn't matter for None values."""
        meta1 = TrackMetadata(artist="Artist", bpm=128.0, key=None)
        meta2 = TrackMetadata(artist="Artist", bpm=None, key="Am")

        # Either merge order produces same result
        merged1 = meta1.merge(meta2)
        merged2 = meta2.merge(meta1)

        assert merged1.bpm == 128.0
        assert merged1.key == "Am"
        assert merged2.bpm == 128.0
        assert merged2.key == "Am"


class TestMetadataAggregator:
    """Test MetadataAggregator."""

    @pytest.mark.asyncio
    async def test_single_provider(self):
        """Test aggregator with single provider."""
        metadata = TrackMetadata(
            artist="Artist",
            title="Title",
            bpm=128.0,
            source=MetadataSource.GETSONGBPM,
            confidence=0.8,
        )
        provider = MockProvider(MetadataSource.GETSONGBPM, metadata)

        aggregator = MetadataAggregator(
            providers=[provider],
            enable_cache=False,
        )

        result = await aggregator.get_metadata("Artist", "Title")

        assert result is not None
        assert result.bpm == 128.0
        assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_multiple_providers_consensus(self):
        """Test consensus from multiple providers."""
        providers = [
            MockProvider(
                MetadataSource.GETSONGBPM,
                TrackMetadata(bpm=128.0, confidence=0.8),
            ),
            MockProvider(
                MetadataSource.TUNEBAT,
                TrackMetadata(bpm=128.5, confidence=0.85),
            ),
            MockProvider(
                MetadataSource.BEATPORT,
                TrackMetadata(bpm=128.2, confidence=0.9),
            ),
        ]

        aggregator = MetadataAggregator(
            providers=providers,
            enable_cache=False,
        )

        result = await aggregator.get_metadata("Artist", "Title")

        assert result is not None
        # BPM should be weighted average, close to 128
        assert 128.0 <= result.bpm <= 128.5
        # High confidence due to agreement
        assert result.confidence > 0.85

    @pytest.mark.asyncio
    async def test_provider_failure_handling(self):
        """Test that one provider failure doesn't break aggregation."""
        good_metadata = TrackMetadata(
            bpm=128.0,
            confidence=0.8,
            source=MetadataSource.TUNEBAT,
        )

        providers = [
            MockProvider(MetadataSource.GETSONGBPM, None),  # Returns None
            MockProvider(MetadataSource.TUNEBAT, good_metadata),  # Returns data
        ]

        aggregator = MetadataAggregator(
            providers=providers,
            enable_cache=False,
        )

        result = await aggregator.get_metadata("Artist", "Title")

        assert result is not None
        assert result.bpm == 128.0

    @pytest.mark.asyncio
    async def test_caching(self, tmp_path):
        """Test that results are cached."""
        cache_file = tmp_path / "test_cache.json"

        metadata = TrackMetadata(
            bpm=128.0,
            confidence=0.8,
            source=MetadataSource.GETSONGBPM,
        )
        provider = MockProvider(MetadataSource.GETSONGBPM, metadata)

        aggregator = MetadataAggregator(
            providers=[provider],
            cache_path=cache_file,
            enable_cache=True,
        )

        # First call - should hit provider
        result1 = await aggregator.get_metadata("Artist", "Title")
        assert result1.bpm == 128.0

        # Cache file should be created
        assert cache_file.exists()

        # Create new aggregator with same cache
        aggregator2 = MetadataAggregator(
            providers=[],  # No providers!
            cache_path=cache_file,
            enable_cache=True,
        )

        # Second call - should use cache
        result2 = await aggregator2.get_metadata("Artist", "Title")
        assert result2 is not None
        assert result2.bpm == 128.0
        assert result2.source == MetadataSource.CACHED

    @pytest.mark.asyncio
    async def test_no_results(self):
        """Test when no providers return results."""
        providers = [
            MockProvider(MetadataSource.GETSONGBPM, None),
            MockProvider(MetadataSource.TUNEBAT, None),
        ]

        aggregator = MetadataAggregator(
            providers=providers,
            enable_cache=False,
        )

        result = await aggregator.get_metadata("Unknown", "Track")

        assert result is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test aggregator as async context manager."""
        metadata = TrackMetadata(bpm=128.0)
        provider = MockProvider(MetadataSource.GETSONGBPM, metadata)

        async with MetadataAggregator(providers=[provider], enable_cache=False) as agg:
            result = await agg.get_metadata("Artist", "Title")
            assert result.bpm == 128.0

        # Provider should be closed after context exit
        # (In real providers with HTTP sessions)


class TestLocalFileProvider:
    """Test LocalFileProvider."""

    @pytest.mark.asyncio
    async def test_file_not_found(self):
        """Test handling of non-existent file."""
        from edm_cue_analyzer.metadata.local import LocalFileProvider

        provider = LocalFileProvider()
        result = await provider.get_metadata(file_path=Path("/nonexistent/file.flac"))

        assert result is None

    def test_filename_parsing(self):
        """Test filename parsing for artist/title."""
        from edm_cue_analyzer.metadata.local import LocalFileProvider

        provider = LocalFileProvider()

        # Test standard format
        result = provider._parse_filename(Path("Artist - Title.flac"))
        assert result["artist"] == "Artist"
        assert result["title"] == "Title"

        # Test with track number
        result = provider._parse_filename(Path("01. Artist - Title.mp3"))
        assert result["artist"] == "Artist"
        assert result["title"] == "Title"

        # Test no separator
        result = provider._parse_filename(Path("JustTitle.wav"))
        assert result.get("artist") is None
        assert result["title"] == "JustTitle"


class TestOnlineProviders:
    """Test online provider base functionality."""

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test that rate limiting delays requests."""
        from edm_cue_analyzer.metadata.online import OnlineProvider

        provider = OnlineProvider(rate_limit_delay=0.1)

        start = asyncio.get_event_loop().time()
        await provider._rate_limit()
        await provider._rate_limit()
        end = asyncio.get_event_loop().time()

        # Second call should be delayed by ~0.1 seconds
        assert end - start >= 0.1

    @pytest.mark.asyncio
    async def test_session_management(self):
        """Test HTTP session is created and reused."""
        from edm_cue_analyzer.metadata.online import OnlineProvider

        provider = OnlineProvider()

        session1 = await provider._get_session()
        session2 = await provider._get_session()

        # Should return same session
        assert session1 is session2

        await provider.close()
        assert session1.closed


# Integration tests (commented out by default - require network)
"""
class TestIntegration:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_getsongbpm(self):
        from edm_cue_analyzer.metadata import GetSongBPMProvider

        async with GetSongBPMProvider() as provider:
            result = await provider.get_metadata(
                artist="Adam Beyer",
                title="Your Mind"
            )

            assert result is not None
            assert result.bpm is not None
            assert result.bpm > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_aggregator(self):
        from edm_cue_analyzer.metadata import MetadataAggregator

        async with MetadataAggregator() as agg:
            result = await agg.get_metadata(
                artist="Charlotte de Witte",
                title="Selected"
            )

            assert result is not None
            assert result.bpm is not None
            assert result.confidence > 0
"""
