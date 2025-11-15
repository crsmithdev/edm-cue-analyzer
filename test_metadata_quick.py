"""Quick test of metadata provider system."""

import asyncio
from pathlib import Path

from src.edm_cue_analyzer.metadata import (
    GetSongBPMProvider,
    LocalFileProvider,
    MetadataAggregator,
    MetadataSource,
    TrackMetadata,
)


async def test_track_metadata():
    """Test TrackMetadata creation and merging."""
    print("Testing TrackMetadata...")

    # Create metadata
    meta1 = TrackMetadata(
        artist="Test Artist",
        title="Test Track",
        bpm=128.0,
        source=MetadataSource.LOCAL_FILE,
    )
    print(f"✓ Created: {meta1.artist} - {meta1.title} @ {meta1.bpm} BPM")

    # Test merging
    meta2 = TrackMetadata(
        artist="Test Artist",
        title="Test Track",
        bpm=130.0,
        key="Am",
        source=MetadataSource.GETSONGBPM,
        confidence=0.9,
    )

    merged = meta1.merge(meta2)
    print(f"✓ Merged: BPM={merged.bpm}, Key={merged.key}, Confidence={merged.confidence}")
    assert merged.bpm == 130.0  # Online BPM preferred
    assert merged.key == "Am"

    print("✓ TrackMetadata tests passed\n")


async def test_local_provider():
    """Test LocalFileProvider filename parsing."""
    print("Testing LocalFileProvider...")

    provider = LocalFileProvider()

    # Test filename parsing
    test_cases = [
        ("Artist - Title.flac", "Artist", "Title"),
        ("01. Artist - Title.mp3", "Artist", "Title"),
        ("Artist_-_Title.wav", "Artist", "Title"),
        ("JustTitle.mp3", None, "JustTitle"),
    ]

    for filename, expected_artist, expected_title in test_cases:
        result = provider._parse_filename(Path(filename))
        assert result.get("artist") == expected_artist, f"Failed for {filename}"
        assert result["title"] == expected_title, f"Failed for {filename}"
        print(f"✓ Parsed '{filename}' → artist='{expected_artist}', title='{expected_title}'")

    print("✓ LocalFileProvider tests passed\n")


async def test_aggregator_mock():
    """Test MetadataAggregator with mock providers."""
    print("Testing MetadataAggregator...")

    # Create mock provider
    from src.edm_cue_analyzer.metadata.base import MetadataProvider

    class MockProvider(MetadataProvider):
        def __init__(self, source, bpm):
            self._source = source
            self._bpm = bpm

        @property
        def source(self):
            return self._source

        async def get_metadata(self, **kwargs):
            return TrackMetadata(
                artist=kwargs.get("artist"),
                title=kwargs.get("title"),
                bpm=self._bpm,
                source=self._source,
                confidence=0.8,
            )

    # Create aggregator with mock providers
    providers = [
        MockProvider(MetadataSource.GETSONGBPM, 128.0),
        MockProvider(MetadataSource.TUNEBAT, 128.5),
        MockProvider(MetadataSource.BEATPORT, 128.2),
    ]

    aggregator = MetadataAggregator(
        providers=providers,
        enable_cache=False,
    )

    result = await aggregator.get_metadata("Test Artist", "Test Track")

    assert result is not None
    assert result.artist == "Test Artist"
    assert result.title == "Test Track"
    assert 128.0 <= result.bpm <= 128.5
    assert result.confidence > 0.8

    print(f"✓ Consensus BPM: {result.bpm:.1f} (confidence: {result.confidence:.2f})")
    print("✓ MetadataAggregator tests passed\n")


async def test_online_provider_real():
    """Test real online provider (optional - requires network)."""
    print("Testing real online provider (GetSongBPM)...")
    print("This requires network access and may take a few seconds...")

    try:
        async with GetSongBPMProvider() as provider:
            result = await provider.get_metadata(
                artist="Adam Beyer",
                title="Your Mind"
            )

            if result:
                print(f"✓ Found: {result.artist} - {result.title}")
                print(f"  BPM: {result.bpm}")
                print(f"  Key: {result.key}")
                print(f"  Confidence: {result.confidence}")
                print("✓ Online provider test passed\n")
            else:
                print("⚠ No results found (site may be unavailable)\n")
    except Exception as e:
        print(f"⚠ Skipped online test: {e}\n")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Metadata Provider System - Quick Tests")
    print("=" * 60)
    print()

    await test_track_metadata()
    await test_local_provider()
    await test_aggregator_mock()

    # Uncomment to test real online provider
    # await test_online_provider_real()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
