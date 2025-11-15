"""Example usage of the metadata provider system."""

import asyncio
from pathlib import Path

from edm_cue_analyzer.metadata import (
    BeatportProvider,
    GetSongBPMProvider,
    LocalFileProvider,
    MetadataAggregator,
    TunebatProvider,
)


async def example_local_file():
    """Example: Read metadata from local file."""
    print("=" * 60)
    print("Example 1: Local File Metadata")
    print("=" * 60)
    
    provider = LocalFileProvider()
    
    # Read from a local audio file
    file_path = Path("/music/Artist - Track.flac")  # Adjust to your file
    if file_path.exists():
        metadata = await provider.get_metadata(file_path=file_path)
        if metadata:
            print(f"Artist: {metadata.artist}")
            print(f"Title: {metadata.title}")
            print(f"Album: {metadata.album}")
            print(f"BPM: {metadata.bpm}")
            print(f"Duration: {metadata.duration}s")
            print(f"Source: {metadata.source.value}")
        else:
            print("Could not read metadata")
    else:
        print(f"File not found: {file_path}")
    
    print()


async def example_online_single():
    """Example: Fetch metadata from a single online provider."""
    print("=" * 60)
    print("Example 2: Single Online Provider")
    print("=" * 60)
    
    provider = GetSongBPMProvider()
    
    try:
        metadata = await provider.get_metadata(
            artist="Adam Beyer",
            title="Your Mind"
        )
        
        if metadata:
            print(f"Artist: {metadata.artist}")
            print(f"Title: {metadata.title}")
            print(f"BPM: {metadata.bpm}")
            """Removed

            This metadata usage example has been removed. See `docs/ARCHITECTURE.md`
            for the consolidated architecture and metadata provider pattern.
            """
    finally:
