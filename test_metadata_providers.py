#!/usr/bin/env python3
"""
Simple script to test metadata providers.

Usage:
    python test_metadata_providers.py "Artist Name" "Track Title"
    python test_metadata_providers.py --file /path/to/track.flac
"""

import argparse
import asyncio
import sys
from pathlib import Path

from src.edm_cue_analyzer.metadata import (
    BeatportProvider,
    GetSongBPMProvider,
    LocalFileProvider,
    MetadataAggregator,
    TunebatProvider,
)


async def test_local_file(file_path: Path):
    """Test local file provider."""
    print(f"\n{'=' * 60}")
    print(f"Testing LocalFileProvider")
    print(f"File: {file_path}")
    print('=' * 60)

    provider = LocalFileProvider()
    metadata = await provider.get_metadata(file_path=file_path)

    if metadata:
        print(f"✓ Artist: {metadata.artist}")
        print(f"✓ Title: {metadata.title}")
        print(f"✓ Album: {metadata.album}")
        print(f"✓ BPM: {metadata.bpm}")
        print(f"✓ Key: {metadata.key}")
        print(f"✓ Genre: {metadata.genre}")
        print(f"✓ Duration: {metadata.duration:.1f}s" if metadata.duration else "✓ Duration: None")
        print(f"✓ Sample Rate: {metadata.sample_rate}Hz" if metadata.sample_rate else "✓ Sample Rate: None")
        print(f"✓ Source: {metadata.source.value}")
    else:
        print("✗ Failed to read metadata")

    return metadata


async def test_online_provider(provider_class, artist: str, title: str):
    """Test a single online provider."""
    provider_name = provider_class.__name__
    print(f"\n{'=' * 60}")
    print(f"Testing {provider_name}")
    print(f"Query: {artist} - {title}")
    print('=' * 60)

    async with provider_class() as provider:
        metadata = await provider.get_metadata(artist=artist, title=title)

        if metadata:
            print(f"✓ BPM: {metadata.bpm}")
            print(f"✓ Key: {metadata.key}")
            print(f"✓ Genre: {metadata.genre}" if metadata.genre else "✗ Genre: Not found")
            print(f"✓ Duration: {metadata.duration:.1f}s" if metadata.duration else "✗ Duration: Not found")
            print(f"✓ Confidence: {metadata.confidence:.2f}")
            print(f"✓ Source: {metadata.source.value}")
        else:
            print(f"✗ No metadata found")

    return metadata


async def test_aggregator(artist: str, title: str):
    """Test metadata aggregator."""
    print(f"\n{'=' * 60}")
    print(f"Testing MetadataAggregator (All Sources)")
    print(f"Query: {artist} - {title}")
    print('=' * 60)

    async with MetadataAggregator() as aggregator:
        metadata = await aggregator.get_metadata(artist=artist, title=title)

        if metadata:
            print(f"✓ BPM: {metadata.bpm}")
            print(f"✓ Key: {metadata.key}")
            print(f"✓ Genre: {metadata.genre}" if metadata.genre else "✗ Genre: Not found")
            print(f"✓ Duration: {metadata.duration:.1f}s" if metadata.duration else "✗ Duration: Not found")
            print(f"✓ Confidence: {metadata.confidence:.2f}")
            print(f"✓ Source: {metadata.source.value}")

            # Test cache on second call
            print("\n--- Testing cache ---")
            metadata2 = await aggregator.get_metadata(artist=artist, title=title)
            if metadata2.source.value == "cached":
                print("✓ Cache is working!")
            else:
                print(f"✗ Cache not used (source: {metadata2.source.value})")
        else:
            print(f"✗ No metadata found")

    return metadata


async def test_all_online(artist: str, title: str):
    """Test all online providers individually."""
    results = {}

    # Test each provider
    results['getsongbpm'] = await test_online_provider(GetSongBPMProvider, artist, title)
    results['tunebat'] = await test_online_provider(TunebatProvider, artist, title)
    results['beatport'] = await test_online_provider(BeatportProvider, artist, title)

    # Test aggregator
    results['aggregated'] = await test_aggregator(artist, title)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print('=' * 60)

    found_count = sum(1 for v in results.values() if v is not None)
    print(f"Providers with results: {found_count}/{len(results)}")

    if results['aggregated']:
        print(f"\nFinal aggregated BPM: {results['aggregated'].bpm}")
        print(f"Final confidence: {results['aggregated'].confidence:.2f}")

    # Compare BPMs if we got results
    bpms = [r.bpm for r in results.values() if r and r.bpm]
    if len(bpms) > 1:
        avg_bpm = sum(bpms) / len(bpms)
        max_diff = max(bpms) - min(bpms)
        print(f"\nBPM range: {min(bpms):.1f} - {max(bpms):.1f}")
        print(f"Average BPM: {avg_bpm:.1f}")
        print(f"Max difference: {max_diff:.1f} BPM")

        if max_diff < 1.0:
            print("✓ Excellent agreement!")
        elif max_diff < 3.0:
            print("✓ Good agreement")
        else:
            print("⚠ Sources disagree significantly")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test metadata providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test online providers
  python test_metadata_providers.py "Adam Beyer" "Your Mind"
  
  # Test local file
  python test_metadata_providers.py --file /music/track.flac
  
  # Test both
  python test_metadata_providers.py "Artist" "Title" --file /music/track.flac
        """
    )

    parser.add_argument('artist', nargs='?', help='Artist name')
    parser.add_argument('title', nargs='?', help='Track title')
    parser.add_argument('--file', '-f', type=Path, help='Path to audio file')

    args = parser.parse_args()

    if not args.file and not (args.artist and args.title):
        parser.error("Either --file or both artist and title are required")

    # Test local file if provided
    if args.file:
        if not args.file.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            return 1

        local_meta = await test_local_file(args.file)

        # Use artist/title from file if not provided
        if local_meta and not (args.artist and args.title):
            if local_meta.artist and local_meta.title:
                args.artist = local_meta.artist
                args.title = local_meta.title
                print(f"\n→ Using artist/title from file: {args.artist} - {args.title}")
            else:
                print("\nError: Could not extract artist/title from file tags", file=sys.stderr)
                return 1

    # Test online providers if we have artist/title
    if args.artist and args.title:
        await test_all_online(args.artist, args.title)

    return 0


if __name__ == '__main__':
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
