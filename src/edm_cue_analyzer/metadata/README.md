# Metadata Provider System

This module provides a unified, extensible architecture for retrieving music track metadata from various sources.

## Architecture

The metadata system uses a **provider pattern** with the following components:

### Core Classes

- **`MetadataProvider`** (ABC): Abstract base class defining the interface all providers must implement
- **`TrackMetadata`**: Dataclass holding track information (artist, title, BPM, key, etc.)
- **`MetadataSource`**: Enum identifying the source of metadata (local file, online database, etc.)
- **`MetadataAggregator`**: Combines results from multiple providers with caching and consensus logic

### Providers

#### Local Providers

- **`LocalFileProvider`**: Extracts metadata from audio file tags using `mutagen` and `soundfile`
  - Reads ID3/Vorbis/MP4 tags
  - Parses artist/title from filenames as fallback
  - Extracts duration and sample rate

#### Online Providers

- **`GetSongBPMProvider`**: Fetches BPM and key from GetSongBPM.com
- **`TunebatProvider`**: Fetches BPM, key, and duration from Tunebat.com
- **`BeatportProvider`**: Fetches BPM, key, and genre from Beatport.com

All online providers:
- Use async HTTP requests with `aiohttp`
- Implement rate limiting to avoid being blocked
- Handle errors gracefully
- Return confidence scores

## Usage Examples

### Basic Local File Reading

```python
from pathlib import Path
from edm_cue_analyzer.metadata import LocalFileProvider

provider = LocalFileProvider()
metadata = await provider.get_metadata(file_path=Path("track.flac"))

print(f"{metadata.artist} - {metadata.title}")
print(f"BPM: {metadata.bpm}")
```

### Single Online Provider

```python
from edm_cue_analyzer.metadata import GetSongBPMProvider

async with GetSongBPMProvider() as provider:
    metadata = await provider.get_metadata(
        artist="Adam Beyer",
        title="Your Mind"
    )
    print(f"BPM: {metadata.bpm}")
```

### Aggregating Multiple Sources

```python
from edm_cue_analyzer.metadata import MetadataAggregator

async with MetadataAggregator() as aggregator:
    # Queries GetSongBPM, Tunebat, and Beatport concurrently
    metadata = await aggregator.get_metadata(
        artist="Charlotte de Witte",
        title="Selected"
    )

    # Results are cached automatically
    print(f"BPM: {metadata.bpm} (confidence: {metadata.confidence:.2f})")
```

### Custom Provider Selection

```python
from edm_cue_analyzer.metadata import (
    MetadataAggregator,
    BeatportProvider,
    TunebatProvider,
)

# Only use specific providers
providers = [
    BeatportProvider(rate_limit_delay=2.0),
    TunebatProvider(),
]

async with MetadataAggregator(providers=providers) as aggregator:
    metadata = await aggregator.get_metadata(
        artist="Artist Name",
        title="Track Title"
    )
```

### Merging Local and Online Metadata

```python
from pathlib import Path
from edm_cue_analyzer.metadata import (
    LocalFileProvider,
    MetadataAggregator,
)

# Get local file metadata
local_provider = LocalFileProvider()
local_meta = await local_provider.get_metadata(file_path=Path("track.flac"))

# Get online metadata
async with MetadataAggregator() as aggregator:
    online_meta = await aggregator.get_metadata(
        artist=local_meta.artist,
        title=local_meta.title
    )

    # Merge: online BPM takes priority, local duration is kept
    merged = local_meta.merge(online_meta)
    print(f"Final BPM: {merged.bpm}")
```

## Features

### Caching

The `MetadataAggregator` automatically caches results to disk (`bpm_cache.json` by default):

```python
aggregator = MetadataAggregator(
    cache_path=Path("my_cache.json"),
    enable_cache=True
)

# First call: queries online sources
metadata1 = await aggregator.get_metadata("Artist", "Title")

# Second call: returns cached result instantly
metadata2 = await aggregator.get_metadata("Artist", "Title")
assert metadata2.source == MetadataSource.CACHED
```

### Consensus Algorithm

When multiple providers return different BPM values, the aggregator computes a consensus:

1. **Weighted Average**: Each source's BPM is weighted by its confidence score
2. **Agreement Bonus**: If sources agree closely (stdev < 1 BPM), confidence increases to 0.95
3. **Disagreement Penalty**: If sources disagree significantly (stdev > 3 BPM), confidence drops to 0.7

### Rate Limiting

Online providers implement configurable rate limiting:

```python
provider = GetSongBPMProvider(
    timeout=15.0,           # Request timeout in seconds
    rate_limit_delay=2.0    # Minimum delay between requests
)
```

### Error Handling

All providers handle errors gracefully:
- Network timeouts return `None`
- HTML parsing errors are logged but don't crash
- Failed providers in aggregator don't prevent other providers from running

## Extending

### Adding a New Online Provider

```python
from edm_cue_analyzer.metadata.online import OnlineProvider
from edm_cue_analyzer.metadata import MetadataSource, TrackMetadata

class MyMusicDBProvider(OnlineProvider):
    @property
    def source(self) -> MetadataSource:
        return MetadataSource.MUSICBRAINZ  # or add new enum value

    async def get_metadata(
        self,
        artist: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> Optional[TrackMetadata]:
        if not artist or not title:
            return None

        await self._rate_limit()
        session = await self._get_session()

        # Your API/scraping logic here
        async with session.get(f"https://api.example.com/search?q={artist}+{title}") as resp:
            data = await resp.json()

            return TrackMetadata(
                artist=artist,
                title=title,
                bpm=data['bpm'],
                source=self.source,
                confidence=0.8
            )
```

### Adding a New Local Provider

```python
from edm_cue_analyzer.metadata import MetadataProvider, MetadataSource, TrackMetadata

class RekordboxXMLProvider(MetadataProvider):
    """Read metadata from Rekordbox XML export."""

    @property
    def source(self) -> MetadataSource:
        return MetadataSource.LOCAL_FILE

    @property
    def requires_network(self) -> bool:
        return False

    async def get_metadata(self, xml_path: Path, track_id: str, **kwargs) -> Optional[TrackMetadata]:
        # Parse XML and extract metadata
        # ...
        return TrackMetadata(...)
```

## Dependencies

### Required
- `soundfile` - Audio file reading

### Optional (for full functionality)
- `mutagen` - Audio tag reading (install with `pip install edm-cue-analyzer[validation]`)
- `aiohttp` - Async HTTP requests (install with `pip install edm-cue-analyzer[validation]`)

## Thread Safety

All providers are async-safe and can be used concurrently:

```python
async with MetadataAggregator() as agg:
    # Query multiple tracks concurrently
    results = await asyncio.gather(
        agg.get_metadata("Artist 1", "Track 1"),
        agg.get_metadata("Artist 2", "Track 2"),
        agg.get_metadata("Artist 3", "Track 3"),
    )
```

However, each provider instance should not be shared across event loops. Create separate instances if needed.

## See Also

- `examples/metadata_usage.py` - Complete working examples
- `BPM_VALIDATION.md` - Documentation of the previous ad-hoc validation approach
- `tests/test_metadata.py` - Unit tests for the metadata system
