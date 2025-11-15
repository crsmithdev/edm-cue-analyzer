# Removed

This document has been removed. See `docs/ARCHITECTURE.md` for the
consolidated architecture overview and the metadata provider pattern that
the project uses.
    @property
    def requires_network(self) -> bool:
        """Whether this provider needs internet access"""
        return False
```

#### `TrackMetadata` (Dataclass)
Holds track information with a `merge()` method for combining data:

```python
@dataclass
class TrackMetadata:
    artist: Optional[str] = None
    title: Optional[str] = None
    album: Optional[str] = None
    bpm: Optional[float] = None
    key: Optional[str] = None
    duration: Optional[float] = None
    genre: Optional[str] = None
    source: MetadataSource = MetadataSource.LOCAL_FILE
    confidence: float = 1.0
    file_path: Optional[Path] = None
    sample_rate: Optional[int] = None
```

#### `MetadataSource` (Enum)
Identifies where metadata came from:

```python
class MetadataSource(Enum):
    LOCAL_FILE = "local_file"
    LOCAL_ANALYSIS = "local_analysis"
    GETSONGBPM = "getsongbpm"
    TUNEBAT = "tunebat"
    BEATPORT = "beatport"
    DISCOGS = "discogs"
    MUSICBRAINZ = "musicbrainz"
    CACHED = "cached"
```

### 2. Local Providers (`metadata/local.py`)

#### `LocalFileProvider`
Extracts metadata from audio file tags using `mutagen` and `soundfile`:

- Reads ID3/Vorbis/MP4 tags for artist, title, album, genre, BPM
- Gets audio properties (duration, sample rate) with `soundfile`
- Falls back to parsing filenames for artist/title
- Handles various filename formats: `"Artist - Title.flac"`, `"01. Artist - Title.mp3"`, etc.

### 3. Online Providers (`metadata/online.py`)

#### Base Class: `OnlineProvider`
Common functionality for all online sources:

- Async HTTP session management with `aiohttp`
- Configurable rate limiting
- Timeout handling
- User-Agent headers

#### `GetSongBPMProvider`
Scrapes GetSongBPM.com:
- Parses HTML for BPM and key information
- Confidence: 0.8 (generally reliable)

#### `TunebatProvider`
Scrapes Tunebat.com:
- Parses BPM, key, and duration
- Confidence: 0.85 (very reliable)

#### `BeatportProvider`
Scrapes Beatport.com:
- Parses BPM, key, and genre
- Confidence: 0.9 (highly reliable for EDM)

All online providers:
- Run concurrently without blocking
- Return `None` on errors (graceful degradation)
- Log warnings/errors for debugging

### 4. Aggregator (`metadata/aggregator.py`)

#### `MetadataAggregator`
Combines multiple providers with intelligent merging:

**Features:**
- **Concurrent Queries**: Queries all providers in parallel using `asyncio.gather()`
- **Consensus Algorithm**: Computes weighted average BPM based on confidence scores
- **Agreement Detection**: Increases confidence when sources agree (stdev < 1 BPM)
- **Caching**: Automatically caches results to disk (`bpm_cache.json`)
- **Graceful Failures**: Continues even if some providers fail

**Consensus Logic:**
```python
# Weighted average by confidence
weighted_bpm = sum(bpm * conf for bpm, conf in bpm_values) / total_weight

# Confidence based on agreement
if bpm_stdev < 1.0:
    confidence = 0.95  # High agreement
elif bpm_stdev < 3.0:
    confidence = 0.85  # Moderate agreement
else:
    confidence = 0.7   # Low agreement
```

## Usage Patterns

### Pattern 1: Local File Only

```python
provider = LocalFileProvider()
metadata = await provider.get_metadata(file_path=Path("track.flac"))
```

### Pattern 2: Single Online Source

```python
async with GetSongBPMProvider() as provider:
    metadata = await provider.get_metadata(
        artist="Adam Beyer",
        title="Your Mind"
    )
```

### Pattern 3: Multi-Source Consensus

```python
async with MetadataAggregator() as aggregator:
    metadata = await aggregator.get_metadata(
        artist="Charlotte de Witte",
        title="Selected"
    )
    # Automatically queries all providers and computes consensus
```

### Pattern 4: Custom Provider Selection

```python
providers = [
    BeatportProvider(rate_limit_delay=2.0),
    TunebatProvider(),
]

async with MetadataAggregator(providers=providers) as aggregator:
    metadata = await aggregator.get_metadata(artist, title)
```

### Pattern 5: Local + Online Merge

```python
# Get local metadata
local_meta = await LocalFileProvider().get_metadata(file_path=path)

# Get online metadata
async with MetadataAggregator() as agg:
    online_meta = await agg.get_metadata(
        artist=local_meta.artist,
        title=local_meta.title
    )
    
# Merge: online BPM preferred, local duration kept
merged = local_meta.merge(online_meta)
```

## Benefits

### 1. **Extensibility**
Adding a new provider is simple - just implement the `MetadataProvider` interface:

```python
class SpotifyProvider(OnlineProvider):
    @property
    def source(self) -> MetadataSource:
        return MetadataSource.SPOTIFY
    
    async def get_metadata(self, artist, title, **kwargs):
        # Implementation
        pass
```

### 2. **Testability**
Each provider can be tested independently. Mock providers can be created for testing:

```python
class MockProvider(MetadataProvider):
    def __init__(self, return_value):
        self._return_value = return_value
    
    async def get_metadata(self, **kwargs):
        return self._return_value
```

### 3. **Maintainability**
- Clear separation of concerns
- Each provider handles one source
- Common HTTP logic in base class
- Aggregation logic isolated

### 4. **Reusability**
The same providers can be used for:
- BPM validation (original use case)
- Track analysis
- Library management
- Cue point generation with online metadata
- Any other metadata needs

### 5. **Performance**
- Concurrent queries via async/await
- Automatic caching reduces redundant requests
- Rate limiting prevents IP bans
- Graceful degradation on failures

## Migration Path

For existing code using ad-hoc BPM fetching:

**Before:**
```python
# Direct scraping with no abstraction
bpm = fetch_bpm_from_getsongbpm(artist, title)
if not bpm:
    bpm = fetch_bpm_from_tunebat(artist, title)
```

**After:**
```python
async with MetadataAggregator() as agg:
    metadata = await agg.get_metadata(artist, title)
    bpm = metadata.bpm
```

## Future Enhancements

### Potential New Providers

1. **MusicBrainz Provider**: Structured database with API
2. **Discogs Provider**: Extensive electronic music catalog
3. **Spotify Provider**: Official API with audio features
4. **Local Analysis Provider**: Run BPM detection as a provider
5. **Rekordbox XML Provider**: Read from exported playlists
6. **Serato Provider**: Read from Serato library

### Additional Features

1. **Provider Priority**: Allow configuring which sources are most trustworthy
2. **Batch Operations**: Efficient bulk metadata fetching
3. **Webhooks/Callbacks**: Notify on metadata updates
4. **Historical Tracking**: Track metadata changes over time
5. **Conflict Resolution UI**: Let users choose when sources disagree

## File Structure

```
src/edm_cue_analyzer/metadata/
├── __init__.py           # Package exports
├── base.py               # Core abstractions
├── local.py              # Local file providers
├── online.py             # Online source providers
├── aggregator.py         # Multi-source aggregation
└── README.md             # Usage documentation

examples/
└── metadata_usage.py     # Complete working examples

docs/
└── METADATA_ARCHITECTURE.md  # This file
```

## Dependencies

- **Core**: `soundfile` (already required)
- **Tags**: `mutagen` (optional, in `validation` extras)
- **Online**: `aiohttp` (optional, in `validation` extras)

Install full functionality:
```bash
pip install edm-cue-analyzer[validation]
```

## Testing

Unit tests should cover:

1. **Provider Interface**: Each provider implements the contract
2. **Local Reading**: Tag parsing, filename parsing, fallbacks
3. **Online Scraping**: HTML parsing, error handling, rate limiting
4. **Aggregation**: Consensus algorithm, caching, concurrent execution
5. **Merging**: TrackMetadata.merge() logic

Integration tests should verify:
- End-to-end metadata fetching
- Cache persistence
- Real HTTP requests (with mocking option)

## Conclusion

This refactoring transforms ad-hoc metadata fetching into a clean, extensible architecture that:

- **Encapsulates** local and online metadata sources behind a common interface
- **Enables** easy addition of new providers
- **Improves** code reusability across the project
- **Maintains** backward compatibility through similar APIs
- **Enhances** testability and maintainability

The provider pattern is a well-established design approach that scales well as the project grows and new metadata sources are needed.
