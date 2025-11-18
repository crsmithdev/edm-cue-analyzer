"""Base classes for metadata providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class MetadataSource(Enum):
    """Source of metadata."""

    LOCAL_FILE = "local_file"
    LOCAL_ANALYSIS = "local_analysis"
    GETSONGBPM = "getsongbpm"
    TUNEBAT = "tunebat"
    BEATPORT = "beatport"
    DISCOGS = "discogs"
    MUSICBRAINZ = "musicbrainz"
    CACHED = "cached"


@dataclass
class TrackMetadata:
    """Container for track metadata from any source."""

    # Identifiers
    artist: str | None = None
    title: str | None = None
    album: str | None = None
    file_path: Path | None = None

    # Audio properties
    bpm: float | None = None
    key: str | None = None
    duration: float | None = None  # seconds
    sample_rate: int | None = None
    energy: float | None = None  # Spotify energy (0.0 to 1.0)

    # Genre/style
    genre: str | None = None
    tags: list[str] = field(default_factory=list)

    # Metadata about the metadata
    source: MetadataSource | None = None
    confidence: float = 1.0  # 0.0 to 1.0
    timestamp: str | None = None  # ISO format

    # Additional data (provider-specific)
    extra: dict = field(default_factory=dict)

    def merge(self, other: "TrackMetadata", prefer_other: bool = False) -> "TrackMetadata":
        """
        Merge this metadata with another, filling in missing fields.

        Args:
            other: Another TrackMetadata to merge with
            prefer_other: If True, prefer values from 'other' when both exist

        Returns:
            New TrackMetadata with merged values
        """
        def choose(ours, theirs):
            # If prefer_other is explicit, use other's value when present
            if prefer_other and theirs is not None:
                return theirs

            # If both are present, prefer the value from the metadata with
            # higher confidence (common-case: online providers have higher
            # confidence than local file tags).
            if ours is not None and theirs is not None:
                try:
                    # Prefer 'theirs' if it has higher confidence
                    if getattr(other, 'confidence', 0.0) > getattr(self, 'confidence', 0.0):
                        return theirs
                except Exception:
                    pass
                return ours

            return ours if ours is not None else theirs

        return TrackMetadata(
            artist=choose(self.artist, other.artist),
            title=choose(self.title, other.title),
            album=choose(self.album, other.album),
            file_path=choose(self.file_path, other.file_path),
            bpm=choose(self.bpm, other.bpm),
            key=choose(self.key, other.key),
            duration=choose(self.duration, other.duration),
            sample_rate=choose(self.sample_rate, other.sample_rate),
            genre=choose(self.genre, other.genre),
            tags=list(set(self.tags + other.tags)),  # Combine tags
            source=choose(self.source, other.source),
            # Confidence should reflect the most confident source unless
            # prefer_other forces the other value.
            confidence=(other.confidence if prefer_other else max(self.confidence, other.confidence)),
            timestamp=choose(self.timestamp, other.timestamp),
            extra={**self.extra, **other.extra},  # Merge extra
        )


class MetadataProvider(ABC):
    """Abstract base class for metadata providers."""

    @abstractmethod
    async def get_metadata(
        self,
        artist: str | None = None,
        title: str | None = None,
        file_path: Path | None = None,
        **kwargs
    ) -> TrackMetadata | None:
        """
        Fetch metadata for a track.

        Args:
            artist: Artist name (optional)
            title: Track title (optional)
            file_path: Path to audio file (optional)
            **kwargs: Provider-specific parameters

        Returns:
            TrackMetadata if found, None otherwise
        """
        pass

    @property
    @abstractmethod
    def source(self) -> MetadataSource:
        """Return the metadata source this provider represents."""
        pass

    @property
    def requires_network(self) -> bool:
        """Return True if this provider needs network access."""
        return False
