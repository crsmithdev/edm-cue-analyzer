"""Music metadata providers and sources."""

from .aggregator import MetadataAggregator
from .base import MetadataProvider, MetadataSource, TrackMetadata
from .local import LocalFileProvider
from .online import (
    BeatportProvider,
    GetSongBPMProvider,
    TunebatProvider,
)
from .spotify_api import SpotifyAPIProvider
from .spotify import parse_spotify_url, track_to_metadata

__all__ = [
    "MetadataProvider",
    "MetadataSource",
    "TrackMetadata",
    "LocalFileProvider",
    "GetSongBPMProvider",
    "TunebatProvider",
    "BeatportProvider",
    "SpotifyAPIProvider",
    "parse_spotify_url",
    "track_to_metadata",
    "MetadataAggregator",
]
