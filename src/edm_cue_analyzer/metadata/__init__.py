"""Music metadata providers and sources."""

from .aggregator import MetadataAggregator
from .base import MetadataProvider, MetadataSource, TrackMetadata
from .local import LocalFileProvider
from .online import (
    BeatportProvider,
    GetSongBPMProvider,
    TunebatProvider,
)

__all__ = [
    "MetadataProvider",
    "MetadataSource",
    "TrackMetadata",
    "LocalFileProvider",
    "GetSongBPMProvider",
    "TunebatProvider",
    "BeatportProvider",
    "MetadataAggregator",
]
