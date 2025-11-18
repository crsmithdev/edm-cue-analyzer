"""Site-level Spotify helpers.

This module contains convenience functions for mapping site-level Spotify
URLs or metadata to our internal TrackMetadata model. It intentionally
keeps logic separate from API provider code to allow future scraping or
site-specific fallbacks.
"""

from __future__ import annotations

import logging

from .base import MetadataSource, TrackMetadata

logger = logging.getLogger(__name__)


def parse_spotify_url(url: str) -> dict | None:
    """Parse a Spotify track URL and return a minimal identifier dict.

    Returns {'track_id': ...} on success or None.
    """
    # Basic handling for URLs like https://open.spotify.com/track/<id>
    try:
        if "/track/" in url:
            track_id = url.rstrip("/\n ").split("/track/")[-1].split("?")[0]
            return {"track_id": track_id}
    except Exception:
        logger.debug("Failed to parse Spotify URL: %s", url)
    return None


def track_to_metadata(artist: str, title: str, bpm: float | None = None) -> TrackMetadata:
    """Create a TrackMetadata instance from simple Spotify-derived values."""
    return TrackMetadata(
        artist=artist,
        title=title,
        bpm=float(bpm) if bpm is not None else None,
        key=None,
        genre=None,
        duration=None,
        source=MetadataSource.ONLINE,
        confidence=0.5,
    )
