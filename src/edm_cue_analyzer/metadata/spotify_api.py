"""Spotify API metadata provider.

This provider attempts to fetch track metadata (tempo, duration, key, etc.)
using the Spotify Web API. It is implemented as a best-effort, optional
provider that will be a no-op if the `spotipy` library or credentials are
not available.

API spec summary (for maintainers):
- Requires Spotify API credentials available through environment variables:
  SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET (or an already-configured
  spotipy credentials manager).
- Searches for a track by "artist - title" using the Search API.
- If a matching track is found, uses the Audio Features endpoint to obtain
  tempo (BPM), key, mode and duration_ms.

Return shape: returns a `TrackMetadata` instance (from metadata.base) or
None if no metadata is found or on error.
"""

from __future__ import annotations

import logging
import os

from .base import MetadataProvider, MetadataSource, TrackMetadata

logger = logging.getLogger(__name__)


class SpotifyAPIProvider(MetadataProvider):
    """Provider that queries the Spotify Web API using spotipy (optional).

    This implementation is conservative:
    - If `spotipy` isn't installed or credentials aren't present it returns None.
    - If a result is found, it maps Spotify's audio_features fields to TrackMetadata.
    """

    @property
    def source(self) -> MetadataSource:
        return MetadataSource.ONLINE

    async def get_metadata(self, artist: str, title: str, **kwargs) -> TrackMetadata | None:
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyClientCredentials
        except Exception:
            logger.debug("spotipy not available; SpotifyAPIProvider disabled")
            return None

        # Credentials via environment variables are the simplest path
        client_id = os.getenv("SPOTIFY_CLIENT_ID")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        if not client_id or not client_secret:
            logger.debug("Spotify credentials not configured; skipping Spotify API provider")
            return None

        try:
            creds = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
            sp = spotipy.Spotify(auth_manager=creds)

            query = f"artist:{artist} track:{title}"
            res = sp.search(q=query, type="track", limit=1)
            items = res.get("tracks", {}).get("items", [])
            if not items:
                logger.debug("Spotify: no match for %s - %s", artist, title)
                return None

            track = items[0]
            track_id = track.get("id")
            features = sp.audio_features([track_id])[0]
            if not features:
                return None

            bpm = features.get("tempo")
            duration_ms = features.get("duration_ms")
            key = features.get("key")
            mode = features.get("mode")
            energy = features.get("energy")  # 0.0 to 1.0

            meta = TrackMetadata(
                artist=artist,
                title=title,
                bpm=float(bpm) if bpm is not None else None,
                key=int(key) if key is not None else None,
                genre=None,
                duration=(duration_ms / 1000.0) if duration_ms else None,
                energy=float(energy) if energy is not None else None,
                source=self.source,
                confidence=0.85,
            )

            return meta
        except Exception as e:
            logger.debug("SpotifyAPIProvider error: %s", e)
            return None
