"""Metadata aggregator that combines results from multiple providers."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from .base import MetadataProvider, MetadataSource, TrackMetadata
from .online import BeatportProvider, GetSongBPMProvider, TunebatProvider

logger = logging.getLogger(__name__)


class MetadataAggregator:
    """
    Aggregate metadata from multiple providers with caching and consensus logic.
    
    This aggregator:
    - Queries multiple providers concurrently
    - Caches results to disk
    - Computes consensus BPM from multiple sources
    - Handles provider failures gracefully
    """

    def __init__(
        self,
        providers: Optional[list[MetadataProvider]] = None,
        cache_path: Optional[Path] = None,
        enable_cache: bool = True,
    ):
        """
        Initialize metadata aggregator.
        
        Args:
            providers: List of metadata providers to use (defaults to all online providers)
            cache_path: Path to cache file (defaults to bpm_cache.json in current dir)
            enable_cache: Whether to use caching
        """
        self.providers = providers or [
            GetSongBPMProvider(),
            TunebatProvider(),
            BeatportProvider(),
        ]
        self.cache_path = cache_path or Path("bpm_cache.json")
        self.enable_cache = enable_cache
        self._cache = self._load_cache() if enable_cache else {}

    def _load_cache(self) -> dict:
        """Load cache from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r') as f:
                    cache = json.load(f)
                    logger.debug(f"Loaded {len(cache)} cached entries from {self.cache_path}")
                    return cache
            except Exception as e:
                logger.warning(f"Could not load cache from {self.cache_path}: {e}")
        return {}

    def _save_cache(self):
        """Save cache to disk."""
        if not self.enable_cache:
            return

        try:
            with open(self.cache_path, 'w') as f:
                json.dump(self._cache, f, indent=2)
            logger.debug(f"Saved cache with {len(self._cache)} entries to {self.cache_path}")
        except Exception as e:
            logger.warning(f"Could not save cache to {self.cache_path}: {e}")

    def _get_cache_key(self, artist: str, title: str) -> str:
        """Generate cache key from artist and title."""
        return f"{artist.lower().strip()}|{title.lower().strip()}"

    async def get_metadata(
        self,
        artist: str,
        title: str,
        use_cache: bool = True,
        **kwargs
    ) -> Optional[TrackMetadata]:
        """
        Get aggregated metadata from multiple providers.
        
        Args:
            artist: Artist name
            title: Track title
            use_cache: Whether to check/update cache
            **kwargs: Additional arguments passed to providers
            
        Returns:
            Merged TrackMetadata with consensus values
        """
        cache_key = self._get_cache_key(artist, title)

        # Check cache first
        if use_cache and self.enable_cache and cache_key in self._cache:
            logger.debug(f"Cache hit for {artist} - {title}")
            cached = self._cache[cache_key]
            return TrackMetadata(
                artist=artist,
                title=title,
                bpm=cached.get('bpm'),
                key=cached.get('key'),
                genre=cached.get('genre'),
                duration=cached.get('duration'),
                source=MetadataSource.CACHED,
                confidence=cached.get('confidence', 1.0),
            )

        # Query all providers concurrently
        logger.info(f"Fetching metadata for {artist} - {title} from {len(self.providers)} providers")
        tasks = [
            provider.get_metadata(artist=artist, title=title, **kwargs)
            for provider in self.providers
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results
        metadata_list = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Provider {self.providers[i].source.value} failed: {result}")
            elif result is not None:
                metadata_list.append(result)
        
        if not metadata_list:
            logger.warning(f"No metadata found for {artist} - {title}")
            return None

        # If only a single provider returned a result, return it directly so
        # the provider's confidence is preserved (avoids fabricating higher
        # consensus confidence for single-source results).
        if len(metadata_list) == 1:
            logger.debug("Single provider returned metadata; returning source result directly")
            single = metadata_list[0]
            # Cache single-provider results as well when enabled
            if use_cache and self.enable_cache and single:
                self._cache[cache_key] = {
                    'bpm': single.bpm,
                    'key': single.key,
                    'genre': single.genre,
                    'duration': single.duration,
                    'confidence': single.confidence,
                    'sources': [single.source.value if single.source else None],
                }
                self._save_cache()
            return single

        # Merge results
        merged = self._merge_metadata(metadata_list, artist, title)

        # Cache result
        if use_cache and self.enable_cache and merged:
            self._cache[cache_key] = {
                'bpm': merged.bpm,
                'key': merged.key,
                'genre': merged.genre,
                'duration': merged.duration,
                'confidence': merged.confidence,
                'sources': [m.source.value for m in metadata_list],
            }
            self._save_cache()

        return merged

    def _merge_metadata(
        self,
        metadata_list: list[TrackMetadata],
        artist: str,
        title: str
    ) -> Optional[TrackMetadata]:
        """
        Merge metadata from multiple sources using confidence-weighted consensus.
        
        Args:
            metadata_list: List of TrackMetadata from different providers
            artist: Artist name for result
            title: Track title for result
            
        Returns:
            Merged TrackMetadata with consensus values
        """
        if not metadata_list:
            return None

        # Start with first result
        merged = TrackMetadata(
            artist=artist,
            title=title,
            source=MetadataSource.CACHED,  # Aggregated result
        )

        # Compute consensus BPM
        bpm_values = [(m.bpm, m.confidence) for m in metadata_list if m.bpm is not None]
        if bpm_values:
            # Use weighted average by confidence
            total_weight = sum(conf for _, conf in bpm_values)
            if total_weight > 0:
                weighted_bpm = sum(bpm * conf for bpm, conf in bpm_values) / total_weight
                merged.bpm = round(weighted_bpm, 1)
                
                # Confidence is higher when sources agree
                bpm_stdev = self._compute_stdev([bpm for bpm, _ in bpm_values])
                if bpm_stdev < 1.0:
                    merged.confidence = 0.95
                elif bpm_stdev < 3.0:
                    merged.confidence = 0.85
                else:
                    merged.confidence = 0.7

        # Take most confident key
        key_values = [(m.key, m.confidence) for m in metadata_list if m.key is not None]
        if key_values:
            key_values.sort(key=lambda x: x[1], reverse=True)
            merged.key = key_values[0][0]

        # Take most confident genre
        genre_values = [(m.genre, m.confidence) for m in metadata_list if m.genre is not None]
        if genre_values:
            genre_values.sort(key=lambda x: x[1], reverse=True)
            merged.genre = genre_values[0][0]

        # Average duration if available
        duration_values = [m.duration for m in metadata_list if m.duration is not None]
        if duration_values:
            merged.duration = sum(duration_values) / len(duration_values)

        logger.info(
            f"Merged metadata from {len(metadata_list)} sources: "
            f"BPM={merged.bpm}, Key={merged.key}, Confidence={merged.confidence:.2f}"
        )

        return merged

    def _compute_stdev(self, values: list[float]) -> float:
        """Compute standard deviation of values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    async def close(self):
        """Close all provider sessions."""
        for provider in self.providers:
            if hasattr(provider, 'close'):
                await provider.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
