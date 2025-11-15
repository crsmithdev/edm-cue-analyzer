"""Online metadata providers for fetching track information from web sources."""

import asyncio
import logging
import re
from typing import Optional
from urllib.parse import quote

import aiohttp

from .base import MetadataProvider, MetadataSource, TrackMetadata

logger = logging.getLogger(__name__)


class OnlineProvider(MetadataProvider):
    """Base class for online metadata providers with HTTP client management."""

    def __init__(self, timeout: float = 10.0, rate_limit_delay: float = 1.0):
        """
        Initialize online provider.
        
        Args:
            timeout: HTTP request timeout in seconds
            rate_limit_delay: Delay between requests to avoid rate limiting
        """
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0.0
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def requires_network(self) -> bool:
        return True

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={'User-Agent': 'EDM-Cue-Analyzer/1.0'}
            )
        return self._session

    async def _rate_limit(self):
        """Apply rate limiting between requests."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self._last_request_time = asyncio.get_event_loop().time()

    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    # Provide default implementations so the base OnlineProvider can be
    # instantiated for session-management tests. Subclasses should override
    # these with real implementations.
    @property
    def source(self) -> MetadataSource:
        """Default source placeholder for base class (override in subclasses)."""
        return MetadataSource.CACHED

    async def get_metadata(self, **kwargs) -> Optional[TrackMetadata]:
        """Base get_metadata raises NotImplementedError by default."""
        raise NotImplementedError("OnlineProvider.get_metadata must be implemented by subclasses")


class GetSongBPMProvider(OnlineProvider):
    """Fetch BPM from GetSongBPM.com."""

    BASE_URL = "https://getsongbpm.com"

    @property
    def source(self) -> MetadataSource:
        return MetadataSource.GETSONGBPM

    async def get_metadata(
        self,
        artist: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> Optional[TrackMetadata]:
        """
        Search GetSongBPM for track metadata.
        
        Args:
            artist: Artist name (required)
            title: Track title (required)
            
        Returns:
            TrackMetadata with BPM and key if found
        """
        if not artist or not title:
            logger.warning("GetSongBPM requires both artist and title")
            return None

        await self._rate_limit()
        
        # Build search URL
        query = f"{artist} {title}".strip()
        search_url = f"{self.BASE_URL}/search?q={quote(query)}"

        try:
            session = await self._get_session()
            async with session.get(search_url) as response:
                if response.status != 200:
                    logger.debug(f"GetSongBPM returned status {response.status}")
                    return None

                html = await response.text()
                
                # Parse BPM from HTML
                bpm_match = re.search(r'<span[^>]*>(\d+)\s*BPM</span>', html, re.IGNORECASE)
                key_match = re.search(r'<span[^>]*>([A-G][#♯♭b]?\s*(?:Major|Minor|maj|min))</span>', html, re.IGNORECASE)

                if bpm_match:
                    metadata = TrackMetadata(
                        artist=artist,
                        title=title,
                        bpm=float(bpm_match.group(1)),
                        source=self.source,
                        confidence=0.8  # GetSongBPM is generally reliable
                    )
                    
                    if key_match:
                        metadata.key = key_match.group(1).strip()
                    
                    logger.debug(f"GetSongBPM found: {metadata.bpm} BPM")
                    return metadata

                logger.debug(f"No BPM found on GetSongBPM for {artist} - {title}")
                return None

        except asyncio.TimeoutError:
            logger.warning(f"GetSongBPM request timed out for {artist} - {title}")
            return None
        except Exception as e:
            logger.error(f"GetSongBPM error for {artist} - {title}: {e}")
            return None


class TunebatProvider(OnlineProvider):
    """Fetch BPM and key from Tunebat.com."""

    BASE_URL = "https://tunebat.com"

    @property
    def source(self) -> MetadataSource:
        return MetadataSource.TUNEBAT

    async def get_metadata(
        self,
        artist: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> Optional[TrackMetadata]:
        """
        Search Tunebat for track metadata.
        
        Args:
            artist: Artist name (required)
            title: Track title (required)
            
        Returns:
            TrackMetadata with BPM, key, and duration if found
        """
        if not artist or not title:
            logger.warning("Tunebat requires both artist and title")
            return None

        await self._rate_limit()
        
        # Build search URL
        query = f"{artist} {title}".strip()
        search_url = f"{self.BASE_URL}/Search?q={quote(query)}"

        try:
            session = await self._get_session()
            async with session.get(search_url) as response:
                if response.status != 200:
                    logger.debug(f"Tunebat returned status {response.status}")
                    return None

                html = await response.text()
                
                # Parse BPM, key, and duration from HTML
                bpm_match = re.search(r'<div[^>]*>(\d+)\s*BPM</div>', html, re.IGNORECASE)
                key_match = re.search(r'<div[^>]*>([A-G][#♯♭b]?\s*(?:Major|Minor))</div>', html, re.IGNORECASE)
                duration_match = re.search(r'<div[^>]*>(\d+):(\d+)</div>', html)

                if bpm_match:
                    metadata = TrackMetadata(
                        artist=artist,
                        title=title,
                        bpm=float(bpm_match.group(1)),
                        source=self.source,
                        confidence=0.85  # Tunebat is very reliable
                    )
                    
                    if key_match:
                        metadata.key = key_match.group(1).strip()
                    
                    if duration_match:
                        minutes = int(duration_match.group(1))
                        seconds = int(duration_match.group(2))
                        metadata.duration = minutes * 60 + seconds
                    
                    logger.debug(f"Tunebat found: {metadata.bpm} BPM")
                    return metadata

                logger.debug(f"No BPM found on Tunebat for {artist} - {title}")
                return None

        except asyncio.TimeoutError:
            logger.warning(f"Tunebat request timed out for {artist} - {title}")
            return None
        except Exception as e:
            logger.error(f"Tunebat error for {artist} - {title}: {e}")
            return None


class BeatportProvider(OnlineProvider):
    """Fetch BPM and key from Beatport.com."""

    BASE_URL = "https://www.beatport.com"

    @property
    def source(self) -> MetadataSource:
        return MetadataSource.BEATPORT

    async def get_metadata(
        self,
        artist: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> Optional[TrackMetadata]:
        """
        Search Beatport for track metadata.
        
        Args:
            artist: Artist name (required)
            title: Track title (required)
            
        Returns:
            TrackMetadata with BPM, key, genre if found
        """
        if not artist or not title:
            logger.warning("Beatport requires both artist and title")
            return None

        await self._rate_limit()
        
        # Build search URL
        query = f"{artist} {title}".strip()
        search_url = f"{self.BASE_URL}/search?q={quote(query)}"

        try:
            session = await self._get_session()
            async with session.get(search_url) as response:
                if response.status != 200:
                    logger.debug(f"Beatport returned status {response.status}")
                    return None

                html = await response.text()
                
                # Parse BPM, key, and genre from HTML
                # Note: Beatport's HTML structure may require more sophisticated parsing
                bpm_match = re.search(r'"bpm":\s*(\d+)', html)
                key_match = re.search(r'"key":\s*"([^"]+)"', html)
                genre_match = re.search(r'"genre":\s*"([^"]+)"', html)

                if bpm_match:
                    metadata = TrackMetadata(
                        artist=artist,
                        title=title,
                        bpm=float(bpm_match.group(1)),
                        source=self.source,
                        confidence=0.9  # Beatport is highly reliable for electronic music
                    )
                    
                    if key_match:
                        metadata.key = key_match.group(1).strip()
                    
                    if genre_match:
                        metadata.genre = genre_match.group(1).strip()
                    
                    logger.debug(f"Beatport found: {metadata.bpm} BPM")
                    return metadata

                logger.debug(f"No BPM found on Beatport for {artist} - {title}")
                return None

        except asyncio.TimeoutError:
            logger.warning(f"Beatport request timed out for {artist} - {title}")
            return None
        except Exception as e:
            logger.error(f"Beatport error for {artist} - {title}: {e}")
            return None
