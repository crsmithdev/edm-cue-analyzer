"""Local file metadata providers."""

import contextlib
import logging
from pathlib import Path

import soundfile as sf

from .base import MetadataProvider, MetadataSource, TrackMetadata

logger = logging.getLogger(__name__)

# Try to import mutagen for tag reading
try:
    import mutagen
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    logger.debug("mutagen not available - tag reading will be limited")


class LocalFileProvider(MetadataProvider):
    """Extract metadata from local audio file tags."""

    @property
    def source(self) -> MetadataSource:
        return MetadataSource.LOCAL_FILE

    async def get_metadata(
        self,
        artist: str | None = None,
        title: str | None = None,
        file_path: Path | None = None,
        **kwargs
    ) -> TrackMetadata | None:
        """
        Read metadata from local file tags.

        Args:
            file_path: Path to audio file (required)

        Returns:
            TrackMetadata with file tag information
        """
        if file_path is None:
            raise ValueError("file_path is required for LocalFileProvider")

        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None

        metadata = TrackMetadata(
            file_path=file_path,
            source=self.source,
        )

        # Get audio properties with soundfile
        try:
            info = sf.info(str(file_path))
            metadata.duration = info.duration
            metadata.sample_rate = info.samplerate
        except Exception as e:
            logger.warning(f"Could not read audio info from {file_path}: {e}")

        # Get tags with mutagen if available
        if MUTAGEN_AVAILABLE:
            try:
                audio = mutagen.File(str(file_path))
                if audio is not None:
                    metadata.artist = self._get_tag(audio, ['artist', 'ARTIST', 'TPE1'])
                    metadata.title = self._get_tag(audio, ['title', 'TITLE', 'TIT2'])
                    metadata.album = self._get_tag(audio, ['album', 'ALBUM', 'TALB'])
                    metadata.genre = self._get_tag(audio, ['genre', 'GENRE', 'TCON'])

                    # Try to get BPM from tags
                    bpm_str = self._get_tag(audio, ['bpm', 'BPM', 'TBPM'])
                    if bpm_str:
                        with contextlib.suppress(ValueError, TypeError):
                            metadata.bpm = float(bpm_str)

            except Exception as e:
                logger.debug(f"Could not read tags from {file_path}: {e}")

        # Fallback: parse filename for artist/title
        if not metadata.artist or not metadata.title:
            parsed = self._parse_filename(file_path)
            metadata.artist = metadata.artist or parsed.get('artist')
            metadata.title = metadata.title or parsed.get('title')

        return metadata

    def _get_tag(self, audio, tag_names: list[str]) -> str | None:
        """Try multiple tag names and return first found."""
        for tag_name in tag_names:
            if tag_name in audio:
                value = audio[tag_name]
                if isinstance(value, list) and len(value) > 0:
                    return str(value[0])
                elif value:
                    return str(value)
        return None

    def _parse_filename(self, file_path: Path) -> dict:
        """
        Parse artist/title from filename.

        Handles formats like:
        - "Artist - Title.flac"
        - "01. Artist - Title.mp3"
        - "Artist_-_Title.wav"
        """
        name = file_path.stem

        # Remove track numbers
        import re
        name = re.sub(r'^\d+[\.\s]*', '', name)

        # Try common separators
        for sep in [' - ', ' – ', ' — ', '_-_', '-']:
            if sep in name:
                parts = name.split(sep, 1)
                return {
                    'artist': parts[0].strip(),
                    'title': parts[1].strip()
                }

        # No separator found, use whole name as title
        return {'title': name.strip()}
