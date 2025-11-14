#!/usr/bin/env python3
"""
BPM Validation Script

Validates BPM detection accuracy by comparing analyzer results against
online BPM databases (GetSongBPM, Tunebat, Beatport).

Usage:
    python validate_bpm_accuracy.py /path/to/music/library [--max-files N] [--skip-cache]
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import aiohttp
from bs4 import BeautifulSoup

from edm_cue_analyzer import AudioAnalyzer
from edm_cue_analyzer.config import get_default_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler('bpm_validation.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BpmValidation:
    """Results from validating a single track."""
    filepath: str
    artist: str
    title: str
    detected_bpm: float
    reference_bpm: Optional[float]
    source: Optional[str]
    error_bpm: Optional[float]
    error_percent: Optional[float]
    analysis_time: float
    success: bool
    error_message: Optional[str] = None


class BpmFetcher:
    """Fetches reference BPM data from online sources."""
    
    def __init__(self, cache_file: Path = Path("bpm_cache.json")):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.session: Optional[aiohttp.ClientSession] = None
        
    def _load_cache(self) -> dict:
        """Load cached BPM lookups."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save BPM cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
        self._save_cache()
    
    def _parse_filename(self, filepath: Path) -> tuple[str, str]:
        """
        Extract artist and title from filename.
        
        Handles common patterns:
        - "Artist - Title.flac"
        - "01. Artist - Title.mp3"
        - "Artist_-_Title.wav"
        """
        name = filepath.stem
        
        # Remove track numbers (e.g., "01. ", "01 - ", "1. ")
        name = re.sub(r'^\d+[\.\-\s]+', '', name)
        
        # Try to split on " - " or " – " (em dash)
        for sep in [' - ', ' – ', ' -  ', '_-_']:
            if sep in name:
                parts = name.split(sep, 1)
                return parts[0].strip(), parts[1].strip()
        
        # If no separator, use whole name as title
        return "", name.strip()
    
    async def fetch_bpm(self, artist: str, title: str) -> Optional[tuple[float, str]]:
        """
        Fetch BPM from online sources.
        
        Returns:
            Tuple of (bpm, source) or None if not found
        """
        # Check cache first
        cache_key = f"{artist.lower()}|{title.lower()}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if cached is not None:
                return cached['bpm'], cached['source']
            return None
        
        # Try multiple sources
        bpm_result = None
        
        # 1. GetSongBPM
        bpm_result = await self._fetch_getsongbpm(artist, title)
        if bpm_result:
            self.cache[cache_key] = {'bpm': bpm_result[0], 'source': bpm_result[1]}
            return bpm_result
        
        # 2. Tunebat
        await asyncio.sleep(0.5)  # Rate limiting
        bpm_result = await self._fetch_tunebat(artist, title)
        if bpm_result:
            self.cache[cache_key] = {'bpm': bpm_result[0], 'source': bpm_result[1]}
            return bpm_result
        
        # 3. Beatport (if available)
        await asyncio.sleep(0.5)
        bpm_result = await self._fetch_beatport(artist, title)
        if bpm_result:
            self.cache[cache_key] = {'bpm': bpm_result[0], 'source': bpm_result[1]}
            return bpm_result
        
        # Cache negative result to avoid repeated lookups
        self.cache[cache_key] = None
        return None
    
    async def _fetch_getsongbpm(self, artist: str, title: str) -> Optional[tuple[float, str]]:
        """Fetch BPM from GetSongBPM.com"""
        try:
            search_query = f"{artist} {title}".strip()
            if not search_query:
                return None
            
            url = f"https://getsongbpm.com/search?q={quote(search_query)}"
            
            async with self.session.get(url, timeout=10) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find BPM in search results
                for item in soup.select('.search-result-item'):
                    bpm_elem = item.select_one('.bpm')
                    if bpm_elem:
                        bpm_text = bpm_elem.text.strip()
                        match = re.search(r'(\d+(?:\.\d+)?)', bpm_text)
                        if match:
                            bpm = float(match.group(1))
                            logger.debug(f"GetSongBPM: {artist} - {title} = {bpm} BPM")
                            return bpm, "GetSongBPM"
        except Exception as e:
            logger.debug(f"GetSongBPM lookup failed: {e}")
        
        return None
    
    async def _fetch_tunebat(self, artist: str, title: str) -> Optional[tuple[float, str]]:
        """Fetch BPM from Tunebat.com"""
        try:
            search_query = f"{artist} {title}".strip()
            if not search_query:
                return None
            
            url = f"https://tunebat.com/Search?q={quote(search_query)}"
            
            async with self.session.get(url, timeout=10) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find BPM in results
                for item in soup.select('.search-item'):
                    bpm_elem = item.select_one('.bpm')
                    if bpm_elem:
                        bpm_text = bpm_elem.text.strip()
                        match = re.search(r'(\d+(?:\.\d+)?)', bpm_text)
                        if match:
                            bpm = float(match.group(1))
                            logger.debug(f"Tunebat: {artist} - {title} = {bpm} BPM")
                            return bpm, "Tunebat"
        except Exception as e:
            logger.debug(f"Tunebat lookup failed: {e}")
        
        return None
    
    async def _fetch_beatport(self, artist: str, title: str) -> Optional[tuple[float, str]]:
        """Fetch BPM from Beatport.com"""
        try:
            search_query = f"{artist} {title}".strip()
            if not search_query:
                return None
            
            url = f"https://www.beatport.com/search?q={quote(search_query)}"
            
            async with self.session.get(url, timeout=10) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Beatport shows BPM in track details
                for item in soup.select('.track'):
                    bpm_elem = item.select_one('.bpm-value')
                    if bpm_elem:
                        bpm_text = bpm_elem.text.strip()
                        match = re.search(r'(\d+(?:\.\d+)?)', bpm_text)
                        if match:
                            bpm = float(match.group(1))
                            logger.debug(f"Beatport: {artist} - {title} = {bpm} BPM")
                            return bpm, "Beatport"
        except Exception as e:
            logger.debug(f"Beatport lookup failed: {e}")
        
        return None


async def validate_file(
    filepath: Path,
    analyzer: AudioAnalyzer,
    fetcher: BpmFetcher
) -> BpmValidation:
    """Validate BPM detection for a single file."""
    
    # Parse filename
    artist, title = fetcher._parse_filename(filepath)
    
    logger.info(f"Processing: {artist} - {title}")
    
    try:
        # Fetch reference BPM from online sources
        reference_result = await fetcher.fetch_bpm(artist, title)
        
        if reference_result is None:
            logger.warning(f"No online BPM found for: {artist} - {title}")
            return BpmValidation(
                filepath=str(filepath),
                artist=artist,
                title=title,
                detected_bpm=0.0,
                reference_bpm=None,
                source=None,
                error_bpm=None,
                error_percent=None,
                analysis_time=0.0,
                success=False,
                error_message="No reference BPM found online"
            )
        
        reference_bpm, source = reference_result
        logger.info(f"Reference BPM: {reference_bpm} ({source})")
        
        # Detect BPM using analyzer
        start_time = time.perf_counter()
        structure = await analyzer.analyze_with(filepath, analyses="bpm-only")
        analysis_time = time.perf_counter() - start_time
        
        detected_bpm = structure.bpm
        
        # Calculate error
        error_bpm = detected_bpm - reference_bpm
        error_percent = (error_bpm / reference_bpm) * 100
        
        # Check for octave errors (double/half time)
        if abs(detected_bpm - reference_bpm * 2) < abs(error_bpm):
            error_bpm = detected_bpm - reference_bpm * 2
            error_percent = (error_bpm / reference_bpm) * 100
        elif abs(detected_bpm - reference_bpm / 2) < abs(error_bpm):
            error_bpm = detected_bpm - reference_bpm / 2
            error_percent = (error_bpm / reference_bpm) * 100
        
        logger.info(
            f"Detected: {detected_bpm:.1f} BPM | "
            f"Reference: {reference_bpm:.1f} BPM | "
            f"Error: {error_bpm:+.1f} BPM ({error_percent:+.1f}%)"
        )
        
        return BpmValidation(
            filepath=str(filepath),
            artist=artist,
            title=title,
            detected_bpm=detected_bpm,
            reference_bpm=reference_bpm,
            source=source,
            error_bpm=error_bpm,
            error_percent=error_percent,
            analysis_time=analysis_time,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}", exc_info=True)
        return BpmValidation(
            filepath=str(filepath),
            artist=artist,
            title=title,
            detected_bpm=0.0,
            reference_bpm=None,
            source=None,
            error_bpm=None,
            error_percent=None,
            analysis_time=0.0,
            success=False,
            error_message=str(e)
        )


async def validate_library(
    library_path: Path,
    max_files: Optional[int] = None,
    skip_cache: bool = False
) -> list[BpmValidation]:
    """Validate BPM detection across a music library."""
    
    # Find audio files
    audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.aac', '.ogg'}
    audio_files = [
        f for f in library_path.rglob('*')
        if f.suffix.lower() in audio_extensions
    ]
    
    if not audio_files:
        logger.error(f"No audio files found in {library_path}")
        return []
    
    if max_files:
        audio_files = audio_files[:max_files]
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Initialize analyzer
    config = get_default_config()
    analyzer = AudioAnalyzer(config.analysis)
    
    # Process files
    results = []
    
    cache_file = Path("bpm_cache.json") if not skip_cache else None
    async with BpmFetcher(cache_file) as fetcher:
        for i, filepath in enumerate(audio_files, 1):
            logger.info(f"\n[{i}/{len(audio_files)}] {filepath.name}")
            
            result = await validate_file(filepath, analyzer, fetcher)
            results.append(result)
            
            # Rate limiting for online lookups
            if i < len(audio_files):
                await asyncio.sleep(1)
    
    return results


def print_report(results: list[BpmValidation]):
    """Print comprehensive validation report."""
    
    # Filter successful validations
    successful = [r for r in results if r.success and r.reference_bpm is not None]
    
    if not successful:
        print("\n❌ No successful validations to report")
        return
    
    # Calculate statistics
    total_tested = len(successful)
    total_files = len(results)
    
    errors = [abs(r.error_bpm) for r in successful]
    error_percents = [abs(r.error_percent) for r in successful]
    analysis_times = [r.analysis_time for r in successful]
    
    mean_error = sum(errors) / len(errors)
    mean_error_percent = sum(error_percents) / len(error_percents)
    mean_analysis_time = sum(analysis_times) / len(analysis_times)
    
    # Count accuracy within thresholds
    within_1bpm = sum(1 for e in errors if e <= 1.0)
    within_2bpm = sum(1 for e in errors if e <= 2.0)
    within_5bpm = sum(1 for e in errors if e <= 5.0)
    within_5pct = sum(1 for e in error_percents if e <= 5.0)
    
    # Print report
    print("\n" + "="*80)
    print("BPM VALIDATION REPORT")
    print("="*80)
    
    print(f"\n📊 Summary")
    print(f"  Total files scanned: {total_files}")
    print(f"  Successfully validated: {total_tested}")
    print(f"  No reference found: {total_files - total_tested}")
    
    print(f"\n🎯 Accuracy Statistics")
    print(f"  Mean error: {mean_error:.2f} BPM ({mean_error_percent:.2f}%)")
    print(f"  Within ±1 BPM: {within_1bpm}/{total_tested} ({within_1bpm/total_tested*100:.1f}%)")
    print(f"  Within ±2 BPM: {within_2bpm}/{total_tested} ({within_2bpm/total_tested*100:.1f}%)")
    print(f"  Within ±5 BPM: {within_5bpm}/{total_tested} ({within_5bpm/total_tested*100:.1f}%)")
    print(f"  Within ±5%: {within_5pct}/{total_tested} ({within_5pct/total_tested*100:.1f}%)")
    
    print(f"\n⏱️  Performance")
    print(f"  Mean analysis time: {mean_analysis_time:.2f}s per track")
    print(f"  Total analysis time: {sum(analysis_times):.1f}s")
    
    # Source breakdown
    sources = {}
    for r in successful:
        sources[r.source] = sources.get(r.source, 0) + 1
    
    print(f"\n🔍 Reference Sources")
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count} tracks ({count/total_tested*100:.1f}%)")
    
    # Worst errors
    print(f"\n❌ Top 10 Largest Errors")
    worst_errors = sorted(successful, key=lambda r: abs(r.error_bpm), reverse=True)[:10]
    
    for i, r in enumerate(worst_errors, 1):
        print(f"\n  {i}. {r.artist} - {r.title}")
        print(f"     Detected: {r.detected_bpm:.1f} BPM | Reference: {r.reference_bpm:.1f} BPM ({r.source})")
        print(f"     Error: {r.error_bpm:+.1f} BPM ({r.error_percent:+.1f}%)")
    
    # Perfect matches
    perfect = [r for r in successful if abs(r.error_bpm) < 0.5]
    if perfect:
        print(f"\n✅ Perfect Matches (±0.5 BPM): {len(perfect)}")
        for r in perfect[:5]:  # Show first 5
            print(f"  • {r.artist} - {r.title}: {r.detected_bpm:.1f} BPM")
    
    print("\n" + "="*80)
    
    # Save detailed results to JSON
    output_file = Path("bpm_validation_results.json")
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print(f"📝 Log file: bpm_validation.log\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate BPM detection accuracy against online databases"
    )
    parser.add_argument(
        "library_path",
        type=Path,
        help="Path to music library directory"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to test (default: all)"
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip BPM lookup cache (re-fetch all)"
    )
    
    args = parser.parse_args()
    
    if not args.library_path.exists():
        print(f"❌ Error: Directory not found: {args.library_path}")
        sys.exit(1)
    
    if not args.library_path.is_dir():
        print(f"❌ Error: Not a directory: {args.library_path}")
        sys.exit(1)
    
    print(f"\n🎵 Starting BPM validation for: {args.library_path}")
    print(f"⏳ This may take a while...\n")
    
    # Run validation
    results = asyncio.run(validate_library(
        args.library_path,
        max_files=args.max_files,
        skip_cache=args.skip_cache
    ))
    
    # Print report
    print_report(results)


if __name__ == "__main__":
    main()
