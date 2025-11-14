# Library Usage Guide

This document explains how to use `edm-cue-analyzer` as a library in your own applications.

## Design Philosophy

The library provides **single-file analysis primitives**. Batch processing and parallelization across multiple files should be handled by your application code, giving you full control over concurrency, resource management, and orchestration.

**What the library does:**
- Analyzes individual audio files (async)
- Parallelizes internal operations (e.g., BPM consensus detection)
- Provides modular analysis selection

**What your application should do:**
- Orchestrate batch processing
- Control concurrency limits
- Implement progress tracking
- Handle errors and retries

## Basic Usage

### Single File Analysis

```python
import asyncio
from pathlib import Path
from edm_cue_analyzer import AudioAnalyzer
from edm_cue_analyzer.config import get_default_config

async def analyze_single_file():
    # Load configuration
    config = get_default_config()
    
    # Create analyzer
    analyzer = AudioAnalyzer(config.analysis)
    
    # Analyze file (full analysis)
    audio_path = Path("track.flac")
    structure = await analyzer.analyze_file(audio_path)
    
    print(f"BPM: {structure.bpm}")
    print(f"Drops: {structure.drops}")
    print(f"Breakdowns: {structure.breakdowns}")

# Run
asyncio.run(analyze_single_file())
```

### Selective Analysis

```python
async def analyze_bpm_only():
    config = get_default_config()
    analyzer = AudioAnalyzer(config.analysis)
    
    # Only detect BPM (faster)
    audio_path = Path("track.flac")
    structure = await analyzer.analyze_with(audio_path, analyses="bpm-only")
    
    print(f"BPM: {structure.bpm}")

asyncio.run(analyze_bpm_only())
```

## Batch Processing

The library does NOT provide batch processing. Your application controls how multiple files are processed.

### Example 1: Simple Sequential Processing

```python
async def batch_sequential(files: list[Path]):
    """Process files one at a time."""
    config = get_default_config()
    analyzer = AudioAnalyzer(config.analysis)
    
    results = []
    for file in files:
        try:
            structure = await analyzer.analyze_file(file)
            results.append((file, structure))
        except Exception as e:
            print(f"Error analyzing {file}: {e}")
            results.append((file, None))
    
    return results
```

### Example 2: Concurrent Processing with Semaphore

```python
async def batch_concurrent(files: list[Path], max_concurrent: int = 4):
    """Process multiple files concurrently with limit."""
    config = get_default_config()
    analyzer = AudioAnalyzer(config.analysis)
    
    # Limit concurrent operations
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def analyze_with_limit(file: Path):
        async with semaphore:
            try:
                return await analyzer.analyze_file(file)
            except Exception as e:
                print(f"Error analyzing {file}: {e}")
                return None
    
    # Process all files concurrently (up to max_concurrent at a time)
    results = await asyncio.gather(*[analyze_with_limit(f) for f in files])
    return list(zip(files, results))
```

### Example 3: Progress Tracking

```python
async def batch_with_progress(files: list[Path]):
    """Process files with progress tracking."""
    config = get_default_config()
    analyzer = AudioAnalyzer(config.analysis)
    
    total = len(files)
    completed = 0
    
    async def analyze_with_progress(file: Path):
        nonlocal completed
        result = await analyzer.analyze_file(file)
        completed += 1
        print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
        return result
    
    results = []
    for file in files:
        result = await analyze_with_progress(file)
        results.append((file, result))
    
    return results
```

### Example 4: Task Queue Integration (Celery/RQ)

```python
from celery import Celery

app = Celery('audio_analysis', broker='redis://localhost:6379')

@app.task
def analyze_track_task(file_path: str):
    """Celery task for asynchronous processing."""
    import asyncio
    from pathlib import Path
    from edm_cue_analyzer import AudioAnalyzer
    from edm_cue_analyzer.config import get_default_config
    
    async def analyze():
        config = get_default_config()
        analyzer = AudioAnalyzer(config.analysis)
        structure = await analyzer.analyze_file(Path(file_path))
        return {
            'bpm': structure.bpm,
            'drops': structure.drops,
            'breakdowns': structure.breakdowns,
        }
    
    return asyncio.run(analyze())

# Usage
result = analyze_track_task.delay('/path/to/track.flac')
```

### Example 5: Web API (FastAPI)

```python
from fastapi import FastAPI, UploadFile, BackgroundTasks
from pathlib import Path
import asyncio

app = FastAPI()

# Shared analyzer instance
from edm_cue_analyzer import AudioAnalyzer
from edm_cue_analyzer.config import get_default_config

config = get_default_config()
analyzer = AudioAnalyzer(config.analysis)

@app.post("/analyze")
async def analyze_upload(file: UploadFile):
    """Analyze uploaded audio file."""
    # Save upload to temp file
    temp_path = Path(f"/tmp/{file.filename}")
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # Analyze
    try:
        structure = await analyzer.analyze_file(temp_path)
        return {
            "bpm": structure.bpm,
            "duration": structure.duration,
            "drops": structure.drops,
            "breakdowns": structure.breakdowns,
        }
    finally:
        temp_path.unlink()  # Cleanup

@app.post("/batch")
async def analyze_batch(files: list[UploadFile], max_concurrent: int = 4):
    """Analyze multiple files concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_file(file: UploadFile):
        async with semaphore:
            temp_path = Path(f"/tmp/{file.filename}")
            with open(temp_path, "wb") as f:
                f.write(await file.read())
            
            try:
                structure = await analyzer.analyze_file(temp_path)
                return {
                    "filename": file.filename,
                    "bpm": structure.bpm,
                    "drops": structure.drops,
                }
            finally:
                temp_path.unlink()
    
    results = await asyncio.gather(*[process_file(f) for f in files])
    return {"results": results}
```

## Analysis Options

### Available Analyses

- `"bpm"` - BPM and beat detection only
- `"energy"` - Energy curve calculation
- `"drops"` - Drop detection
- `"breakdowns"` - Breakdown detection
- `"builds"` - Build-up detection

### Presets

- `"bpm-only"` - Just BPM (fastest)
- `"structure"` - Full structural analysis
- `"full"` - All analyses (default)

### Custom Selection

```python
# Run specific analyses
structure = await analyzer.analyze_with(
    audio_path, 
    analyses={"bpm", "energy", "drops"}  # Just these three
)

# Use preset
structure = await analyzer.analyze_with(
    audio_path,
    analyses="structure"
)
```

## Configuration

```python
from edm_cue_analyzer.config import load_config, Config

# Load custom config
config = load_config(Path("my_config.yaml"))
analyzer = AudioAnalyzer(config.analysis)

# Or modify default config
config = get_default_config()
config.analysis.drop_min_spacing_bars = 12  # Customize detection
analyzer = AudioAnalyzer(config.analysis)
```

## Best Practices

1. **Reuse Analyzer Instance**: Create one `AudioAnalyzer` and reuse it for multiple files
2. **Control Concurrency**: Use `asyncio.Semaphore` to limit parallel operations
3. **Error Handling**: Always wrap analysis in try/except for production use
4. **Resource Cleanup**: Clean up temporary files in `finally` blocks
5. **Progress Tracking**: Implement at application level, not in library
6. **Async All The Way**: Use async functions throughout your call stack
7. **Memory Management**: For very large batches, process in chunks

## Why This Design?

**Separation of Concerns:**
- Library focuses on accurate, fast single-file analysis
- Application controls orchestration, concurrency, and infrastructure

**Flexibility:**
- CLI can use simple `asyncio.gather()`
- Web API can use connection pools and rate limiting
- Task queue can use distributed workers
- Desktop app can use background threads with callbacks

**Performance:**
- Internal optimizations (BPM consensus parallelization) benefit all consumers
- Application chooses optimal concurrency for its environment
- No one-size-fits-all batch processing that might not fit your use case
