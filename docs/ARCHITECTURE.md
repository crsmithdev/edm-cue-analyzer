# EDM Cue Analyzer — Architecture Overview

This document concisely describes the architecture and core components of
`edm-cue-analyzer`. It's intended to give a quick mental model for contributors
and integrators.

## High-level goals

- Provide single-file audio analysis primitives (async) that callers can
  orchestrate for batch processing.
- Produce reproducible, testable structure outputs: BPM (detected), optional
  reference BPM (from tags/metadata), energy curve, and structural events
  (drops, breakdowns, builds).
- Be modular: analyses are pluggable, metadata providers are swappable,
  and feature extractors are extensible.

## Major components

- `AudioAnalyzer` (library entry point)
  - Loads audio and prepares an analysis context.
  - Orchestrates requested analyses in dependency order.
  - Integrates local metadata lookup (tags) and exposes `detected_bpm`
    and `reference_bpm` explicitly on `TrackStructure`.

- `analyses` (pluggable analysis modules)
  - Implemented as small units (bpm, energy, drops, breakdowns, builds).
  - Each analysis declares dependencies and a callable `func(context)` that
    returns a result object placed into the analysis context.

- `consensus` (BPM aggregation)
  - Combines multiple BPM estimators (librosa/aubio/essentia) into a
    consensus `bpm` estimate with confidence scoring.

- `metadata` (provider abstraction)
  - Providers implement a simple interface to fetch `TrackMetadata`.
  - `LocalFileProvider` reads file tags (mutagen). Online providers
    (GetSongBPM/Tunebat/Beatport) are optional and behind async clients.
  - `MetadataAggregator` merges results and caches them for validation runs.

- `TrackStructure` (analysis result)
  - Contains explicit fields: `detected_bpm`, `reference_bpm`, `duration`,
    `beats`, `bar_duration`, `energy_curve`, `energy_times`, `drops`,
    `breakdowns`, `builds`, and `features`.
  - Callers should choose `reference_bpm` when present; otherwise use
    `detected_bpm`.

- `CueGenerator` and exporters
  - Generate cue points (hot + memory) from the `TrackStructure` and
    configuration.
  - Exporters (Rekordbox XML) format results for DJ software.

## Config & CLI

- `Config` contains analysis tuning parameters (energy thresholds, spacing,
  bpm precision). CLI builds on the library and passes a full `Config` to
  `AudioAnalyzer` for consistent rounding and behavior.

## Design notes

- Explicit BPM fields (detected vs reference) avoid ambiguity and make tests
  deterministic. The library no longer exposes a single `structure.bpm`
  write-once field; callers must select which BPM to use for downstream
  calculations.
- The analysis pipeline emphasizes small, focused modules with minimal
  side-effects. This simplifies unit testing and allows swapping
  implementations (e.g., Essentia vs librosa) without broad changes.

## Testing & validation

- Unit tests cover analysis utilities and metadata aggregation. The project
  includes a validation pipeline for measuring BPM accuracy at scale and
  analyzing rounding/precision effects.

## Where to start contributing

1. Read `src/edm_cue_analyzer/analyzer.py` and `src/edm_cue_analyzer/analyses`.
2. Add small, focused tests in `tests/` for any new analysis or metadata
   provider.
3. Keep changes backward-compatible for public callers, or clearly mark
   breaking changes in the changelog.

---

If you want this expanded into a longer developer guide or to include a
diagram, tell me which areas to expand first (metadata, BPM consensus,
or analyses internals).
