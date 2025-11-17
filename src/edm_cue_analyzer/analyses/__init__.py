"""Analysis registry and dependency resolution system."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .bpm import analyze_bpm
from .breakdowns import analyze_breakdowns
from .builds import analyze_builds
from .drops import analyze_drops
from .energy import analyze_energy


@dataclass
class Analysis:
    """Definition of an analysis with its dependencies."""

    name: str
    func: Callable[[dict], Awaitable[Any]]
    dependencies: set[str]
    description: str


# Registry of all available analyses
ANALYSES = {
    "bpm": Analysis(
        name="bpm",
        func=analyze_bpm,
        dependencies=set(),
        description="BPM and beat detection using consensus algorithm",
    ),
    "energy": Analysis(
        name="energy",
        func=analyze_energy,
        dependencies={"bpm"},
        description="RMS energy curve calculation",
    ),
    "drops": Analysis(
        name="drops",
        func=analyze_drops,
        dependencies={"bpm", "energy"},
        description="Drop point detection (beat/bass returns)",
    ),
    "breakdowns": Analysis(
        name="breakdowns",
        func=analyze_breakdowns,
        dependencies={"bpm", "energy"},
        description="Breakdown detection (energy/complexity drops)",
    ),
    "builds": Analysis(
        name="builds",
        func=analyze_builds,
        dependencies={"bpm", "energy"},
        description="Build-up detection (increasing energy/tension)",
    ),
}

# Preset combinations for common use cases
ANALYSIS_PRESETS = {
    "bpm": {"bpm"},
    "structure": {"bpm", "energy", "drops", "breakdowns", "builds"},
    "full": set(ANALYSES.keys()),
}


def resolve_dependencies(requested: set[str]) -> list[str]:
    """
    Topologically sort analyses based on dependencies.

    Args:
        requested: Set of analysis names to run

    Returns:
        List of analysis names in execution order

    Raises:
        ValueError: If unknown analysis requested or circular dependency detected
    """
    # Validate all requested analyses exist
    unknown = requested - set(ANALYSES.keys())
    if unknown:
        raise ValueError(f"Unknown analyses: {unknown}")

    resolved = []
    seen = set()
    visiting = set()

    def visit(name: str):
        if name in seen:
            return
        if name in visiting:
            raise ValueError(f"Circular dependency detected involving: {name}")

        visiting.add(name)
        analysis = ANALYSES[name]

        for dep in analysis.dependencies:
            visit(dep)

        visiting.remove(name)
        seen.add(name)
        resolved.append(name)

    for name in requested:
        visit(name)

    return resolved


def expand_preset(analyses: str | set[str]) -> set[str]:
    """
    Expand preset name to set of analyses, or pass through explicit set.

    Args:
        analyses: Either a preset name string or explicit set of analysis names

    Returns:
        Set of analysis names to execute
    """
    if isinstance(analyses, str):
        if analyses in ANALYSIS_PRESETS:
            return ANALYSIS_PRESETS[analyses].copy()
        # Single analysis name
        return {analyses}
    return analyses


__all__ = [
    "Analysis",
    "ANALYSES",
    "ANALYSIS_PRESETS",
    "resolve_dependencies",
    "expand_preset",
]
