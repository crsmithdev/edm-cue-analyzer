"""Generate cue points from track structure and configuration."""

from dataclasses import dataclass

from .analyzer import TrackStructure, bars_to_seconds
from .config import Config, CueConfig


@dataclass
class CuePoint:
    """Represents a single cue point."""

    label: str
    position: float  # seconds
    cue_type: str  # 'hot' or 'memory'
    hot_cue_number: int | None = None  # 0-7 for hot cues A-H
    color: str | None = None
    loop_length: float | None = None  # seconds


class CueGenerator:
    """Generates cue points based on track structure and configuration."""

    def __init__(self, config: Config):
        self.config = config

    def generate_cues(self, structure: TrackStructure) -> list[CuePoint]:
        """
        Generate all cue points for a track.

        Args:
            structure: Analyzed track structure

        Returns:
            List of CuePoint objects
        """
        cues = []

        # Generate hot cues
        hot_cue_mapping = ["A", "B", "C", "D", "E", "F", "G", "H"]
        for cue_id in hot_cue_mapping:
            if cue_id in self.config.hot_cues:
                cue_config = self.config.hot_cues[cue_id]
                position = self._calculate_position(cue_config, structure)

                if position is not None:
                    loop_length = None
                    if cue_config.loop_bars:
                        loop_length = bars_to_seconds(cue_config.loop_bars, structure.bpm)

                    cues.append(
                        CuePoint(
                            label=cue_config.name,
                            position=position,
                            cue_type="hot",
                            hot_cue_number=hot_cue_mapping.index(cue_id),
                            color=cue_config.color,
                            loop_length=loop_length,
                        )
                    )

        # Generate memory cues
        for mem_cue_config in self.config.memory_cues:
            position = self._calculate_position_from_dict(mem_cue_config, structure)

            if position is not None:
                cues.append(
                    CuePoint(label=mem_cue_config["name"], position=position, cue_type="memory")
                )

        return cues

    def _calculate_position(self, cue_config: CueConfig, structure: TrackStructure) -> float | None:
        """Calculate position for a cue based on its configuration."""
        position = None

        # Position based on percentage
        if cue_config.position_percent is not None:
            position = structure.duration * cue_config.position_percent

        # Position based on detected structure
        elif cue_config.position_method:
            position = self._position_from_method(cue_config.position_method, structure)

        # Apply offset if position was found
        if position is not None and cue_config.offset_bars != 0:
            offset_seconds = bars_to_seconds(abs(cue_config.offset_bars), structure.bpm)
            if cue_config.offset_bars < 0:
                position -= offset_seconds
            else:
                position += offset_seconds

        # Clamp to track duration
        if position is not None:
            position = max(0.0, min(position, structure.duration))

        return position

    def _calculate_position_from_dict(
        self, cue_dict: dict, structure: TrackStructure
    ) -> float | None:
        """Calculate position from a dictionary configuration (for memory cues)."""
        position = None

        # Position based on percentage
        if "position_percent" in cue_dict:
            position = structure.duration * cue_dict["position_percent"]

        # Position based on detected structure
        elif "position_method" in cue_dict:
            position = self._position_from_method(cue_dict["position_method"], structure)

        # Apply offset if specified
        if position is not None and "offset_bars" in cue_dict:
            offset_bars = cue_dict["offset_bars"]
            offset_seconds = bars_to_seconds(abs(offset_bars), structure.bpm)
            if offset_bars < 0:
                position -= offset_seconds
            else:
                position += offset_seconds

        # Clamp to track duration
        if position is not None:
            position = max(0.0, min(position, structure.duration))

        return position

    def _position_from_method(self, method: str, structure: TrackStructure) -> float | None:
        """
        Calculate position based on detection method.

        Supported methods:
        - first_drop, second_drop, third_drop
        - after_first_drop, after_second_drop
        - before_first_drop, before_second_drop
        - first_breakdown, second_breakdown
        - first_build, second_build
        """
        if "drop" in method:
            return self._get_drop_position(method, structure)
        elif "breakdown" in method:
            return self._get_breakdown_position(method, structure)
        elif "build" in method:
            return self._get_build_position(method, structure)

        return None

    def _get_drop_position(self, method: str, structure: TrackStructure) -> float | None:
        """Get position of drop based on method."""
        if not structure.drops:
            # Fallback: estimate based on typical EDM structure
            return structure.duration * 0.35  # Drops typically around 35% mark

        # Determine which drop
        drop_index = 0
        if "second" in method:
            drop_index = 1
        elif "third" in method:
            drop_index = 2

        if drop_index >= len(structure.drops):
            # Use last drop if requested drop doesn't exist
            drop_index = len(structure.drops) - 1

        position = structure.drops[drop_index]

        # Handle before/after modifiers
        if "before" in method:
            # Position is already at the drop, which is what "before" means
            pass
        elif "after" in method:
            # "after" means the same as the drop position in our detection
            # since we detect the peak itself
            pass

        return position

    def _get_breakdown_position(self, method: str, structure: TrackStructure) -> float | None:
        """Get position of breakdown based on method."""
        if not structure.breakdowns:
            # Fallback: estimate based on typical EDM structure
            return structure.duration * 0.55  # Breakdowns typically around 55% mark

        # Determine which breakdown
        breakdown_index = 0
        if "second" in method:
            breakdown_index = 1

        if breakdown_index >= len(structure.breakdowns):
            breakdown_index = len(structure.breakdowns) - 1

        return structure.breakdowns[breakdown_index]

    def _get_build_position(self, method: str, structure: TrackStructure) -> float | None:
        """Get position of build based on method."""
        if not structure.builds:
            # No detected builds
            return None

        # Determine which build
        build_index = 0
        if "second" in method:
            build_index = 1

        if build_index >= len(structure.builds):
            build_index = len(structure.builds) - 1

        return structure.builds[build_index]
