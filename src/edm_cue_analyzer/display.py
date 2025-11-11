"""Terminal display with color support."""

from colorama import Fore, Style, init

from .analyzer import TrackStructure
from .cue_generator import CuePoint

# Initialize colorama for cross-platform color support
init(autoreset=True)


class TerminalDisplay:
    """Display cue analysis results in terminal with colors."""

    # Color mapping for terminal display
    COLOR_MAP = {
        "BLUE": Fore.BLUE,
        "GREEN": Fore.GREEN,
        "TEAL": Fore.CYAN,
        "YELLOW": Fore.YELLOW,
        "ORANGE": Fore.LIGHTYELLOW_EX,
        "PURPLE": Fore.MAGENTA,
        "RED": Fore.RED,
    }

    def __init__(self, color_enabled: bool = True, waveform_width: int = 80):
        self.color_enabled = color_enabled
        self.waveform_width = waveform_width

    def display_analysis(self, filepath: str, structure: TrackStructure, cues: list[CuePoint]):
        """Display complete analysis results."""
        self._print_header()
        self._print_track_info(filepath, structure)
        self._print_structure_info(structure)
        self._print_cues(cues, structure)
        self._print_waveform(cues, structure)

    def _print_header(self):
        """Print header banner."""
        print("\n" + "=" * 80)
        print(f"{Fore.CYAN}{'EDM CUE ANALYZER':^80}{Style.RESET_ALL}")
        print("=" * 80 + "\n")

    def _print_track_info(self, filepath: str, structure: TrackStructure):
        """Print basic track information."""
        print(f"{Fore.GREEN}Track:{Style.RESET_ALL} {filepath}")
        print(f"{Fore.GREEN}BPM:{Style.RESET_ALL} {structure.bpm:.1f}")
        print(f"{Fore.GREEN}Duration:{Style.RESET_ALL} {self._format_time(structure.duration)}")
        print(f"{Fore.GREEN}Bar Duration:{Style.RESET_ALL} {structure.bar_duration:.2f}s")
        print()

    def _print_structure_info(self, structure: TrackStructure):
        """Print detected structure elements."""
        print(f"{Fore.YELLOW}Detected Structure:{Style.RESET_ALL}")

        if structure.drops:
            drop_times = [self._format_time(d) for d in structure.drops]
            print(f"  Drops: {', '.join(drop_times)}")
        else:
            print(f"  Drops: {Fore.RED}None detected{Style.RESET_ALL}")

        if structure.breakdowns:
            breakdown_times = [self._format_time(b) for b in structure.breakdowns]
            print(f"  Breakdowns: {', '.join(breakdown_times)}")
        else:
            print(f"  Breakdowns: {Fore.RED}None detected{Style.RESET_ALL}")

        if structure.builds:
            build_times = [self._format_time(b) for b in structure.builds]
            print(f"  Builds: {', '.join(build_times)}")
        else:
            print(f"  Builds: {Fore.RED}None detected{Style.RESET_ALL}")

        print()

    def _print_cues(self, cues: list[CuePoint], structure: TrackStructure):
        """Print cue point table."""
        print(f"{Fore.YELLOW}Generated Cue Points:{Style.RESET_ALL}\n")

        # Separate hot cues and memory cues
        hot_cues = [c for c in cues if c.cue_type == "hot"]
        memory_cues = [c for c in cues if c.cue_type == "memory"]

        # Print hot cues
        if hot_cues:
            print(f"{Fore.CYAN}Hot Cues:{Style.RESET_ALL}")
            print(f"{'ID':<4} {'Label':<20} {'Time':<10} {'Loop':<10} {'Color':<10}")
            print("-" * 65)

            for cue in sorted(hot_cues, key=lambda x: x.hot_cue_number):
                cue_id = chr(65 + cue.hot_cue_number)  # Convert 0-7 to A-H
                time_str = self._format_time(cue.position)
                loop_str = f"{cue.loop_length:.1f}s" if cue.loop_length else "N/A"

                # Apply color
                color = self.COLOR_MAP.get(cue.color, Fore.WHITE)
                color_display = f"{color}●{Style.RESET_ALL} {cue.color}"

                print(f"{cue_id:<4} {cue.label:<20} {time_str:<10} {loop_str:<10} {color_display}")

            print()

        # Print memory cues
        if memory_cues:
            print(f"{Fore.CYAN}Memory Cues:{Style.RESET_ALL}")
            print(f"{'Label':<30} {'Time':<10}")
            print("-" * 40)

            for cue in sorted(memory_cues, key=lambda x: x.position):
                time_str = self._format_time(cue.position)
                print(f"{cue.label:<30} {time_str:<10}")

            print()

    def _print_waveform(self, cues: list[CuePoint], structure: TrackStructure):
        """Print ASCII waveform with cue markers."""
        if not self.color_enabled:
            return

        print(f"{Fore.YELLOW}Track Timeline:{Style.RESET_ALL}")

        # Create timeline
        width = self.waveform_width
        timeline = [" "] * width

        # Mark hot cues on timeline
        hot_cues = [c for c in cues if c.cue_type == "hot"]
        for cue in hot_cues:
            pos = int((cue.position / structure.duration) * width)
            if 0 <= pos < width:
                cue_id = chr(65 + cue.hot_cue_number)
                color = self.COLOR_MAP.get(cue.color, Fore.WHITE)
                timeline[pos] = f"{color}{cue_id}{Style.RESET_ALL}"

        # Print timeline
        print("|" + "".join(timeline) + "|")

        # Print time markers
        start = "0:00"
        end = self._format_time(structure.duration)
        padding = width - len(start) - len(end)
        print(f"{start}{' ' * padding}{end}")
        print()

    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"


def display_results(
    filepath: str,
    structure: TrackStructure,
    cues: list[CuePoint],
    color_enabled: bool = True,
    waveform_width: int = 80,
):
    """
    Convenience function to display analysis results.

    Args:
        filepath: Path to audio file
        structure: Analyzed track structure
        cues: Generated cue points
        color_enabled: Enable colored output
        waveform_width: Width of ASCII waveform
    """
    display = TerminalDisplay(color_enabled, waveform_width)
    display.display_analysis(filepath, structure, cues)
