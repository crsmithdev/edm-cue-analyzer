"""Export cue points to Rekordbox XML format."""

import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from xml.dom import minidom

from .analyzer import TrackStructure
from .cue_generator import CuePoint


class RekordboxXMLExporter:
    """Export cue points to Rekordbox XML format."""

    # Rekordbox color codes (integer values)
    COLOR_CODES = {
        "PINK": 0,
        "RED": 1,
        "ORANGE": 2,
        "YELLOW": 3,
        "GREEN": 4,
        "AQUA": 5,
        "BLUE": 6,
        "PURPLE": 7,
        "TEAL": 5,  # Map TEAL to AQUA
    }

    def __init__(self, color_mapping: dict = None):
        """
        Initialize exporter.

        Args:
            color_mapping: Optional custom color mapping dict
        """
        self.color_mapping = color_mapping or self.COLOR_CODES

    def export_track(
        self, filepath: Path, cues: list[CuePoint], structure: TrackStructure, output_path: Path
    ):
        """
        Export a single track with cues to Rekordbox XML.

        Args:
            filepath: Path to the audio file
            cues: List of cue points
            structure: Track structure info
            output_path: Where to save the XML file
        """
        # Create root element
        root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")

        # Create PRODUCT element
        ET.SubElement(
            root, "PRODUCT", Name="rekordbox", Version="6.0.0", Company="Pioneer DJ"
        )

        # Create COLLECTION element
        collection = ET.SubElement(root, "COLLECTION", Entries="1")

        # Create TRACK element
        track = self._create_track_element(collection, filepath, structure)

        # Add cue points
        for cue in cues:
            self._add_cue_point(track, cue, structure)

        # Create PLAYLISTS element (required but can be empty)
        ET.SubElement(root, "PLAYLISTS")

        # Write to file with proper formatting
        xml_string = self._prettify_xml(root)
        output_path.write_text(xml_string)

    def _create_track_element(
        self, parent: ET.Element, filepath: Path, structure: TrackStructure
    ) -> ET.Element:
        """Create TRACK element with basic metadata."""
        track_attrs = {
            "TrackID": "1",
            "Name": filepath.stem,
            "Artist": "",
            "Album": "",
            "Genre": "EDM",
            "Kind": filepath.suffix.upper().replace(".", ""),
            "Size": str(filepath.stat().st_size) if filepath.exists() else "0",
            "TotalTime": str(int(structure.duration)),
            "BitRate": "320",  # Default assumption
            "SampleRate": "44100",
            "Location": f"file://localhost/{filepath.as_posix()}",
            "TempoInBpm": f"{structure.bpm:.2f}",
            "DateAdded": datetime.now().strftime("%Y-%m-%d"),
        }

        return ET.SubElement(parent, "TRACK", **track_attrs)

    def _add_cue_point(self, track: ET.Element, cue: CuePoint, structure: TrackStructure):
        """Add a cue point to the track element."""
        # Convert position to milliseconds
        position_ms = cue.position * 1000

        if cue.cue_type == "hot":
            # Hot cue
            color_code = self.color_mapping.get(cue.color, 4)  # Default to GREEN

            cue_attrs = {
                "Name": cue.label,
                "Type": "0",  # 0 = cue point
                "Start": f"{position_ms:.3f}",
                "Num": str(cue.hot_cue_number),
                "Red": "255",
                "Green": "255",
                "Blue": "255",
                "ColorID": str(color_code),
            }

            # Add loop information if present
            if cue.loop_length:
                end_ms = (cue.position + cue.loop_length) * 1000
                cue_attrs["End"] = f"{end_ms:.3f}"

            ET.SubElement(track, "POSITION_MARK", **cue_attrs)

        else:
            # Memory cue
            cue_attrs = {
                "Name": cue.label,
                "Type": "0",
                "Start": f"{position_ms:.3f}",
                "Num": "-1",  # -1 indicates memory cue
                "Red": "255",
                "Green": "255",
                "Blue": "255",
            }

            ET.SubElement(track, "POSITION_MARK", **cue_attrs)

    def _prettify_xml(self, elem: ET.Element) -> str:
        """Return a pretty-printed XML string."""
        rough_string = ET.tostring(elem, encoding="utf-8")
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


def export_to_rekordbox(
    filepath: Path,
    cues: list[CuePoint],
    structure: TrackStructure,
    output_path: Path,
    color_mapping: dict = None,
):
    """
    Convenience function to export cues to Rekordbox XML.

    Args:
        filepath: Path to audio file
        cues: List of cue points
        structure: Track structure
        output_path: Output XML file path
        color_mapping: Optional custom color mapping
    """
    exporter = RekordboxXMLExporter(color_mapping)
    exporter.export_track(filepath, cues, structure, output_path)
