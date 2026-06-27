"""Traktor NML exporter for hot cues and track analysis."""

import xml.etree.ElementTree as ET
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class HotCue:
    """Hot cue marker for Traktor."""
    name: str  # HotCue_1, HotCue_2, etc.
    start_ms: int  # Start time in milliseconds
    type_str: str  # 'hotcue' for Traktor
    description: str  # Label/description


def parse_traktor_nml(nml_path: str) -> ET.Element:
    """
    Parse Traktor collection.nml file.

    Loads NML XML and validates structure.

    Args:
        nml_path: Path to Traktor collection.nml file

    Returns:
        Root XML element (COLLECTION)

    Raises:
        FileNotFoundError: If file doesn't exist
        ET.ParseError: If XML is malformed

    Example:
        >>> root = parse_traktor_nml("Collection.nml")
        >>> print(f"Entries: {root.get('ENTRIES')}")
    """
    path = Path(nml_path)

    if not path.exists():
        raise FileNotFoundError(f"Traktor NML not found: {nml_path}")

    try:
        tree = ET.parse(nml_path)
        root = tree.getroot()

        if root.tag != 'COLLECTION':
            raise ValueError(f"Invalid NML: root tag is {root.tag}, expected COLLECTION")

        logger.info(f"Parsed Traktor NML: {root.get('ENTRIES')} entries")
        return root

    except ET.ParseError as e:
        raise ET.ParseError(f"Failed to parse NML: {e}")


def _get_or_create_element(parent: ET.Element, tag: str, **attrs) -> ET.Element:
    """Get existing child element or create new one."""
    existing = parent.find(tag)
    if existing is not None:
        return existing

    element = ET.Element(tag, attrs)
    parent.append(element)
    return element


def _ms_to_traktor_offset(ms: int) -> int:
    """
    Convert milliseconds to Traktor offset format (in samples at 48kHz).

    Traktor stores cue points as sample offsets at 48kHz sample rate.
    """
    # 48000 samples/second = 48 samples/ms
    return ms * 48


def add_track_analysis(
    nml_root: ET.Element,
    track_id: int,
    track_analysis: Dict,
    db_path: str = "data/djia.db"
) -> Optional[ET.Element]:
    """
    Add analysis data to a track entry in NML tree.

    Adds:
    - BPM from Phase 2
    - Key (if available)
    - Hot cues (Pad 1=drop, Pad 2=breakdown, Pad 4=outro) from Phase 3
    - Metadata (brightness, danceability, mood)

    Args:
        nml_root: Root COLLECTION element from parse_traktor_nml()
        track_id: ID of track to update
        track_analysis: Analysis results dict with keys:
            - bpm (float)
            - cue_points (list of dicts with 'time' and 'type')
            - brightness (float 0-100)
            - danceability (float 0-100)
            - mood (str, optional)
        db_path: Path to database for track info

    Returns:
        Updated ENTRY element or None if track not found in NML

    Example:
        >>> root = parse_traktor_nml("Collection.nml")
        >>> analysis = {
        ...     'bpm': 128.5,
        ...     'brightness': 75.5,
        ...     'danceability': 88.3,
        ...     'cue_points': [
        ...         {'time': 45.2, 'type': 'drop'},
        ...         {'time': 120.5, 'type': 'breakdown'},
        ...     ]
        ... }
        >>> track_entry = add_track_analysis(root, 42, analysis)
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # Get track info from database
        cursor = conn.execute(
            """
            SELECT t.file_path, t.file_name, t.title, t.artist
            FROM tracks t
            WHERE t.id = ?
            """,
            (track_id,)
        )
        track_row = cursor.fetchone()

        if track_row is None:
            logger.warning(f"Track not found in database: {track_id}")
            return None

        track_dict = dict(track_row)

        # Find matching entry in NML
        entry = None
        for e in nml_root.findall('ENTRY'):
            # Match by file path or title
            title_elem = e.find('TITLE')
            artist_elem = e.find('ARTIST')

            if (title_elem is not None and
                artist_elem is not None and
                title_elem.text == track_dict.get('title') and
                artist_elem.text == track_dict.get('artist')):
                entry = e
                break

        if entry is None:
            logger.warning(f"Track not found in NML: {track_dict.get('title')}")
            return None

        # Add/update TEMPO (BPM)
        if 'bpm' in track_analysis and track_analysis['bpm']:
            tempo_elem = _get_or_create_element(entry, 'TEMPO')
            tempo_elem.set('BPM', str(round(float(track_analysis['bpm']), 1)))

        # Add/update INFO with brightness and other metadata
        info_elem = _get_or_create_element(entry, 'INFO')

        if 'brightness' in track_analysis and track_analysis['brightness']:
            brightness = int(track_analysis['brightness'])
            info_elem.set('BRIGHTNESS', str(brightness))

        if 'danceability' in track_analysis and track_analysis['danceability']:
            danceability = int(track_analysis['danceability'])
            info_elem.set('DANCEABILITY', str(danceability))

        if 'mood' in track_analysis and track_analysis['mood']:
            info_elem.set('MOOD', str(track_analysis['mood']))

        # Add hot cues from structure points
        if 'cue_points' in track_analysis and track_analysis['cue_points']:
            cue_points = track_analysis['cue_points']

            # Map structure types to hot cue pads
            type_to_pad = {
                'drop': 1,
                'breakdown': 2,
                'outro': 4,
            }

            for cue in cue_points:
                cue_type = cue.get('type', '').lower()
                if cue_type not in type_to_pad:
                    continue

                pad_num = type_to_pad[cue_type]
                time_seconds = cue.get('time', 0.0)
                time_ms = int(time_seconds * 1000)

                # Create CUE_V2 element (Traktor Pro 3 format)
                cue_name = f"HotCue_{pad_num}"
                cue_elem = ET.Element(
                    'CUE_V2',
                    {
                        'NAME': cue_name,
                        'DISPL_ORDER': str(pad_num - 1),
                        'TYPE': '0',  # 0 = regular cue
                        'START': str(_ms_to_traktor_offset(time_ms)),
                        'LEN': '0',
                        'REPEATS': '-1',
                        'HOTCUE': str(pad_num),
                    }
                )

                # Remove any existing cue with same name
                existing_cues = entry.findall('CUE_V2')
                for existing in existing_cues:
                    if existing.get('NAME') == cue_name:
                        entry.remove(existing)

                entry.append(cue_elem)

        logger.info(f"Added analysis to track: {track_dict.get('title')}")
        return entry

    finally:
        conn.close()


def export_nml(nml_root: ET.Element, output_path: str) -> bool:
    """
    Write modified NML tree back to file.

    Validates XML is well-formed before writing.

    Args:
        nml_root: Root COLLECTION element (modified)
        output_path: Where to save the NML file

    Returns:
        True if successful

    Raises:
        IOError: If file write fails

    Example:
        >>> root = parse_traktor_nml("Collection.nml")
        >>> success = export_nml(root, "Collection_analyzed.nml")
        >>> print(f"Exported: {success}")
    """
    try:
        # Create tree from root
        tree = ET.ElementTree(nml_root)

        # Register default namespace to preserve Traktor format
        ET.register_namespace('', 'http://www.native-instruments.com/')

        # Write to file
        tree.write(output_path, encoding='utf-8', xml_declaration=True)

        logger.info(f"Exported NML to: {output_path}")

        # Validate by re-parsing
        try:
            test_parse = ET.parse(output_path)
            logger.info(f"Validated exported NML: {test_parse.getroot().tag}")
        except ET.ParseError as e:
            logger.error(f"Exported NML failed validation: {e}")
            return False

        return True

    except IOError as e:
        logger.error(f"Failed to export NML: {e}")
        raise


def export_all_tracks(
    traktor_nml_path: str,
    db_path: str = "data/djia.db",
    output_path: str = "results/collection_analyzed.nml"
) -> str:
    """
    Batch export: read Traktor collection, add analysis for all tracks, write back.

    Reads Traktor NML, fetches analysis from database for each track,
    adds hot cues + metadata, validates XML, writes to output_path.

    Args:
        traktor_nml_path: Path to original Traktor Collection.nml
        db_path: Path to DJIA SQLite database
        output_path: Where to save analyzed collection NML

    Returns:
        Path to exported NML file

    Raises:
        FileNotFoundError: If input NML not found
        IOError: If export fails

    Example:
        >>> output = export_all_tracks(
        ...     "Collection.nml",
        ...     "data/djia.db",
        ...     "results/collection_analyzed.nml"
        ... )
        >>> print(f"Saved to: {output}")
    """
    # Parse input NML
    nml_root = parse_traktor_nml(traktor_nml_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # Get all analyzed tracks
        cursor = conn.execute(
            """
            SELECT
                t.id,
                t.title,
                t.artist,
                f.bpm,
                s.segment_type,
                s.start_time
            FROM tracks t
            LEFT JOIN features f ON t.id = f.track_id
            LEFT JOIN segments s ON t.id = s.track_id
            ORDER BY t.id
            """
        )

        # Group segments by track_id
        track_segments = {}
        for row in cursor.fetchall():
            row_dict = dict(row)
            track_id = row_dict['id']

            if track_id not in track_segments:
                track_segments[track_id] = {
                    'id': track_id,
                    'bpm': row_dict['bpm'],
                    'cue_points': [],
                }

            if row_dict['segment_type'] and row_dict['start_time'] is not None:
                track_segments[track_id]['cue_points'].append({
                    'time': row_dict['start_time'],
                    'type': row_dict['segment_type'],
                })

        # Add analysis to each track
        updated_count = 0
        for track_id, analysis in track_segments.items():
            result = add_track_analysis(nml_root, track_id, analysis, db_path)
            if result is not None:
                updated_count += 1

        logger.info(f"Updated {updated_count} tracks in NML")

        # Export
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        success = export_nml(nml_root, output_path)

        if not success:
            raise IOError("Failed to export NML")

        return output_path

    finally:
        conn.close()
