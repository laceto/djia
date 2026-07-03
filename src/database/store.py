"""Database operations for DJIA."""

import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional

from .schema import get_connection, init_db

logger = logging.getLogger(__name__)


class TrackStore:
    """Database operations for tracks and features."""

    def __init__(self, db_path: str = "data/djia.db"):
        """Initialize store with database path."""
        self.db_path = db_path
        # Ensure database is initialized
        init_db(db_path)

    def insert_track(
        self,
        file_path: str,
        file_name: str,
        format: str,
        duration: float,
        artist: Optional[str] = None,
        title: Optional[str] = None,
        album: Optional[str] = None,
    ) -> int:
        """
        Insert a new track into the database.

        Args:
            file_path: Full path to audio file
            file_name: Name of the file
            format: Audio format (mp3, wav, etc.)
            duration: Track duration in seconds
            artist: Artist name
            title: Track title
            album: Album name

        Returns:
            Track ID
        """
        try:
            conn = get_connection(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO tracks
                (file_path, file_name, format, duration, artist, title, album)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (file_path, file_name, format, duration, artist, title, album))

            conn.commit()
            track_id = cursor.lastrowid
            logger.info(f"Inserted track {track_id}: {file_name}")
            return track_id
        except sqlite3.IntegrityError:
            logger.warning(f"Track already exists: {file_path}")
            # Return existing track ID
            return self.get_track_id(file_path)
        except Exception as e:
            logger.error(f"Error inserting track: {e}")
            raise
        finally:
            conn.close()

    def insert_features(
        self,
        track_id: int,
        features: Dict[str, Any],
    ) -> int:
        """
        Insert audio features for a track.

        Args:
            track_id: ID of the track
            features: Dictionary of audio features

        Returns:
            Feature record ID
        """
        try:
            conn = get_connection(self.db_path)
            cursor = conn.cursor()

            # Extract MFCC vector if present (will be serialized to JSON)
            mfcc_vector = None
            if 'mfcc_vector' in features:
                try:
                    mfcc_vector = json.dumps(features['mfcc_vector'].tolist()
                                            if hasattr(features['mfcc_vector'], 'tolist')
                                            else features['mfcc_vector'])
                except Exception as e:
                    logger.warning(f"Could not serialize MFCC vector: {e}")

            cursor.execute("""
                INSERT OR REPLACE INTO features
                (track_id, bpm, spectral_centroid_mean, spectral_centroid_std,
                 spectral_rolloff_mean, spectral_flux_mean, harmonic_ratio,
                 percussive_ratio, mfcc_mean, mfcc_std, mfcc_delta_mean,
                 chroma_variance, chroma_entropy, rms_mean, rms_std, rms_peak,
                 key, camelot_key, key_confidence, mfcc_vector)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                track_id,
                features.get('tempo'),
                features.get('spectral_centroid_mean'),
                features.get('spectral_centroid_std'),
                features.get('spectral_rolloff_mean'),
                features.get('spectral_flux_mean'),
                features.get('harmonic_ratio'),
                features.get('percussive_ratio'),
                features.get('mfcc_mean'),
                features.get('mfcc_std'),
                features.get('mfcc_delta_mean'),
                features.get('chroma_variance'),
                features.get('chroma_entropy'),
                features.get('rms_mean'),
                features.get('rms_std'),
                features.get('rms_peak'),
                features.get('key'),
                features.get('camelot_key'),
                features.get('key_confidence'),
                mfcc_vector,
            ))

            conn.commit()
            logger.debug(f"Inserted features for track {track_id}")
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error inserting features: {e}")
            raise
        finally:
            conn.close()

    def insert_mood(
        self,
        track_id: int,
        mood_scores: Dict[str, float],
    ) -> int:
        """
        Insert mood classification for a track.

        Args:
            track_id: ID of the track
            mood_scores: Dictionary with mood confidence scores

        Returns:
            Mood record ID
        """
        try:
            conn = get_connection(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO mood
                (track_id, dark, hypnotic, euphoric, aggressive, industrial, minimal)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                track_id,
                mood_scores.get('dark', 0.0),
                mood_scores.get('hypnotic', 0.0),
                mood_scores.get('euphoric', 0.0),
                mood_scores.get('aggressive', 0.0),
                mood_scores.get('industrial', 0.0),
                mood_scores.get('minimal', 0.0),
            ))

            conn.commit()
            logger.debug(f"Inserted mood for track {track_id}")
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error inserting mood: {e}")
            raise
        finally:
            conn.close()

    def insert_segment(
        self,
        track_id: int,
        segment_type: str,
        start_time: float,
        end_time: float,
        confidence: float = 1.0,
        method: str = "spectral",
    ) -> int:
        """
        Insert a segment for a track.

        Args:
            track_id: ID of the track
            segment_type: Type of segment (intro, build, drop, breakdown, outro, etc.)
            start_time: Start time in seconds
            end_time: End time in seconds
            confidence: Confidence score (0-1)
            method: How the segment was derived ('spectral' or 'phrase<N>')

        Returns:
            Segment record ID
        """
        try:
            conn = get_connection(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO segments
                (track_id, segment_type, start_time, end_time, confidence, method)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (track_id, segment_type, start_time, end_time, confidence, method))

            conn.commit()
            logger.debug(f"Inserted segment for track {track_id}")
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error inserting segment: {e}")
            raise
        finally:
            conn.close()

    def replace_segments(
        self,
        track_id: int,
        segments: List[Dict[str, Any]],
        method: str = "spectral",
    ) -> int:
        """
        Replace all segments of a given method for a track (idempotent re-analysis).

        Args:
            track_id: ID of the track
            segments: Dicts with segment_type, start_time, end_time, confidence
            method: How the segments were derived ('spectral' or 'phrase<N>')

        Returns:
            Number of segments inserted
        """
        try:
            conn = get_connection(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "DELETE FROM segments WHERE track_id = ? AND method = ?",
                (track_id, method),
            )
            cursor.executemany("""
                INSERT INTO segments
                (track_id, segment_type, start_time, end_time, confidence, method)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                (
                    track_id,
                    seg["segment_type"],
                    seg["start_time"],
                    seg["end_time"],
                    seg.get("confidence", 1.0),
                    method,
                )
                for seg in segments
            ])

            conn.commit()
            logger.debug(f"Replaced {len(segments)} '{method}' segments for track {track_id}")
            return len(segments)
        except Exception as e:
            logger.error(f"Error replacing segments: {e}")
            raise
        finally:
            conn.close()

    def get_track(self, track_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a track by ID.

        Args:
            track_id: ID of the track

        Returns:
            Track data as dictionary or None
        """
        try:
            conn = get_connection(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM tracks WHERE id = ?", (track_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting track: {e}")
            return None
        finally:
            conn.close()

    def get_track_id(self, file_path: str) -> Optional[int]:
        """
        Get track ID by file path.

        Args:
            file_path: Path to audio file

        Returns:
            Track ID or None
        """
        try:
            conn = get_connection(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM tracks WHERE file_path = ?", (file_path,))
            row = cursor.fetchone()
            return row['id'] if row else None
        except Exception as e:
            logger.error(f"Error getting track ID: {e}")
            return None
        finally:
            conn.close()

    def get_all_tracks(self) -> List[Dict[str, Any]]:
        """
        Get all tracks from database.

        Returns:
            List of track dictionaries
        """
        try:
            conn = get_connection(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM tracks ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting all tracks: {e}")
            return []
        finally:
            conn.close()

    def get_tracks_count(self) -> int:
        """Get total number of tracks in database."""
        try:
            conn = get_connection(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) as count FROM tracks")
            row = cursor.fetchone()
            return row['count'] if row else 0
        except Exception as e:
            logger.error(f"Error getting tracks count: {e}")
            return 0
        finally:
            conn.close()

    def get_track_features(self, track_id: int) -> Optional[Dict[str, Any]]:
        """
        Get features for a track.

        Args:
            track_id: ID of the track

        Returns:
            Features as dictionary or None
        """
        try:
            conn = get_connection(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM features WHERE track_id = ?", (track_id,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                # Parse MFCC vector if present
                if result.get('mfcc_vector'):
                    try:
                        result['mfcc_vector'] = json.loads(result['mfcc_vector'])
                    except Exception as e:
                        logger.warning(f"Could not parse MFCC vector: {e}")
                return result
            return None
        except Exception as e:
            logger.error(f"Error getting track features: {e}")
            return None
        finally:
            conn.close()

    def get_track_mood(self, track_id: int) -> Optional[Dict[str, Any]]:
        """
        Get mood classification for a track.

        Args:
            track_id: ID of the track

        Returns:
            Mood data as dictionary or None
        """
        try:
            conn = get_connection(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM mood WHERE track_id = ?", (track_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting track mood: {e}")
            return None
        finally:
            conn.close()

    def get_track_segments(self, track_id: int) -> List[Dict[str, Any]]:
        """
        Get all segments for a track.

        Args:
            track_id: ID of the track

        Returns:
            List of segment dictionaries
        """
        try:
            conn = get_connection(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM segments WHERE track_id = ?
                ORDER BY start_time
            """, (track_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting track segments: {e}")
            return []
        finally:
            conn.close()

    def delete_track(self, track_id: int) -> bool:
        """
        Delete a track and all related data.

        Args:
            track_id: ID of the track

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = get_connection(self.db_path)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM tracks WHERE id = ?", (track_id,))
            conn.commit()
            logger.info(f"Deleted track {track_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting track: {e}")
            return False
        finally:
            conn.close()

    def search_tracks(self, query: str) -> List[Dict[str, Any]]:
        """
        Search tracks by artist, title, or album.

        Args:
            query: Search query string

        Returns:
            List of matching track dictionaries
        """
        try:
            conn = get_connection(self.db_path)
            cursor = conn.cursor()

            search_term = f"%{query}%"
            cursor.execute("""
                SELECT * FROM tracks
                WHERE artist LIKE ? OR title LIKE ? OR album LIKE ?
                ORDER BY artist, title
            """, (search_term, search_term, search_term))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error searching tracks: {e}")
            return []
        finally:
            conn.close()

    def get_segments_by_type(self, track_id: int, segment_type: str) -> Optional[Dict[str, Any]]:
        """
        Get the first segment of a specific type for a track.

        Used for Traktor export (finding drop, breakdown, outro times).

        Args:
            track_id: ID of the track
            segment_type: Type of segment to find (drop, breakdown, outro, etc.)

        Returns:
            First matching segment as dictionary or None
        """
        try:
            conn = get_connection(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM segments
                WHERE track_id = ? AND segment_type = ?
                ORDER BY start_time
                LIMIT 1
            """, (track_id, segment_type))
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting segments by type: {e}")
            return None
        finally:
            conn.close()

    def get_all_tracks_with_features(self) -> List[Dict[str, Any]]:
        """
        Get all tracks with their features joined.

        Used for similarity matching and batch operations.

        Returns:
            List of track dictionaries with features included
        """
        try:
            conn = get_connection(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT t.*, f.*
                FROM tracks t
                LEFT JOIN features f ON t.id = f.track_id
                ORDER BY t.id
            """)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting all tracks with features: {e}")
            return []
        finally:
            conn.close()
