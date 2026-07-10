"""Master orchestrator that ties together all analysis phases."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm

from .ingestion.scanner import AudioScanner
from .ingestion.loader import AudioLoader
from .audio_analysis import analyze_track as analyze_audio
from .dsp.config import get_config
from .dsp.groove_engine import analyze_groove
from .dsp.mood_engine import analyze_mood as analyze_tonality
from .dsp.curation_engine import compute_spectral_flatness, compute_crest_factor
from .dsp.phrasing_engine import analyze_structure, create_phrase_locked_segments
from .ai.classifier import MoodClassifier
from .database.store import TrackStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Orchestrator:
    """Master orchestrator for complete DJIA pipeline."""

    def __init__(
        self,
        db_path: str = "data/djia.db",
        segment_preset: str = "minimal",
        bars_per_phrase: int = 16,
    ):
        """
        Initialize orchestrator with database.

        Args:
            db_path: Path to SQLite database
            segment_preset: DSP config preset for spectral segment detection
            bars_per_phrase: Phrase length (bars) for the phrase-locked segment grid
        """
        self.db_path = db_path
        self.store = TrackStore(db_path)
        self.loader = AudioLoader()
        self.mood_classifier = MoodClassifier()
        self.segment_config = get_config(segment_preset)
        self.bars_per_phrase = bars_per_phrase

    @staticmethod
    def _segment_dicts(segments) -> List[Dict[str, Any]]:
        """Convert phrasing Segment objects to store dicts, stripping beat ranges from labels."""
        return [
            {
                # "drop (beats 32-64)" -> "drop"
                'segment_type': seg.label.split('(')[0].strip(),
                'start_time': seg.start_time,
                'end_time': seg.end_time,
                'confidence': seg.confidence,
            }
            for seg in segments
        ]

    def _add_segments(self, track_id: int, y, sr, bpm, file_path) -> None:
        """Detect and persist structure segments — spectral + phrase-locked grid (best-effort)."""
        if not bpm:
            logger.warning(f"No BPM for {file_path}; skipping segment detection")
            return
        try:
            phrasing_cfg = self.segment_config.phrasing
            result = analyze_structure(
                y, sr, bpm,
                hop_length=self.segment_config.hop_length,
                novelty_threshold=phrasing_cfg.novelty_threshold,
                min_segment_duration=phrasing_cfg.min_segment_duration,
                breakdown_threshold=phrasing_cfg.breakdown_duration_threshold,
            )
            self.store.replace_segments(
                track_id, self._segment_dicts(result.segments), method="spectral"
            )

            locked = create_phrase_locked_segments(
                duration=len(y) / sr,
                bpm=bpm,
                bars_per_phrase=self.bars_per_phrase,
            )
            self.store.replace_segments(
                track_id, self._segment_dicts(locked), method=f"phrase{self.bars_per_phrase}"
            )
        except Exception as e:
            logger.warning(f"Failed to detect segments for {file_path}: {e}")

    def _add_tonality(self, features: Dict[str, Any], y, sr, file_path) -> None:
        """Detect musical key/Camelot/timbral roughness/ZCR and merge into the features dict
        (best-effort)."""
        try:
            tonality = analyze_tonality(y, sr)
            features['key'] = tonality.key
            features['camelot_key'] = tonality.camelot_key
            features['key_confidence'] = tonality.key_confidence
            features['zero_crossing_rate'] = tonality.zero_crossing_rate
            features['roughness'] = tonality.roughness
        except Exception as e:
            logger.warning(f"Failed to detect key for {file_path}: {e}")

    def _add_swing(self, features: Dict[str, Any], y, sr, file_path) -> None:
        """Measure swing, onset strength, and beat strength; merge into the features dict
        (best-effort)."""
        try:
            groove = analyze_groove(y, sr)
            features['swing_score'] = groove.swing_score
            features['onset_strength_mean'] = groove.onset_strength_mean
            features['onset_strength_std'] = groove.onset_strength_std
            features['beat_strength'] = groove.beat_strength
        except Exception as e:
            logger.warning(f"Failed to measure swing for {file_path}: {e}")

    def _add_density(self, features: Dict[str, Any], y, sr, file_path) -> None:
        """Measure spectral density (flatness, crest factor) and merge into the features dict
        (best-effort). Crest factor reuses rms_mean/rms_peak already extracted by analyze_audio."""
        try:
            features['spectral_flatness'] = compute_spectral_flatness(y, sr)
            features['crest_factor'] = compute_crest_factor(
                rms_mean=features.get('rms_mean'),
                rms_peak=features.get('rms_peak'),
            )
        except Exception as e:
            logger.warning(f"Failed to measure spectral density for {file_path}: {e}")

    def analyze_library(
        self,
        data_dir: str = "data",
        skip_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Analyze entire audio library through all phases.

        Pipeline:
        1. Scan directory for audio files
        2. Load audio and extract metadata
        3. Extract DSP features (audio analysis)
        4. Detect structure segments (spectral + phrase-locked grid)
        5. Classify mood
        6. Store in database

        Args:
            data_dir: Directory containing audio files
            skip_existing: Skip tracks already in database

        Returns:
            Summary dict with analyzed/skipped/error counts
        """
        # Phase 1: Scan
        logger.info(f"Scanning directory: {data_dir}")
        scanner = AudioScanner(data_dir)
        audio_files = scanner.scan()

        if not audio_files:
            logger.warning(f"No audio files found in {data_dir}")
            return {
                'analyzed': 0,
                'skipped': 0,
                'errors': 0,
                'db_path': self.db_path,
            }

        # Filter out existing tracks if skip_existing
        if skip_existing:
            existing_paths = set()
            for track in self.store.get_all_tracks():
                existing_paths.add(track['file_path'])

            audio_files = [
                f for f in audio_files
                if str(f['path']) not in existing_paths
            ]

        logger.info(f"Found {len(audio_files)} audio file(s)")

        # Analyze each track
        analyzed = 0
        skipped = 0
        errors = 0

        for file_info in tqdm(audio_files, desc="Analyzing tracks", unit="track"):
            try:
                file_path = file_info['path']

                # Phase 2: Load metadata
                metadata = self.loader.extract_metadata(file_path)

                # Check if file is in database already
                existing_id = self.store.get_track_id(str(file_path))
                if existing_id:
                    if skip_existing:
                        skipped += 1
                        continue
                    track_id = existing_id
                else:
                    # Insert track into database
                    track_id = self.store.insert_track(
                        file_path=str(file_path),
                        file_name=metadata['file_name'],
                        format=metadata['format'],
                        duration=metadata.get('duration', 0),
                        artist=metadata.get('artist'),
                        title=metadata.get('title'),
                        album=metadata.get('album'),
                    )

                # Phase 3: Extract DSP features
                audio_data = self.loader.load_audio(file_path)
                if not audio_data:
                    logger.error(f"Failed to load audio: {file_path}")
                    errors += 1
                    continue

                y = audio_data['audio_array']
                sr = audio_data['sample_rate']
                features = analyze_audio(str(file_path), sr, None)

                if not features:
                    logger.error(f"Failed to extract features: {file_path}")
                    errors += 1
                    continue

                # analyze_audio() only sets 'tempo', not 'bpm' — alias so downstream
                # callers that read features.get('bpm') (e.g. _add_segments) see it too.
                features.setdefault('bpm', features.get('tempo'))

                # Detect musical key (Camelot), timbre, swing, and density; merge into features
                self._add_tonality(features, y, sr, file_path)
                self._add_swing(features, y, sr, file_path)
                self._add_density(features, y, sr, file_path)

                # Store features
                self.store.insert_features(track_id, features)

                # Detect structure segments (spectral + 16-bar phrase grid) and persist
                self._add_segments(track_id, y, sr, features.get('bpm'), file_path)

                # Phase 4: Classify mood
                try:
                    mood_result = self.mood_classifier.classify_mood(y, sr)
                    if mood_result and 'moods' in mood_result:
                        self.store.insert_mood(track_id, mood_result['moods'])
                except Exception as e:
                    logger.warning(f"Failed to classify mood for {file_path}: {e}")

                analyzed += 1
                logger.info(f"✓ Analyzed: {Path(file_path).name}")

            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
                errors += 1

        return {
            'analyzed': analyzed,
            'skipped': skipped,
            'errors': errors,
            'db_path': self.db_path,
        }

    def analyze_single_track(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a single track through complete pipeline.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with track features, or None if failed
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        try:
            # Load metadata
            metadata = self.loader.extract_metadata(file_path)

            # Load audio
            audio_data = self.loader.load_audio(file_path)
            if not audio_data:
                logger.error(f"Failed to load audio: {file_path}")
                return None

            y = audio_data['audio_array']
            sr = audio_data['sample_rate']

            # Extract features
            features = analyze_audio(str(file_path), sr, None)
            if not features:
                logger.error(f"Failed to extract features: {file_path}")
                return None

            # analyze_audio() only sets 'tempo', not 'bpm' — alias for downstream callers.
            features.setdefault('bpm', features.get('tempo'))

            # Detect musical key (Camelot), timbre, swing, and density; merge into features
            self._add_tonality(features, y, sr, file_path)
            self._add_swing(features, y, sr, file_path)
            self._add_density(features, y, sr, file_path)

            # Classify mood
            try:
                mood_result = self.mood_classifier.classify_mood(y, sr)
                if mood_result and 'moods' in mood_result:
                    features['mood'] = mood_result['moods']
            except Exception as e:
                logger.warning(f"Failed to classify mood: {e}")

            # Add metadata
            features.update(metadata)

            return features

        except Exception as e:
            logger.error(f"Error analyzing track: {e}")
            return None

    def get_all_tracks_dict(self) -> Dict[int, Dict[str, Any]]:
        """
        Get all tracks from database as a dictionary with features.

        Returns:
            Dict mapping track_id -> features_dict
        """
        tracks_dict = {}
        all_tracks = self.store.get_all_tracks()

        for track in all_tracks:
            track_id = track['id']

            # Get features
            features = self.store.get_track_features(track_id)

            if features:
                # Get mood
                mood = self.store.get_track_mood(track_id)

                # Build features dict
                track_features = {
                    'id': track_id,
                    'file_name': track['file_name'],
                    'duration': track.get('duration', 0),
                    'artist': track.get('artist'),
                    'title': track.get('title'),
                    'tempo': features.get('bpm'),
                    'rms_mean': features.get('rms_mean'),
                    'key': features.get('key'),
                    'mood': mood if mood else {},
                }

                # Add other features
                track_features.update(features)
                tracks_dict[track_id] = track_features

        return tracks_dict

    def get_track_count(self) -> int:
        """Get total number of analyzed tracks."""
        return self.store.get_tracks_count()
