"""
Integration example: Using ingestion and database modules together.

This script demonstrates the complete pipeline for ingesting audio files,
extracting their features, and storing them in the SQLite database.
"""

import logging
from pathlib import Path
from src.ingestion import AudioScanner, AudioLoader
from src.database import TrackStore
from src.audio_analysis import analyze_track

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ingest_all_tracks(data_dir: str = "data", db_path: str = "data/djia.db") -> None:
    """
    Complete ingestion pipeline: scan, load, analyze, and store tracks.

    Args:
        data_dir: Directory containing audio files
        db_path: Path to SQLite database
    """
    logger.info("Starting DJIA ingestion pipeline...")

    # Initialize components
    scanner = AudioScanner(data_dir)
    loader = AudioLoader(target_sr=22050)
    store = TrackStore(db_path)

    # Scan for audio files
    logger.info(f"Scanning {data_dir} for audio files...")
    audio_files = scanner.scan()
    logger.info(f"Found {len(audio_files)} audio file(s)")

    if not audio_files:
        logger.warning("No audio files found!")
        return

    # Process each file
    processed = 0
    failed = 0

    for file_info in audio_files:
        file_path = file_info['path']
        file_name = file_info['name']

        try:
            logger.info(f"Processing: {file_name}")

            # Extract metadata
            metadata = loader.extract_metadata(file_path)

            # Insert track into database
            track_id = store.insert_track(
                file_path=str(file_path),
                file_name=file_name,
                format=file_info['format'],
                duration=metadata.get('duration', 0.0) or 0.0,
                artist=metadata.get('artist'),
                title=metadata.get('title'),
                album=metadata.get('album'),
            )
            logger.debug(f"Inserted track {track_id}: {file_name}")

            # Load audio and extract features
            audio_data = loader.load_audio(file_path)
            if audio_data:
                y, sr = audio_data
                features = analyze_track(str(file_path))

                if features:
                    # Store features in database
                    store.insert_features(track_id, features)
                    logger.debug(f"Stored features for track {track_id}")
                    logger.info(f"  BPM: {features.get('tempo', 'N/A'):.1f}")
                    logger.info(f"  Duration: {features.get('duration', 'N/A'):.1f}s")
                    processed += 1
                else:
                    logger.warning(f"Failed to extract features from {file_name}")
                    failed += 1
            else:
                logger.warning(f"Failed to load audio: {file_name}")
                failed += 1

        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
            failed += 1

    # Summary
    logger.info("=" * 70)
    logger.info("INGESTION COMPLETE")
    logger.info(f"Total files: {len(audio_files)}")
    logger.info(f"Successfully processed: {processed}")
    logger.info(f"Failed: {failed}")
    logger.info("=" * 70)

    # Database statistics
    total_tracks = store.get_tracks_count()
    logger.info(f"Total tracks in database: {total_tracks}")


def query_database(db_path: str = "data/djia.db", search_query: str = None) -> None:
    """
    Query and display database contents.

    Args:
        db_path: Path to SQLite database
        search_query: Optional search query (artist/title/album)
    """
    store = TrackStore(db_path)

    if search_query:
        logger.info(f"Searching for: {search_query}")
        tracks = store.search_tracks(search_query)
    else:
        logger.info("Retrieving all tracks...")
        tracks = store.get_all_tracks()

    if not tracks:
        logger.warning("No tracks found")
        return

    logger.info(f"Found {len(tracks)} track(s)")
    logger.info("=" * 70)

    for track in tracks:
        logger.info(f"ID: {track['id']}")
        logger.info(f"  Title: {track['title']}")
        logger.info(f"  Artist: {track['artist']}")
        logger.info(f"  Album: {track['album']}")
        logger.info(f"  Duration: {track['duration']:.1f}s")
        logger.info(f"  Format: {track['format']}")

        # Get features if available
        features = store.get_track_features(track['id'])
        if features:
            logger.info(f"  BPM: {features.get('bpm', 'N/A')}")
            logger.info(f"  Spectral Centroid: {features.get('spectral_centroid_mean', 'N/A')}")
        logger.info("-" * 70)


if __name__ == "__main__":
    # Run the ingestion pipeline
    ingest_all_tracks(data_dir="data", db_path="data/djia.db")

    # Query some results
    print("\n")
    query_database(db_path="data/djia.db")
