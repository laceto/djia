"""
Phase 4 Example: Track Similarity Engine and Traktor NML Export

Demonstrates:
1. Normalizing track features for similarity computation
2. Finding similar tracks using cosine similarity
3. Exporting hot cues and analysis to Traktor NML format
"""

from pathlib import Path
from src.database.schema import init_db
from src.database.store import TrackStore
from src.matching.similarity import find_similar_tracks, normalize_features
from src.traktor.exporter import export_all_tracks, parse_traktor_nml
import numpy as np


def example_similarity_search():
    """
    Example: Find similar tracks to a given track.

    This workflow:
    1. Loads track features from database
    2. Normalizes all features to zero mean, unit variance
    3. Computes cosine similarity between query track and all others
    4. Returns top-K most similar tracks sorted by score
    """
    print("\n=== Phase 4: Similarity Search Example ===\n")

    db_path = "db/djia.db"
    store = TrackStore(db_path)

    # Get all tracks
    all_tracks = store.get_all_tracks()
    if len(all_tracks) < 2:
        print(f"Need at least 2 tracks in database. Found: {len(all_tracks)}")
        return

    query_track_id = all_tracks[0]['id']
    query_track = store.get_track(query_track_id)

    print(f"Query Track: {query_track.get('title', 'Unknown')} by {query_track.get('artist', 'Unknown')}")
    print(f"Track ID: {query_track_id}\n")

    # Find similar tracks - no filters
    print("Finding 5 most similar tracks (all)...")
    similar_all = find_similar_tracks(query_track_id, top_k=5, db_path=db_path)

    if similar_all:
        print(f"Found {len(similar_all)} matches:\n")
        for track, score in similar_all:
            print(f"  • {track.get('title', 'Unknown')} ({track.get('artist', 'Unknown')})")
            print(f"    BPM: {track.get('bpm', 'N/A'):.1f}, Similarity: {score:.3f}\n")
    else:
        print("No similar tracks found.")

    # Find similar with BPM tolerance
    query_bpm = store.get_track_features(query_track_id).get('bpm', 128.0)
    print(f"\nFinding similar tracks with BPM ±2 (±{query_bpm - 2:.0f}-{query_bpm + 2:.0f})...")

    similar_bpm = find_similar_tracks(
        query_track_id,
        top_k=5,
        bpm_tolerance=2.0,
        db_path=db_path
    )

    if similar_bpm:
        print(f"Found {len(similar_bpm)} matches:\n")
        for track, score in similar_bpm:
            bpm = track.get('bpm', 'N/A')
            print(f"  • {track.get('title', 'Unknown')} - BPM: {bpm}, Similarity: {score:.3f}")
    else:
        print("No similar tracks found with BPM filter.")


def example_traktor_export():
    """
    Example: Export analyzed tracks to Traktor NML.

    This workflow:
    1. Loads original Traktor Collection.nml
    2. Reads analyzed features and structure from database
    3. Adds hot cues (drop=Pad1, breakdown=Pad2, outro=Pad4)
    4. Adds metadata (BPM, brightness, danceability)
    5. Exports to results/collection_analyzed.nml

    The exported file can be opened in Traktor Pro 3+
    """
    print("\n=== Phase 4: Traktor NML Export Example ===\n")

    # Check if Traktor NML exists
    traktor_nml = Path("Collection.nml")

    if not traktor_nml.exists():
        print(f"Note: Traktor Collection.nml not found at {traktor_nml}")
        print("To use Traktor export:")
        print("  1. Export Collection.nml from Traktor Pro (File > Export Collection)")
        print("  2. Place it in the project root directory")
        print("  3. Re-run this example\n")
        return

    db_path = "db/djia.db"
    output_nml = "results/collection_analyzed.nml"

    print(f"Input NML: {traktor_nml}")
    print(f"Database: {db_path}")
    print(f"Output NML: {output_nml}\n")

    # Parse input
    print("Parsing Traktor Collection.nml...")
    root = parse_traktor_nml(str(traktor_nml))
    entry_count = root.get('ENTRIES', 'Unknown')
    print(f"Found {entry_count} tracks in collection\n")

    # Export with analysis
    print("Exporting with analysis...")
    try:
        result_path = export_all_tracks(
            str(traktor_nml),
            db_path,
            output_nml
        )
        print(f"✓ Exported to: {result_path}")
        print("\nHot Cues Added:")
        print("  Pad 1: Drop (first sudden energy increase)")
        print("  Pad 2: Breakdown (first reduction in percussion)")
        print("  Pad 4: Outro (end section)\n")
        print("To use in Traktor:")
        print("  1. Open Traktor Pro")
        print("  2. File > Import Collection")
        print("  3. Select collection_analyzed.nml")
        print("  4. Hot cues will appear automatically on playback\n")
    except Exception as e:
        print(f"✗ Export failed: {e}")


def example_feature_normalization():
    """
    Example: Feature normalization for similarity.

    Shows how individual track features are normalized
    to ensure equal contribution to similarity metric.
    """
    print("\n=== Phase 4: Feature Normalization Example ===\n")

    db_path = "db/djia.db"
    store = TrackStore(db_path)

    all_tracks = store.get_all_tracks()
    if not all_tracks:
        print("No tracks in database")
        return

    track_id = all_tracks[0]['id']
    features = store.get_track_features(track_id)

    if not features:
        print(f"No features for track {track_id}")
        return

    track = store.get_track(track_id)
    print(f"Track: {track.get('title', 'Unknown')}\n")

    # Show raw features
    print("Raw Features (selected):")
    raw_features = {
        'BPM': features.get('bpm', 0),
        'Spectral Centroid (Hz)': features.get('spectral_centroid_mean', 0),
        'RMS Peak': features.get('rms_peak', 0),
        'Harmonic Ratio': features.get('harmonic_ratio', 0),
    }

    for name, value in raw_features.items():
        print(f"  {name}: {value}")

    # Normalize
    print("\nNormalized Features (Z-score):")
    normalized = normalize_features(features)

    print(f"  Normalized vector shape: {normalized.shape}")
    print(f"  Vector mean: {np.mean(normalized):.6f} (should be ~0)")
    print(f"  Vector std: {np.std(normalized):.6f} (should be ~1)")
    print("\nNormalization ensures:")
    print("  • All features have equal weight in similarity")
    print("  • Large values (BPM) don't dominate small values (ratios)")
    print("  • Cross-track comparisons are fair\n")


def main():
    """Run all Phase 4 examples."""
    print("="*70)
    print("DJIA Phase 4: Similarity Engine & Traktor Export")
    print("="*70)

    # Initialize database
    db_path = "db/djia.db"
    if not Path(db_path).exists():
        print(f"\nInitializing database: {db_path}")
        init_db(db_path)

    # Run examples
    example_feature_normalization()
    example_similarity_search()
    example_traktor_export()

    print("\n" + "="*70)
    print("Phase 4 Examples Complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
