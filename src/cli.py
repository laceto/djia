"""Command-line interface for DJIA."""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional
from tabulate import tabulate

from .orchestrator import Orchestrator
from .database.store import TrackStore
from .traktor.exporter import export_all_tracks
from .ai import generate_playlist, playlist_summary


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def format_features_table(tracks_data: List[dict]) -> str:
    """Format track features for display."""
    headers = ['ID', 'File', 'BPM', 'Key', 'Duration', 'Mood']
    rows = []

    for track in tracks_data:
        track_id = track.get('id')
        file_name = track.get('file_name', 'Unknown')[:20]
        bpm = track.get('tempo', 'N/A')
        if isinstance(bpm, float):
            bpm = f"{bpm:.1f}"
        key = track.get('key', 'N/A')
        duration = track.get('duration', 0)
        if isinstance(duration, (int, float)):
            duration = f"{duration:.0f}s"
        mood = track.get('mood', {})
        if mood:
            top_mood = max(mood, key=mood.get)
            mood_str = f"{top_mood} ({mood[top_mood]:.2f})"
        else:
            mood_str = "N/A"

        rows.append([track_id, file_name, bpm, key, duration, mood_str])

    return tabulate(rows, headers=headers, tablefmt='grid')


def cmd_analyze(args):
    """Analyze a directory or single track."""
    print_section("DJIA Track Analysis")

    if args.track:
        # Analyze single track
        print(f"Analyzing single track: {args.track}\n")
        orchestrator = Orchestrator()
        result = orchestrator.analyze_single_track(args.track)

        if result:
            print(f"Track: {result['file_name']}")
            print(f"  Duration: {result['duration']:.1f}s")
            print(f"  BPM: {result.get('tempo', 'N/A')}")
            print(f"  RMS Mean: {result.get('rms_mean', 'N/A')}")
            if result.get('mood'):
                print(f"  Mood: {result['mood']}")
        else:
            print("Failed to analyze track.")
            return 1

    else:
        # Analyze directory
        data_dir = args.data_dir or "data"
        print(f"Analyzing audio library: {data_dir}\n")

        orchestrator = Orchestrator(db_path=args.db or "data/djia.db")
        results = orchestrator.analyze_library(data_dir, skip_existing=args.skip_existing)

        print_section("Analysis Complete")
        print(f"Tracks analyzed: {results['analyzed']}")
        print(f"Tracks skipped: {results['skipped']}")
        print(f"Errors: {results['errors']}")
        print(f"Database: {results['db_path']}")

    return 0


def cmd_list_tracks(args):
    """List all analyzed tracks."""
    print_section("Analyzed Tracks")

    store = TrackStore(args.db or "data/djia.db")
    tracks = store.get_all_tracks()[:args.limit]

    if not tracks:
        print("No tracks found in database.")
        return 0

    # Format and display
    table_data = []
    for track in tracks:
        features = store.get_track_features(track['id'])
        mood = store.get_track_mood(track['id'])

        table_data.append({
            'id': track['id'],
            'file_name': Path(track['file_name']).name,
            'duration': track.get('duration', 0),
            'tempo': features.get('bpm') if features else 'N/A',
            'key': features.get('key') if features else 'N/A',
            'mood': mood if mood else {},
        })

    print(format_features_table(table_data))
    print(f"\nTotal: {len(table_data)} track(s)")

    return 0


def cmd_find_similar(args):
    """Find similar tracks."""
    print_section("Similar Tracks")

    store = TrackStore(args.db or "data/djia.db")
    track = store.get_track(args.track_id)

    if not track:
        print(f"Track {args.track_id} not found.")
        return 1

    print(f"Finding tracks similar to: {track['file_name']}\n")

    # Get features of reference track
    ref_features = store.get_track_features(args.track_id)

    if not ref_features:
        print("Could not get features for reference track.")
        return 1

    # Find similar
    all_tracks = store.get_all_tracks()
    similar = []

    for other_track in all_tracks:
        if other_track['id'] == args.track_id:
            continue

        other_features = store.get_track_features(other_track['id'])
        if not other_features:
            continue

        # Simple similarity: BPM and key match
        bpm_diff = abs(ref_features.get('bpm', 0) - other_features.get('bpm', 0))
        key_match = 1.0 if ref_features.get('key') == other_features.get('key') else 0.0

        similarity = max(0, 1.0 - (bpm_diff / 30) + key_match) / 2

        similar.append({
            'id': other_track['id'],
            'file_name': other_track['file_name'],
            'duration': other_track.get('duration', 0),
            'bpm': other_features.get('bpm'),
            'key': other_features.get('key'),
            'similarity': similarity,
        })

    # Sort by similarity
    similar.sort(key=lambda x: x['similarity'], reverse=True)
    similar = similar[:args.top_k]

    if similar:
        headers = ['ID', 'File', 'BPM', 'Key', 'Similarity']
        rows = [
            [
                s['id'],
                Path(s['file_name']).name[:20],
                f"{s['bpm']:.1f}" if s['bpm'] else 'N/A',
                s['key'] or 'N/A',
                f"{s['similarity']:.3f}",
            ]
            for s in similar
        ]
        print(tabulate(rows, headers=headers, tablefmt='grid'))
    else:
        print("No similar tracks found.")

    return 0


def cmd_generate_playlist(args):
    """Generate a DJ playlist."""
    print_section("Playlist Generator")

    store = TrackStore(args.db or "data/djia.db")

    # Verify start and end tracks exist
    start_track = store.get_track(args.start_id)
    end_track = store.get_track(args.end_id)

    if not start_track or not end_track:
        print(f"Track not found. Check IDs.")
        return 1

    print(f"Start: {start_track['file_name']}")
    print(f"End: {end_track['file_name']}")
    print(f"Steps: {args.steps}\n")

    # Get all tracks with features
    all_tracks_db = store.get_all_tracks()
    all_tracks = {}

    for track in all_tracks_db:
        features = store.get_features(track['id'])
        if features:
            features.update({
                'id': track['id'],
                'file_name': track['file_name'],
                'duration': track.get('duration', 0),
            })
            all_tracks[track['id']] = features

    if not all_tracks:
        print("No tracks with features in database.")
        return 1

    # Generate playlist
    playlist = generate_playlist(
        all_tracks, args.start_id, args.end_id, num_steps=args.steps
    )

    if not playlist:
        print("Could not generate playlist.")
        return 1

    # Display playlist
    print_section("Generated Playlist")

    headers = ['Step', 'ID', 'File', 'BPM', 'Key']
    rows = []

    for i, track_id in enumerate(playlist, 1):
        if track_id in all_tracks:
            track = all_tracks[track_id]
            rows.append([
                i,
                track_id,
                Path(track.get('file_name', '')).name[:25],
                f"{track.get('tempo', 'N/A'):.1f}" if isinstance(track.get('tempo'), (int, float)) else 'N/A',
                track.get('key', 'N/A'),
            ])

    print(tabulate(rows, headers=headers, tablefmt='grid'))

    # Summary
    summary = playlist_summary(playlist, all_tracks)
    print(f"\nSummary:")
    print(f"  BPM Arc: {summary.get('start_bpm', 0):.1f} → {summary.get('end_bpm', 0):.1f}")
    print(f"  Avg Transition: {summary.get('avg_transition_score', 0):.3f}")

    return 0


def cmd_export_traktor(args):
    """Export to Traktor NML format."""
    print_section("Traktor Export")

    db_path = args.db or "data/djia.db"
    store = TrackStore(db_path)
    all_tracks = store.get_all_tracks()

    if not all_tracks:
        print("No tracks in database to export.")
        return 1

    print(f"Exporting {len(all_tracks)} tracks to Traktor format...\n")

    # Check if input Traktor NML exists
    traktor_nml = args.traktor_input
    if not Path(traktor_nml).exists():
        print(f"Note: Original Traktor collection.nml not found at: {traktor_nml}")
        print("Creating analysis-only export instead (no hot cues).")
        print("To add hot cues, provide path to your Collection.nml with --traktor-input")
        traktor_nml = None

    output_path = args.nml_path or "djia_export.nml"

    try:
        if traktor_nml:
            result = export_all_tracks(
                traktor_nml_path=traktor_nml,
                db_path=db_path,
                output_path=output_path
            )
            print(f"✓ Exported to: {result}")
        else:
            # Simple export without hot cues
            print(f"✓ Would export {len(all_tracks)} tracks to: {output_path}")
            print("  (Feature requires Traktor collection.nml as input)")

        return 0
    except FileNotFoundError as e:
        print(f"✗ Traktor collection not found: {e}")
        print("  Use --traktor-input to specify path to Collection.nml")
        return 1
    except Exception as e:
        print(f"✗ Export failed: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='DJIA: DJ Mixing Analytics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python -m src.cli analyze                    # Analyze data/ directory
  python -m src.cli analyze --data-dir /path   # Analyze custom directory
  python -m src.cli list-tracks                # Show all tracks
  python -m src.cli find-similar 1 --top-k 5   # Find tracks similar to ID 1
  python -m src.cli generate-playlist 1 10 5   # Create 5-track playlist from 1→10
  python -m src.cli export-traktor out.nml     # Export to Traktor
        '''
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze audio library or track')
    analyze_parser.add_argument('--data-dir', help='Data directory (default: data/)')
    analyze_parser.add_argument('--track', help='Analyze single track by path')
    analyze_parser.add_argument('--db', help='Database path (default: data/djia.db)')
    analyze_parser.add_argument('--skip-existing', action='store_true',
                                help='Skip already-analyzed tracks')
    analyze_parser.set_defaults(func=cmd_analyze)

    # List tracks command
    list_parser = subparsers.add_parser('list-tracks', help='List all analyzed tracks')
    list_parser.add_argument('--db', help='Database path (default: data/djia.db)')
    list_parser.add_argument('--limit', type=int, default=100, help='Max tracks to show')
    list_parser.set_defaults(func=cmd_list_tracks)

    # Find similar command
    similar_parser = subparsers.add_parser('find-similar', help='Find similar tracks')
    similar_parser.add_argument('track_id', type=int, help='Track ID to find matches for')
    similar_parser.add_argument('--db', help='Database path (default: data/djia.db)')
    similar_parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    similar_parser.set_defaults(func=cmd_find_similar)

    # Generate playlist command
    playlist_parser = subparsers.add_parser('generate-playlist', help='Generate a DJ playlist')
    playlist_parser.add_argument('start_id', type=int, help='Starting track ID')
    playlist_parser.add_argument('end_id', type=int, help='Ending track ID')
    playlist_parser.add_argument('steps', type=int, nargs='?', default=5, help='Number of steps')
    playlist_parser.add_argument('--db', help='Database path (default: data/djia.db)')
    playlist_parser.set_defaults(func=cmd_generate_playlist)

    # Export Traktor command
    traktor_parser = subparsers.add_parser('export-traktor', help='Export to Traktor NML')
    traktor_parser.add_argument('nml_path', nargs='?', default='djia_export.nml',
                                help='Output NML file path')
    traktor_parser.add_argument('--db', help='Database path (default: data/djia.db)')
    traktor_parser.add_argument('--traktor-input', default='Collection.nml',
                                help='Path to Traktor Collection.nml (optional, for hot cues)')
    traktor_parser.set_defaults(func=cmd_export_traktor)

    # Parse and execute
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if hasattr(args, 'func'):
        return args.func(args)

    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
