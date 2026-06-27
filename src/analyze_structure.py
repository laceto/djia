"""Analyze track structure (breakdowns, drops, transitions)."""

import sys
from pathlib import Path
from src.audio_analysis import analyze_track
from src.mixing_metrics import score_track
from src.structure_detection import analyze_structure


def main(data_dir: str = "data"):
    """Analyze structure of all tracks."""
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Data directory '{data_dir}' not found.")
        return

    # Find audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg'}
    audio_files = sorted([f for f in data_path.glob('*') if f.suffix.lower() in audio_extensions])

    if not audio_files:
        print(f"No audio files found in '{data_dir}'.")
        return

    print(f"Found {len(audio_files)} track(s).\n")

    for audio_file in audio_files[:5]:  # Analyze first 5 for demo
        print(f"\n{'='*70}")
        print(f"TRACK: {audio_file.name}")
        print(f"{'='*70}")

        # Extract features
        features = analyze_track(str(audio_file))
        if not features:
            print("  [Failed to analyze]")
            continue

        # Analyze structure
        structure = analyze_structure(str(audio_file), features['tempo'])

        # Print summary
        print(f"Tempo: {features['tempo']:.1f} BPM | Duration: {features['duration']:.1f}s")
        print(f"\nStructure Elements:")
        print(f"  Drops: {structure['structure_summary']['num_drops']}")
        print(f"  Breakdowns: {structure['structure_summary']['num_breakdowns']}")
        print(f"  Transitions: {structure['structure_summary']['num_transitions']}")

        # Print all structure points
        if structure['all_points']:
            print(f"\nTimeline:")
            for point in structure['all_points']:
                print(f"  [{point.time:6.1f}s] {point.type.upper():12} (beat {point.beat:6.0f}, confidence: {point.confidence:.1%})")
        else:
            print("  [No major structure points detected]")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    main(data_dir)
