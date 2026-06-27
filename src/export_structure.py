"""Export structure analysis (drops, breakdowns, transitions) to organized files."""

import sys
import csv
from pathlib import Path
from datetime import datetime
from src.audio_analysis import analyze_track
from src.structure_detection import analyze_structure


def export_track_structure(file_path: str, output_dir: str = "results") -> dict:
    """Analyze a track and export its structure data."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Analyze track
    features = analyze_track(file_path)
    if not features:
        return None

    # Get structure
    structure = analyze_structure(file_path, features['tempo'])

    # Prepare data
    track_name = Path(file_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create separate CSV for this track
    csv_file = output_path / f"{track_name}_structure.csv"

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Type', 'Time (s)', 'Beat', 'Confidence'])

        for point in structure['all_points']:
            writer.writerow([
                point.type.upper(),
                f"{point.time:.1f}",
                f"{point.beat:.0f}",
                f"{point.confidence:.1%}"
            ])

    return {
        'track': track_name,
        'file_path': file_path,
        'duration': features['duration'],
        'tempo': features['tempo'],
        'drops': structure['drops'],
        'breakdowns': structure['breakdowns'],
        'transitions': structure['transitions'],
        'csv_file': str(csv_file)
    }


def main(data_dir: str = "data", output_dir: str = "results"):
    """Analyze all tracks and export structure data."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

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

    # Master summary file
    summary_file = output_path / "structure_summary_all_tracks.txt"

    with open(summary_file, 'w', encoding='utf-8') as summary:
        summary.write("=" * 80 + "\n")
        summary.write("DJ MIXING ANALYTICS - COMPLETE STRUCTURE BREAKDOWN\n")
        summary.write("=" * 80 + "\n\n")

        for i, audio_file in enumerate(audio_files, 1):
            print(f"[{i}/{len(audio_files)}] Analyzing: {audio_file.name}...")

            result = export_track_structure(str(audio_file), output_dir)
            if not result:
                print(f"  [Failed to analyze]")
                continue

            # Write to summary file
            summary.write(f"\n{'='*80}\n")
            summary.write(f"TRACK {i}: {result['track']}\n")
            summary.write(f"{'='*80}\n")
            summary.write(f"Duration: {result['duration']:.1f}s | Tempo: {result['tempo']:.1f} BPM\n")
            summary.write(f"Drops: {len(result['drops'])} | Breakdowns: {len(result['breakdowns'])} | Transitions: {len(result['transitions'])}\n\n")

            # DROPS
            if result['drops']:
                summary.write("DROPS:\n")
                summary.write("-" * 80 + "\n")
                for drop in result['drops']:
                    summary.write(f"  {drop.time:7.1f}s (beat {drop.beat:6.0f}) - Confidence: {drop.confidence:.1%}\n")
                summary.write("\n")

            # BREAKDOWNS
            if result['breakdowns']:
                summary.write("BREAKDOWNS:\n")
                summary.write("-" * 80 + "\n")
                for breakdown in result['breakdowns']:
                    summary.write(f"  {breakdown.time:7.1f}s (beat {breakdown.beat:6.0f}) - Confidence: {breakdown.confidence:.1%}\n")
                summary.write("\n")

            # TRANSITIONS
            if result['transitions']:
                summary.write("TRANSITIONS:\n")
                summary.write("-" * 80 + "\n")
                for transition in result['transitions']:
                    summary.write(f"  {transition.time:7.1f}s (beat {transition.beat:6.0f}) - Confidence: {transition.confidence:.1%}\n")
                summary.write("\n")

            print(f"  Drops: {len(result['drops'])}, Breakdowns: {len(result['breakdowns'])}, Transitions: {len(result['transitions'])}")

    print(f"\nComplete summary saved to: {summary_file}")
    print(f"Individual CSV files saved in: {output_dir}/")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    main(data_dir)
