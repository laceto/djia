"""Helper to display segments with beat counts."""


def count_beats_in_segment(beat_times, segment_start, segment_end):
    """Count how many beats fall within a segment."""
    beats_in_segment = [
        beat_time for beat_time in beat_times
        if segment_start <= beat_time < segment_end
    ]
    return len(beats_in_segment), beats_in_segment


def display_segments_with_beats(track, show_beat_times=False):
    """
    Display segments with beat counts.

    Args:
        track: Track object from extract_track_features
        show_beat_times: If True, show individual beat times within each segment
    """
    print("\n" + "="*80)
    print("SEGMENTS WITH BEAT COUNTS")
    print("="*80)
    print(f"{'#':<3} {'Type':<15} {'Start':<10} {'End':<10} {'Duration':<10} {'Beats':<8} {'Confidence'}")
    print("-"*80)

    for i, seg in enumerate(track.phrasing.segments, 1):
        beat_count, beat_times = count_beats_in_segment(
            track.groove.beat_times,
            seg.start_time,
            seg.end_time
        )
        duration = seg.end_time - seg.start_time
        confidence_pct = seg.confidence * 100

        print(f"{i:<3} {seg.label:<15} {seg.start_time:<10.2f} {seg.end_time:<10.2f} "
              f"{duration:<10.2f} {beat_count:<8} {confidence_pct:.0f}%")

        # Show individual beat times if requested
        if show_beat_times and beat_times:
            beat_times_str = ", ".join(f"{bt:.2f}s" for bt in beat_times[:5])
            if len(beat_times) > 5:
                beat_times_str += f" ... ({len(beat_times)} total)"
            print(f"    Beats: {beat_times_str}")

    print("-"*80)
    print(f"Total segments: {len(track.phrasing.segments)}")
    print(f"Total beats: {len(track.groove.beat_times)}")
    print(f"Avg beats per segment: {len(track.groove.beat_times) / max(1, len(track.phrasing.segments)):.1f}")


# Example usage:
if __name__ == "__main__":
    from src.dsp.extractor import extract_track_features
    import os

    # Get first audio file
    data_dir = 'data'
    audio_files = [f for f in os.listdir(data_dir) if f.endswith('.mp3')][:1]

    if audio_files:
        audio_path = os.path.join(data_dir, audio_files[0])
        print(f"Extracting features from: {audio_files[0]}")

        track = extract_track_features(audio_path)

        # Display segments with beats
        display_segments_with_beats(track, show_beat_times=True)

        # Also show summary statistics
        print("\n" + "="*80)
        print("BEAT STATISTICS BY SEGMENT TYPE")
        print("="*80)

        segment_types = {}
        for seg in track.phrasing.segments:
            if seg.label not in segment_types:
                segment_types[seg.label] = {"count": 0, "total_beats": 0}

            beat_count, _ = count_beats_in_segment(
                track.groove.beat_times,
                seg.start_time,
                seg.end_time
            )
            segment_types[seg.label]["count"] += 1
            segment_types[seg.label]["total_beats"] += beat_count

        print(f"{'Type':<15} {'Segments':<12} {'Total Beats':<15} {'Avg Beats/Seg'}")
        print("-"*80)
        for seg_type, stats in sorted(segment_types.items()):
            avg_beats = stats["total_beats"] / stats["count"] if stats["count"] > 0 else 0
            print(f"{seg_type:<15} {stats['count']:<12} {stats['total_beats']:<15} {avg_beats:.1f}")
