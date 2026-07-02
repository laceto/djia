"""
Compare spectral detection vs phrase-locked segmentation.

Shows what happens when you fix phrases to exactly 16 bars.
"""

import glob
from src.dsp.extractor import extract_track_features
from src.dsp.config import get_config
from src.dsp.phrasing_engine import (
    analyze_structure,
    create_phrase_locked_segments,
    time_to_bar,
    bar_to_time
)
import librosa
import numpy as np

# Find Marrakech
files = glob.glob("data/**/*.mp3", recursive=True)
marrakech = [f for f in files if "marrakech" in f.lower()]

if not marrakech:
    print("❌ Marrakech track not found")
else:
    track_path = marrakech[0]
    print(f"✅ Analyzing: {track_path}\n")

    # Load audio
    y, sr = librosa.load(track_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)

    # Get BPM from groove engine
    from src.dsp.groove_engine import analyze_groove
    groove = analyze_groove(y, sr)
    bpm = groove.bpm

    print(f"Track Properties:")
    print(f"  Duration: {duration:.2f}s ({duration/60:.2f} min)")
    print(f"  BPM: {bpm:.2f}")
    print()

    # ========== TEST 1: Spectral Detection (Current) ==========
    print("=" * 70)
    print("TEST 1: SPECTRAL DETECTION (Current)")
    print("=" * 70)

    config = get_config("minimal")
    phrasing = analyze_structure(
        y, sr,
        bpm=bpm,
        novelty_threshold=config.phrasing.novelty_threshold,
        min_segment_duration=config.phrasing.min_segment_duration,
        breakdown_threshold=config.phrasing.breakdown_duration_threshold,
        include_beats=True
    )

    print(f"Total segments: {len(phrasing.segments)}")
    print(f"Hot cues: {len(phrasing.cue_points)}")

    # Count bars in each segment
    bars_per_segment = []
    for seg in phrasing.segments:
        start_bar = time_to_bar(seg.start_time, bpm)
        end_bar = time_to_bar(seg.end_time, bpm)
        bars = end_bar - start_bar
        bars_per_segment.append(bars)

    print(f"\nBar distribution:")
    print(f"  Avg bars/segment: {np.mean(bars_per_segment):.1f}")
    print(f"  Min bars/segment: {np.min(bars_per_segment):.1f}")
    print(f"  Max bars/segment: {np.max(bars_per_segment):.1f}")
    print(f"  Std dev: {np.std(bars_per_segment):.1f}")

    print(f"\nFirst 10 segments (spectral detection):")
    for i, seg in enumerate(phrasing.segments[:10], 1):
        start_bar = time_to_bar(seg.start_time, bpm)
        end_bar = time_to_bar(seg.end_time, bpm)
        bars = end_bar - start_bar
        print(f"  {i:2d}. {seg.label:35s} | {bars:5.1f} bars")

    print()

    # ========== TEST 2: Phrase-Locked (16 bars) ==========
    print("=" * 70)
    print("TEST 2: PHRASE-LOCKED SEGMENTATION (16 bars/phrase)")
    print("=" * 70)

    segments_16 = create_phrase_locked_segments(
        duration=duration,
        bpm=bpm,
        bars_per_phrase=16,
        include_beats=True
    )

    print(f"Total segments: {len(segments_16)}")
    print(f"Hot cues: {len(segments_16)}")  # One per segment

    # Verify all segments are exactly 16 bars
    bars_per_segment_16 = []
    for seg in segments_16:
        start_bar = time_to_bar(seg.start_time, bpm)
        end_bar = time_to_bar(seg.end_time, bpm)
        bars = end_bar - start_bar
        bars_per_segment_16.append(bars)

    print(f"\nBar distribution (should all be ~16):")
    print(f"  Avg bars/segment: {np.mean(bars_per_segment_16):.1f}")
    print(f"  Min bars/segment: {np.min(bars_per_segment_16):.1f}")
    print(f"  Max bars/segment: {np.max(bars_per_segment_16):.1f}")
    print(f"  Std dev: {np.std(bars_per_segment_16):.1f}")

    print(f"\nFirst 10 segments (16-bar phrases):")
    for i, seg in enumerate(segments_16[:10], 1):
        start_bar = time_to_bar(seg.start_time, bpm)
        end_bar = time_to_bar(seg.end_time, bpm)
        bars = end_bar - start_bar
        print(f"  {i:2d}. {seg.label:60s} | {bars:5.1f} bars")

    print()

    # ========== TEST 3: Phrase-Locked (8 bars) ==========
    print("=" * 70)
    print("TEST 3: PHRASE-LOCKED SEGMENTATION (8 bars/phrase)")
    print("=" * 70)

    segments_8 = create_phrase_locked_segments(
        duration=duration,
        bpm=bpm,
        bars_per_phrase=8,
        include_beats=True
    )

    print(f"Total segments: {len(segments_8)}")

    bars_per_segment_8 = []
    for seg in segments_8:
        start_bar = time_to_bar(seg.start_time, bpm)
        end_bar = time_to_bar(seg.end_time, bpm)
        bars = end_bar - start_bar
        bars_per_segment_8.append(bars)

    print(f"\nBar distribution (should all be ~8):")
    print(f"  Avg bars/segment: {np.mean(bars_per_segment_8):.1f}")
    print(f"  Min bars/segment: {np.min(bars_per_segment_8):.1f}")
    print(f"  Max bars/segment: {np.max(bars_per_segment_8):.1f}")

    print(f"\nFirst 15 segments (8-bar phrases):")
    for i, seg in enumerate(segments_8[:15], 1):
        start_bar = time_to_bar(seg.start_time, bpm)
        end_bar = time_to_bar(seg.end_time, bpm)
        bars = end_bar - start_bar
        print(f"  {i:2d}. {seg.label:60s} | {bars:5.1f} bars")

    print()

    # ========== COMPARISON TABLE ==========
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Approach':<30} | {'Segments':<10} | {'Bars/Seg':<10} | {'Regularity':<15}")
    print("-" * 70)

    regularity_spectral = f"±{np.std(bars_per_segment):.1f}"
    regularity_16 = f"±{np.std(bars_per_segment_16):.1f}"
    regularity_8 = f"±{np.std(bars_per_segment_8):.1f}"

    print(f"{'Spectral Detection':<30} | {len(phrasing.segments):<10} | {np.mean(bars_per_segment):<10.1f} | {regularity_spectral:<15}")
    print(f"{'16-bar phrases':<30} | {len(segments_16):<10} | {np.mean(bars_per_segment_16):<10.1f} | {regularity_16:<15}")
    print(f"{'8-bar phrases':<30} | {len(segments_8):<10} | {np.mean(bars_per_segment_8):<10.1f} | {regularity_8:<15}")

    print()

    # ========== WHAT HAPPENS ==========
    print("=" * 70)
    print("WHAT HAPPENS WHEN YOU FIX TO 16 BARS?")
    print("=" * 70)

    total_bars_16 = np.mean(bars_per_segment_16) * len(segments_16)
    phrase_duration_16 = bar_to_time(16, bpm)

    print(f"\n1. STRUCTURE BECOMES PREDICTABLE")
    print(f"   - Every segment is exactly 16 bars ({phrase_duration_16:.2f}s)")
    print(f"   - No irregular boundaries")
    print(f"   - No false detections")
    print(f"   - Problem: Misses subtle structure changes")

    print(f"\n2. SEGMENT COUNT CHANGES")
    print(f"   - Spectral: {len(phrasing.segments)} segments")
    print(f"   - 16-bar:   {len(segments_16)} segments")
    print(f"   - Ratio:    {len(segments_16) / len(phrasing.segments):.1%}")

    print(f"\n3. HOT CUE MAPPING CHANGES")
    print(f"   - Spectral: {len(phrasing.cue_points)} cues")
    print(f"   - 16-bar:   {len(segments_16)} cues (1 per segment)")
    print(f"   - Interval: Every {phrase_duration_16:.2f}s")

    print(f"\n4. MUSICAL ALIGNMENT")
    print(f"   ✅ Pros:")
    print(f"      - Segments align with 4/4 music structure")
    print(f"      - Easy to count (8 bars = 1 drop, 16 bars = 1 phrase)")
    print(f"      - Predictable DJ cue spacing")
    print(f"      - Matches Traktor beat grid")

    print(f"\n   ❌ Cons:")
    print(f"      - Doesn't adapt to actual track structure")
    print(f"      - Can't detect short breakdowns/breaks")
    print(f"      - Loses fine-grained structural detail")
    print(f"      - May place cues in wrong musical spots")

    print()

    # ========== RECOMMENDED APPROACH ==========
    print("=" * 70)
    print("RECOMMENDED HYBRID APPROACH")
    print("=" * 70)

    print(f"""
For Marrakech (minimal house):

1. Use SPECTRAL DETECTION to find actual boundaries
   (Gets: {len(phrasing.segments)} segments with real structure)

2. Then SNAP boundaries to nearest bar
   (Makes cue positions beat-grid aligned)

3. Result: Structure + Precision
   ✅ Real structural points (not forced regularity)
   ✅ Beat-grid aligned cues (DJ friendly)
   ✅ Flexible segment lengths (adapts to track)
""")
