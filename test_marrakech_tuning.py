"""
Comprehensive test: Spectral detection vs Phrase-locked segmentation
Compares parameter tuning on Hermanez - Marrakech (minimal house)
"""

import glob
import numpy as np
from collections import Counter
from src.dsp.extractor import extract_track_features
from src.dsp.config import get_config, custom_config
from src.dsp.phrasing_engine import (
    create_phrase_locked_segments,
    time_to_bar,
    bar_to_time
)
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from pathlib import Path

# Find Marrakech in data folder
files = glob.glob("data/**/*.mp3", recursive=True)
marrakech = [f for f in files if "retroflex" in f.lower()]

if not marrakech:
    print("[ERROR] Marrakech track not found in data/")
    print("Looking for files with 'marrakech' in the name...")
    print(f"Available MP3s: {files[:5]}")
else:
    track_path = marrakech[0]
    print(f"[FOUND] {track_path}")

    # Load for duration/BPM info
    y, sr = librosa.load(track_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)

    from src.dsp.groove_engine import analyze_groove
    groove = analyze_groove(y, sr)
    bpm = groove.bpm

    print(f"   Duration: {duration:.2f}s ({duration/60:.2f}m)")
    print(f"   BPM: {bpm:.2f}\n")

    # Helper function to analyze segments
    def analyze_segments(segments, config_name, bpm_val):
        """Analyze segment distribution"""
        bars_per_seg = []
        for seg in segments:
            start_bar = time_to_bar(seg.start_time, bpm_val)
            end_bar = time_to_bar(seg.end_time, bpm_val)
            bars = end_bar - start_bar
            bars_per_seg.append(bars)

        return {
            'total': len(segments),
            'avg_bars': np.mean(bars_per_seg) if bars_per_seg else 0,
            'min_bars': np.min(bars_per_seg) if bars_per_seg else 0,
            'max_bars': np.max(bars_per_seg) if bars_per_seg else 0,
            'std_bars': np.std(bars_per_seg) if bars_per_seg else 0,
            'bars_list': bars_per_seg
        }

    # Helper function to display segment bar numbers
    def print_segment_bars(segments, bpm_val, max_segments=15):
        """Print segment boundaries in bars and time"""
        print(f"{'#':>2} | {'Label':20} | {'Start Bar':>12} | {'End Bar':>12} | {'Bars':>8} | {'Time Range':20}")
        print("-" * 100)
        for i, seg in enumerate(segments[:max_segments], 1):
            start_bar = time_to_bar(seg.start_time, bpm_val)
            end_bar = time_to_bar(seg.end_time, bpm_val)
            bars = end_bar - start_bar
            time_range = f"{seg.start_time:.1f}-{seg.end_time:.1f}s"
            # Extract label type (before parentheses)
            label_type = seg.label.split('(')[0].strip()
            print(f"{i:2} | {label_type:20} | {start_bar:12.1f} | {end_bar:12.1f} | {bars:8.1f} | {time_range:20}")
        if len(segments) > max_segments:
            print(f"... and {len(segments) - max_segments} more segments")

    # Helper function to get all boundaries
    def get_segment_boundaries(segments, bpm_val):
        """Extract all segment start/end times and bars"""
        boundaries = []
        for i, seg in enumerate(segments, 1):
            start_bar = time_to_bar(seg.start_time, bpm_val)
            end_bar = time_to_bar(seg.end_time, bpm_val)
            start_beat = int(start_bar * 4)
            end_beat = int(end_bar * 4)

            boundaries.append({
                'index': i,
                'label': seg.label,
                'start_time': seg.start_time,
                'end_time': seg.end_time,
                'duration': seg.end_time - seg.start_time,
                'start_bar': start_bar,
                'end_bar': end_bar,
                'start_beat': start_beat,
                'end_beat': end_beat,
                'confidence': seg.confidence
            })
        return boundaries

    # Helper function to create spectrogram plots with bar numbers on x-axis
    def plot_spectrogram_with_boundaries(y_audio, sr_val, segments, config_name, bpm_val, save_path=None):
        """Create spectrogram plot with segment boundaries overlaid, x-axis in bars"""
        try:
            # Compute STFT
            D = librosa.stft(y_audio, hop_length=512)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            # Get duration in seconds
            duration_sec = librosa.get_duration(y=y_audio, sr=sr_val)

            # Create figure
            fig, ax = plt.subplots(figsize=(16, 7))

            # Plot spectrogram with time axis (we'll convert to bars)
            img = librosa.display.specshow(S_db, sr=sr_val, hop_length=512, x_axis='time', y_axis='log', ax=ax, cmap='magma')
            fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Magnitude (dB)')

            # Get current x-axis limits (in seconds)
            x_lim = ax.get_xlim()

            # Convert time axis to bars
            # Create new x-axis ticks at bar boundaries every 16 bars
            total_bars = int(time_to_bar(duration_sec, bpm_val)) + 1
            bar_interval = 16  # Show ticks every 16 bars (1 phrase)

            # Generate bar tick positions (in seconds)
            bar_positions_time = []
            bar_labels = []
            for bar_num in range(0, total_bars + 1, bar_interval):
                bar_time = bar_to_time(bar_num, bpm_val)
                if bar_time <= duration_sec:
                    bar_positions_time.append(bar_time)
                    bar_labels.append(str(bar_num))

            # Set x-axis ticks and labels in bars
            ax.set_xticks(bar_positions_time)
            ax.set_xticklabels(bar_labels, rotation=45, ha='right')
            ax.set_xlabel('Bars', fontsize=12, fontweight='bold')

            # Overlay segment boundaries with bar labels
            colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'white', 'orange', 'purple']
            for i, seg in enumerate(segments):
                color = colors[i % len(colors)]
                start_bar = time_to_bar(seg.start_time, bpm_val)
                end_bar = time_to_bar(seg.end_time, bpm_val)

                # Draw vertical lines at boundaries
                ax.axvline(x=seg.start_time, color=color, linestyle='--', linewidth=2.5, alpha=0.8)

                # Add segment label with bar numbers
                label_text = f"[{i+1}]\nbar {start_bar:.0f}-{end_bar:.0f}"
                ax.text(seg.start_time + 5, ax.get_ylim()[1] * 0.90, label_text,
                       color=color, fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            ax.set_title(f'Spectrogram with Segments (Bar-based X-axis) - {config_name}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
            plt.tight_layout()

            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"  Saved: {save_path}")
                plt.close()
                return save_path
            else:
                return fig

        except Exception as e:
            print(f"  Error creating spectrogram: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ========== TEST 1: Default Config ==========
    print("=" * 80)
    print("TEST 1: SPECTRAL DETECTION - Default Config")
    print("=" * 80)
    print("Parameters: novelty_threshold=0.5, min_segment_duration=8.0s")
    print("-" * 80)
    config_default = get_config("default")
    track_default = extract_track_features(track_path, config=config_default)
    stats_default = analyze_segments(track_default.phrasing.segments, "Default", bpm)

    print(f"Total segments: {stats_default['total']}")
    print(f"Hot cues: {len(track_default.phrasing.cue_points)}")
    print(f"\nBar distribution:")
    print(f"  Avg bars/segment: {stats_default['avg_bars']:.1f}")
    print(f"  Min/Max: {stats_default['min_bars']:.1f} - {stats_default['max_bars']:.1f}")
    print(f"  Std dev: {stats_default['std_bars']:.1f}")

    labels = [seg.label for seg in track_default.phrasing.segments]
    counts = Counter(labels)
    print(f"\nSegment types: {dict(counts)}")

    print("\nSegment Boundaries (in Bars):")
    print_segment_bars(track_default.phrasing.segments, bpm, max_segments=10)
    print()

    # ========== TEST 2: Minimal Config ==========
    print("=" * 80)
    print("TEST 2: SPECTRAL DETECTION - Minimal Config (RECOMMENDED)")
    print("=" * 80)
    print("Parameters: novelty_threshold=0.65, min_segment_duration=12.0s")
    print("-" * 80)
    config_minimal = get_config("minimal")
    track_minimal = extract_track_features(track_path, config=config_minimal)
    stats_minimal = analyze_segments(track_minimal.phrasing.segments, "Minimal", bpm)

    print(f"Total segments: {stats_minimal['total']}")
    print(f"Hot cues: {len(track_minimal.phrasing.cue_points)}")
    print(f"\nBar distribution:")
    print(f"  Avg bars/segment: {stats_minimal['avg_bars']:.1f}")
    print(f"  Min/Max: {stats_minimal['min_bars']:.1f} - {stats_minimal['max_bars']:.1f}")
    print(f"  Std dev: {stats_minimal['std_bars']:.1f}")

    labels = [seg.label for seg in track_minimal.phrasing.segments]
    counts = Counter(labels)
    print(f"\nSegment types: {dict(counts)}")

    print("\nSegment Boundaries (in Bars):")
    print_segment_bars(track_minimal.phrasing.segments, bpm, max_segments=10)
    print()

    # ========== TEST 3: Custom Aggressive Minimal ==========
    print("=" * 80)
    print("TEST 3: SPECTRAL DETECTION - Ultra-Minimal Config")
    print("=" * 80)
    print("Parameters: novelty_threshold=0.7, min_segment_duration=16.0s")
    print("-" * 80)
    config_custom = custom_config(
        novelty_threshold=0.7,
        min_segment_duration=16.0,
        breakdown_duration_threshold=40.0
    )
    track_custom = extract_track_features(track_path, config=config_custom)
    stats_custom = analyze_segments(track_custom.phrasing.segments, "Custom", bpm)

    print(f"Total segments: {stats_custom['total']}")
    print(f"Hot cues: {len(track_custom.phrasing.cue_points)}")
    print(f"\nBar distribution:")
    print(f"  Avg bars/segment: {stats_custom['avg_bars']:.1f}")
    print(f"  Min/Max: {stats_custom['min_bars']:.1f} - {stats_custom['max_bars']:.1f}")
    print(f"  Std dev: {stats_custom['std_bars']:.1f}")

    labels = [seg.label for seg in track_custom.phrasing.segments]
    counts = Counter(labels)
    print(f"\nSegment types: {dict(counts)}")

    print("\nSegment Boundaries (in Bars):")
    print_segment_bars(track_custom.phrasing.segments, bpm, max_segments=10)
    print()

    # ========== TEST 4: Phrase-Locked 16 bars ==========
    print("=" * 80)
    print("TEST 4: PHRASE-LOCKED - 16 bars/phrase")
    print("=" * 80)
    print("All segments fixed to exactly 16 bars")
    print("-" * 80)
    segments_16 = create_phrase_locked_segments(
        duration=duration,
        bpm=bpm,
        bars_per_phrase=16,
        include_beats=True
    )
    stats_16 = analyze_segments(segments_16, "16-bar", bpm)

    print(f"Total segments: {stats_16['total']}")
    print(f"Hot cues: {stats_16['total']}")
    print(f"\nBar distribution:")
    print(f"  Avg bars/segment: {stats_16['avg_bars']:.1f}")
    print(f"  Min/Max: {stats_16['min_bars']:.1f} - {stats_16['max_bars']:.1f}")
    print(f"  Std dev: {stats_16['std_bars']:.1f}")
    print(f"  Cue interval: {bar_to_time(16, bpm):.2f}s (every 16 bars)")
    print()

    # ========== TEST 5: Phrase-Locked 8 bars ==========
    print("=" * 80)
    print("TEST 5: PHRASE-LOCKED - 8 bars/phrase")
    print("=" * 80)
    print("All segments fixed to exactly 8 bars")
    print("-" * 80)
    segments_8 = create_phrase_locked_segments(
        duration=duration,
        bpm=bpm,
        bars_per_phrase=8,
        include_beats=True
    )
    stats_8 = analyze_segments(segments_8, "8-bar", bpm)

    print(f"Total segments: {stats_8['total']}")
    print(f"Hot cues: {stats_8['total']}")
    print(f"\nBar distribution:")
    print(f"  Avg bars/segment: {stats_8['avg_bars']:.1f}")
    print(f"  Min/Max: {stats_8['min_bars']:.1f} - {stats_8['max_bars']:.1f}")
    print(f"  Std dev: {stats_8['std_bars']:.1f}")
    print(f"  Cue interval: {bar_to_time(8, bpm):.2f}s (every 8 bars)")
    print()

    # ========== DETAILED BREAKDOWN (Minimal Config) ==========
    print("=" * 100)
    print("DETAILED: Minimal Config Segments (RECOMMENDED)")
    print("=" * 100)
    print(f"{'#':>2} | {'Segment Label':40} | {'Time':20} | {'Duration':10} | {'Bars':10}")
    print("-" * 100)

    boundaries_minimal = get_segment_boundaries(track_minimal.phrasing.segments, bpm)
    for b in boundaries_minimal:
        time_str = f"{b['start_time']:.1f}-{b['end_time']:.1f}s"
        bars = b['end_bar'] - b['start_bar']
        print(f"{b['index']:2} | {b['label']:40} | {time_str:20} | {b['duration']:9.2f}s | {bars:9.1f}")
    print()

    # ========== COMPARISON TABLE ==========
    print("=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Approach':<25} | {'Segments':>8} | {'Avg Bars':>10} | {'Regularity':>12} | {'Cue Space':>10}")
    print("-" * 80)

    regularity_default = f"±{stats_default['std_bars']:.1f}"
    regularity_minimal = f"±{stats_minimal['std_bars']:.1f}"
    regularity_custom = f"±{stats_custom['std_bars']:.1f}"
    regularity_16 = f"±{stats_16['std_bars']:.1f}"
    regularity_8 = f"±{stats_8['std_bars']:.1f}"

    cue_default = "irregular"
    cue_minimal = "irregular"
    cue_custom = "irregular"
    cue_16 = f"{bar_to_time(16, bpm):.2f}s"
    cue_8 = f"{bar_to_time(8, bpm):.2f}s"

    print(f"{'Spectral (Default)':<25} | {stats_default['total']:8} | {stats_default['avg_bars']:10.1f} | {regularity_default:>12} | {cue_default:>10}")
    print(f"{'Spectral (Minimal)*':<25} | {stats_minimal['total']:8} | {stats_minimal['avg_bars']:10.1f} | {regularity_minimal:>12} | {cue_minimal:>10}")
    print(f"{'Spectral (Ultra-Min)':<25} | {stats_custom['total']:8} | {stats_custom['avg_bars']:10.1f} | {regularity_custom:>12} | {cue_custom:>10}")
    print(f"{'Phrase-Locked (16-bar)':<25} | {stats_16['total']:8} | {stats_16['avg_bars']:10.1f} | {regularity_16:>12} | {cue_16:>10}")
    print(f"{'Phrase-Locked (8-bar)':<25} | {stats_8['total']:8} | {stats_8['avg_bars']:10.1f} | {regularity_8:>12} | {cue_8:>10}")

    print("\n* = RECOMMENDED for Marrakech (real structure + minimal artifacts)")
    print()

    # ========== ANALYZE SPECTRAL ENERGY PER SEGMENT ==========
    print("=" * 100)
    print("ANALYZING SPECTRAL ENERGY - 16-bar Segments")
    print("=" * 100)

    def classify_segment_energy(y_audio, sr_val, segment, bpm_val, idx):
        """Analyze spectral energy and classify segment type"""
        # Convert segment times to samples
        start_sample = int(segment.start_time * sr_val)
        end_sample = int(segment.end_time * sr_val)
        segment_audio = y_audio[start_sample:end_sample]

        if len(segment_audio) == 0:
            return None

        # Compute spectrogram for this segment
        D = librosa.stft(segment_audio, hop_length=512)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # Extract energy metrics across frequency bands
        bass_energy = np.mean(S_db[0:50])  # Sub-bass to bass (0-1kHz approx)
        mid_energy = np.mean(S_db[50:150])  # Mids (1-3kHz approx)
        high_energy = np.mean(S_db[150:])  # Presence/highs (3k+ Hz)
        total_energy = np.mean(S_db)

        # Energy trend (start vs end of segment)
        mid_point = len(S_db) // 2
        start_energy = np.mean(S_db[:mid_point])
        end_energy = np.mean(S_db[mid_point:])
        energy_trend = end_energy - start_energy  # Positive = building, negative = dropping

        start_bar = time_to_bar(segment.start_time, bpm_val)
        end_bar = time_to_bar(segment.end_time, bpm_val)

        return {
            'segment_idx': idx,
            'bars': f"{start_bar:.0f}-{end_bar:.0f}",
            'start_bar': start_bar,
            'end_bar': end_bar,
            'bass_energy': bass_energy,
            'mid_energy': mid_energy,
            'high_energy': high_energy,
            'total_energy': total_energy,
            'energy_trend': energy_trend,
            'duration_sec': segment.end_time - segment.start_time
        }

    def classify_region_type(seg_data, prev_energy=None, next_energy=None):
        """Classify segment as: quiet, building, peak, breakdown, sustained"""
        total_e = seg_data['total_energy']
        trend = seg_data['energy_trend']

        # Thresholds (tuned for Marrakech minimal house - much more sensitive)
        # Minimal house stays in -60 to -30 dB range, so detect subtle shifts
        quiet_threshold = -57      # Below this = quiet sections
        sustained_threshold = -54  # Middle ground = sustained/stable
        peak_threshold = -51       # Above this = relative peak for minimal house

        if total_e < quiet_threshold:
            return 'QUIET/BREAKDOWN'
        elif trend > 2.5 and total_e > sustained_threshold:
            return 'BUILDING/RAMP-UP'
        elif trend < -2.5 and total_e > quiet_threshold:
            return 'DROP/RELEASE'
        elif total_e > peak_threshold:
            return 'PEAK/HIGH-ENERGY'
        else:
            return 'SUSTAINED'

    # Analyze all 16-bar segments
    segment_energies = []
    for i, seg in enumerate(segments_16):
        seg_data = classify_segment_energy(y, sr, seg, bpm, i + 1)
        if seg_data:
            seg_type = classify_region_type(seg_data)
            seg_data['type'] = seg_type
            segment_energies.append(seg_data)

    # Print detailed analysis
    print(f"\n{'#':>2} | {'Bars':>8} | {'Type':>20} | {'Total dB':>8} | {'Trend':>7} | {'Bass':>7} | {'Mid':>7} | {'High':>7}")
    print("-" * 100)
    for seg in segment_energies:
        print(f"{seg['segment_idx']:2} | {seg['bars']:>8} | {seg['type']:>20} | {seg['total_energy']:8.1f} | {seg['energy_trend']:7.1f} | {seg['bass_energy']:7.1f} | {seg['mid_energy']:7.1f} | {seg['high_energy']:7.1f}")

    # Summary by type
    print("\n" + "=" * 100)
    print("SPECTRAL REGION CLASSIFICATION SUMMARY")
    print("=" * 100)
    type_counts = {}
    type_bars = {}
    for seg in segment_energies:
        seg_type = seg['type']
        if seg_type not in type_counts:
            type_counts[seg_type] = 0
            type_bars[seg_type] = []
        type_counts[seg_type] += 1
        type_bars[seg_type].append(f"{seg['bars']}")

    for seg_type in sorted(type_counts.keys()):
        count = type_counts[seg_type]
        bars_list = ", ".join(type_bars[seg_type])
        print(f"\n{seg_type}:")
        print(f"  Count: {count} segments")
        print(f"  Bar ranges: {bars_list}")

    print()

    # ========== SPECTROGRAM PLOTS WITH BOUNDARIES ==========
    print("=" * 100)
    print("GENERATING SPECTROGRAMS WITH SEGMENT BOUNDARIES")
    print("=" * 100)

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("\nGenerating spectrograms (this may take a moment)...")

    # Plot 1: Default Config
    print("\n[1] Default Config Spectrogram")
    print("-" * 100)
    plot_path_1 = plot_spectrogram_with_boundaries(
        y, sr, track_default.phrasing.segments,
        "Default Config (22 segments)",
        bpm, save_path=str(results_dir / "spectrogram_01_default.png")
    )

    # Plot 2: Minimal Config
    print("\n[2] Minimal Config Spectrogram (RECOMMENDED)")
    print("-" * 100)
    plot_path_2 = plot_spectrogram_with_boundaries(
        y, sr, track_minimal.phrasing.segments,
        "Minimal Config (2 segments)",
        bpm, save_path=str(results_dir / "spectrogram_02_minimal.png")
    )

    # Plot 3: 16-bar Phrase-Locked
    print("\n[3] 16-bar Phrase-Locked Spectrogram")
    print("-" * 100)
    plot_path_3 = plot_spectrogram_with_boundaries(
        y, sr, segments_16,
        "Phrase-Locked 16-bar (22 segments)",
        bpm, save_path=str(results_dir / "spectrogram_03_16bar.png")
    )

    print("\nSpectrogram files saved to results/ directory")
    print()

    # ========== ALL SEGMENT BOUNDARIES ==========
    print("=" * 100)
    print("ALL SEGMENT BOUNDARIES - FORMAT REFERENCE")
    print("=" * 100)

    boundaries = get_segment_boundaries(track_minimal.phrasing.segments, bpm)

    print("\n[1] COMPACT TABLE (Time and Bars)")
    print("-" * 100)
    print(f"{'#':>2} | {'Start Time':>10} | {'End Time':>10} | {'Start Bar':>10} | {'End Bar':>10} | {'Label':40}")
    print("-" * 100)
    for b in boundaries:
        print(f"{b['index']:2} | {b['start_time']:10.2f}s | {b['end_time']:10.2f}s | {b['start_bar']:10.1f} | {b['end_bar']:10.1f} | {b['label']}")

    print("\n[2] BOUNDARY LIST (Time)")
    print("-" * 100)
    print("Segment boundaries (times in seconds):")
    for b in boundaries:
        print(f"  Segment {b['index']}: {b['start_time']:.2f}s -> {b['end_time']:.2f}s")

    print("\n[3] BOUNDARY LIST (Bars)")
    print("-" * 100)
    print("Segment boundaries (bars):")
    for b in boundaries:
        print(f"  Segment {b['index']}: bar {b['start_bar']:.1f} -> bar {b['end_bar']:.1f}")

    print("\n[4] BOUNDARY LIST (Beats)")
    print("-" * 100)
    print("Segment boundaries (beats):")
    for b in boundaries:
        print(f"  Segment {b['index']}: beat {b['start_beat']} -> beat {b['end_beat']}")

    print("\n[5] PYTHON DICT/JSON FORMAT")
    print("-" * 100)
    import json
    print(json.dumps(boundaries, indent=2))

    print("\n[6] CSV FORMAT")
    print("-" * 100)
    print("index,label,start_time,end_time,duration,start_bar,end_bar,start_beat,end_beat,confidence")
    for b in boundaries:
        print(f"{b['index']},{b['label']},{b['start_time']:.2f},{b['end_time']:.2f},{b['duration']:.2f},{b['start_bar']:.1f},{b['end_bar']:.1f},{b['start_beat']},{b['end_beat']},{b['confidence']}")

    print("\n[7] TIMELINE (ASCII VISUALIZATION)")
    print("-" * 100)
    timeline_length = 80
    max_time = duration
    print(f"Timeline (0-{max_time:.0f}s):")
    print("[" + "=" * timeline_length + "]")
    for b in boundaries:
        start_pos = int((b['start_time'] / max_time) * timeline_length)
        end_pos = int((b['end_time'] / max_time) * timeline_length)
        marker = f"[{b['index']}]"
        if start_pos < timeline_length:
            print(" " * start_pos + marker + " " + b['label'][:40])
    print()

    # Summary statistics
    print("=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    print(f"Total segments: {len(boundaries)}")
    print(f"Total duration: {sum(b['duration'] for b in boundaries):.2f}s")
    print(f"Track duration: {duration:.2f}s")
    print(f"\nSegment durations:")
    print(f"  Min: {min(b['duration'] for b in boundaries):.2f}s")
    print(f"  Max: {max(b['duration'] for b in boundaries):.2f}s")
    print(f"  Avg: {np.mean([b['duration'] for b in boundaries]):.2f}s")
    print()

    print(f"Segment bar counts:")
    bar_counts = [b['end_bar'] - b['start_bar'] for b in boundaries]
    print(f"  Min: {min(bar_counts):.1f} bars")
    print(f"  Max: {max(bar_counts):.1f} bars")
    print(f"  Avg: {np.mean(bar_counts):.1f} bars")
    print()

    # ========== RED PEAK IDENTIFICATION ==========
    print("=" * 100)
    print("RED PEAK MOMENTS (Spectral Analysis)")
    print("=" * 100)
    print("These correspond to HIGH-ENERGY regions visible as RED in the spectrogram:\n")

    # Find high-energy segments (peaks)
    peaks = [seg for seg in segment_energies if seg['type'] in ['PEAK/HIGH-ENERGY', 'DROP/RELEASE']]
    peaks_sorted = sorted(peaks, key=lambda x: x['total_energy'], reverse=True)

    if peaks_sorted:
        for i, peak in enumerate(peaks_sorted[:5], 1):  # Top 5 energy peaks
            print(f"{i}. Segment {peak['segment_idx']} (Bar {peak['bars']})")
            print(f"   Type: {peak['type']}")
            print(f"   Total Energy: {peak['total_energy']:.1f} dB")
            print(f"   Trend: {peak['energy_trend']:+.1f} dB (0=stable, +ve=rising, -ve=falling)")
            print(f"   Bass:Mid:High ratio → {peak['bass_energy']:.1f}:{peak['mid_energy']:.1f}:{peak['high_energy']:.1f}")
            print()
    else:
        print("No clear peak regions detected. This suggests a smooth, evolving track with gentle energy variations.")
        print()

    # ========== INTERPRETATION ==========
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print(f"""
Marrakech is a MINIMAL HOUSE track:
• Constant groove with subtle variations
• No sudden structural changes
• Standard 4/4 beat structure

RECOMMENDATIONS:

1. FOR ACCURATE ANALYSIS (recommended):
   [*] Use: Minimal Config (spectral detection)
   Why: Captures real track structure while filtering noise
   Segments: {stats_minimal['total']} (sensible number)
   Result: Beat-range labels show actual drops/builds

2. FOR DJ PREDICTABILITY:
   [*] Use: Phrase-Locked 16-bar
   Why: Cues every {bar_to_time(16, bpm):.2f}s (perfectly regular)
   Segments: {stats_16['total']} (very regular)
   Result: Easy to count along, predictable spacing

3. FOR MAXIMUM DETAIL:
   [*] Use: Spectral Default
   Why: Detects all variations
   Segments: {stats_default['total']} (may have noise)
   Result: Detailed but irregular

CHOICE FOR MARRAKECH:
→ Use MINIMAL Config (Test 2)
→ Labels include beat/bar ranges automatically
→ Real structure + beat-grid aligned cues
→ Best balance for DJ workflow
""")
