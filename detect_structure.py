"""Detect breakdowns and drops (in BARS) from kick+bass low-band energy.

For techno, the musically meaningful structure is driven by the low end:
when the kick+bass is present you're in a DROP; when it falls away you're in a
BREAKDOWN. This is far more reliable than generic spectral-novelty segmentation
for four-on-the-floor material.

Method:
  1. Detect BPM  ->  seconds-per-bar = (60 / BPM) * 4   (4/4)
  2. Low-band (20-150 Hz) energy envelope from the STFT
  3. Smooth over ~1 bar, threshold to get kick "on" / "off"
  4. Keep sections >= MIN_BARS long, report each transition as a bar number
  5. Optionally snap bar numbers to the nearest PHRASE grid (4/8/16 bars)

Usage:
    python detect_structure.py [path/to/track.mp3] [--phrase 8] [--min-bars 4] [--no-plot]
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import librosa

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Single source of truth for structure detection lives in the DSP engine.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.dsp.phrasing_engine import (  # noqa: E402
    compute_lowband_energy,
    smooth_lowband_energy,
    detect_energy_sections,
)


def detect_bpm(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    bpm, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    return float(np.atleast_1d(bpm)[0])


def to_bar(t, spb):
    return int(round(t / spb)) + 1  # 1-indexed


def snap(bar, phrase):
    """Snap a bar number to the nearest phrase-grid boundary (bar 1, 1+phrase, ...)."""
    if phrase <= 1:
        return bar
    k = round((bar - 1) / phrase)
    return int(k * phrase + 1)


def main():
    ap = argparse.ArgumentParser(description="Detect breakdown/drop bars from low-band energy")
    ap.add_argument("audio", nargs="?", default="data/2000_and_one-pak_pak.mp3")
    ap.add_argument("--phrase", type=int, default=8,
                    help="Snap bars to this phrase grid (0 = no snapping; try 4/8/16)")
    ap.add_argument("--min-bars", type=int, default=4,
                    help="Ignore sections shorter than this many bars")
    ap.add_argument("--thresh", type=float, default=0.4,
                    help="Kick-on threshold as a fraction of peak low-band energy")
    ap.add_argument("--pads", type=int, default=0,
                    help="Limit hot cues to N physical pads (e.g. 4). 0 = one cue per section")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.audio):
        print(f"ERROR: not found: {args.audio}")
        sys.exit(1)

    y, sr = librosa.load(args.audio, sr=22050)
    dur = librosa.get_duration(y=y, sr=sr)
    bpm = detect_bpm(y, sr)
    hop = 512

    # Structure comes from the DSP engine (same code the Traktor hot cues use).
    energy = compute_lowband_energy(y, sr, hop_length=hop)
    sections = detect_energy_sections(energy, sr, bpm, hop_length=hop,
                                      min_bars=args.min_bars, thresh_frac=args.thresh)
    # Smoothed envelope + bar length from the engine, for the plot / bar math.
    energy_s, spb = smooth_lowband_energy(energy, sr, bpm, hop_length=hop)
    times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop)

    print(f"Track   : {os.path.basename(args.audio)}")
    print(f"BPM     : {bpm:.2f}   sec/bar: {spb:.3f}   total bars: ~{int(dur/spb)}")
    snap_note = f"snapped to {args.phrase}-bar phrases" if args.phrase > 1 else "raw (no snapping)"
    print(f"Sections: {len(sections)} (>= {args.min_bars} bars, {snap_note})\n")

    rows = []
    print(f"{'BAR':>5}  {'TIME':>8}  {'EVENT':<22}  LENGTH")
    print("-" * 52)
    for is_drop, t0, t1 in sections:
        bar_raw = to_bar(t0, spb)
        bar = snap(bar_raw, args.phrase)
        length = int(round((t1 - t0) / spb))
        label = "DROP (kick in)" if is_drop else "BREAKDOWN (kick out)"
        m, s = divmod(t0, 60)
        print(f"{bar:>5}  {int(m):02d}:{s:05.2f}  {label:<22}  {length} bars")
        rows.append((bar, t0, t1, label, length))

    drops = [r for r in rows if "DROP" in r[3]]
    print(f"\n{len(drops)} DROP point(s) — bars: {', '.join(str(r[0]) for r in drops)}")

    # --- optional: reduce to N physical pads (e.g. a 4-pad controller) ---
    pad_cues = []
    if args.pads > 0:
        from src.dsp.phrasing_engine import label_energy_sections, map_segments_to_hotcues
        segments = label_energy_sections(sections)
        cues = map_segments_to_hotcues(segments, max_pads=args.pads)
        print(f"\nHOT CUES on {args.pads} pads (most important sections):")
        print(f"  {'PAD':>4}  {'BAR':>5}  {'TIME':>8}  TYPE")
        print("  " + "-" * 32)
        for c in cues:
            bar = snap(to_bar(c.time, spb), args.phrase)
            m, s = divmod(c.time, 60)
            print(f"  {c.label:>4}  {bar:>5}  {int(m):02d}:{s:05.2f}  {c.type}")
            pad_cues.append((c.label, bar, c.time, c.type))

    # write text report
    os.makedirs("results", exist_ok=True)
    rpt = Path("results/structure_bars.txt")
    with open(rpt, "w") as f:
        f.write(f"Track: {args.audio}\nBPM: {bpm:.2f}  sec/bar: {spb:.3f}  total bars ~{int(dur/spb)}\n")
        f.write(f"Snapping: {snap_note}\n\n")
        f.write(f"{'BAR':>5}  {'TIME':>8}  {'EVENT':<22}  LENGTH\n")
        f.write("-" * 52 + "\n")
        for bar, t0, t1, label, length in rows:
            m, s = divmod(t0, 60)
            f.write(f"{bar:>5}  {int(m):02d}:{s:05.2f}  {label:<22}  {length} bars\n")
        f.write(f"\nDROP bars: {', '.join(str(r[0]) for r in drops)}\n")
        if pad_cues:
            f.write(f"\nHOT CUES on {args.pads} pads:\n")
            for pad, bar, t, typ in pad_cues:
                m, s = divmod(t, 60)
                f.write(f"  {pad:>4}  bar {bar:>4}  {int(m):02d}:{s:05.2f}  {typ}\n")
    print(f"\nText report -> {rpt}")

    if not args.no_plot:
        plt.figure(figsize=(13, 4))
        plt.plot(times, energy_s, color="#333", linewidth=0.8, label="Kick+bass energy (20-150 Hz)")
        plt.axhline(args.thresh * energy_s.max(), color="grey", ls="--", lw=0.8, label="kick-on threshold")
        for bar, t0, t1, label, length in rows:
            color = "#2ca02c" if "DROP" in label else "#d62728"
            plt.axvspan(t0, t1, color=color, alpha=0.12)
            plt.axvline(t0, color=color, lw=1.2)
            plt.text(t0, energy_s.max() * 0.95, f"b{bar}", rotation=90,
                     va="top", ha="right", fontsize=8, color=color)
        # mark the pads selected for the controller
        for pad, bar, t, typ in pad_cues:
            plt.axvline(t, color="#1f77b4", lw=1.6, ls=":")
            plt.text(t, energy_s.max() * 0.55, pad, rotation=90, va="top", ha="left",
                     fontsize=8, color="#1f77b4", fontweight="bold")
        plt.title(f"Structure from low-band energy — green=DROP, red=BREAKDOWN  ({bpm:.1f} BPM)")
        plt.xlabel("Time (s)"); plt.ylabel("Low-band energy")
        plt.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        os.makedirs("results/plots", exist_ok=True)
        out = "results/plots/structure_bars.png"
        plt.savefig(out, dpi=110)
        plt.close()
        print(f"Plot        -> {out}")


if __name__ == "__main__":
    main()
