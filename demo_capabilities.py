"""DJIA — DJ Mixing Analytics: full capability demo (standalone script).

Runs the ACTUAL repository code end-to-end on a single techno track and
demonstrates every one of the 5 implemented phases:

    Phase 1  Ingestion & library scanning   (ingestion/scanner, ingestion/loader)
    Phase 2  DSP core — 4 analysis engines   (dsp/groove, phrasing, mood, curation)
    Phase 3  AI layer — mood classification  (ai/classifier)
    Phase 4  Database, similarity, Traktor    (database/store, matching/similarity, traktor/exporter)
    Phase 5  Advanced AI — transitions/sets   (ai/transition_mapper, ai/playlist_generator)

Every function called below is the project's own code — nothing is reimplemented.

Usage:
    python demo_capabilities.py [path/to/track.mp3]

If no path is given it defaults to data/2000_and_one-pak_pak.mp3.
Plots are written to results/plots/ (headless-safe); pass --show to open windows.
"""

import argparse
import copy
import os
import sys
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")  # headless: save figures to files instead of opening windows
import matplotlib.pyplot as plt

import librosa
import librosa.display

# Repo is importable from its root (this script lives at the repo root).
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

PLOT_DIR = Path("results/plots")


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def savefig(name: str, show: bool) -> None:
    """Save the current figure to results/plots/, and optionally display it."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    print(f"    [plot] saved -> {path}")
    if show:
        plt.show()
    plt.close()


# ----------------------------------------------------------------------------
# Phase 1 — Ingestion & Library Scanning
# ----------------------------------------------------------------------------
def phase1_ingestion(audio: str, show: bool):
    banner("PHASE 1 · Ingestion & Library Scanning")
    from src.ingestion.scanner import AudioScanner
    from src.ingestion.loader import AudioLoader

    scanner = AudioScanner(data_dir="data")
    found = scanner.scan()
    print(f"Scanner found {scanner.get_file_count()} audio file(s) in data/:")
    for f in found:
        print("  -", f.get("file_name", f))

    loader = AudioLoader()  # TARGET_SR = 22050, mono
    meta = loader.extract_metadata(Path(audio))
    print("\nMetadata extracted by AudioLoader:")
    for k, v in meta.items():
        print(f"  {k:>10}: {v}")

    # Load the waveform exactly as the pipeline does (22.05 kHz mono)
    y, sr = loader.load_audio(Path(audio))
    print(f"\nLoaded {len(y):,} samples @ {sr} Hz  ->  {len(y)/sr/60:.2f} min mono")

    t = np.linspace(0, len(y) / sr, num=len(y))
    plt.figure()
    plt.plot(t, y, linewidth=0.4, color="#1f77b4")
    plt.title("Waveform (22.05 kHz mono)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    savefig("01_waveform.png", show)

    return y, sr


# ----------------------------------------------------------------------------
# Phase 2 — DSP Core (4 engines)
# ----------------------------------------------------------------------------
def phase2_dsp(audio: str, y, sr, show: bool):
    banner("PHASE 2 · DSP Core — the 4 analysis engines")
    from src.dsp.extractor import extract_track_features, extract_feature_vector

    track = extract_track_features(audio)  # runs all 4 engines
    print("Track DNA extracted.")
    print(f"  Duration    : {track.duration:.1f}s")
    print(f"  Sample rate : {track.sample_rate} Hz")
    print(f"  Analysed at : {track.analysis_timestamp}")

    # --- 2a Groove ---
    g = track.groove
    print("\n[2a] Groove Engine")
    print(f"  BPM (decimal) : {g.bpm:.2f}")
    print(f"  Swing score   : {g.swing_score:.2f}   (0=stiff, 1=groovy)")
    print(f"  Tempo stable  : {g.tempo_stability}   (variance={g.stability_variance:.4f})")
    print(f"  Beats detected: {len(g.beat_times)}")

    mask = np.array(g.beat_times) < 20
    plt.figure()
    seg = y[: 20 * sr]
    plt.plot(np.linspace(0, 20, len(seg)), seg, linewidth=0.4, color="#999")
    for bt in np.array(g.beat_times)[mask]:
        plt.axvline(bt, color="#d62728", alpha=0.6, linewidth=0.8)
    plt.title(f"Beat grid (first 20s) — {g.bpm:.2f} BPM")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    savefig("02_beat_grid.png", show)

    # --- 2b Phrasing ---
    p = track.phrasing
    print("\n[2b] Phrasing Engine")
    print(f"  Segment boundaries  : {len(p.segment_boundaries)}")
    print(f"  Segments labelled   : {len(p.segments)}")
    print(f"  Hot cues generated  : {len(p.cue_points)}")
    print(f"  Structure confidence: {p.structure_confidence:.2f}")
    print("  Segment labels:", dict(Counter(s.label for s in p.segments)))
    print("  First 8 hot cues:")
    for c in p.cue_points[:8]:
        m, s = divmod(c.time, 60)
        print(f"    {c.label:<6} {c.type:<10} @ {int(m):02d}:{s:05.2f}")

    from src.dsp.phrasing_engine import compute_novelty_curve
    nov = compute_novelty_curve(y, sr)
    nov_t = librosa.frames_to_time(np.arange(len(nov)), sr=sr, hop_length=512)
    plt.figure()
    plt.plot(nov_t, nov, color="#2ca02c", linewidth=0.7, label="Spectral novelty")
    for b in p.segment_boundaries:
        plt.axvline(b, color="#ff7f0e", alpha=0.35, linewidth=0.8)
    plt.title("Phrasing engine — novelty curve & detected section boundaries")
    plt.xlabel("Time (s)")
    plt.ylabel("Novelty (0-1)")
    plt.legend(loc="upper right")
    savefig("03_novelty.png", show)

    # --- 2c Mood ---
    mo = track.mood
    print("\n[2c] Mood Engine")
    print(f"  Musical key   : {mo.key}")
    print(f"  Camelot key   : {mo.camelot_key}   (for harmonic mixing)")
    print(f"  Key confidence: {mo.key_confidence:.2f}")
    print(f"  Brightness    : {mo.brightness:.2f}   (0=dark, 1=bright)")

    chroma = librosa.feature.chroma_cqt(y=y[: 60 * sr], sr=sr)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(chroma, y_axis="chroma", x_axis="time", sr=sr, cmap="magma")
    plt.colorbar(label="Intensity")
    plt.title(f"Chromagram (first 60s) -> detected {mo.key} ({mo.camelot_key})")
    savefig("04_chromagram.png", show)

    # --- 2d Curation ---
    cu = track.curation
    print("\n[2d] Curation Engine")
    print(f"  Danceability : {cu.danceability:.2f}")
    print(f"  Energy profile: {cu.energy_type}")
    print(f"  Complexity   : {cu.complexity_score:.2f}")
    print(f"  Semantic tags: {', '.join(cu.semantic_tags)}")

    ec = cu.energy_curve
    ec_t = librosa.frames_to_time(np.arange(len(ec)), sr=sr, hop_length=512)
    plt.figure()
    plt.plot(ec_t, ec, color="#9467bd", linewidth=0.6)
    plt.fill_between(ec_t, ec, color="#9467bd", alpha=0.25)
    plt.title(f"Energy (RMS) curve — profile classified as '{cu.energy_type}'")
    plt.xlabel("Time (s)")
    plt.ylabel("RMS energy")
    savefig("05_energy.png", show)

    # --- 2e feature vector ---
    print("\n[2e] Flat feature vector (for ML / similarity)")
    fv = extract_feature_vector(track)
    for k, v in fv.items():
        print(f"  {k:<22}: {v:.3f}" if isinstance(v, float) else f"  {k:<22}: {v}")

    return track


# ----------------------------------------------------------------------------
# Phase 3 — AI mood classification
# ----------------------------------------------------------------------------
def phase3_classifier(y, sr, show: bool):
    banner("PHASE 3 · AI Layer — mood classification")
    from src.ai.classifier import classify_mood

    result = classify_mood(y, sr)
    moods = result["moods"]
    print("Energy level :", result["energy"])
    print("Danceability : %.2f" % result["danceability"])
    print("\nMood scores:")
    for k, v in sorted(moods.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<12}: {v:.2f}")

    labels = list(moods.keys())
    vals = list(moods.values())
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    vals_c = vals + vals[:1]
    ang_c = ang + ang[:1]
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(ang_c, vals_c, color="#d62728", linewidth=2)
    ax.fill(ang_c, vals_c, color="#d62728", alpha=0.25)
    ax.set_xticks(ang)
    ax.set_xticklabels(labels)
    ax.set_title("Mood profile (rule-based classifier)", pad=20)
    savefig("06_mood_radar.png", show)

    return moods


# ----------------------------------------------------------------------------
# Phase 4 — Database, similarity, Traktor export
# ----------------------------------------------------------------------------
def phase4_database(audio: str, track, moods, sr):
    banner("PHASE 4 · Database, Similarity Search & Traktor Export")
    from src.database.schema import init_db
    from src.database.store import TrackStore
    from src.matching.similarity import find_similar_tracks
    from src.traktor.exporter import add_track_analysis, export_nml

    DB = "results/demo_djia.db"
    os.makedirs("results", exist_ok=True)
    if os.path.exists(DB):
        os.remove(DB)
    init_db(DB).close()  # create tables
    store = TrackStore(DB)

    def store_track(tr, artist, title, mood_scores):
        tid = store.insert_track(
            file_path=tr.file_path, file_name=os.path.basename(tr.file_path),
            format="mp3", duration=tr.duration, artist=artist, title=title)
        store.insert_features(tid, {
            "tempo": float(tr.groove.bpm),
            "spectral_centroid_mean": float(tr.mood.brightness * (sr / 2)),
            "spectral_flux_mean": float(tr.curation.complexity_score),
            "rms_mean": float(np.mean(tr.curation.energy_curve)),
            "rms_std": float(np.std(tr.curation.energy_curve)),
            "rms_peak": float(np.max(tr.curation.energy_curve)),
            "chroma_variance": float(tr.mood.key_confidence),
        })
        store.insert_mood(tid, {k: float(v) for k, v in mood_scores.items()})
        for s in tr.phrasing.segments:
            store.insert_segment(tid, segment_type=s.label, start_time=s.start_time,
                                 end_time=s.end_time, confidence=s.confidence)
        return tid

    # 4a store the real track
    real_id = store_track(track, "2000 and One", "Pak Pak", moods)
    print(f"[4a] Inserted real track -> id={real_id}")
    print("     Tracks in DB:", store.get_tracks_count())

    # 4b similarity — insert synthetic neighbours to rank against
    print("\n[4b] Similarity search (cosine)")
    for i, (dbpm, dbright) in enumerate(
        [(0.5, 0.02), (4.0, 0.10), (-2.0, -0.05), (8.0, 0.25), (1.0, -0.02)], start=1
    ):
        tr2 = copy.deepcopy(track)
        tr2.groove.bpm = track.groove.bpm + dbpm
        tr2.mood.brightness = float(np.clip(track.mood.brightness + dbright, 0, 1))
        tr2.file_path = f"data/_synthetic_neighbour_{i}.mp3"
        store_track(tr2, f"Demo Artist {i}", f"Synthetic Neighbour {i}", moods)
    print("     DB now holds", store.get_tracks_count(), "tracks (1 real + 5 synthetic).")

    matches = find_similar_tracks(real_id, top_k=5, db_path=DB)
    print(f"     Top matches for track {real_id} (Pak Pak):")
    for tdict, score in matches:
        print(f"       score={score:.3f}  ->  {tdict.get('title')}  (bpm {tdict.get('bpm')})")

    # 4c Traktor NML export
    print("\n[4c] Traktor NML export")
    root = ET.Element("NML", VERSION="19")
    coll = ET.SubElement(root, "COLLECTION", ENTRIES="1")
    entry = ET.SubElement(coll, "ENTRY", TITLE="Pak Pak", ARTIST="2000 and One")
    ET.SubElement(entry, "LOCATION", DIR="/data/", FILE=os.path.basename(audio),
                  VOLUME="Macintosh HD")
    analysis = {
        "bpm": round(track.groove.bpm, 2),
        "brightness": round(track.mood.brightness * 100, 1),
        "danceability": round(track.curation.danceability * 100, 1),
        "mood": track.curation.semantic_tags[0] if track.curation.semantic_tags else "techno",
        "cue_points": [{"time": c.time, "type": c.type} for c in track.phrasing.cue_points[:6]],
    }
    add_track_analysis(root, real_id, analysis, db_path=DB)
    NML_OUT = "results/demo_collection.nml"
    export_nml(root, NML_OUT)
    print("     Wrote", NML_OUT)

    return store, DB


# ----------------------------------------------------------------------------
# Phase 5 — Transition scoring & playlist generation
# ----------------------------------------------------------------------------
def phase5_advanced(track, store, real_id_hint=1):
    banner("PHASE 5 · Advanced AI — Transition Scoring & Playlist Generation")
    from src.ai.transition_mapper import score_transition
    from src.ai.playlist_generator import generate_playlist

    # 5a transition scoring
    print("[5a] Transition mapper (BPM 40% / key 30% / mood 20% / energy 10%)")
    A = {
        "id": real_id_hint, "bpm": track.groove.bpm, "key": track.mood.key,
        "mood": {t: 1.0 for t in track.curation.semantic_tags},
        "rms_mean": float(np.mean(track.curation.energy_curve)),
    }
    B = dict(A)
    B["bpm"] = A["bpm"] + 6
    B["key"] = "C#/Db minor"
    ts = score_transition(A, B)
    print(f"     BPM score   : {ts.bpm_score}")
    print(f"     Key score   : {ts.key_score}")
    print(f"     Mood score  : {ts.mood_score}")
    print(f"     Energy score: {ts.energy_score}")
    print(f"     OVERALL     : {ts.overall_score}")

    # 5b playlist generation over the demo library
    print("\n[5b] Playlist generator (Dijkstra over transition graph)")
    all_tracks = {}
    for row in store.get_all_tracks_with_features():
        tid = row["id"]
        all_tracks[tid] = {
            "id": tid,
            "bpm": row.get("bpm") or 120,
            "key": "F#/Gb minor",
            "mood": {"dark": 0.7, "techno": 1.0},
            "rms_mean": row.get("rms_mean") or 0.2,
            "title": row.get("title"),
        }
    ids = sorted(all_tracks)
    path = generate_playlist(all_tracks, start_track_id=ids[0],
                             end_track_id=ids[-1], num_steps=4)
    print("     Track IDs available:", ids)
    print("     Optimal playlist path:", path)
    if path:
        print("     Set order:")
        for pos, tid in enumerate(path, 1):
            t = all_tracks[tid]
            print(f"       {pos}. {t['title']:<24} {t['bpm']:.1f} BPM")


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DJIA full capability demo")
    parser.add_argument("audio", nargs="?", default="data/2000_and_one-pak_pak.mp3",
                        help="Path to an audio file (default: data/2000_and_one-pak_pak.mp3)")
    parser.add_argument("--show", action="store_true",
                        help="Open plot windows instead of only saving them")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"ERROR: track not found: {args.audio}")
        print("Pass a path, e.g.:  python demo_capabilities.py path/to/track.mp3")
        sys.exit(1)

    banner("DJIA — DJ MIXING ANALYTICS · FULL CAPABILITY DEMO")
    print("Track:", args.audio, "(%.1f MB)" % (os.path.getsize(args.audio) / 1e6))

    y, sr = phase1_ingestion(args.audio, args.show)
    track = phase2_dsp(args.audio, y, sr, args.show)
    moods = phase3_classifier(y, sr, args.show)
    store, _ = phase4_database(args.audio, track, moods, sr)
    phase5_advanced(track, store)

    banner("DEMO COMPLETE — all 5 phases exercised on real repo code")
    print("Plots  -> results/plots/*.png")
    print("DB     -> results/demo_djia.db")
    print("Traktor-> results/demo_collection.nml")


if __name__ == "__main__":
    main()
