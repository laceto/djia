"""Build a per-track structural-signature table from the segments table.

For each (track, method) pair, derive:
  - signature_raw: ordered segment_type labels joined by '-'
  - signature_collapsed: run-length-collapsed labels, e.g. intro-build(5)-drop(6)-outro
  - bar_signature: each segment's exact bar range, e.g. intro:0-16|build:16-32|drop:32-96|outro:96-101
  - n_segments, total_bars

signature_collapsed groups tracks by section *shape* (label pattern) regardless of
timing; bar_signature additionally exposes exactly where (in bars) each section falls,
for comparing tracks that share the same shape AND land at the same bar counts.

Why bar_signature has whole numbers for 'phrase16' but decimals for 'spectral':
the two segmentation methods (src/dsp/phrasing_engine.py) derive boundaries completely
differently.
  - phrase16 segments come from create_phrase_locked_segments(), which cuts the track
    into fixed 16-bar blocks from time zero — its boundaries ARE bar numbers by
    construction, so start_bar/end_bar always land on integers (0, 16, 32, ...).
  - spectral segments come from detect_energy_sections(), which finds transitions in
    the actual low-band (kick+bass) energy curve and converts the crossing point to a
    timestamp via librosa.frames_to_time() (STFT-frame resolution, ~23ms at the default
    512-sample hop / 22,050 Hz) — a real audio event with no relationship to the bar
    grid. Dividing that timestamp by bar duration to get a "bar number" here is just a
    unit conversion for readability; it essentially never lands on a whole number,
    because the kick/bass actually drops out or returns whenever it does in the mix,
    not on a phrase boundary. A value like 73.2 means the transition happened ~20% of
    the way into bar 74 (bpm-dependent fractional offset, not a rounding artifact).

Also builds an intro/outro pairing table: for each (track, method), the intro and
outro segment's bar length (rounded to the nearest whole bar for grouping, since
spectral bar lengths are fractional), and groups of 2+ tracks that share the same
(intro_bars_rounded, outro_bars_rounded) pair.

Writes results/track_structures.csv and prints structure-sharing groups (2+ tracks
with an identical collapsed signature) per method.
"""
import sqlite3
from itertools import groupby

import pandas as pd

DB_PATH = "db/djia.db"
OUT_CSV = "results/track_structures.csv"
OUT_XLSX = "results/track_structures.xlsx"


def collapse(labels):
    parts = []
    for label, group in groupby(labels):
        n = len(list(group))
        parts.append(f"{label}({n})" if n > 1 else label)
    return "-".join(parts)


def build_tables(conn):
    """Returns (structures, segments_bars): the per-(track,method) summary table, and
    a long table with one row per segment carrying its exact bar boundaries."""
    query = """
        SELECT s.track_id, t.file_name, s.method, s.segment_type, s.start_time, s.end_time, f.bpm
        FROM segments s
        JOIN tracks t ON t.id = s.track_id
        JOIN features f ON f.track_id = s.track_id
        ORDER BY s.track_id, s.method, s.start_time
    """
    df = pd.read_sql_query(query, conn)

    struct_rows = []
    segment_rows = []
    for (track_id, file_name, method), group in df.groupby(["track_id", "file_name", "method"]):
        group = group.sort_values("start_time")
        labels = group["segment_type"].tolist()
        bpm = group["bpm"].iloc[0]
        bar_seconds = (60.0 / bpm) * 4
        track_start = group["start_time"].min()

        # Whole numbers for 'phrase16' (bar-locked by construction), fractional for
        # 'spectral' (real audio-transition timestamps divided by bar duration) — see
        # module docstring for why that's expected, not a rounding bug.
        bar_ranges = []  # (segment_type, start_bar, end_bar) per segment, in order
        for idx, seg in enumerate(group.itertuples(), start=1):
            start_bar = round((seg.start_time - track_start) / bar_seconds, 1)
            end_bar = round((seg.end_time - track_start) / bar_seconds, 1)
            bar_ranges.append((seg.segment_type, start_bar, end_bar))
            # Signed distance from start_bar to the nearest whole bar line, in
            # [-0.5, 0.5]; always 0 for phrase16. phrase_aligned flags transitions
            # within 1 beat (0.25 bar) of a bar line, i.e. likely an intentional
            # phrase-locked edit rather than a sound-design-driven transition.
            start_bar_offset = round(start_bar - round(start_bar), 2)
            segment_rows.append({
                "track_id": track_id,
                "file_name": file_name,
                "method": method,
                "segment_index": idx,
                "segment_type": seg.segment_type,
                "start_bar": start_bar,
                "end_bar": end_bar,
                "n_bars": round(end_bar - start_bar, 1),
                "start_bar_offset": start_bar_offset,
                "phrase_aligned": abs(start_bar_offset) <= 0.25,
                "start_time": round(seg.start_time, 2),
                "end_time": round(seg.end_time, 2),
            })

        total_bars = bar_ranges[-1][2] if bar_ranges else 0.0
        bar_signature = "|".join(f"{t}:{s:g}-{e:g}" for t, s, e in bar_ranges)

        struct_rows.append({
            "track_id": track_id,
            "file_name": file_name,
            "method": method,
            "n_segments": len(labels),
            "signature_raw": "-".join(labels),
            "signature_collapsed": collapse(labels),
            "bar_signature": bar_signature,
            "total_bars": round(total_bars, 1),
        })

    structures = pd.DataFrame(struct_rows).sort_values(["method", "track_id"]).reset_index(drop=True)
    segments_bars = pd.DataFrame(segment_rows).sort_values(["method", "track_id", "segment_index"]).reset_index(drop=True)
    return structures, segments_bars


def intro_outro_table(segments_bars: pd.DataFrame) -> pd.DataFrame:
    """One row per (track, method): intro/outro bar length, if present, rounded to
    the nearest whole bar for grouping (spectral bar lengths are fractional — see
    module docstring — so exact-decimal matches would almost never happen)."""
    rows = []
    for (track_id, file_name, method), group in segments_bars.groupby(["track_id", "file_name", "method"]):
        intro = group[group["segment_type"] == "intro"]
        outro = group[group["segment_type"] == "outro"]
        intro_bars = intro["n_bars"].iloc[0] if len(intro) else None
        outro_bars = outro["n_bars"].iloc[-1] if len(outro) else None
        rows.append({
            "track_id": track_id,
            "file_name": file_name,
            "method": method,
            "has_intro": intro_bars is not None,
            "intro_bars": intro_bars,
            "intro_bars_rounded": round(intro_bars) if intro_bars is not None else None,
            "has_outro": outro_bars is not None,
            "outro_bars": outro_bars,
            "outro_bars_rounded": round(outro_bars) if outro_bars is not None else None,
        })
    return pd.DataFrame(rows).sort_values(["method", "track_id"]).reset_index(drop=True)


def intro_outro_pair_groups(io_table: pd.DataFrame, method: str) -> pd.DataFrame:
    """Tracks sharing the same (intro_bars_rounded, outro_bars_rounded) pair — i.e.
    same-length intro AND same-length outro — for the given method. Only tracks with
    both an intro and an outro segment are eligible."""
    sub = io_table[(io_table["method"] == method) & io_table["has_intro"] & io_table["has_outro"]]
    grouped = sub.groupby(["intro_bars_rounded", "outro_bars_rounded"]).agg(
        n_tracks=("file_name", "count"),
        file_names=("file_name", lambda s: "; ".join(sorted(s))),
    ).reset_index()
    grouped = grouped[grouped["n_tracks"] >= 2]
    return grouped.sort_values(["n_tracks", "intro_bars_rounded", "outro_bars_rounded"],
                                ascending=[False, True, True]).reset_index(drop=True)


def report_groups(structures: pd.DataFrame):
    for method in sorted(structures["method"].unique()):
        sub = structures[structures["method"] == method]
        print(f"\n{'='*70}\nMethod: {method}  ({len(sub)} tracks)\n{'='*70}")
        grouped = sub.groupby("signature_collapsed")["file_name"].apply(list).sort_values(key=lambda s: s.str.len(), ascending=False)
        shared = grouped[grouped.apply(len) >= 2]
        print(f"{len(shared)} shared structure group(s) covering {shared.apply(len).sum()} tracks "
              f"(out of {len(sub)}); {len(grouped) - len(shared)} unique/singleton structures")
        for sig, files in shared.items():
            print(f"\n  [{len(files)} tracks] {sig}")
            for fn in files[:10]:
                print(f"    - {fn}")
            if len(files) > 10:
                print(f"    ... and {len(files) - 10} more")


def group_summary(structures: pd.DataFrame, method: str) -> pd.DataFrame:
    sub = structures[structures["method"] == method]
    grouped = sub.groupby("signature_collapsed").agg(
        n_tracks=("file_name", "count"),
        file_names=("file_name", lambda s: "; ".join(sorted(s))),
    ).reset_index()
    return grouped.sort_values(["n_tracks", "signature_collapsed"], ascending=[False, True]).reset_index(drop=True)


def by_track_wide(structures: pd.DataFrame) -> pd.DataFrame:
    """One row per track: both methods' signatures side by side, plus each spectral
    group's size, so sorting by spectral_signature_collapsed in Excel groups
    same-structure tracks together."""
    wide = structures.pivot(index=["track_id", "file_name"], columns="method",
                             values=["n_segments", "signature_collapsed", "bar_signature", "total_bars"])
    wide.columns = [f"{method}_{col}" for col, method in wide.columns]
    wide = wide.reset_index()

    spectral_counts = structures[structures["method"] == "spectral"] \
        .groupby("signature_collapsed")["file_name"].transform("count")
    spectral_group_size = structures[structures["method"] == "spectral"].assign(
        spectral_group_size=spectral_counts
    )[["track_id", "spectral_group_size"]]

    wide = wide.merge(spectral_group_size, on="track_id", how="left")
    return wide.sort_values(["spectral_signature_collapsed", "track_id"]).reset_index(drop=True)


def write_xlsx(structures: pd.DataFrame, segments_bars: pd.DataFrame, io_table: pd.DataFrame):
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        by_track_wide(structures).to_excel(writer, sheet_name="by_track", index=False)
        structures.to_excel(writer, sheet_name="long_format", index=False)
        segments_bars.to_excel(writer, sheet_name="segments_bars", index=False)
        group_summary(structures, "spectral").to_excel(writer, sheet_name="spectral_groups", index=False)
        group_summary(structures, "phrase16").to_excel(writer, sheet_name="phrase16_groups", index=False)
        io_table.to_excel(writer, sheet_name="intro_outro", index=False)
        intro_outro_pair_groups(io_table, "spectral").to_excel(writer, sheet_name="intro_outro_pairs_spectral", index=False)
        intro_outro_pair_groups(io_table, "phrase16").to_excel(writer, sheet_name="intro_outro_pairs_phrase16", index=False)

        for sheet in writer.sheets.values():
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for col_cells in sheet.columns:
                length = max((len(str(c.value)) for c in col_cells if c.value is not None), default=10)
                sheet.column_dimensions[col_cells[0].column_letter].width = min(length + 2, 60)

    print(f"Wrote {OUT_XLSX}")


def main():
    conn = sqlite3.connect(DB_PATH)
    structures, segments_bars = build_tables(conn)
    io_table = intro_outro_table(segments_bars)
    structures.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(structures)} rows to {OUT_CSV}")
    write_xlsx(structures, segments_bars, io_table)
    report_groups(structures)


if __name__ == "__main__":
    main()
