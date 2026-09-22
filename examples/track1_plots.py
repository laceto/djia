import sqlite3
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TRACK_ID = 1

# Categorical palette (fixed order, validated for CVD-safety) — see dataviz skill.
CAT = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
}
SEGMENT_COLOR = {
    "intro": CAT["blue"],
    "build": CAT["orange"],
    "drop": CAT["aqua"],
    "breakdown": CAT["yellow"],
    "outro": CAT["magenta"],
}
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

conn = sqlite3.connect("db/djia.db")

track = pd.read_sql_query("SELECT * FROM tracks WHERE id = ?", conn, params=(TRACK_ID,)).iloc[0]
features = pd.read_sql_query("SELECT * FROM features WHERE track_id = ?", conn, params=(TRACK_ID,)).iloc[0]
mood = pd.read_sql_query("SELECT * FROM mood WHERE track_id = ?", conn, params=(TRACK_ID,)).iloc[0]
segments = pd.read_sql_query(
    "SELECT * FROM segments WHERE track_id = ? ORDER BY method, start_time", conn, params=(TRACK_ID,)
)

TITLE_SUFFIX = f"{track['file_name'].strip()} (track_id={TRACK_ID})"


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=SECONDARY_INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# --- 1. Segment timeline: spectral vs phrase16, one row per method ---
methods = segments["method"].unique().tolist()
fig, ax = plt.subplots(figsize=(12, 2.5 + 0.6 * len(methods)))

for row_i, method in enumerate(methods):
    method_segs = segments[segments["method"] == method]
    bars = [
        (r.start_time, r.end_time - r.start_time) for r in method_segs.itertuples()
    ]
    colors = [SEGMENT_COLOR.get(r.segment_type, MUTED) for r in method_segs.itertuples()]
    ax.broken_barh(bars, (row_i - 0.4, 0.8), facecolors=colors, edgecolor="white", linewidth=1.5)
    for r in method_segs.itertuples():
        dur = r.end_time - r.start_time
        if dur > 15:
            ax.text(
                r.start_time + dur / 2, row_i, r.segment_type,
                ha="center", va="center", fontsize=8, color="white", fontweight="bold",
            )

ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods)
ax.set_xlabel("Time (s)", color=SECONDARY_INK)
ax.set_title(f"Segment structure — {TITLE_SUFFIX}", color=INK)
ax.set_xlim(0, track["duration"])
style_axes(ax)
ax.yaxis.grid(False)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEGMENT_COLOR.values()]
ax.legend(handles, SEGMENT_COLOR.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.25),
          ncol=len(SEGMENT_COLOR), frameon=False)
plt.tight_layout()
plt.savefig("examples/track1_segments.png", dpi=150)
plt.close()
print("saved examples/track1_segments.png")


# --- 2. Mood profile: single-series bar chart (one hue = magnitude, not identity) ---
mood_dims = ["dark", "hypnotic", "euphoric", "aggressive", "industrial", "minimal"]
mood_values = [mood[d] for d in mood_dims]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(mood_dims, mood_values, color=CAT["blue"], width=0.6)
for b, v in zip(bars, mood_values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", color=SECONDARY_INK, fontsize=9)
ax.set_ylim(0, max(mood_values) * 1.25)
ax.set_ylabel("Score (0-1)", color=SECONDARY_INK)
ax.set_title(f"Mood profile — {TITLE_SUFFIX}", color=INK)
style_axes(ax)
plt.tight_layout()
plt.savefig("examples/track1_mood.png", dpi=150)
plt.close()
print("saved examples/track1_mood.png")


# --- 3. Spectral/energy feature summary: means with std as error bars ---
feature_pairs = [
    ("spectral_centroid", "spectral_centroid_mean", "spectral_centroid_std"),
    ("spectral_flux", "spectral_flux_mean", None),
    ("onset_strength", "onset_strength_mean", "onset_strength_std"),
    ("rms", "rms_mean", "rms_std"),
    ("mfcc", "mfcc_mean", "mfcc_std"),
]
labels = [p[0] for p in feature_pairs]
means = [features[p[1]] for p in feature_pairs]
stds = [features[p[2]] if p[2] else 0 for p in feature_pairs]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(labels, means, yerr=stds, capsize=5, color=CAT["blue"], width=0.55,
       error_kw={"ecolor": SECONDARY_INK, "elinewidth": 1.2})
ax.set_ylabel("Value (mean ± std)", color=SECONDARY_INK)
ax.set_title(f"Spectral / energy features — {TITLE_SUFFIX}", color=INK)
style_axes(ax)
plt.tight_layout()
plt.savefig("examples/track1_features.png", dpi=150)
plt.close()
print("saved examples/track1_features.png")


# --- 4. Track vs library: z-score profile across shared features ---
FEATURES = ["bpm", "swing_score", "dark", "hypnotic", "euphoric", "aggressive", "industrial", "minimal"]
df = pd.read_sql_query(
    """
    SELECT t.id, f.bpm, f.swing_score, m.dark, m.hypnotic, m.euphoric, m.aggressive, m.industrial, m.minimal
    FROM tracks t
    LEFT JOIN features f ON f.track_id = t.id
    LEFT JOIN mood m ON m.track_id = t.id
    """,
    conn,
).dropna(subset=FEATURES)

if len(df) > 1:
    z = (df[FEATURES] - df[FEATURES].mean()) / df[FEATURES].std()
    track_z = z.loc[df["id"] == TRACK_ID].iloc[0]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [CAT["blue"] if v >= 0 else "#e34948" for v in track_z.values]
    ax.bar(FEATURES, track_z.values, color=colors, width=0.55)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_ylabel("z-score vs library mean", color=SECONDARY_INK)
    ax.set_title(f"Track vs library (n={len(df)}) — {TITLE_SUFFIX}", color=INK)
    style_axes(ax)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("examples/track1_vs_library.png", dpi=150)
    plt.close()
    print("saved examples/track1_vs_library.png")
else:
    print("skipped track1_vs_library.png — need >1 track in library for a meaningful z-score")

conn.close()
