"""Data-driven 5-phase setlist generator.

Builds a club-set energy arc (warm-up -> build -> peak -> breakdown -> comeback)
from the analyzed library: each track is scored for phase suitability from its
measured mood/energy/brightness/BPM, phases get proportional quotas, ordering
inside and across phases maximizes transition compatibility, and every
consecutive pair gets a mix sheet with element-onset mix points (where to start
the incoming deck, where its bass lands, when the blend must be done).
"""

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .transition_mapper import (
    _calculate_bpm_compatibility,
    _calculate_energy_arc,
    _calculate_mood_continuity,
)

logger = logging.getLogger(__name__)

MOOD_DIMS = ["dark", "hypnotic", "euphoric", "aggressive", "industrial", "minimal"]

PHASE_ORDER = ["warmup", "build", "peak", "breakdown", "comeback"]

# Share of the set (breakdown handled separately: 1-2 tracks, never proportional)
PHASE_SHARE = {"warmup": 0.20, "build": 0.29, "peak": 0.36, "comeback": 0.15}

# Phases where the energy should rise track-over-track
ASCENDING_PHASES = {"warmup", "build", "comeback"}

TRANSITION_WEIGHTS = {"bpm": 0.35, "key": 0.30, "mood": 0.20, "energy": 0.15}
ENERGY_DIRECTION_BONUS = 0.06

# Groove/swing is applied as a multiplicative penalty on top of the weighted
# blend above, not a 5th competing weight (mirrors the Pairing notebook's
# groove_score, but as a modifier rather than a share of the total). Unknown
# swing on either track applies no penalty (factor 1.0, same as not having the
# term at all); a full swing clash floors the score at GROOVE_PENALTY_FLOOR.
GROOVE_PENALTY_FLOOR = 0.7

PHASE_STRATEGY = {
    "warmup": ("32-48 bars", "warm lows, restrained highs — let grooves breathe"),
    "build": ("16-32 bars", "push high-mids gradually, layer percussion"),
    "peak": ("8-16 bars", "full spectrum, minimal effects — swap on drops"),
    "breakdown": ("clean 4-8 bar break", "no blend: cut on a phrase, reverb tail, strip the kick"),
    "comeback": ("12-16 bars", "reintroduce bass gradually (low-pass opening over 16 bars)"),
}


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def load_library(db_path: str = "db/djia.db") -> List[Dict]:
    """
    Load every fully-analyzed track (has BPM) as a flat dict.

    Returns dicts with: id, file_name, file_path, title, artist, duration,
    bpm, camelot_key, rms_mean, brightness (spectral centroid), swing_score,
    and the six mood dimensions.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT t.id, t.file_name, t.file_path, t.title, t.artist, t.duration,
                   f.bpm, f.camelot_key, f.rms_mean, f.swing_score,
                   f.spectral_centroid_mean AS brightness,
                   m.dark, m.hypnotic, m.euphoric, m.aggressive, m.industrial, m.minimal
            FROM tracks t
            JOIN features f ON t.id = f.track_id
            LEFT JOIN mood m ON t.id = m.track_id
            WHERE f.bpm IS NOT NULL
            ORDER BY t.id
            """
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def track_name(track: Dict) -> str:
    """Display name: 'Artist - Title' when tagged, else the file stem."""
    artist, title = track.get("artist"), track.get("title")
    if title:
        return f"{artist} - {title}" if artist else str(title)
    return Path(str(track.get("file_name", f"id{track.get('id')}"))).stem


# ---------------------------------------------------------------------------
# Phase scoring & assignment
# ---------------------------------------------------------------------------

def _minmax(values: List[Optional[float]]) -> List[float]:
    """Min-max normalize to 0-1; None maps to 0.5 (neutral)."""
    present = [v for v in values if v is not None]
    if not present:
        return [0.5] * len(values)
    lo, hi = min(present), max(present)
    span = hi - lo
    if span <= 0:
        return [0.5] * len(values)
    return [0.5 if v is None else (v - lo) / span for v in values]


def phase_quotas(n_tracks: int) -> Dict[str, int]:
    """
    Split n_tracks across the five phases.

    Breakdown is a gesture, not an intermission: 1 track below 26, 2 from 26 up.
    The remainder is split by PHASE_SHARE using largest-remainder rounding so
    quotas always sum exactly to n_tracks.
    """
    if n_tracks < 5:
        raise ValueError(f"Need at least 5 tracks for a 5-phase set, got {n_tracks}")

    breakdown = 1 if n_tracks < 26 else 2
    remaining = n_tracks - breakdown

    raw = {p: remaining * share for p, share in PHASE_SHARE.items()}
    quotas = {p: max(1, int(raw[p])) for p in PHASE_SHARE}
    # Largest remainder for the leftover slots
    leftover = remaining - sum(quotas.values())
    remainders = sorted(PHASE_SHARE, key=lambda p: raw[p] - int(raw[p]), reverse=True)
    for p in remainders:
        if leftover <= 0:
            break
        quotas[p] += 1
        leftover -= 1
    # If max(1, ...) overshot (tiny sets), trim from the largest phase
    while sum(quotas.values()) > remaining:
        biggest = max(quotas, key=quotas.get)
        quotas[biggest] -= 1

    quotas["breakdown"] = breakdown
    return quotas


def score_phases(tracks: List[Dict]) -> Dict[int, Dict[str, float]]:
    """
    Score every track's suitability for each phase (higher = better fit).

    Uses measured data only: the six mood dimensions, RMS energy, spectral
    brightness, and BPM — energy/brightness/BPM min-max normalized over the
    candidate pool so the profiles are library-relative.
    """
    energy = _minmax([t.get("rms_mean") for t in tracks])
    bright = _minmax([t.get("brightness") for t in tracks])
    bpm = _minmax([t.get("bpm") for t in tracks])

    raw: Dict[str, List[float]] = {p: [] for p in PHASE_ORDER}
    for i, t in enumerate(tracks):
        m = {d: (t.get(d) if t.get(d) is not None else 0.0) for d in MOOD_DIMS}
        e, b, tempo = energy[i], bright[i], bpm[i]

        raw["warmup"].append(
            1.2 * m["hypnotic"] + 0.8 * m["minimal"] + 0.3 * m["dark"]
            - 0.8 * m["aggressive"] - 0.6 * m["euphoric"] - 0.8 * e - 0.3 * b
        )
        raw["build"].append(
            0.8 * m["dark"] + 0.5 * m["hypnotic"] + 0.4 * m["industrial"]
            - 0.3 * m["euphoric"] + 0.6 * (1.0 - abs(e - 0.55) * 2.0)
        )
        raw["peak"].append(
            1.0 * m["euphoric"] + 0.8 * m["aggressive"] + 0.3 * m["industrial"]
            + 0.9 * e + 0.3 * b
        )
        raw["breakdown"].append(
            0.9 * (1.0 - e) + 0.6 * m["hypnotic"] + 0.3 * m["minimal"]
            + 0.5 * (1.0 - tempo) - 0.5 * m["aggressive"]
        )
        raw["comeback"].append(
            0.6 * m["euphoric"] + 0.4 * m["dark"] + 0.3 * m["aggressive"] + 0.5 * e
        )

    # Normalize each phase column so phases compete on equal footing
    normed = {p: _minmax(raw[p]) for p in PHASE_ORDER}
    return {
        t["id"]: {p: normed[p][i] for p in PHASE_ORDER}
        for i, t in enumerate(tracks)
    }


def assign_phases(
    tracks: List[Dict], quotas: Dict[str, int]
) -> Dict[str, List[Dict]]:
    """
    Assign tracks to phases by global greedy matching: the strongest
    (track, phase) affinities claim their slots first, so no phase hoards
    tracks another phase needs more.
    """
    total = sum(quotas.values())
    if total > len(tracks):
        raise ValueError(f"Setlist needs {total} tracks but library has {len(tracks)}")

    scores = score_phases(tracks)
    by_id = {t["id"]: t for t in tracks}

    candidates = sorted(
        ((scores[tid][p], tid, p) for tid in scores for p in PHASE_ORDER),
        reverse=True,
    )

    assignment: Dict[str, List[Dict]] = {p: [] for p in PHASE_ORDER}
    used: set = set()
    for score, tid, phase in candidates:
        if tid in used or len(assignment[phase]) >= quotas[phase]:
            continue
        assignment[phase].append(by_id[tid])
        used.add(tid)
        if len(used) == total:
            break
    return assignment


# ---------------------------------------------------------------------------
# Transition scoring & ordering
# ---------------------------------------------------------------------------

def camelot_score(a: Optional[str], b: Optional[str]) -> float:
    """Camelot-wheel compatibility for codes like '7A' (0-1; 0.5 when unknown)."""
    def parse(code):
        if not code:
            return None
        s = str(code).strip().upper()
        if len(s) < 2 or s[-1] not in "AB" or not s[:-1].isdigit():
            return None
        n = int(s[:-1])
        return (n, s[-1]) if 1 <= n <= 12 else None

    pa, pb = parse(a), parse(b)
    if pa is None or pb is None:
        return 0.5
    (na, la), (nb, lb) = pa, pb
    dist = min(abs(na - nb), 12 - abs(na - nb))
    if na == nb and la == lb:
        return 1.00
    if na == nb:
        return 0.90  # relative major/minor
    if dist == 1 and la == lb:
        return 0.90  # adjacent, same mode
    if dist == 2 and la == lb:
        return 0.70
    if dist == 1:
        return 0.60  # diagonal neighbour
    return 0.30


def _mood_vec(track: Dict) -> Optional[Dict[str, float]]:
    vec = {d: track.get(d) for d in MOOD_DIMS if track.get(d) is not None}
    return vec or None


def _groove_score(a: Dict, b: Dict) -> Optional[float]:
    """
    Swing compatibility (0-1); None when either track lacks a stored
    swing_score (0=straight/stiff grid, 1=loose/swung). Same groove -> 1.0;
    a straight track against a heavily swung one -> hats flam during a long
    blend, so the score falls off linearly and hits 0 at a 0.5 gap.
    """
    sa, sb = a.get("swing_score"), b.get("swing_score")
    if sa is None or sb is None:
        return None
    return max(0.0, 1.0 - abs(float(sa) - float(sb)) / 0.5)


def transition_score(a: Dict, b: Dict, ascending: bool = False) -> float:
    """Blended a->b transition quality (0-1), notebook-compatible weights."""
    score = (
        TRANSITION_WEIGHTS["bpm"] * _calculate_bpm_compatibility(a["bpm"], b["bpm"])
        + TRANSITION_WEIGHTS["key"] * camelot_score(a.get("camelot_key"), b.get("camelot_key"))
        + TRANSITION_WEIGHTS["mood"] * _calculate_mood_continuity(_mood_vec(a), _mood_vec(b))
        + TRANSITION_WEIGHTS["energy"] * _calculate_energy_arc(a.get("rms_mean"), b.get("rms_mean"))
    )
    groove = _groove_score(a, b)
    if groove is not None:
        score *= GROOVE_PENALTY_FLOOR + (1 - GROOVE_PENALTY_FLOOR) * groove
    if ascending:
        ra, rb = a.get("rms_mean"), b.get("rms_mean")
        if ra is not None and rb is not None and rb >= ra:
            score += ENERGY_DIRECTION_BONUS
    return score


def order_setlist(assignment: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Chain the whole set: phases in canonical order, tracks within each phase
    picked greedily as the best follower of the current last track (with an
    energy-ascent bonus in warm-up/build/comeback). The opener is the
    lowest-energy warm-up track. Each track gains a 'phase' key.
    """
    order: List[Dict] = []
    for phase in PHASE_ORDER:
        pool = list(assignment.get(phase, []))
        ascending = phase in ASCENDING_PHASES
        while pool:
            if not order:
                nxt = min(pool, key=lambda t: t.get("rms_mean") or 0.0)
            else:
                prev = order[-1]
                nxt = max(pool, key=lambda t: transition_score(prev, t, ascending))
            pool.remove(nxt)
            entry = dict(nxt)
            entry["phase"] = phase
            order.append(entry)
    return order


def build_setlist(tracks: List[Dict], n_tracks: int) -> Dict:
    """Quota -> assignment -> ordering. Returns {'order', 'quotas'}."""
    quotas = phase_quotas(n_tracks)
    assignment = assign_phases(tracks, quotas)
    return {"order": order_setlist(assignment), "quotas": quotas}


# ---------------------------------------------------------------------------
# Mix points (element onsets, cached)
# ---------------------------------------------------------------------------

DEFAULT_CACHE = "results/mix_points_cache.json"


def mix_points_cached(track: Dict, cache_path: str = DEFAULT_CACHE) -> Dict:
    """
    Element-onset mix points for a track, cached to JSON keyed by file name.

    Loads the full audio on a cache miss (seconds per track), then never again.
    Returns {} when the audio file can't be found.
    """
    cache: Dict[str, Dict] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning(f"Unreadable mix-points cache {cache_path}; rebuilding")

    key = str(track.get("file_name"))
    if key in cache:
        return cache[key]

    candidates = [track.get("file_path"), os.path.join("data", key)]
    path = next((p for p in candidates if p and os.path.exists(str(p))), None)
    if path is None or not track.get("bpm"):
        logger.warning(f"No audio/BPM for {key}; mix sheet will be partial")
        return {}

    import librosa

    from ..dsp.phrasing_engine import derive_mix_points, detect_element_onsets

    logger.info(f"Computing mix points for {key}")
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    bpm = float(track["bpm"])
    duration = float(track.get("duration") or librosa.get_duration(y=y, sr=sr))
    onsets = detect_element_onsets(y, sr, bpm=bpm)
    points = derive_mix_points(onsets, bpm=bpm, duration=duration)
    points["n_onsets"] = len(onsets)

    cache[key] = points
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1)
    return points


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _mmss(t: Optional[float]) -> str:
    return "—" if t is None else f"{int(t // 60)}:{int(t % 60):02d}"


def _transition_sheet(a: Dict, b: Dict, pts_a: Dict, pts_b: Dict) -> List[str]:
    """Markdown lines for one a->b transition mix sheet."""
    pitch = (a["bpm"] - b["bpm"]) / b["bpm"] * 100
    key_s = camelot_score(a.get("camelot_key"), b.get("camelot_key"))
    overall = transition_score(a, b)
    blend, eq = PHASE_STRATEGY[b["phase"]] if b["phase"] != a["phase"] else PHASE_STRATEGY[a["phase"]]

    mix_out = pts_a.get("mix_out")
    mix_in, bass_in = pts_b.get("mix_in"), pts_b.get("bass_in")
    full_on = pts_b.get("full_on")

    lines = [
        f"**{track_name(a)}  →  {track_name(b)}**  "
        f"(score {overall:.2f} | key {a.get('camelot_key') or '?'}→{b.get('camelot_key') or '?'} {key_s:.2f})",
        "",
    ]
    if abs(pitch) > 8:
        lines.append(
            f"- ⚠️ BPM gap too wide to beatmatch ({a['bpm']:.1f} → {b['bpm']:.1f}, "
            f"{pitch:+.1f}%) — **don't blend**: let the outgoing track end or cut on a "
            "phrase, then restart the groove with the incoming track"
        )
    else:
        lines.append(
            f"- Beatmatch: pitch the incoming track **{pitch:+.1f}%** "
            f"({a['bpm']:.1f} → {b['bpm']:.1f} BPM)"
        )
    lines.append(f"- Blend length: **{blend}** — {eq}")

    if b["phase"] == "breakdown":
        lines.append(
            f"- **Clean break**: cut the outgoing track on a phrase near {_mmss(mix_out)}, "
            "reverb tail into space, let the breakdown track stand alone."
        )
        return lines

    if mix_out is not None and mix_in is not None:
        offset = (bass_in - mix_in) if bass_in is not None else 0.0
        start_at = max(0.0, mix_out - offset)
        lines.append(
            f"- Start the incoming deck (from its {_mmss(mix_in)} cue) when the outgoing "
            f"track reaches **{_mmss(start_at)}**"
        )
        if bass_in is not None and offset > 0:
            lines.append(
                f"- **Bass swap at {_mmss(mix_out)}** on the outgoing clock — the incoming "
                f"bass lands ({_mmss(bass_in)} on its own clock) right at the mix-out point"
            )
        else:
            lines.append(f"- **Bass swap at {_mmss(mix_out)}** (incoming bass is live from its start)")
        if full_on is not None:
            lines.append(f"- Incoming track fully on by {_mmss(full_on)} — blend must be done")
    else:
        lines.append("- No element mix points available — mix on phrase boundaries by ear")

    return lines


def render_report(
    setlist: Dict,
    mix_points_fn: Optional[Callable[[Dict], Dict]] = None,
    title: str = "DJIA — Data-Driven 5-Phase Setlist",
) -> str:
    """
    Render the phase plan + per-transition mix sheets as markdown.

    Args:
        setlist: Output of build_setlist
        mix_points_fn: track -> mix-points dict (defaults to the cached
                       element-onset computation; pass a stub in tests)
        title: Report title
    """
    if mix_points_fn is None:
        mix_points_fn = mix_points_cached

    order = setlist["order"]
    total_s = sum(t.get("duration") or 0 for t in order)
    bpms = [t["bpm"] for t in order]

    out = [
        f"# {title}",
        "",
        f"**Tracks:** {len(order)}  |  **Total duration:** "
        f"{int(total_s // 3600)}:{int(total_s % 3600 // 60):02d}:{int(total_s % 60):02d}  |  "
        f"**BPM:** {min(bpms):.0f}–{max(bpms):.0f} (avg {sum(bpms) / len(bpms):.1f})",
        "",
        "Phases are chosen from measured mood/energy/brightness/BPM; ordering maximizes",
        "pairwise transition compatibility (BPM 35% / Camelot key 30% / mood 20% / energy 15%).",
        "Mix points come from element-onset detection on the audio itself.",
        "",
        "## Phase plan",
        "",
    ]

    pos = 0
    for phase in PHASE_ORDER:
        phase_tracks = [t for t in order if t["phase"] == phase]
        if not phase_tracks:
            continue
        dur = sum(t.get("duration") or 0 for t in phase_tracks)
        blend, eq = PHASE_STRATEGY[phase]
        out += [
            f"### {phase.upper()} — {len(phase_tracks)} tracks | {_mmss(dur)} | blends {blend}",
            "",
            f"*{eq}*",
            "",
            "| # | Track | BPM | Key | Energy | Top mood |",
            "|---|---|---|---|---|---|",
        ]
        for t in phase_tracks:
            pos += 1
            moods = {d: t.get(d) for d in MOOD_DIMS if t.get(d) is not None}
            top = max(moods, key=moods.get) if moods else "?"
            rms = t.get("rms_mean")
            out.append(
                f"| {pos} | {track_name(t)} | {t['bpm']:.1f} | {t.get('camelot_key') or '?'} | "
                f"{rms:.3f} | {top} |" if rms is not None else
                f"| {pos} | {track_name(t)} | {t['bpm']:.1f} | {t.get('camelot_key') or '?'} | ? | {top} |"
            )
        out.append("")

    out += ["## Transition mix sheets", ""]
    points = [mix_points_fn(t) or {} for t in order]
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        marker = "" if a["phase"] == b["phase"] else f"  `[{a['phase']} → {b['phase']}]`"
        out.append(f"### {i + 1} → {i + 2}{marker}")
        out.append("")
        out += _transition_sheet(a, b, points[i], points[i + 1])
        out.append("")

    return "\n".join(out)


def generate_setlist(
    db_path: str = "db/djia.db",
    n_tracks: int = 28,
    output_path: str = "results/setlist_5phase.md",
    with_mix_sheets: bool = True,
) -> str:
    """
    End-to-end: load library, build the 5-phase setlist, write the markdown
    report (phase plan + per-transition mix sheets) and return its path.

    with_mix_sheets=False skips the audio-based element-onset computation
    (fast, but transitions fall back to phrase-boundary advice).
    """
    tracks = load_library(db_path)
    setlist = build_setlist(tracks, n_tracks)
    fn = mix_points_cached if with_mix_sheets else (lambda t: {})
    report = render_report(setlist, mix_points_fn=fn)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
    except PermissionError:
        # the file is open elsewhere (e.g. a viewer) - write a timestamped
        # copy instead of dying
        from datetime import datetime

        stem = Path(output_path)
        output_path = str(
            stem.with_name(f"{stem.stem}_{datetime.now():%Y%m%d-%H%M%S}{stem.suffix}")
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
    logger.info(f"Wrote {n_tracks}-track 5-phase setlist to {output_path}")
    return output_path
