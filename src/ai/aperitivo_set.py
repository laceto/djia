"""DJ set builder with profiles (aperitivo, latenight): ~2 hour sets with two energy peaks.

Pool -> energy/vocal/mood scoring -> beam search over track orders -> Excel tracklist.
Everything except `load_pool` and `export_set_xlsx` is pure so it can be unit tested.
"""

import math
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

BASSLINE_KEYS = ("bassline_rolling", "bassline_deep_sub", "bassline_offbeat",
                 "bassline_staccato", "bassline_drone")

MIX_BEATS = 32                 # overlap per transition
KEY_WEIGHT = 1.0
BASS_REPEAT_COST = (0.2, 0.4)  # same type as previous, and as previous two
ARTIST_REPEAT_COST = 0.5



@dataclass(frozen=True)
class Profile:
    """Everything that makes one kind of set differ from another."""
    name: str
    max_rms: Optional[float] = None      # pool: loudness ceiling
    min_rms: Optional[float] = None      # pool: loudness floor
    bpm_min: float = 108.0
    bpm_max: float = 126.0
    min_minutes: float = 4.0
    peaks: Tuple[Tuple[float, float, float], ...] = ((0.33, 0.08, 0.80), (0.72, 0.09, 1.00))
    curve_base: float = 0.15             # target energy between peaks (0-1)
    ramp_floor: float = 0.0              # opening level as a share of the curve
    fade_floor: float = 0.35             # closing level (higher = ends hotter)
    vocal_weight: float = 0.4            # >0 favours vocal tracks, <0 avoids them
    mood_weights: Dict[str, float] = field(default_factory=dict)
    mood_weight: float = 0.0             # cost bonus for matching the mood mix
    max_bpm_step: float = 4.0
    energy_weight: float = 3.0           # how tightly tracks must follow the curve
    bpm_weight: float = 0.15
    title: str = ""


APERITIVO = Profile(name="aperitivo", max_rms=0.15, title="Aperitivo set")
LATENIGHT = Profile(
    name="latenight", min_rms=0.20, bpm_min=120.0, bpm_max=132.0,
    peaks=((0.30, 0.09, 0.85), (0.74, 0.10, 1.00)), curve_base=0.40,
    ramp_floor=0.40, fade_floor=0.55, vocal_weight=-0.4,
    mood_weights={"dark": 1.0, "hypnotic": 1.0, "industrial": 1.0},
    mood_weight=0.5, max_bpm_step=6.5, bpm_weight=0.10, energy_weight=10.0,
    title="Late night set",
)
PROFILES = {p.name: p for p in (APERITIVO, LATENIGHT)}

_CAMELOT = re.compile(r"^(\d{1,2})([ABab])$")


def parse_camelot(key: Optional[str]) -> Optional[Tuple[int, str]]:
    m = _CAMELOT.match((key or "").strip())
    if not m or not 1 <= int(m.group(1)) <= 12:
        return None
    return int(m.group(1)), m.group(2).upper()


def key_cost(a: Optional[str], b: Optional[str]) -> float:
    """0 same key, 0.15 energy-boost/swap neighbours, 0.5 two steps, 1.0 clash."""
    pa, pb = parse_camelot(a), parse_camelot(b)
    if pa is None or pb is None:
        return 0.6
    (na, la), (nb, lb) = pa, pb
    step = min((na - nb) % 12, (nb - na) % 12)
    if step == 0:
        return 0.0 if la == lb else 0.15
    if la == lb:
        return {1: 0.15, 2: 0.5}.get(step, 1.0)
    return 1.0


def energy_curve(frac: float, profile: Profile = APERITIVO) -> float:
    """Target energy 0-1 at set fraction `frac`: opening ramp, two peaks, wind-down."""
    v = profile.curve_base
    for centre, width, height in profile.peaks:
        v += (height - profile.curve_base) * math.exp(-0.5 * ((frac - centre) / width) ** 2)
    ramp = profile.ramp_floor + (1.0 - profile.ramp_floor) * min(1.0, frac / 0.15)
    fade = min(1.0, (1.0 - frac) / 0.12 + profile.fade_floor)
    return max(0.0, min(1.0, v * ramp * fade))


def _minmax(vals: List[float]) -> List[float]:
    lo, hi = min(vals), max(vals)
    return [0.0 if hi == lo else (v - lo) / (hi - lo) for v in vals]


def _zscores(vals: List[float]) -> List[float]:
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
    return [(v - m) / sd for v in vals]


def dominant_bassline(track: Dict) -> str:
    best = max(BASSLINE_KEYS, key=lambda k: track.get(k) or 0.0)
    return best.replace("bassline_", "").replace("_", " ")


def _norm_name(t: Dict) -> str:
    text = f"{t.get('artist') or ''} {t.get('title') or t.get('file_name') or ''}"
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def load_pool(db_path: str, profile: Profile = APERITIVO) -> List[Dict]:
    """Mixable tracks for the profile, with energy/vocal/mood (0-1) scores attached."""
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        """SELECT t.id, t.artist, t.title, t.file_name, t.duration,
                  f.bpm, f.camelot_key, f.rms_mean, f.onset_strength_mean, f.kick_rate,
                  f.sub_presence, f.vocal_presence, f.bassline_rolling, f.bassline_deep_sub,
                  f.bassline_offbeat, f.bassline_staccato, f.bassline_drone,
                  m.dark, m.hypnotic, m.euphoric, m.aggressive, m.industrial, m.minimal
           FROM tracks t JOIN features f ON f.track_id = t.id
           LEFT JOIN mood m ON m.track_id = t.id
           WHERE f.bpm BETWEEN ? AND ? AND f.rms_mean BETWEEN ? AND ?
             AND t.duration >= ? AND f.camelot_key IS NOT NULL
           ORDER BY t.id""",
        (profile.bpm_min, profile.bpm_max,
         profile.min_rms if profile.min_rms is not None else 0.0,
         profile.max_rms if profile.max_rms is not None else 9.0,
         profile.min_minutes * 60.0),
    ).fetchall()
    db.close()

    seen, pool = set(), []
    for r in rows:                      # drop duplicate files (same artist+title text)
        d = dict(r)
        key = _norm_name(d)
        if key in seen:
            continue
        seen.add(key)
        pool.append(d)
    if not pool:
        return pool
    return score_pool(pool, profile)


def score_pool(pool: List[Dict], profile: Profile = APERITIVO) -> List[Dict]:
    """Attach 'energy', 'vocal' and 'mood' (each 0-1 within the pool)."""
    cols = ("rms_mean", "onset_strength_mean", "kick_rate", "sub_presence")
    zs = [_zscores([float(t.get(c) or 0.0) for t in pool]) for c in cols]
    energy = _minmax([sum(z[i] for z in zs) / len(cols) for i in range(len(pool))])
    vocal = _minmax([float(t.get("vocal_presence") or 0.0) for t in pool])
    if profile.mood_weights:
        raw = [sum(w * float(t.get(k) or 0.0) for k, w in profile.mood_weights.items())
               for t in pool]
        mood = _minmax(raw)
    else:
        mood = [0.0] * len(pool)
    for t, e, v, m in zip(pool, energy, vocal, mood):
        t["energy"], t["vocal"], t["mood"] = e, v, m
    return pool


def overlap_seconds(bpm: float) -> float:
    return MIX_BEATS * 60.0 / bpm


def transition_cost(prev: Dict, cur: Dict, history: List[Dict],
                    profile: Profile = APERITIVO) -> float:
    step = abs(prev["bpm"] - cur["bpm"])
    cost = KEY_WEIGHT * key_cost(prev.get("camelot_key"), cur.get("camelot_key"))
    cost += profile.bpm_weight * step + (2.0 if step > profile.max_bpm_step else 0.0)
    kind = dominant_bassline(cur)
    if dominant_bassline(prev) == kind:
        cost += BASS_REPEAT_COST[0]
        if len(history) >= 2 and dominant_bassline(history[-2]) == kind:
            cost += BASS_REPEAT_COST[1]
    if prev.get("artist") and prev.get("artist") == cur.get("artist"):
        cost += ARTIST_REPEAT_COST
    return cost


def energy_target_fn(pool: List[Dict], profile: Profile):
    """Curve scaled into the pool's 10th-90th percentile energy range."""
    energies = sorted(t["energy"] for t in pool)
    lo = energies[int(0.10 * (len(energies) - 1))]
    hi = energies[int(0.90 * (len(energies) - 1))]
    return lambda frac: lo + (hi - lo) * energy_curve(frac, profile)


def build_set(pool: List[Dict], minutes: float = 120.0, vocal_weight: Optional[float] = None,
              beam_width: int = 40, max_tracks: int = 30,
              profile: Profile = APERITIVO) -> List[Dict]:
    """Beam search for the lowest-cost ordering that fills `minutes`."""
    if not pool:
        return []
    if vocal_weight is None:
        vocal_weight = profile.vocal_weight
    target_s = minutes * 60.0
    target = energy_target_fn(pool, profile)

    beams = [(0.0, [], 0.0)]            # (cost, sequence, elapsed seconds)
    finished = []
    for _ in range(max_tracks):
        nxt = []
        for cost, seq, elapsed in beams:
            used = {t["id"] for t in seq}
            for cand in pool:
                if cand["id"] in used:
                    continue
                frac = min(1.0, elapsed / target_s)
                c = cost + profile.energy_weight * (cand["energy"] - target(frac)) ** 2
                c -= vocal_weight * cand["vocal"] + profile.mood_weight * cand["mood"]
                if seq:
                    c += transition_cost(seq[-1], cand, seq, profile)
                new_elapsed = elapsed + cand["duration"] - (overlap_seconds(cand["bpm"]) if seq else 0.0)
                entry = (c, seq + [cand], new_elapsed)
                (finished if new_elapsed >= target_s else nxt).append(entry)
        if not nxt:
            break
        nxt.sort(key=lambda e: e[0] / len(e[1]))
        beams = nxt[:beam_width]
    candidates = finished or beams
    best = min(candidates, key=lambda e: e[0] / len(e[1]) + 0.01 * abs(e[2] - target_s) / 60.0)
    return best[1]


def layout(seq: List[Dict]) -> List[Dict]:
    """Per-slot start/end times (seconds) and transition notes."""
    out, t = [], 0.0
    for i, tr in enumerate(seq):
        start = t if i == 0 else t - overlap_seconds(tr["bpm"])
        end = start + tr["duration"]
        row = dict(tr, position=i + 1, start=start, end=end, bassline=dominant_bassline(tr))
        if i:
            prev = seq[i - 1]
            row["key_move"] = f"{prev['camelot_key']} -> {tr['camelot_key']} (cost {key_cost(prev['camelot_key'], tr['camelot_key']):.2f})"
            row["bpm_step"] = tr["bpm"] - prev["bpm"]
            row["blend_s"] = overlap_seconds(tr["bpm"])
        out.append(row)
        t = end
    return out


def _clock(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def export_set_xlsx(rows: List[Dict], out_path: str, minutes: float,
                    profile: Profile = APERITIVO) -> None:
    import os
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = Workbook()
    ws = wb.active
    ws.title = profile.title or profile.name
    headers = ["#", "Start", "End", "Track ID", "Artist", "Title", "BPM", "Key", "Energy",
               "Target", "Vocal", "Bassline", "Key move", "BPM step", "Blend (s)"]
    ws.append(headers)
    for c in ws[1]:
        c.font = Font(name="Arial", bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1F3864")
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    total = rows[-1]["end"] if rows else 0.0
    for r in rows:
        ws.append([
            r["position"], _clock(r["start"]), _clock(r["end"]), r["id"], r.get("artist") or "",
            r.get("title") or r.get("file_name"), round(r["bpm"], 1), r["camelot_key"],
            round(r["energy"], 2), round(r.get("target", 0.0), 2), round(r["vocal"], 2),
            r["bassline"], r.get("key_move", ""),
            round(r["bpm_step"], 1) if "bpm_step" in r else None,
            round(r["blend_s"]) if "blend_s" in r else None,
        ])
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.font = Font(name="Arial", size=10)
    for i, w in enumerate([4, 9, 9, 9, 26, 46, 7, 7, 8, 8, 8, 12, 24, 9, 10], 1):
        ws.column_dimensions[chr(64 + i)].width = w
    ws.freeze_panes = "A2"

    notes = wb.create_sheet("Notes")
    lines = [
        f"{profile.title}: {len(rows)} tracks, {_clock(total)} total (target {minutes:.0f} min).",
        f"Pool: rms_mean {profile.min_rms or 0:g} to {profile.max_rms or 'max'}, "
        f"BPM {profile.bpm_min:g}-{profile.bpm_max:g}, >= {profile.min_minutes:g} min, duplicates removed.",
        "Energy = mean z-score of rms, onset strength, kick rate and sub presence, scaled 0-1 within the pool.",
        "Target = the two-soft-peak curve scaled into the pool's 10th-90th percentile energy range.",
        ("Vocal tracks are favoured through a cost bonus" if profile.vocal_weight > 0
         else "Vocal tracks are penalised" if profile.vocal_weight < 0
         else "Vocals are ignored") + "; key moves follow the Camelot wheel.",
        "Mood (dark/hypnotic/industrial) is rewarded." if profile.mood_weight else "Mood is not used.",
        f"Each transition overlaps {MIX_BEATS} beats; times account for the overlap.",
        "Bassline = the highest of the five bassline scores for that track.",
    ]
    for i, text in enumerate(lines, 1):
        notes.cell(row=i, column=1, value=text).font = Font(name="Arial", size=10)
    notes.column_dimensions["A"].width = 110
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    wb.save(out_path)


def make_set(db_path: str, profile: Profile = APERITIVO, minutes: float = 120.0,
             vocal_weight: Optional[float] = None, beam_width: int = 40) -> List[Dict]:
    """Convenience: pool -> sequence -> layout rows with their energy target."""
    pool = load_pool(db_path, profile)
    seq = build_set(pool, minutes=minutes, vocal_weight=vocal_weight,
                    beam_width=beam_width, profile=profile)
    rows = layout(seq)
    if rows:
        target = energy_target_fn(pool, profile)
        total = minutes * 60.0
        for r in rows:
            r["target"] = target(min(1.0, r["start"] / total))
    return rows
