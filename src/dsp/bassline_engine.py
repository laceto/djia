"""Bassline-type scores — what the bass *line* does, not just how much low end there is.

`sub_engine` answers "what kind of sub?" (presence, fundamental, rumble, pump). This
module scores five bassline archetypes a DJ reasons about, each 0-1 and **multi-label**
(a track can be rolling *and* deep-sub):

* ``bassline_rolling``  — continuous 16th-note motion between the kicks ("chugger").
* ``bassline_deep_sub`` — clean, low, tonal sub that dominates the 75-250 Hz bass body.
* ``bassline_offbeat``  — quiet right after the kick, with the energy landing late in the
  beat (the "and" and the pick-up into the next kick).
* ``bassline_staccato`` — several short notes separated by real silence.
* ``bassline_drone``    — a held low end that is barely articulated.

Everything is derived from the **mix** (no stem separation), so kick and bass overlap
below ~150 Hz. Two things keep that tolerable: the rhythm band starts at 70 Hz, above
most kick fundamentals (a 50 Hz floor let the kick's sustained tone fill the post-kick
slot and erased every offbeat pattern), and the rhythm measures look only at the three
16th slots *after* the kick slot, where the kick body has decayed.

An ``acid`` (303) score is deliberately absent: mid-band descriptors of a resonant,
swept lead (peak share, flatness, centroid sweep, peak prominence) were tested against
known acid tracks and did not separate them from the rest of the library (AUC 0.38-0.51,
i.e. chance). It needs stems or a model trained on labelled examples. The scores are heuristic and the anchors below are first-pass —
re-calibrate them against labelled tracks rather than treating them as physical
constants (see `tests/test_dsp.py::TestBasslineEngine` for the synthetic extremes).

Best-effort like the other engines: a missing/silent/corrupt signal yields ``None`` for
every score, never an exception.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import librosa
from scipy.signal import butter, hilbert, sosfiltfilt

from . import sub_engine

logger = logging.getLogger(__name__)

BASSLINE_KEYS = (
    "bassline_rolling",
    "bassline_deep_sub",
    "bassline_offbeat",
    "bassline_staccato",
    "bassline_drone",
)

BASS_SR = 2000                  # decimated rate for the rhythm analysis
BASS_BAND = (70.0, 250.0)       # rhythm band: above most kick fundamentals, below the mids
BODY_BAND = (75.0, 250.0)       # "bass body" for the sub-vs-body balance
_ENV_SMOOTH_MS = 10.0

BINS_PER_BEAT = 16              # attack-detection resolution (64th notes)
SLOTS = 4                       # 16th-note slots per beat; slot 0 holds the kick
BINS_PER_SLOT = BINS_PER_BEAT // SLOTS
_PRESENT_FRAC = 0.20            # a beat "has bass" above this fraction of the loud-level ref
_REF_PERCENTILE = 90            # loud-level reference = this percentile of bin envelopes
_ATTACK_RISE = 1.35             # an attack: bin level >= this x the recent minimum...
_ATTACK_MIN_STEP = 0.30         # ...and at least this fraction of the loud reference higher
_SILENT_FRAC = 0.30             # gap bin counts as silence below this x the beat's gap peak
_KICK_SNAP = 8                 # kick phase is snapped to multiples of 1/_KICK_SNAP beat
MIN_BEATS = 16                  # fewer usable beats than this -> not enough to score rhythm

# Anchors (value scoring 0.0, value scoring 1.0).
_DENSITY_ANCHORS = (0.40, 0.90)       # rolling: share of gap slots carrying an attack
_GAP_LEVEL_ANCHORS = (0.30, 0.70)     # rolling: gap level relative to the loud reference
_LATE_ENERGY_ANCHORS = (0.60, 0.85)   # offbeat: share of between-kick bass energy in the last two slots
_POST_KICK_DIP_ANCHORS = (0.30, 0.80)  # offbeat: how far the bass drops right after the kick
_SILENCE_ANCHORS = (0.45, 0.80)       # staccato: fraction of the gap that is silent
_NOTES_ANCHORS = (0.20, 0.60)         # staccato: needs actual notes, not just silence
_SUSTAIN_ANCHORS = (0.40, 0.80)       # drone: bass still up between kicks
_ARTIC_ANCHORS = (0.45, 0.10)         # drone: few attacks (inverted: fewer -> higher)
_SUB_BALANCE_DB_ANCHORS = (0.0, 9.0)   # deep_sub: sub minus body power, dB (library median ~3)
_PEAK_SHARE_ANCHORS = (0.20, 0.55)    # deep_sub: tonal sub (share of band at the peak)
_SUB_TONAL_WEIGHT = 0.4
_OFFBEAT_SUB_BLEND = 0.3              # weight of sub_engine's offbeat label in offbeat



def _empty_profile() -> Dict[str, Optional[float]]:
    return {k: None for k in BASSLINE_KEYS}


def _lerp01(value: float, zero_at: float, one_at: float) -> float:
    """Map `value` onto 0-1 between the anchors, clamped. Handles inverted anchors."""
    if one_at == zero_at:
        return 0.0
    return float(np.clip((value - zero_at) / (one_at - zero_at), 0.0, 1.0))


def _bass_envelope(y, sr: int, band) -> tuple:
    """Band-pass envelope at BASS_SR. Returns (envelope, band signal)."""
    y_ds = librosa.resample(y, orig_sr=sr, target_sr=BASS_SR, res_type="soxr_hq")
    sos = butter(4, list(band), btype="band", fs=BASS_SR, output="sos")
    x = sosfiltfilt(sos, y_ds)
    env = np.abs(hilbert(x))
    w = max(1, int(_ENV_SMOOTH_MS / 1000.0 * BASS_SR))
    return np.convolve(env, np.ones(w) / w, mode="same"), x


def _beat_matrix(env: np.ndarray, anchors: np.ndarray, kick_offset: float) -> Optional[np.ndarray]:
    """Envelope sampled on a (beats x BINS_PER_BEAT) grid, bin 0 on the kick.

    Each beat is interpolated between its own anchors, so tempo drift stretches the grid
    instead of smearing it. Beats whose span is far from the median (tracker glitches)
    are skipped.
    """
    spans = np.diff(anchors)
    spans = spans[np.isfinite(spans) & (spans > 0)]
    if spans.size < MIN_BEATS:
        return None
    median_span = float(np.median(spans))
    times = np.arange(env.size) / BASS_SR
    offsets = (np.arange(BINS_PER_BEAT) + 0.5) / BINS_PER_BEAT
    rows = []
    for start, end in zip(anchors[:-1], anchors[1:]):
        span = end - start
        if not (0.5 * median_span < span < 1.5 * median_span):
            continue
        t = start + (kick_offset + offsets) * span
        if t[-1] >= times[-1]:
            break
        rows.append(np.interp(t, times, env))
    if len(rows) < MIN_BEATS:
        return None
    return np.asarray(rows)


def _rhythm_measures(y, sr: int, anchors: np.ndarray) -> Optional[Dict[str, float]]:
    """Kick-aligned beat grid of the bass band -> raw rhythm measures (see `_rhythm_scores`)."""
    env, _ = _bass_envelope(y, sr, BASS_BAND)
    kick_phase = sub_engine._kick_phase_bin(y, sr, anchors) / sub_engine.PHASE_BINS
    # The onset detector reads a few frames *after* the transient, which would shift every
    # slot boundary by about a bin. Snap to the nearest 1/8 beat: that keeps the real
    # ambiguity it exists to resolve (a beat grid sitting on the offbeat) and drops the lag.
    kick_offset = (round(kick_phase * _KICK_SNAP) / _KICK_SNAP) % 1.0
    grid = _beat_matrix(env, anchors, kick_offset)
    return None if grid is None else _rhythm_scores(grid)


def _rhythm_scores(grid: np.ndarray) -> Optional[Dict[str, float]]:
    """rolling / staccato / offbeat / drone from the kick-aligned beat grid."""
    ref = float(np.percentile(grid, _REF_PERCENTILE))
    if ref <= 0:
        return None

    n_beats = grid.shape[0]
    slot_means = grid.reshape(n_beats, SLOTS, BINS_PER_SLOT).mean(axis=2)  # (beats, 4)
    gap = grid[:, BINS_PER_SLOT:]                                           # slots 1-3

    # --- attacks: a bin that jumps well above the previous two, in the gap slots only.
    flat = grid.reshape(-1)
    prev = np.minimum(np.roll(flat, 1), np.roll(flat, 2))
    attack_flat = (
        (flat >= _ATTACK_RISE * prev)
        & (flat - prev >= _ATTACK_MIN_STEP * ref)
        & (flat >= _PRESENT_FRAC * ref)
    )
    attack = attack_flat.reshape(n_beats, BINS_PER_BEAT)
    attack[:, -1] = False  # the kick's own rising edge pre-rings into the last pre-kick bin
    slot_has_attack = attack.reshape(n_beats, SLOTS, BINS_PER_SLOT).any(axis=2)
    density = float(slot_has_attack[:, 1:].mean())            # share of gap slots with a note

    gap_level = float(min(1.0, slot_means[:, 1:].mean() / ref))

    # --- silence inside the gap, only over beats that actually carry bass.
    gap_peak = gap.max(axis=1)
    present = gap_peak >= _PRESENT_FRAC * ref
    if present.sum() < MIN_BEATS // 2:
        # Essentially no bass between kicks: nothing to articulate.
        return {"rolling": 0.0, "staccato": 0.0, "offbeat": 0.0, "drone": 0.0}
    silence = float(
        (gap[present] < _SILENT_FRAC * gap_peak[present, None]).mean()
    )

    # Where the bass sits between the kicks. Averaged energy, not attack counts: the kick
    # tail beating against the sub throws false attacks into slot 1, but cannot fake a
    # sustained quiet slot 1. An offbeat line leaves slot 1 quiet and fills the rest of
    # the beat; a roll fills all three slots.
    gap_mean = slot_means[:, 1:].mean(axis=0)                           # mean level of slots 1, 2, 3
    late_energy_share = float(gap_mean[1:].sum() / gap_mean.sum()) if gap_mean.sum() > 0 else 0.0
    late_peak = float(gap_mean[1:].max())
    post_kick_dip = float(1.0 - gap_mean[0] / late_peak) if late_peak > 0 else 0.0

    return {
        "density": density,
        "gap_level": gap_level,
        "silence": silence,
        "late_energy_share": late_energy_share,
        "post_kick_dip": max(0.0, post_kick_dip),
    }


def _balance_and_tonality(y, sr: int) -> Optional[float]:
    """Sub (20-75 Hz) minus bass-body (75-250 Hz) power, in dB."""
    sub = sub_engine._sub_band_signal(y, sr)
    y_ds = librosa.resample(y, orig_sr=sr, target_sr=BASS_SR, res_type="soxr_hq")
    sos = butter(4, list(BODY_BAND), btype="band", fs=BASS_SR, output="sos")
    body = sosfiltfilt(sos, y_ds)
    p_sub, p_body = float(np.mean(sub ** 2)), float(np.mean(body ** 2))
    if p_sub <= 0 or p_body <= 0:
        return None
    return 10.0 * np.log10(p_sub / p_body)


def compute_bassline_profile(
    y,
    sr: int,
    bpm: Optional[float] = None,
    beat_times: Optional[Sequence[float]] = None,
    sub_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[float]]:
    """Score six bassline archetypes (0-1, multi-label) for one track.

    Args:
        y: mono (or multi-channel, mixed down here) audio samples.
        sr: sample rate in Hz.
        bpm: known tempo, used for the beat grid when `beat_times` is absent.
        beat_times: known beat positions (seconds) — preferred, they track tempo drift.
        sub_profile: optional `sub_engine.compute_sub_profile` output (or the matching
            DB columns). Used only to nudge ``offbeat`` (sub_character) and
            ``deep_sub`` (sub_peak_share); the scores are computed without it.

    Returns:
        Dict with keys ``BASSLINE_KEYS``, each 0-1 or ``None`` when it cannot be
        computed. The rhythm scores (rolling/staccato/offbeat/drone) are ``None`` when
        no usable beat grid exists; ``deep_sub`` does not need one.
    """
    profile = _empty_profile()

    if y is None or sr is None or sr <= 0:
        return profile
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = librosa.to_mono(y)
    if y.size == 0 or not np.any(np.isfinite(y)) or np.allclose(y, 0.0):
        return profile

    sub_profile = sub_profile or {}

    try:
        balance_db = _balance_and_tonality(y, sr)
        if balance_db is not None:
            score = _lerp01(balance_db, *_SUB_BALANCE_DB_ANCHORS)
            peak_share = sub_profile.get("sub_peak_share")
            if peak_share is not None:
                score = ((1.0 - _SUB_TONAL_WEIGHT) * score
                         + _SUB_TONAL_WEIGHT * _lerp01(peak_share, *_PEAK_SHARE_ANCHORS))
            profile["bassline_deep_sub"] = float(score)
    except Exception as e:  # noqa: BLE001 - best-effort, mirrors the DSP engines
        logger.warning(f"bassline deep_sub failed: {e}")

    try:
        anchors = sub_engine._beat_anchors(y, sr, bpm, beat_times)
        if anchors is not None:
            m = _rhythm_measures(y, sr, anchors)
            if m is not None:
                if "density" not in m:  # no bass between kicks
                    profile.update({
                        "bassline_rolling": 0.0, "bassline_staccato": 0.0,
                        "bassline_offbeat": 0.0, "bassline_drone": 0.0,
                    })
                else:
                    d, g, s = m["density"], m["gap_level"], m["silence"]
                    profile["bassline_rolling"] = float(
                        0.5 * _lerp01(d, *_DENSITY_ANCHORS) + 0.5 * _lerp01(g, *_GAP_LEVEL_ANCHORS)
                    )
                    profile["bassline_staccato"] = float(np.sqrt(
                        _lerp01(s, *_SILENCE_ANCHORS) * _lerp01(d, *_NOTES_ANCHORS)
                    ))
                    # Gated by the post-kick dip: a roll attacks late too, but never goes quiet.
                    offbeat = float(np.sqrt(
                        _lerp01(m["post_kick_dip"], *_POST_KICK_DIP_ANCHORS)
                        * _lerp01(m["late_energy_share"], *_LATE_ENERGY_ANCHORS)
                    ))
                    label = sub_profile.get("sub_character")
                    if label in ("offbeat", "syncopated"):
                        flag = 1.0 if label == "offbeat" else 0.3
                        offbeat = (1.0 - _OFFBEAT_SUB_BLEND) * offbeat + _OFFBEAT_SUB_BLEND * flag
                    elif label is not None:
                        offbeat = (1.0 - _OFFBEAT_SUB_BLEND) * offbeat
                    profile["bassline_offbeat"] = float(offbeat)
                    profile["bassline_drone"] = float(np.sqrt(
                        _lerp01(g, *_SUSTAIN_ANCHORS) * _lerp01(d, *_ARTIC_ANCHORS)
                    ))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"bassline rhythm scores failed: {e}")

    return profile


def bassline_tags(profile: Dict[str, Any], threshold: float = 0.5) -> List[str]:
    """Tags for every archetype scoring at or above `threshold` (strongest first)."""
    hits = [
        (profile.get(k), k.replace("bassline_", "").replace("_", "-") + "-bass")
        for k in BASSLINE_KEYS
        if profile.get(k) is not None and profile[k] >= threshold
    ]
    return [tag for _, tag in sorted(hits, reverse=True)]
