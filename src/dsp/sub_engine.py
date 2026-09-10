"""Sub-bass character features — presence, fundamental, rumble, and sidechain pump.

`stem_profile.sub_ratio` answers "how much low end is there?". This module answers
"what *kind* of low end is it?" — the questions a DJ actually asks of a track's
bottom octave:

* **Sub present?**  ``sub_presence`` — mean-square share of the 20-75 Hz band.
* **Sub fundamental?**  ``sub_f0_hz`` / ``sub_note`` — the pitch the sub sits on,
  resolved finely enough to name the note (compare it with `mood.camelot_key`).
* **Rumble?**  ``rumble_score`` — is the low end a noisy, reverb-smeared wash
  (rumble kick, 909 tail) rather than a played note?
* **Sub-pump?**  ``pump_depth`` with ``pump_phase`` / ``sub_peak_phase`` — how hard
  the sub moves over the beat and *where* it peaks and bottoms out. Depth alone does
  not separate a sidechained sub from a decaying rumble tail or an offbeat bassline —
  all three modulate at the beat rate — but phase does: a sidechain bottoms out *at*
  the kick (phase ~0), a decay tail bottoms out just *before* the next one (phase
  ~0.9), and an offbeat bass peaks halfway between kicks (phase ~0.5).

Two things make this cheap and honest:

* All analysis runs on the band-limited signal **decimated to 1 kHz** (``SUB_SR``),
  so a 1024-point FFT gives ~1 Hz bins — a semitone at 41 Hz is 2.4 Hz wide. The
  pipeline's stored spectrogram (``spectrogram.compute_spectrogram``, n_fft 2048 at
  22.05 kHz) has 10.8 Hz bins: about four across the whole sub range, enough for
  presence and nothing else.
* The beat fold uses the **actual beat times** where they are known, not a uniform
  grid. Over a 6-minute track even slight tempo drift smears a uniform-grid fold flat
  (measured 0.04 vs 0.78 pump depth on the same track).

Everything here is best-effort: a missing/silent/corrupt signal yields ``None`` for
each feature, never an exception — same contract as ``stem_profile``.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import librosa
from scipy.signal import butter, hilbert, sosfiltfilt

logger = logging.getLogger(__name__)

# Analysis band (Hz). Wider at the top than stem_profile's SUB_BAND (20-60) so a sub
# rooted at C2 (65.4 Hz) still lands inside it.
SUB_LOW = 20.0
SUB_HIGH = 75.0

SUB_SR = 1000          # decimated rate for all sub-band analysis
_N_FFT = 1024          # 0.98 Hz/bin, 1.02 s window at SUB_SR
_HOP = 256             # 0.256 s
_ENV_SMOOTH_MS = 20.0  # envelope smoothing window
_ONSET_HOP = 512       # hop for the full-rate onset envelope (kick phase reference)

PHASE_BINS = 64                 # resolution of the beat-synchronous envelope profile
GAP_WINDOW = (0.35, 0.90)       # beat phase range that sits between kicks
_LOUD_FRAME_FRAC = 0.30         # frame is "sub-active" above this fraction of the median
# Half-width of the "peak region" when measuring how concentrated the band is. Wide
# enough to keep the amplitude-modulation sidebands a sidechained sub throws off
# (+/- the beat rate, ~2 Hz at 125 BPM) with the note they belong to.
_PEAK_HALF_WIDTH_HZ = 3.0

# Below this share of total power the band is empty and every other feature is noise.
SUB_PRESENCE_MIN = 0.01
# sub_character thresholds
RUMBLE_MIN = 0.50
PUMP_DEPTH_MIN = 0.35
PUMP_PHASE_MAX = 0.25           # a sidechain bottoms out within the first quarter-beat
OFFBEAT_PHASE = (0.375, 0.625)  # an offbeat bassline peaks around the half-beat

# rumble_score component anchors: (value scoring 0.0, value scoring 1.0). Calibrated
# between synthetic extremes (a pure sine sub and band-limited noise with a beat-synced
# tail, see tests/test_dsp.py::TestSubEngine) and real techno; re-calibrate against a
# real library rather than treating these as physical constants.
_PEAK_SHARE_ANCHORS = (0.55, 0.20)   # share of band power within +/-3 Hz of the peak
_CREST_ANCHORS = (18.0, 6.0)         # peak-to-median band spectrum, dB
_JITTER_ANCHORS = (8.0, 50.0)        # cents of frame-to-frame f0 wander
_RUMBLE_WEIGHTS = (0.30, 0.25, 0.45)  # peak share, crest, jitter

SUB_PROFILE_KEYS = (
    "sub_presence",
    "sub_f0_hz",
    "sub_note",
    "sub_f0_jitter_cents",
    "sub_flatness",
    "sub_peak_share",
    "sub_peak_crest_db",
    "rumble_score",
    "pump_depth",
    "pump_phase",
    "sub_peak_phase",
    "sub_gap_ratio",
    "sub_character",
)


def _empty_profile() -> Dict[str, Any]:
    return {k: None for k in SUB_PROFILE_KEYS}


def _lerp01(value: float, zero_at: float, one_at: float) -> float:
    """Map `value` onto 0-1 between the anchors, clamped. Handles inverted anchors."""
    if one_at == zero_at:
        return 0.0
    return float(np.clip((value - zero_at) / (one_at - zero_at), 0.0, 1.0))


def _sub_band_signal(y: np.ndarray, sr: int) -> np.ndarray:
    """Decimate to SUB_SR, then band-pass to [SUB_LOW, SUB_HIGH].

    Resampling first is deliberate: it anti-alias-filters everything above 500 Hz, and
    the band-pass is far better conditioned at fs=1000 (0.04-0.15 normalized) than at
    22.05 kHz (0.002-0.007), where a low-order IIR loses numerical stability.
    """
    y_sub = librosa.resample(y, orig_sr=sr, target_sr=SUB_SR, res_type="soxr_hq")
    sos = butter(4, [SUB_LOW, SUB_HIGH], btype="band", fs=SUB_SR, output="sos")
    return sosfiltfilt(sos, y_sub)


def _envelope(x: np.ndarray) -> np.ndarray:
    """Analytic-signal magnitude, moving-average smoothed (at SUB_SR)."""
    env = np.abs(hilbert(x))
    w = max(1, int(_ENV_SMOOTH_MS / 1000.0 * SUB_SR))
    return np.convolve(env, np.ones(w) / w, mode="same")


def _beat_anchors(y, sr: int, bpm, beat_times: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    """Beat onset times to fold on — the tracked beats when known, else a BPM grid."""
    if beat_times is not None and len(beat_times) >= 8:
        anchors = np.asarray(beat_times, dtype=float)
        if np.all(np.isfinite(anchors)) and np.all(np.diff(anchors) > 0):
            return anchors

    period = None
    if bpm:
        candidate = float(np.atleast_1d(bpm)[0])
        if 40.0 < candidate < 250.0:
            period = 60.0 / candidate
    if period is None:
        try:
            est = float(np.atleast_1d(librosa.feature.rhythm.tempo(y=y, sr=sr))[0])
            period = 60.0 / est if 40.0 < est < 250.0 else None
        except Exception:  # noqa: BLE001 - tempo estimation is a fallback, not a contract
            period = None
    if period is None:
        return None
    anchors = np.arange(0.0, len(y) / sr, period)
    return anchors if anchors.size >= 8 else None


def _fold(values: np.ndarray, times: np.ndarray, anchors: np.ndarray) -> Optional[np.ndarray]:
    """Average `values` over every beat cycle onto a PHASE_BINS phase grid.

    Each cycle is interpolated between consecutive anchors, so tempo drift stretches
    the grid with the music instead of smearing the average.
    """
    spans = np.diff(anchors)
    spans = spans[np.isfinite(spans) & (spans > 0)]
    if spans.size < 4:
        return None
    median_span = float(np.median(spans))
    grid = np.linspace(0.0, 1.0, PHASE_BINS, endpoint=False)

    cycles = []
    for start, end in zip(anchors[:-1], anchors[1:]):
        span = end - start
        if not (0.5 * median_span < span < 1.5 * median_span):
            continue  # beat-tracker glitch (dropped or doubled beat)
        cycles.append(np.interp(start + grid * span, times, values))
    if len(cycles) < 4:
        return None
    return np.mean(np.asarray(cycles), axis=0)


def _kick_phase_bin(y, sr: int, anchors: np.ndarray) -> int:
    """Phase bin carrying the transient peak, so phase 0 can be rolled onto the kick.

    Beat *phase* is what separates a sidechained sub from a decaying rumble tail, so it
    is anchored to the actual transient rather than to a beat grid whose origin may sit
    on the offbeat.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_ONSET_HOP)
    if onset_env.size == 0 or not np.any(onset_env > 0):
        return 0
    times = librosa.frames_to_time(np.arange(onset_env.size), sr=sr, hop_length=_ONSET_HOP)
    profile = _fold(onset_env, times, anchors)
    return 0 if profile is None else int(np.argmax(profile))


def _spectral_stats(x: np.ndarray) -> Dict[str, Optional[float]]:
    """Per-frame sub-band fundamental, flatness, peak share, and crest (medians)."""
    stats: Dict[str, Optional[float]] = {
        "sub_f0_hz": None,
        "sub_f0_jitter_cents": None,
        "sub_flatness": None,
        "sub_peak_share": None,
        "sub_peak_crest_db": None,
    }
    spec = np.abs(librosa.stft(x, n_fft=_N_FFT, hop_length=_HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=SUB_SR, n_fft=_N_FFT)
    band = (freqs >= SUB_LOW) & (freqs <= SUB_HIGH)
    if not np.any(band) or spec.shape[1] == 0:
        return stats

    spec_band = spec[band]
    freqs_band = freqs[band]
    bin_hz = float(freqs[1] - freqs[0])
    half_width = max(1, int(round(_PEAK_HALF_WIDTH_HZ / bin_hz)))

    frame_energy = spec_band.sum(axis=0)
    active = frame_energy > 0
    if not np.any(active):
        return stats
    loud = frame_energy > _LOUD_FRAME_FRAC * np.median(frame_energy[active])
    idx = np.flatnonzero(loud)
    if idx.size == 0:
        return stats

    f0s, flatness, peak_share, crest = [], [], [], []
    for i in idx:
        col = spec_band[:, i]
        total = float(col.sum())
        if total <= 0:
            continue
        k = int(np.argmax(col))
        offset = 0.0
        if 0 < k < col.size - 1:
            # Parabolic interpolation on the log magnitudes: recovers the true peak to
            # a fraction of a bin, which is what makes a ~1 cent f0 possible at all.
            a, b, c = np.log(col[k - 1:k + 2] + 1e-20)
            denom = a - 2 * b + c
            if denom != 0:
                offset = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
        f0s.append(float(freqs_band[k]) + offset * bin_hz)

        p = col / total
        flatness.append(float(np.exp(np.mean(np.log(p + 1e-20))) / (np.mean(p) + 1e-20)))
        peak_share.append(
            float(col[max(0, k - half_width):k + half_width + 1].sum() / total)
        )
        median = float(np.median(col))
        crest.append(10.0 * np.log10(float(col[k]) / median) if median > 0 else 60.0)

    if not f0s:
        return stats
    f0_arr = np.asarray(f0s)
    f0 = float(np.median(f0_arr))
    stats["sub_f0_hz"] = f0
    if f0 > 0:
        cents = 1200.0 * np.log2(np.maximum(f0_arr, 1e-9) / f0)
        stats["sub_f0_jitter_cents"] = float(np.median(np.abs(cents)))
    stats["sub_flatness"] = float(np.median(flatness))
    stats["sub_peak_share"] = float(np.median(peak_share))
    stats["sub_peak_crest_db"] = float(np.median(crest))
    return stats


def _rumble_score(stats: Dict[str, Optional[float]]) -> Optional[float]:
    """Blend the three tonality cues into one 0-1 "is this a noisy wash" score.

    Spectral flatness is *reported* but deliberately not scored: a mix whose sub band
    holds a bass note plus the kick fundamental reads as flat as noise, so it cannot
    tell a two-note low end from a rumble.
    """
    share, crest, jitter = (
        stats.get("sub_peak_share"), stats.get("sub_peak_crest_db"),
        stats.get("sub_f0_jitter_cents"),
    )
    if share is None or crest is None or jitter is None:
        return None
    w_share, w_crest, w_jitter = _RUMBLE_WEIGHTS
    return float(
        w_share * _lerp01(share, *_PEAK_SHARE_ANCHORS)
        + w_crest * _lerp01(crest, *_CREST_ANCHORS)
        + w_jitter * _lerp01(jitter, *_JITTER_ANCHORS)
    )


def _classify(presence: float, profile: Dict[str, Any]) -> str:
    """Coarse label: none / rumble / pumped / offbeat / sustained."""
    if presence < SUB_PRESENCE_MIN:
        return "none"
    rumble = profile.get("rumble_score")
    if rumble is not None and rumble >= RUMBLE_MIN:
        return "rumble"
    depth = profile.get("pump_depth")
    if depth is not None and depth >= PUMP_DEPTH_MIN:
        phase, peak_phase = profile.get("pump_phase"), profile.get("sub_peak_phase")
        if phase is not None and phase <= PUMP_PHASE_MAX:
            return "pumped"
        if peak_phase is not None and OFFBEAT_PHASE[0] <= peak_phase <= OFFBEAT_PHASE[1]:
            return "offbeat"
    return "sustained"


def compute_sub_profile(
    y,
    sr: int,
    bpm: Optional[float] = None,
    beat_times: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Compute sub-bass character features for one track.

    Args:
        y: mono (or multi-channel, mixed down here) audio samples.
        sr: sample rate in Hz.
        bpm: known tempo, used for the beat fold when `beat_times` is absent.
        beat_times: known beat positions (seconds) — preferred, they track tempo drift.
            Phase 0 is re-derived from the transient peak either way, so an
            offbeat-locked grid does not skew `pump_phase`.

    Returns:
        Dict with keys ``SUB_PROFILE_KEYS``:
            - ``sub_presence``: mean-square share of 20-75 Hz in the full mix (0-1).
            - ``sub_f0_hz`` / ``sub_note``: sub fundamental in Hz and as a note name.
            - ``sub_f0_jitter_cents``: median frame-to-frame wander of that pitch;
              ~1 cent for a played sub line, tens of cents for noise.
            - ``sub_flatness``: spectral flatness *within* the band (0 tonal, 1 noise).
            - ``sub_peak_share``: share of band power within +/-3 Hz of the peak.
            - ``sub_peak_crest_db``: peak-to-median of the band spectrum, in dB.
            - ``rumble_score``: 0-1 blend of the three cues above; high = noisy wash.
            - ``pump_depth``: (max-min)/max of the beat-folded sub envelope.
            - ``pump_phase`` / ``sub_peak_phase``: beat phase (0-1, 0 = kick) of that
              minimum and maximum.
            - ``sub_gap_ratio``: mean sub level between kicks, relative to its peak.
            - ``sub_character``: "none" | "rumble" | "pumped" | "offbeat" | "sustained".
        Any feature that cannot be computed is ``None`` rather than raising.
    """
    profile = _empty_profile()

    if y is None or sr is None or sr <= 0:
        return profile

    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = librosa.to_mono(y)
    if y.size == 0 or not np.any(np.isfinite(y)) or np.allclose(y, 0.0):
        return profile

    try:
        total_power = float(np.mean(y ** 2))
        if total_power <= 0:
            return profile

        sub = _sub_band_signal(y, sr)
        if sub.size == 0:
            return profile

        presence = float(np.mean(sub ** 2) / total_power)
        profile["sub_presence"] = presence

        if presence < SUB_PRESENCE_MIN:
            # Nothing down there: leave the character metrics None rather than
            # reporting a fundamental fitted to filter leakage.
            profile["sub_character"] = "none"
            return profile

        profile.update(_spectral_stats(sub))
        if profile["sub_f0_hz"]:
            profile["sub_note"] = librosa.hz_to_note(profile["sub_f0_hz"])
        profile["rumble_score"] = _rumble_score(profile)

        anchors = _beat_anchors(y, sr, bpm, beat_times)
        if anchors is not None:
            env = _envelope(sub)
            folded = _fold(env, np.arange(env.size) / SUB_SR, anchors)
            if folded is not None and folded.max() > 0:
                # Roll so phase 0 sits on the kick transient, not on the grid origin.
                folded = np.roll(folded, -_kick_phase_bin(y, sr, anchors))
                peak = float(folded.max())
                profile["pump_depth"] = float((peak - folded.min()) / peak)
                profile["pump_phase"] = float(np.argmin(folded) / PHASE_BINS)
                profile["sub_peak_phase"] = float(np.argmax(folded) / PHASE_BINS)
                lo = int(GAP_WINDOW[0] * PHASE_BINS)
                hi = int(GAP_WINDOW[1] * PHASE_BINS)
                profile["sub_gap_ratio"] = float(folded[lo:hi].mean() / peak)

        profile["sub_character"] = _classify(presence, profile)
    except Exception as e:  # noqa: BLE001 - best-effort, mirrors the DSP engines
        logger.warning(f"sub profile computation failed: {e}")
        return _empty_profile()

    return profile


def sub_tags(profile: Dict[str, Any]) -> List[str]:
    """Semantic tags for a computed profile, for curation/search ("rumble", "sub-pump")."""
    character = profile.get("sub_character")
    if character is None:
        return []
    if character == "none":
        return ["no-sub"]

    tags = ["rumble" if character == "rumble" else "sub-bass"]
    if character == "pumped":
        tags.append("sub-pump")
    elif character == "offbeat":
        tags.append("offbeat-bass")
    if profile.get("sub_note"):
        tags.append(f"sub-{profile['sub_note']}")
    return tags
