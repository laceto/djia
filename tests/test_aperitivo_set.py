"""Tests for the aperitivo set builder (pure functions, no DB)."""

from src.ai.aperitivo_set import (
    build_set, energy_curve, key_cost, layout, overlap_seconds, parse_camelot, score_pool,
)


def _track(i, key="5A", bpm=123.0, rms=0.1, vocal=0.05, dur=420.0, **bass):
    t = {"id": i, "artist": f"A{i}", "title": f"T{i}", "file_name": f"f{i}", "duration": dur,
         "bpm": bpm, "camelot_key": key, "rms_mean": rms, "onset_strength_mean": rms * 10,
         "kick_rate": rms * 30, "sub_presence": rms * 4, "vocal_presence": vocal,
         "bassline_rolling": 0.0, "bassline_deep_sub": 0.0, "bassline_offbeat": 0.0,
         "bassline_staccato": 0.0, "bassline_drone": 0.0}
    t.update(bass)
    return t


def test_parse_camelot():
    assert parse_camelot("8a") == (8, "A")
    assert parse_camelot("13A") is None
    assert parse_camelot(None) is None


def test_key_cost_ordering():
    assert key_cost("5A", "5A") == 0.0
    assert key_cost("5A", "6A") == key_cost("5A", "4A") == key_cost("5A", "5B") == 0.15
    assert key_cost("12A", "1A") == 0.15          # wheel wraps
    assert key_cost("5A", "7A") < key_cost("5A", "9A")
    assert key_cost("5A", "6B") == 1.0


def test_energy_curve_has_two_peaks_and_soft_ends():
    xs = [i / 100 for i in range(101)]
    ys = [energy_curve(x) for x in xs]
    peaks = [i for i in range(1, 100) if ys[i] > ys[i - 1] and ys[i] >= ys[i + 1]]
    assert len(peaks) == 2
    assert ys[0] < 0.1 and ys[-1] < ys[peaks[1]]
    assert ys[peaks[1]] > ys[peaks[0]]            # second peak is higher
    valley = min(ys[peaks[0]:peaks[1]])
    assert valley < 0.7 * ys[peaks[0]]


def test_build_set_respects_length_uniqueness_and_bpm_steps():
    keys = ["5A", "6A", "4A", "5B", "7A", "3A"]
    pool = score_pool([
        _track(i, key=keys[i % 6], bpm=122 + (i % 3) * 0.5, rms=0.05 + 0.1 * ((i * 7) % 20) / 20)
        for i in range(40)
    ])
    seq = build_set(pool, minutes=120, beam_width=20)
    rows = layout(seq)
    assert len({t["id"] for t in seq}) == len(seq)
    assert rows[-1]["end"] >= 120 * 60 - 1
    assert rows[-1]["end"] < 120 * 60 + 12 * 60   # no more than ~one track over
    for r in rows[1:]:
        assert abs(r["bpm_step"]) <= 4.0


def test_vocal_weight_prefers_vocal_tracks():
    pool = score_pool([_track(i, rms=0.1, vocal=0.9 if i >= 25 else 0.0) for i in range(30)])
    with_v = build_set(pool, minutes=60, vocal_weight=2.0, beam_width=10)
    without = build_set(pool, minutes=60, vocal_weight=0.0, beam_width=10)
    share = lambda s: sum(t["vocal"] > 0.5 for t in s) / len(s)
    assert share(with_v) > share(without)


def test_layout_applies_overlap():
    seq = score_pool([_track(1), _track(2)])
    rows = layout(seq)
    assert rows[1]["start"] == rows[0]["end"] - overlap_seconds(123.0)


def test_latenight_profile_differs_from_aperitivo():
    from src.ai.aperitivo_set import APERITIVO, LATENIGHT, PROFILES
    assert set(PROFILES) == {"aperitivo", "latenight"}
    assert LATENIGHT.min_rms and LATENIGHT.min_rms > (APERITIVO.max_rms or 0)
    assert LATENIGHT.vocal_weight < 0 < APERITIVO.vocal_weight
    # late night never drops as low as the aperitivo welcome, and peaks stay hotter
    assert energy_curve(0.0, LATENIGHT) > energy_curve(0.0, APERITIVO)
    mid = [energy_curve(x / 100, LATENIGHT) for x in range(35, 70)]
    assert min(mid) > min(energy_curve(x / 100, APERITIVO) for x in range(35, 70))


def test_latenight_penalises_vocals_and_rewards_mood():
    from src.ai.aperitivo_set import LATENIGHT
    pool = score_pool([
        dict(_track(i, rms=0.25, vocal=0.9 if i < 10 else 0.0),
             dark=0.9 if i >= 20 else 0.1, hypnotic=0.5, industrial=0.5)
        for i in range(30)
    ], LATENIGHT)
    assert all(t["mood"] > 0.9 for t in pool if t["id"] >= 20)
    seq = build_set(pool, minutes=60, beam_width=10, profile=LATENIGHT)
    assert sum(t["vocal"] > 0.5 for t in seq) == 0
    assert sum(t["id"] >= 20 for t in seq) >= len(seq) // 2


def test_bpm_step_limit_comes_from_profile():
    from src.ai.aperitivo_set import LATENIGHT, transition_cost
    a, b = _track(1, bpm=123.0), _track(2, bpm=129.2)
    assert transition_cost(a, b, [a]) > transition_cost(a, b, [a], LATENIGHT) + 1.5
