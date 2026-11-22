"""The Level 4 shot-position check (docs/research/P8_shots2526.md): pure parts on synthetic clocks, scores and tracks."""

import math

import pytest

from tools import p8_shots2526 as S


def test_court_frame_puts_both_baskets_and_the_lateral_sign_right():
    assert S.court_xy(0, 0, "left") == (5.25, 25.0)
    assert S.court_xy(0, 0, "right") == (88.75, 25.0)
    assert S.court_xy(-7, 35, "left") == pytest.approx((8.75, 25.7))       # mirror x, y of a logged layup: 8.72, 25.74
    assert S.court_xy(-225, 31, "right") == pytest.approx((85.65, 2.5))   # a corner three: 85.68, 2.45
    with pytest.raises(ValueError):
        S.court_xy(0, 0, None)


EVENTS = [(1, 700.0, "0-2"), (1, 650.0, "3-2"), (2, 600.0, "3-4")]       # (period, clock, "away-home")


def parsed(clock, period=None, ints=(3, 2)):
    return {"clocks": [clock] if clock is not None else [], "periods": [period] if period else [], "ints": list(ints)}


def test_a_frame_reads_a_moment_only_when_the_score_agrees():
    assert S.frame_moments(parsed(640.0, 1), EVENTS) == {(1, 640.0)}
    assert S.frame_moments(parsed(640.0, 1, ints=(9, 9)), EVENTS) == set()
    assert S.frame_moments(parsed(None, 1), EVENTS) == set()
    m = S.frame_moments(parsed(590.0, ints=(3, 4)), EVENTS)                # no period read: 1 to 4 are tried
    assert (2, 590.0) in m and (1, 590.0) not in m                         # 3-4 only after period 2's 10:00


def test_anchors_keep_single_moment_reads_and_the_estimate_assumes_a_running_clock():
    reads = {0.0: parsed(None), 20.0: parsed(640.0, 1), 40.0: parsed(625.0, 1)}
    an = S.anchors_of(reads, EVENTS)
    assert an == [(20.0, 1, 640.0), (40.0, 1, 625.0)]
    assert S.estimate(an, 1, 630.0) == 30.0              # from the 640 anchor: 10 s of clock later
    assert S.estimate(an, 1, 650.0) is None              # no anchor at or above the shot's clock
    assert S.estimate(an, 2, 600.0) is None


def clock_video(stops=()):
    """A synthetic broadcast: clock 700 at video 0 s, running 1:1, frozen during [a, b) video spans."""
    def clock_at(t):
        frozen = sum(min(max(t - a, 0), b - a) for a, b in stops)
        return 700.0 - (t - frozen)
    def read(t):
        c = clock_at(t)
        return {(1, float(math.ceil(c - 1e-9)))}         # the scorebug shows the clock rounded up to the second
    return read


def test_the_search_brackets_the_change_to_the_logged_second_and_bisects_it():
    read = clock_video()
    an = [(0.0, 1, 700.0)]
    r = S.place_shot(1, 650.0, an, read, 4000)
    assert r["status"] == "placed"
    assert abs(r["t"] - 50.0) <= 0.26 and r["bracket_s"] == 0.25


def test_a_dead_ball_only_moves_the_moment_later_and_the_search_still_finds_it():
    read = clock_video(stops=[(10.0, 30.0)])               # 20 s stopped: the shot at 650 sits at video 70 s
    r = S.place_shot(1, 650.0, [(0.0, 1, 700.0)], read, 4000)
    assert r["status"] == "placed" and abs(r["t"] - 70.0) <= 0.26


def test_a_shot_is_not_placed_when_the_moment_is_unreadable_or_the_clock_jumps():
    base = clock_video()
    hole = lambda t: set() if 47.5 <= t <= 52.5 else base(t)          # replay over the moment
    assert S.place_shot(1, 650.0, [(0.0, 1, 700.0)], hole, 4000)["status"] == "gap at the moment"
    jump = lambda t: base(t) if t < 50 else {(1, float(math.ceil(700.0 - t - 30)))}    # 30 s cut out of the upload
    assert S.place_shot(1, 650.0, [(0.0, 1, 700.0)], jump, 4000)["status"] == "clock jumped"
    assert S.place_shot(1, 650.0, [], base, 4000)["status"] == "no anchor"
    late = lambda t: {(1, 640.0)}                                      # already past the moment at the first read
    assert S.place_shot(1, 650.0, [(0.0, 1, 700.0)], late, 4000)["status"] == "passed before a bracket"


def test_windows_merge_close_shots_and_keep_the_warm_up():
    w = S.windows_for([100.0, 105.0, 200.0], 30.0, "sec")
    assert len(w) == 2
    assert w[0]["f0"] == int(round(97.0 * 30)) and w[0]["f_end"] == int(round(106.0 * 30))
    assert w[0]["f_start"] == int(round(95.5 * 30))                    # PAD_S 1.5 s of warm-up
    assert w[1]["window"] == "sec_w001"
    long = S.windows_for([float(t) for t in range(0, 400, 8)], 30.0, "s")
    assert all((x["f_end"] - x["f0"]) / 30.0 <= S.MAX_WINDOW_S for x in long)


def test_the_oracle_takes_the_nearest_track_over_the_given_frames():
    pts = {0: [(10.0, 10.0), (40.0, 25.0)], 3: [(12.0, 10.0)], 6: []}
    assert S.oracle(pts, (12.0, 10.0), [0]) == (2.0, 0)
    assert S.oracle(pts, (12.0, 10.0), [0, 3, 6]) == (0.0, 3)
    assert S.oracle(pts, (12.0, 10.0), [6, 9]) == (None, None)
    nan = {0: [(float("nan"), 1.0)], 3: [(float("nan"), 1.0), (12.0, 11.0)]}                    # r23 leaves NaN points
    assert S.oracle(nan, (12.0, 10.0), [0, 3]) == (1.0, 3)


def test_the_mirror_must_beat_the_runner_up_by_the_margin():
    assert S.choose_mirror({"identity": 2.0, "flip_x": 30.0, "flip_y": 9.0, "flip_xy": 40.0}) == ("identity", True)
    assert S.choose_mirror({"identity": 2.0, "flip_x": 3.5, "flip_y": None, "flip_xy": None}) == ("identity", False)
    assert S.choose_mirror({"identity": None}) == (None, False)


def test_rows_leave_out_unplaced_shots_and_judge_the_targets_on_the_point_estimate():
    def ok(win, at, w):
        return {"placement": "placed", "status": "ok", "window": w, "period": 2,
                "mirrors": {"identity": {"at": at, "win": win, "best_offset_s": -1.0}}}
    obs = [ok(1.0, 2.0, "a"), ok(2.0, 2.5, "a"), ok(2.5, 3.0, "b"), ok(5.0, 9.0, "c"),
           {"placement": "gap at the moment", "period": 2}]
    r = S.rows_of(obs, "identity")
    assert r["attempts_outside_period_1"] == 5 and r["placed"] == 4 and r["placed_share"] == 0.8
    assert r["windowed"]["p50_ft"] == pytest.approx(2.25) and r["windowed"]["within_3ft"] == 0.75
    assert r["windowed"]["pass_targets"] is True
    assert r["at_placed"]["pass_targets"] is False                       # p90 above 6 ft
    assert r["status_counts"] == {"placed: ok": 4, "gap at the moment": 1}


def test_the_two_games_in_d85_test_arenas_and_any_other_game_stop_the_tool():
    from tools.research_testset import refuse_cv_test
    for gid in S.KEPT_OUT + ("0022500001",):
        with pytest.raises(SystemExit):
            S.checked(gid)
    for home in ("TOR", "LAL"):
        with pytest.raises(SystemExit):
            refuse_cv_test(["x"], S.WHO, homes=[home])
    assert S.PILOT in S.CHECK_SET and len(S.CHECK_SET) == 10 and not set(S.KEPT_OUT) & set(S.CHECK_SET)


def test_the_constants_match_the_pre_registration_and_the_frozen_windows():
    from analytics.shot_frames import WINDOW_AFTER_S
    from tools.p8_scorecard import CONSTANTS
    assert S.WIN_BEFORE_S == WINDOW_AFTER_S == 3.0 and S.WIN_AFTER_S == 0.5
    assert (S.ANCHOR_STEP_S, S.SEARCH_BEFORE_S, S.SEARCH_MAX_READS, S.BISECT_READS) == (20.0, 5.0, 60, 2)
    assert S.TARGETS == {"shooter_p50_ft": CONSTANTS["shooter_p50_ft"], "shooter_p90_ft": CONSTANTS["shooter_p90_ft"]}
    assert S.GATE_MARGIN_FT == {"p50": 1.0, "p90": 2.0} and S.MIRROR_MARGIN_FT == 2.0
