"""P8-E2 (docs/research/P8_E2.md): placing play-by-play shots and rebounds through the clock time map, on synthetic
time maps and events."""

import json

import pytest

from tools import p8_events as E


def span(section, f0, f1, period, c0, c1, fps=60.0):
    return {"section": section, "f0": f0, "f1": f1, "period": period, "c0": c0, "c1": c1, "fps": fps,
            "video_s": round((f1 - f0) / fps, 1)}


def test_the_constants_equal_the_frozen_and_pinned_ones():
    from sportvu.local_sync import RUN_TOL_S
    from tools.p8_scorecard import CONSTANTS
    assert E.RUN_TOL_S == RUN_TOL_S
    assert E.MARGIN_S == CONSTANTS["event_live_wide_s"]
    assert E.TOL_S == CONSTANTS["event_tol_s"]


def test_game_seconds_run_through_quarters_and_overtime():
    assert E.game_s(1, 720.0) == 0.0
    assert E.game_s(1, 0.0) == E.game_s(2, 720.0) == 720.0
    assert E.game_s(4, 0.0) == E.game_s(5, 300.0) == 2880.0
    assert E.game_s(6, 0.0) == 2880.0 + 600.0


def test_the_chain_drops_a_misread_span_whose_clock_runs_backward():
    spans = [span("s00", 0, 180, 1, 78.0, 75.0),          # a misread among spans near 11:00
             span("s00", 400, 700, 1, 670.0, 665.0),
             span("s00", 1000, 1180, 1, 658.0, 655.0),
             span("s01", 0, 600, 1, 650.0, 640.0)]
    assert E.chain(spans) == [1, 2, 3]
    # a misread that outweighs the true spans is kept; the chain rule weighs video seconds, nothing else
    assert E.chain([span("s00", 0, 6000, 1, 78.0, 0.0), span("s00", 6100, 6200, 2, 700.0, 698.0)]) == [0, 1]
    assert E.chain([]) == []


def test_a_span_may_overlap_the_previous_one_by_the_run_tolerance_only():
    a = span("s00", 0, 600, 1, 700.0, 690.0)
    assert E.can_follow(a, span("s00", 700, 800, 1, 691.0, 689.0))
    assert not E.can_follow(a, span("s00", 700, 800, 1, 691.5, 689.0))
    assert E.can_follow(a, span("s00", 700, 800, 2, 720.0, 700.0))


def test_an_event_is_placed_within_half_a_second_of_a_span_and_classified_otherwise():
    chained = [span("s00", 0, 600, 1, 700.0, 690.0),
               span("s00", 900, 1500, 1, 690.0, 680.0),            # 5 s of video, clock stopped: a dead ball
               span("s00", 2100, 2700, 1, 670.0, 660.0),           # 10 s of video, 10 s of clock: clock ran unread
               span("s00", 2760, 3360, 1, 640.0, 630.0),           # 1 s of video, 20 s of clock: the video skips
               span("s01", 0, 600, 1, 600.0, 590.0),
               span("s01", 900, 1500, 3, 500.0, 490.0)]
    assert E.classify(1, 695.0, chained) is None
    assert E.classify(1, 700.5, chained) is None
    assert E.classify(1, 700.6, chained) == "between sections"     # before the first mapped moment
    assert E.classify(1, 675.0, chained) == "clock ran unread"
    assert E.classify(1, 650.0, chained) == "video skips game time"
    assert E.classify(1, 610.0, chained) == "between sections"
    assert E.classify(2, 600.0, chained) == "period not mapped"     # period 2 has no chained span
    assert E.classify(4, 100.0, chained) == "period not mapped"


def test_gap_seconds_split_dead_balls_unread_clock_and_skips():
    chained = [span("s00", 0, 600, 1, 700.0, 690.0),
               span("s00", 900, 1500, 1, 690.0, 680.0),
               span("s00", 2100, 2700, 1, 670.0, 660.0),
               span("s00", 2760, 3360, 1, 640.0, 630.0),
               span("s01", 0, 600, 1, 600.0, 590.0)]
    g = E.gap_seconds(chained)
    assert g["dead_ball_video_s"] == 5.0
    assert g["unread_video_s"] == 10.0 and g["unread_game_s"] == 10.0
    assert g["skipped_game_s"] == 19.0
    assert g["mapped_game_s"] == 50.0 and g["mapped_video_s"] == 50.0


def test_place_window_picks_the_nearest_time_map_frame_and_applies_the_lag():
    frames = [[f, 1, round(100.0 - (f - 60) / 60.0, 3)] for f in range(60, 661, 6)]     # 100.0 .. 90.0
    tm = {"spans": [{"f0": 60, "f1": 660, "period": 1, "c0": 100.0, "c1": 90.0}], "frames": frames}
    ev = [(7, "shot", 1, 95.0), (8, "rebound", 1, 89.6), (9, "shot", 1, 89.4), (10, "shot", 2, 95.0)]
    got = E.place_window(tm, ev)
    assert got == [{"frame": 360, "kind": "shot", "event": 7}, {"frame": 660, "kind": "rebound", "event": 8}]
    lagged = E.place_window(tm, [(7, "shot", 1, 95.0)], lag=0.5)                  # OCR clock 94.5
    assert lagged == [{"frame": 390, "kind": "shot", "event": 7}]


def test_the_m1_lag_never_uses_the_scored_games_own_offsets():
    offs = {"a": [0.1, 0.2, 0.3], "b": [1.0, 1.0], "c": [5.0]}
    assert E.leave_one_out_lag(offs, "a") == 1.0
    assert E.leave_one_out_lag(offs, "c") == 0.3
    assert E.leave_one_out_lag({"a": [0.1]}, "a") is None


def test_window_tallies_match_the_pinned_scorer_one_to_one_within_a_second():
    truth = [(1, 600.0), (1, 590.0), (1, 580.0)]
    pred = [(1, 600.8), (1, 588.5), (1, 580.2), (1, 570.0)]
    obs = E.window_tallies(truth, pred)
    assert sorted(obs) == sorted([("t", 1), ("t", 0), ("t", 1), ("f", 1), ("f", 1)])
    assert E.found_share(obs) == pytest.approx(2 / 3)
    assert E.false_per_true(obs) == pytest.approx(2 / 3)
    assert E.found_share([]) is None and E.false_per_true([("f", 1)]) is None


def test_scored_rows_pool_windows_and_games_and_judge_the_targets():
    def sc(windows):
        return {"methods": {"M0": {"per_window": windows}}}
    hit = {"offset_s": 0.2, "truth": {"shot": [[1, 600.0]], "rebound": [[1, 590.0]]},
           "pred": {"shot": [[1, 600.2]], "rebound": [[1, 590.2]]}}
    miss = {"offset_s": 1.4, "truth": {"shot": [[1, 500.0]], "rebound": [[1, 490.0]]},
            "pred": {"shot": [[1, 501.4]], "rebound": [[1, 491.4]]}}
    files = {"g1": sc({"g1_w0": hit, "g1_w1": hit}), "g2": sc({"g2_w0": hit, "g2_w1": miss})}
    rows = E.scored_rows(files, ["g1", "g2"], "M0")
    assert rows["shot"]["truth_events"] == 4 and rows["shot"]["found_share"] == 0.75
    assert rows["shot"]["false_per_true"] == 0.25 and rows["shot"]["pass"] is False
    assert rows["shot"]["found_share_game_ci"] is not None
    one = E.scored_rows(files, ["g1"], "M0")
    assert one["rebound"]["found_share"] == 1.0 and one["rebound"]["pass"] is True
    assert one["rebound"]["found_share_game_ci"] is None                       # one game: no game bootstrap


def test_the_full_error_row_skips_events_it_cannot_join_uniquely():
    sc = {"methods": {"M0": {"per_window": {"w0": {"offset_s": 0.5, "truth": {"shot": [[1, 600.0], [1, 590.0]],
                                                                               "rebound": []}, "pred": {}}}}}}
    ev = {"shot": [(1, "shot", 1, 600.0), (2, "shot", 1, 590.0), (3, "shot", 1, 590.0)], "rebound": []}
    phys = {"shot": {"1": 601.2}, "rebound": {}}
    r = E.full_error_rows(sc, ev, phys, lag=0.5)
    assert r["shot"]["skipped_not_unique"] == 1
    assert r["shot"]["M0"]["n"] == 1 and r["shot"]["M0"]["share_within_1s"] == 1.0     # 600 + 0.5 - 601.2 = -0.7
    assert r["shot"]["M1"]["share_within_1s"] == 0.0                                   # 600 - 0.5 + 0.5 - 601.2


def test_place_game_writes_an_events_file_for_every_label_window_even_off_the_chain(tmp_path, monkeypatch):
    """A label window off the chain still gets its sidecar links and an empty events file, so the scorer keeps its
    truth events in the denominator."""
    P = {k: tmp_path / k for k in ("sync", "label_build", "builds")}
    for p in P.values():
        p.mkdir()
    tm_sec = {"section": "g_s00", "fps": 60.0, "spans": [
        {"f0": 0, "f1": 600, "period": 1, "c0": 78.0, "c1": 68.0, "reads": 11, "video_s": 10.0},
        {"f0": 1200, "f1": 2400, "period": 1, "c0": 700.0, "c1": 680.0, "reads": 21, "video_s": 20.0}]}
    (P["sync"] / "g_s00_local_timemap.json").write_text(json.dumps(tm_sec))
    wins = [{"window": "g_s00_w00", "section": "g_s00", "f0": 1200, "period": 1},
            {"window": "g_s00_w01", "section": "g_s00", "f0": 0, "period": 1}]
    frames = [[f, 1, round(700.0 - (f - 90) / 60.0, 3)] for f in range(90, 1291, 6)]
    (P["label_build"] / "g_s00_w00_timemap.json").write_text(json.dumps(
        {"spans": [{"f0": 90, "f1": 1290, "period": 1, "c0": 700.0, "c1": 680.0}], "frames": frames}))
    (P["label_build"] / "g_s00_w01_timemap.json").write_text(json.dumps(
        {"spans": [{"f0": 90, "f1": 690, "period": 1, "c0": 78.0, "c1": 68.0}],
         "frames": [[f, 1, round(78.0 - (f - 90) / 60.0, 3)] for f in range(90, 691, 6)]}))
    for w in ("g_s00_w00", "g_s00_w01"):
        (P["label_build"] / f"{w}_frames.json").write_text("{}")
    from tools import p8_frozen as F
    monkeypatch.setattr(F, "label_windows", lambda P_, tag: wins)
    monkeypatch.setattr(E, "guard", lambda items, who=None: items)
    monkeypatch.setattr(E, "pbp_events", lambda sv: {
        "shot": [(1, "shot", 1, 690.0), (2, "shot", 1, 70.0)], "rebound": [(3, "rebound", 1, 650.0)],
        "_contract_frame": {}})
    cov, placed = E.place_game(P, "g", {"sportvu": "sv", "video_id": "v"}, {"g": 0.5})
    assert cov["chained_spans"] == 1 and len(cov["dropped_spans"]) == 1
    assert cov["events"]["shot"]["placed_in_chained_span"] == 1
    assert cov["events"]["shot"]["gaps"]["period not mapped"] == 0
    assert cov["events"]["rebound"]["gaps"]["between sections"] == 1
    assert placed == {1: "g_s00@1200"}
    on = json.loads((P["builds"] / "e2" / "g_s00_w00_events.json").read_text())
    off = json.loads((P["builds"] / "e2" / "g_s00_w01_events.json").read_text())
    assert on["events"] == [{"frame": 690, "kind": "shot", "event": 1}]
    assert off["events"] == []                                       # the misread window places nothing
    m1 = json.loads((P["builds"] / "e2_m1" / "g_s00_w00_events.json").read_text())
    assert m1["lag_s"] == 0.5 and m1["events"] == [{"frame": 720, "kind": "shot", "event": 1}]
    assert (P["builds"] / "e2" / "g_s00_w01_frames.json").is_symlink()


def test_a_spent_heldout_or_cv_test_id_stops_the_tool():
    with pytest.raises(SystemExit):
        E.guard(["12.16.2015.PHX.at.GSW"], E.WHO)
    with pytest.raises(SystemExit):
        E.pbp_events("12.16.2015.PHX.at.GSW")
