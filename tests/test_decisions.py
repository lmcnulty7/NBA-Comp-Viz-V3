"""analytics/decisions on the local synthetic games of tests/_fx_decisions.py (the fallback generator, so the
tests pass without analytics/fixtures.py), a hand-built ball path for pass and drive detection, and the leakage
guards. No network. Every number here is fixture output, not a result."""
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from analytics import schema  # noqa: E402
from analytics.decisions import eda as E  # noqa: E402
from analytics.decisions import features as F  # noqa: E402
from analytics.decisions import train as T  # noqa: E402
from analytics.decisions import validate as V  # noqa: E402
from tests import _fx_decisions as fx  # noqa: E402

Row = namedtuple("Row", "possession start_frame end_frame offense_team_id gap")


# ── helpers ────────────────────────────────────────────────────────────────────────────────────────
def _bundles(n_games=4, n_possessions=12, heldout_every=4, seed=0):
    return V.load_fixture_bundles(seed, n_games, heldout_every, n_possessions, season_fn=fx.local_synth_season)


def _frames(ball, off, dfn, off_team=1, def_team=2):
    """ball: (n, 3); off, dfn: {player_id: (n, 2)} -> a section 4 style frames table (the columns the lane reads)."""
    n = len(ball)
    rows = []
    for team, side in ((off_team, off), (def_team, dfn)):
        for pid, xy in side.items():
            rows.append(pd.DataFrame({"frame": np.arange(n), "period": 1, "game_clock_s": 700 - np.arange(n) / 25.0,
                                      "ball_x": ball[:, 0], "ball_y": ball[:, 1], "ball_z": ball[:, 2],
                                      "team_id": team, "home": team == off_team, "player_id": pid, "x": xy[:, 0], "y": xy[:, 1],
                                      "ts_ms": 1_000_000 + np.arange(n) * 40}))
    return pd.concat(rows, ignore_index=True).sort_values(["frame", "player_id"], kind="stable").reset_index(drop=True)


def _static(xy, n):
    return np.repeat(np.array([xy], dtype=float), n, axis=0)


def _possession(frames, n, off_team=1):
    return F.Possession(frames, frames["frame"].to_numpy(), Row(0, 0, n - 1, off_team, False), 25.0)


def _hand_built(flight_frames=10, hold=20, shot_release=None):
    """A (id 1) holds the ball near the left basket, passes to teammate B (id 2) over flight_frames, B holds it."""
    n = hold + flight_frames + hold
    a_xy, b_xy = (20.0, 25.0), (10.0, 40.0)
    ball = np.zeros((n, 3))
    ball[:hold, :2] = np.array(a_xy) + np.array([0.5, 0.0])
    ball[:hold, 2] = 3.0
    t = np.linspace(0, 1, flight_frames + 2)[1:-1]
    ball[hold:hold + flight_frames, :2] = np.array(a_xy)[None] + (np.array(b_xy) - np.array(a_xy))[None] * t[:, None]
    ball[hold:hold + flight_frames, 2] = 6.0
    ball[hold + flight_frames:, :2] = np.array(b_xy) + np.array([0.5, 0.0])
    ball[hold + flight_frames:, 2] = 3.0
    off = {1: _static(a_xy, n), 2: _static(b_xy, n), 3: _static((30.0, 10.0), n), 4: _static((28.0, 45.0), n), 5: _static((8.0, 10.0), n)}
    dfn = {11: _static((17.0, 25.0), n), 12: _static((9.0, 33.0), n), 13: _static((26.0, 12.0), n),
           14: _static((24.0, 42.0), n), 15: _static((6.0, 14.0), n)}
    frames = _frames(ball, off, dfn)
    return frames, n, hold


# ── fixtures and validation ────────────────────────────────────────────────────────────────────────
def test_local_fixture_tables_conform_to_the_schema():
    g = fx.local_synth_game(3, n_possessions=8)
    for t in ("rosters", "shot_frames", "rebound_frames", "possessions", "states_1hz"):
        assert schema.validate(g[t], t) == [], t
    assert set(V.FRAME_COLUMNS) <= set(g["frames"].columns)
    assert g["possessions"]["points"].sum() > 0 and (g["shot_frames"]["release_frame"] >= 0).all()
    _, rep = V.validate_bundle(V._as_bundle(g))
    assert sum(len(v) for v in rep["schema"].values()) == 0 and rep["range"]["possessions.span"] == 0


def test_validate_counts_duplicates_and_ranges():
    g = fx.local_synth_game(4, n_possessions=6)
    b = V._as_bundle(g)
    b["frames"] = pd.concat([b["frames"], b["frames"].head(10)], ignore_index=True)
    b["frames"].loc[0, "x"] = 500.0
    b["possessions"] = pd.concat([b["possessions"], b["possessions"].head(1)], ignore_index=True)
    cleaned, rep = V.validate_bundle(b)
    assert rep["duplicates"]["frames"] == 10 and rep["duplicates"]["possessions"] == 1
    assert rep["range"]["frames.x"] == 1
    assert len(cleaned["frames"]) == len(g["frames"]) and cleaned["frames"]["frame"].is_monotonic_increasing


# ── detection on a hand-built ball path ────────────────────────────────────────────────────────────
def test_pass_detected_on_hand_built_ball_path():
    frames, n, hold = _hand_built()
    pos = _possession(frames, n)
    assert pos.basket == F.RIM_LEFT
    holder = pos.holders()
    assert holder[0] == 0 and holder[-1] == 1 and (holder[hold + 2:hold + 8] == -1).all()
    last_a = int(np.flatnonzero(holder == 0).max())              # the ball leaves A's 2.5 ft over the first flight frame
    assert hold - 1 <= last_a <= hold + 1
    passes = F.detect_passes(pos, releases=np.array([]))
    assert len(passes) == 1
    i, a, b = passes[0]
    assert i == last_a and pos.off_ids[a] == 1 and pos.off_ids[b] == 2
    assert F.detect_drives(pos) == []
    # the same path with a shot release inside the window is not a pass
    assert F.detect_passes(pos, releases=np.array([hold + 2])) == []
    # the ball arriving after 1.5 s is not a pass either
    frames2, n2, hold2 = _hand_built(flight_frames=45)
    assert F.detect_passes(_possession(frames2, n2), releases=np.array([])) == []


def test_pass_context_and_quality_are_computed_at_the_decision_frame():
    frames, n, hold = _hand_built()
    sf = pd.DataFrame(columns=["sportvu_game", "event", "release_frame", "shooter_id", "shooter_at_release_id"])
    poss = pd.DataFrame([{"sportvu_game": "g", "possession": 0, "start_frame": 0, "end_frame": n - 1,
                          "offense_team_id": 1, "gap": False, "points": 2}])
    bundle = {"sportvu_game": "g", "heldout": False, "frames": frames, "shot_frames": sf, "possessions": poss}
    ctx = F.decision_contexts(bundle)
    assert list(ctx["decision_kind"]) == ["pass"] and hold - 1 <= int(ctx["frame"].iloc[0]) <= hold + 1
    assert int(ctx["player_id"].iloc[0]) == 1 and int(ctx["receiver_id"].iloc[0]) == 2
    assert ctx["nearest_def_dist_ft"].iloc[0] == pytest.approx(3.0)
    assert ctx["n_teammates"].iloc[0] == 4
    ev = F.option_values(ctx, lambda x, y, bx, by: np.full(len(np.atleast_1d(x)), 0.5))
    assert len(ev) == 1 and ev["best_option"].iloc[0].startswith(("pass_to_", "shot"))
    assert ev["decision_quality"].iloc[0] <= 1e-9


def test_drive_detected_when_the_handler_runs_at_the_basket():
    n = 60
    ball = np.zeros((n, 3))
    path = np.zeros((n, 2))
    path[:, 0] = 30.0 - np.minimum(np.arange(n), 40) * (12.0 / 25.0)       # 12 ft/s toward the left basket for 1.6 s
    path[:, 1] = 25.0
    ball[:, :2] = path + np.array([0.5, 0.0])
    ball[:, 2] = 3.0
    off = {1: path, 2: _static((10.0, 40.0), n), 3: _static((30.0, 10.0), n), 4: _static((28.0, 45.0), n), 5: _static((8.0, 10.0), n)}
    dfn = {11: _static((17.0, 20.0), n), 12: _static((9.0, 33.0), n), 13: _static((26.0, 12.0), n),
           14: _static((24.0, 42.0), n), 15: _static((6.0, 14.0), n)}
    pos = _possession(_frames(ball, off, dfn), n)
    drives = F.detect_drives(pos)
    assert len(drives) == 1 and pos.off_ids[drives[0][1]] == 1 and drives[0][0] == 1
    slow = path.copy()
    slow[:, 0] = 30.0 - np.arange(n) * (4.0 / 25.0)
    ball2 = ball.copy()
    ball2[:, :2] = slow + np.array([0.5, 0.0])
    assert F.detect_drives(_possession(_frames(ball2, dict(off, **{1: slow}), dfn), n)) == []


# ── option values ──────────────────────────────────────────────────────────────────────────────────
def test_option_values_follow_the_documented_formulas():
    ctx = pd.DataFrame([{"sportvu_game": "g", "possession": 0, "frame": 10, "event": -1, "player_id": 1, "receiver_id": 2,
                         "decision_kind": "pass", "heldout": False, "basket_x": 5.25, "basket_y": 25.0, "x": 25.25, "y": 25.0,
                         "vx": 0.0, "vy": 0.0, "dist_to_basket_ft": 20.0, "angle_rad": 0.0, "nearest_def_dist_ft": 2.0,
                         "n_teammates": 1, "n_teammates_open": 1, "elapsed_in_possession_s": 0.4, "ball_z": 3.0,
                         "speed_toward_basket_ftps": 0.0,
                         "tm1_id": 2, "tm1_x": 10.25, "tm1_y": 25.0, "tm1_def_ft": 8.0, "tm1_pass_ft": 18.0,
                         "tm2_id": -1, "tm2_x": np.nan, "tm2_y": np.nan, "tm2_def_ft": np.nan, "tm2_pass_ft": np.nan,
                         "tm3_id": -1, "tm3_x": np.nan, "tm3_y": np.nan, "tm3_def_ft": np.nan, "tm3_pass_ft": np.nan,
                         "tm4_id": -1, "tm4_x": np.nan, "tm4_y": np.nan, "tm4_def_ft": np.nan, "tm4_pass_ft": np.nan}])
    p_half = lambda x, y, bx, by: np.full(len(np.atleast_1d(x)), 0.5)
    ev = F.option_values(ctx, p_half)
    shot = 2 * (0.5 - 0.03 * 4)                       # 20 ft, defender at 2 ft: 4 ft below the 6 ft threshold
    pass_v = 2 * 0.5 * 1.0 * (0.95 - 0.01 * 3)       # open teammate at 5 ft, pass of 18 ft
    assert ev["shot_value"].iloc[0] == pytest.approx(shot)
    assert ev["chosen_value"].iloc[0] == pytest.approx(pass_v)
    assert ev["best_option"].iloc[0] == "pass_to_1" and ev["decision_quality"].iloc[0] == pytest.approx(0.0)
    ev2 = F.option_values(ctx, p_half, hold=pd.Series([1.1]))
    assert ev2["best_option"].iloc[0] == "hold" and ev2["decision_quality"].iloc[0] == pytest.approx(pass_v - 1.1)
    ctx3 = ctx.copy()
    ctx3["decision_kind"] = "shot"
    ev3 = F.option_values(ctx3, p_half, p_shot_override=pd.Series([0.6]))
    assert ev3["chosen_value"].iloc[0] == pytest.approx(1.2) and ev3["prior"].iloc[0] == "location"


# ── leakage ────────────────────────────────────────────────────────────────────────────────────────
def test_shifting_a_decision_frame_after_the_outcome_is_caught():
    b = _bundles(n_games=1, n_possessions=8, heldout_every=0)[0]
    ctx = F.decision_contexts(b)
    F.assert_decisions_before_outcome(ctx, b["shot_frames"], b["possessions"])
    shots = ctx.index[ctx["decision_kind"] == "shot"]
    assert len(shots)
    shifted = ctx.copy()
    shifted.loc[shots[0], "frame"] += 10                     # after the release: the ball is in the air
    with pytest.raises(ValueError, match="later than the release"):
        F.assert_decisions_before_outcome(shifted, b["shot_frames"], b["possessions"])
    passes = ctx.index[ctx["decision_kind"] == "pass"]
    late = ctx.copy()
    end = int(b["possessions"].set_index("possession").loc[int(ctx.loc[passes[0], "possession"]), "end_frame"])
    late.loc[passes[0], "frame"] = end + 5
    with pytest.raises(ValueError, match="outside their possession span"):
        F.assert_decisions_before_outcome(late, b["shot_frames"], b["possessions"])
    leaky = ctx.copy()
    leaky["points"] = 2
    with pytest.raises(ValueError, match="never features"):
        schema.assert_time_forward(leaky, "states_1hz")
    assert F.feature_columns() == F.FEATURES and set(F.FEATURES).isdisjoint(schema.ALL_EXCLUDED)


def test_trainer_refuses_a_held_out_id_flagged_as_training():
    b = _bundles(n_games=1, n_possessions=6, heldout_every=0)[0]
    b["sportvu_game"] = "12.16.2015.PHX.at.GSW"
    b["heldout"] = False
    with pytest.raises(SystemExit, match="refuses to run"):
        T.run([b], 2, 0, 5)


def test_fit_value_model_refuses_held_out_rows_and_uses_location_only():
    rows = pd.DataFrame({"sportvu_game": ["g"] * 6, "event": range(6), "made": [True, False, True, False, True, False],
                         "period": 1, "clock_s": 0.0, "loc_x": [0, 50, 100, 150, 200, 240], "loc_y": [20, 60, 100, 160, 200, 230],
                         "heldout": [False] * 5 + [True]})
    with pytest.raises(ValueError, match="held-out row"):
        T.fit_value_model(rows)
    m = T.fit_value_model(rows[~rows["heldout"]])
    p = m(np.array([10.0, 30.0]), np.array([25.0, 25.0]), np.array([5.25, 5.25]), np.array([25.0, 25.0]))
    assert p.shape == (2,) and np.all((p > 0) & (p < 1))


# ── end to end ─────────────────────────────────────────────────────────────────────────────────────
def test_end_to_end_fixture_run_writes_the_contract_tables(tmp_path):
    bundles = _bundles(n_games=4, n_possessions=12, heldout_every=4)
    res = T.run(bundles, n_folds=3, seed=0, n_boot=20)
    ev = res["events"]
    assert list(ev.columns[:10]) == ["sportvu_game", "possession", "frame", "player_id", "decision_kind", "chosen_value",
                                     "best_value", "best_option", "decision_quality", "heldout"]
    assert {"shot", "pass"} <= set(ev["decision_kind"]) and ev["sportvu_game"].nunique() == 3
    assert not ev["heldout"].any() and (ev["fold"] >= 0).all()
    non_drive = ev[ev["decision_kind"] != "drive"]
    assert (non_drive["decision_quality"] <= 1e-9).all()
    assert ev["best_option"].str.match(r"^(shot|hold|pass_to_[1-4])$").all()
    assert ev["chosen_value"].between(0, 3).all() and ev["best_value"].between(0, 3).all()
    held = res["events_heldout"]
    assert len(held) and held["heldout"].all() and (held["fold"] == -1).all() and held["sportvu_game"].nunique() == 1
    players = res["players"]
    assert list(players.columns[:6]) == ["player_id", "n_decisions", "mean_quality", "lo", "hi", "insufficient"]
    assert players["insufficient"].all()                      # fixture players are far under the 300-decision gate
    assert players["n_decisions"].sum() == len(ev)
    v = res["validation"]["location"]
    assert set(v) == {"per_game", "summary", "shuffled"} and v["summary"]["n_games"] <= 3
    assert res["prior"] == "location" and res["fit_rows"] > 0
    written = T.write_outputs(res, tmp_path, "fixture")
    assert (tmp_path / "analytics" / "decisions" / "fixture" / "decision_players.parquet").exists()
    assert (tmp_path / "analytics" / "decisions" / "heldout" / "fixture" / "decision_events").exists()
    back = pd.read_parquet(written[0])
    assert len(back) and set(back["decision_kind"]) <= set(F.KINDS)
    # EDA tables run on the result
    assert len(E.kinds_per_possession(ev)) == 3 and len(E.quality_by_kind(ev)) >= 2
    assert len(E.value_distribution(ev)) >= 4 and len(E.correlation_table(res)) == 3
    assert E.shuffled_check(res, n_shuffles=3).shape == (1, 5)


def test_hold_option_from_an_epv_table_and_xfg_override_join(tmp_path):
    b = _bundles(n_games=1, n_possessions=8, heldout_every=0)[0]
    ctx = F.decision_contexts(b)
    epv = pd.DataFrame({"sportvu_game": b["sportvu_game"], "possession": ctx["possession"], "frame": ctx["frame"] - 1,
                        "elapsed_in_possession_s": 0.0, "epv_points": 1.5, "p_score": 0.5, "points_to_end": 0, "fold": 0,
                        "heldout": False})
    hold = F.hold_values(ctx, epv)
    assert hold.notna().all() and (hold == 1.5).all()
    shots = b["shot_frames"]
    xfg = pd.DataFrame({"sportvu_game": b["sportvu_game"], "event": shots["event"], "player_id": shots["shooter_id"],
                        "p_location": 0.4, "p_contest": 0.4, "p_selected": 0.42, "made": shots["made"], "fold": 0, "heldout": False})
    ov = F.xfg_override(ctx, xfg)
    is_shot = ctx["decision_kind"] == "shot"
    assert ov[is_shot].notna().all() and ov[~is_shot].isna().all()
    b["epv"], b["xfg"] = epv, xfg
    res = T.run([b], 2, 0, 5)
    assert res["prior"] == "xfg_selected" and res["n_hold"] > 0 and (res["events"]["hold_value"] == 1.5).all()
    assert "xfg_selected" in res["events_by_prior"] and len(res["events_by_prior"]["xfg_selected"]) == len(res["events"])


def test_data_root_layout_round_trip(tmp_path):
    games = fx.local_write_fixture_root(tmp_path, seed=1, n_games=2, heldout_every=2, n_possessions=6)
    log = []
    bundles = V.load_root_bundles(tmp_path, fx.SEASON, log=log)
    assert [b["sportvu_game"] for b in bundles] == sorted(games, key=lambda g: g.startswith("02")) and log == []
    assert [b["heldout"] for b in bundles] == [False, True]
    assert bundles[0]["events"] is not None and bundles[0]["states_1hz"] is not None
    rep = V.main(["--data", str(tmp_path), "--season", fx.SEASON])
    assert len(rep) == 2 and all(sum(len(v) for v in r["schema"].values()) == 0 for r in rep)
    res = T.run(bundles, 2, 0, 5)
    assert res["mode"].startswith("IN-SAMPLE") and len(res["events_heldout"])


@pytest.mark.skipif(not fx.SHARED, reason="analytics/fixtures.py (lane I1) not importable; the local generator covers the lane")
def test_end_to_end_on_the_shared_fixtures():
    """The same run on lane I1's generator: held-out game named with the real held-out id, gap possessions skipped."""
    bundles = V.load_fixture_bundles(0, 3, 3, 10)
    assert [b["heldout"] for b in bundles] == [False, False, True]
    assert bundles[2]["sportvu_game"] == "12.16.2015.PHX.at.GSW"
    res = T.run(bundles, n_folds=2, seed=0, n_boot=10)
    ev = res["events"]
    assert len(ev) and {"shot", "pass"} <= set(ev["decision_kind"]) and ev["sportvu_game"].nunique() == 2
    assert len(res["events_heldout"]) and (res["events_heldout"]["sportvu_game"] == "12.16.2015.PHX.at.GSW").all()
    assert (ev[ev["decision_kind"] != "drive"]["decision_quality"] <= 1e-9).all()
    assert any(c.get("possessions_gap", 0) for c in res["counts"].values())


def test_ortg_counts_possessions_on_the_floor():
    g = fx.local_synth_game(5, n_possessions=8)
    o = F.on_court_offensive_rating(g["frames"], g["possessions"], g["sportvu_game"])
    assert o["n_possessions"].sum() == 5 * len(g["possessions"])
    assert o["points"].sum() == 5 * g["possessions"]["points"].sum()
    assert (o["ortg"] >= 0).all()
