"""tests/test_matchups.py: analytics.matchups on the synthetic fixtures of tests/_fx_matchups.py (no network, no data)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analytics import schema  # noqa: E402
from analytics.matchups import eda, features as F, train as T, validate as V  # noqa: E402
from tests import _fx_matchups as FX  # noqa: E402

HELD = "12.16.2015.PHX.at.GSW"


@pytest.fixture(scope="module")
def games():
    return V.fixture_games(0, n_games=3, heldout_every=3, n_possessions=16)


@pytest.fixture(scope="module")
def ctx(games):
    return T.run(games, None, n_folds=3, seed=0, n_boot=20, n_min=5, fixtures=True, command="pytest")


# ── assignment on hand-built frames ────────────────────────────────────────────────────────────────
def test_assignment_nearest_and_midpoint_term():
    basket = (5.25, 25.0)
    off = np.array([[20.0, 20.0], [20.0, 40.0]])
    dfn = np.array([[21.0, 40.0], [21.0, 20.0]])          # defender 0 stands by off 1, defender 1 by off 0
    assert F.assign(dfn, off, basket).tolist() == [1, 0]
    # equidistant to both offensive players (on their perpendicular bisector): only the midpoint term decides,
    # and it favours the player whose midpoint to the basket is nearer the defender (off 0 here)
    off = np.array([[30.0, 10.0], [20.0, 40.0]])
    dfn = np.array([[29.74, 26.58]])
    d1 = np.hypot(dfn[0, 0] - off[:, 0], dfn[0, 1] - off[:, 1])
    assert d1[0] == pytest.approx(d1[1], abs=0.01)
    cost = F.assignment_cost(dfn, off, basket)
    assert cost.shape == (1, 2) and cost[0, 0] < cost[0, 1] - 1.0
    assert F.assign(dfn, off, basket).tolist() == [0]
    # rectangular: more defenders than offensive players leaves one unassigned
    a = F.assign(np.array([[1.0, 1.0], [2.0, 2.0], [40.0, 40.0]]), np.array([[1.0, 1.5], [2.0, 2.5]]), basket)
    assert sorted(a.tolist()) == [-1, 0, 1]
    assert F.basket_for(np.array([[80.0, 20.0], [85.0, 30.0]])) == (88.75, 25.0)


def test_sample_frames_one_per_second():
    fl = pd.DataFrame({"frame": np.arange(90), "game_clock_s": 700.0 - np.arange(90) / 25.0})   # 25 Hz, 3.56 s
    picked = F.sample_frames(fl, 700.0)
    assert picked.tolist() == [0, 25, 50, 75]
    assert F.sample_frames(fl.iloc[:0], 700.0).tolist() == []


def _hand_game():
    """One possession, two sampled frames (1 Hz at 1 Hz frames), the two defenders swap men at the second frame."""
    rows = []
    off = {11: [(20.0, 20.0), (20.0, 20.0)], 12: [(20.0, 40.0), (20.0, 40.0)]}
    dfn = {21: [(21.0, 20.0), (21.0, 40.0)], 22: [(21.0, 40.0), (21.0, 20.0)]}
    for fr in (0, 1):
        for pid, xy in list(off.items()) + list(dfn.items()):
            rows.append({"game": "g", "period": 1, "frame": fr, "game_clock_s": 700.0 - fr, "team_id": 1 if pid < 20 else 2,
                         "player_id": pid, "x": xy[fr][0], "y": xy[fr][1], "ball_x": 20.0, "ball_y": 20.0, "ball_z": 4.0})
    frames = pd.DataFrame(rows)
    poss = pd.DataFrame([{"sportvu_game": "g", "possession": 0, "period": 1, "start_frame": 0, "end_frame": 1, "start_clock_s": 700.0,
                          "end_clock_s": 699.0, "start_event": 0, "end_event": 0, "offense_team_id": 1, "end_kind": "shot_made",
                          "points": 3, "gap": False, "n_frames": 2}])
    events = pd.DataFrame([{"event": 0, "period": 1, "clock_s": 699.0, "team": "home", "desc": "A makes 3-pt jump shot", "points": 3,
                            "kind": "shot_made", "frame": 1}])
    sf = pd.DataFrame([{"sportvu_game": "g", "event": 0, "period": 1, "clock_s": 699.0, "made": True, "pbp_shooter_name": "A", "shooter_id": 11,
                        "release_frame": 1, "apex_frame": 1, "outcome_frame": 1, "shooter_at_release_id": 11, "identity_match": True,
                        "confidence": 0.9, "confidence_flag": "high"}])
    return {"sportvu_game": "g", "heldout": False, "frames": frames, "events": events, "possessions": poss, "shot_frames": sf}


def test_frame_assignments_switch_and_shares_on_hand_built_possession():
    g = _hand_game()
    xfg = pd.DataFrame({"sportvu_game": ["g"], "event": [0], "p_selected": [0.4]})
    r = F.build(g, xfg)
    mp = r["matchup_possessions"].sort_values(["def_player_id", "off_player_id"]).reset_index(drop=True)
    assert list(mp.columns) == F.MATCHUP_COLUMNS
    assert len(mp) == 4 and mp["partial_possession"].tolist() == pytest.approx([0.5] * 4)
    assert mp["switches"].tolist() == [1, 1, 1, 1]
    assert mp.groupby("def_player_id")["partial_possession"].sum().tolist() == pytest.approx([1.0, 1.0])
    # points go to the shooter's rows (off 11) times the share; expected = p * 3 * share
    on_shooter = mp[mp["off_player_id"] == 11]
    assert on_shooter["points_allowed_share"].tolist() == pytest.approx([1.5, 1.5])
    assert on_shooter["xfg_expected_share"].tolist() == pytest.approx([0.6, 0.6])
    assert mp.loc[mp["off_player_id"] == 12, "points_allowed_share"].tolist() == [0.0, 0.0]
    mf = r["matchup_frames"]
    assert mf["changed"].isna().sum() == 2 and mf["changed"].dropna().tolist() == [1.0, 1.0]
    s = r["shots"]
    assert len(s) == 1 and s["value"].iloc[0] == 3 and s["p"].iloc[0] == 0.4 and int(s["assigned_def_id"].iloc[0]) == 22
    assert s["def_dist_ft"].iloc[0] == pytest.approx(1.0)


# ── leakage guards ─────────────────────────────────────────────────────────────────────────────────
def test_leakage_guards(games):
    mp = F.build(games[0])["matchup_possessions"]
    ff = F.feature_frame(mp)
    assert list(ff.columns) == F.ID_COLUMNS + F.feature_columns()
    for bad in ("points", "made", "end_kind", "end_frame"):
        with pytest.raises(ValueError):
            F.feature_frame(mp.assign(**{bad: 0}))
    with pytest.raises(ValueError):
        schema.assert_time_forward(mp.assign(points=1), "possessions")
    # the ridge design has only indicator columns
    D = T.possession_design(mp, T.possession_points([games[0]]))
    assert all(c[0] in "DO" for c in D["columns"]) and D["X"].shape == (len(D["y"]), len(D["columns"]))
    # the assignment is independent of the outcome: permuting points changes no share or switch count
    g2 = dict(games[0])
    p2 = games[0]["possessions"].copy()
    p2["points"] = np.random.default_rng(1).permutation(p2["points"].to_numpy())
    g2["possessions"] = p2
    mp2 = F.build(g2)["matchup_possessions"]
    pd.testing.assert_frame_equal(mp[F.ID_COLUMNS + F.FEATURES], mp2[F.ID_COLUMNS + F.FEATURES])
    # a held-out id in the training set stops the trainer
    held = [g for g in games if g["heldout"]][0]
    with pytest.raises(SystemExit):
        T.split([{**held, "heldout": False}])
    assert V.is_heldout(HELD) and V.is_heldout("01.18.2016.GSW.at.CLE") and not V.is_heldout("01.01.2016.AAA.at.BBB")


def test_xfg_predictions_interface(tmp_path):
    d = tmp_path / "analytics" / "xfg" / "2015-16"
    d.mkdir(parents=True)
    good = pd.DataFrame({"sportvu_game": ["01.01.2016.AAA.at.BBB", HELD], "event": [0, 0], "player_id": [1, 2], "p_selected": [0.5, 0.6], "fold": [0, -1]})
    good.to_parquet(d / "xfg_predictions.parquet", index=False)
    x = V.load_xfg(tmp_path)
    assert x is not None and len(x) == 2 and list(x.columns) == V.XFG_COLUMNS
    good.assign(fold=[-1, -1]).to_parquet(d / "xfg_predictions.parquet", index=False)   # not out-of-fold on a training row: refused
    assert V.load_xfg(tmp_path) is None
    assert V.load_xfg(tmp_path / "nowhere") is None
    assert V.load_xfg(None) is None
    # a game held out by flag whose id carries no token: the heldout column decides, so fold -1 there is not a refusal
    flagged = pd.DataFrame({"sportvu_game": ["01.01.2016.AAA.at.BBB", "01.02.2016.AAA.at.BBB"], "event": [0, 0], "player_id": [1, 2],
                            "p_selected": [0.5, 0.6], "fold": [0, -1], "heldout": [False, True]})
    flagged.to_parquet(d / "xfg_predictions.parquet", index=False)
    x = V.load_xfg(tmp_path)
    assert x is not None and len(x) == 2 and V.xfg_heldout_mask(x).tolist() == [False, True]
    assert T.default_probability(x) == pytest.approx(0.5)
    flagged.assign(fold=[-1, -1]).to_parquet(d / "xfg_predictions.parquet", index=False)      # a flagged-train row without a fold: refused
    assert V.load_xfg(tmp_path) is None
    flagged.assign(fold=[-2, -1]).to_parquet(d / "xfg_predictions.parquet", index=False)      # in-sample (-2) on a training row: refused too
    assert V.load_xfg(tmp_path) is None
    flagged.assign(fold=[np.nan, -1]).to_parquet(d / "xfg_predictions.parquet", index=False)
    assert V.load_xfg(tmp_path) is None


# ── end to end on the fixtures ─────────────────────────────────────────────────────────────────────
def test_end_to_end_fixture_run(ctx, games, tmp_path):
    assert ctx["train_ids"] == sorted(g["sportvu_game"] for g in games if not g["heldout"]) and ctx["held_ids"] == [HELD]
    assert HELD not in ctx["train_ids"]
    mp = ctx["tables"]["train"]["matchup_possessions"]
    assert list(mp.columns) == F.MATCHUP_COLUMNS and mp["partial_possession"].between(0, 1).all()
    assert mp.groupby(["sportvu_game", "possession", "def_player_id"])["partial_possession"].sum().le(1 + 1e-6).all()
    assert HELD not in set(mp["sportvu_game"])
    d = ctx["defenders"]
    assert list(d.columns) == T.DEFENDER_COLUMNS and d["insufficient"].dtype == bool
    assert (~d["insufficient"]).sum() == (d["n_possessions"] >= 5).sum() > 0
    assert np.isfinite(d["ridge_value"]).all() and np.isfinite(d["diff_per100"]).all()
    m = d["lo"].notna()                                   # no interval for a defender seen in one game only
    assert (d.loc[m, "lo"] <= d.loc[m, "diff_per100"] + 1e-9).all() and (d.loc[m, "hi"] >= d.loc[m, "diff_per100"] - 1e-9).all()
    assert d["diff_per100"].tolist() == pytest.approx((d["points_allowed_per100"] - d["expected_per100"]).tolist())
    assert ctx["cv"]["alpha"] in T.ALPHAS and ctx["cv"]["folds"] == min(3, len(ctx["train_ids"]))
    assert set(ctx["compare"]["mse"]) == set(T.MODELS) and all(np.isfinite(v["point"]) for v in ctx["compare"]["mse"].values())
    assert 0.5 < ctx["stability"]["point"] <= 1.0 and ctx["stability"]["lo"] <= ctx["stability"]["point"] <= ctx["stability"]["hi"]
    assert ctx["defended"]["n_near"] + ctx["defended"]["n_far"] > 0
    assert ctx["xfg"]["fold"].ge(0).sum() == (ctx["xfg"]["sportvu_game"] != HELD).sum()    # training attempts are out-of-fold
    assert (ctx["xfg"].loc[ctx["xfg"]["sportvu_game"] == HELD, "fold"] == -1).all()
    h = ctx["heldout"]
    assert h["games"] == [HELD] and h["n_possessions"] > 0 and h["candidate"] in ("raw", "ridge")
    assert set(h["mse"]) == {"constant", h["candidate"]} and all(np.isfinite(v) for v in h["mse"].values())
    assert h["candidate"] == T.selected_candidate(ctx["cv"]) and len(ctx["cv"]["fold_alphas"]) == ctx["cv"]["folds"]
    text = T.write_report(tmp_path / "matchups.md", ctx)
    assert "fixture output, not a result" in text and chr(0x2014) not in text
    for section in ("## Inputs", "## Assignment stability", "## Defended FG%", "## Baseline comparison", "## Defenders", "## Held-out", "## Status"):
        assert section in text


def test_eda_tables(ctx):
    t = ctx["tables"]["train"]
    pp = eda.partial_possession_distribution(t["matchup_possessions"])
    assert len(pp) == 10 and pp["n"].sum() == len(t["matchup_possessions"]) and pp["share"].sum() == pytest.approx(1.0)
    sw = eda.switches_per_possession(t["matchup_possessions"])
    assert sw["share"].sum() == pytest.approx(1.0) and (sw["switches"] >= 0).all()
    fg = eda.defended_fg_by_distance(t["shots"])
    assert fg["attempts"].sum() == t["shots"]["def_dist_ft"].notna().sum()
    sh = eda.shuffled_assignment_check(t["shots"], t["matchup_frames"], seed=0, n_shuffles=5)
    assert sh["kind"].tolist() == ["real assignment", "shuffled assignment"] and np.isfinite(sh["gap"]).all()
    nb = json.loads((Path(__file__).resolve().parents[1] / "notebooks" / "eda_matchups.ipynb").read_text())
    assert nb["nbformat"] == 4 and any("analytics.matchups" in "".join(c["source"]) for c in nb["cells"])


def test_validate_counts_and_cli(games, tmp_path):
    g = dict(games[0])
    fr = g["frames"].copy()
    fr = pd.concat([fr, fr.iloc[:7]], ignore_index=True)        # 7 duplicate (frame, player_id) rows
    fr.loc[fr.index[-10:-7], "x"] = np.nan                      # 3 missing positions
    g["frames"] = fr
    r = V.validate_game(g)
    assert r["counts"]["duplicate_rows"] == 7 and r["counts"]["missing_position"] == 3 and r["problems"] == []
    assert len(g["frames"]) == len(games[0]["frames"]) - 3      # duplicates dropped, the three rows without a position dropped
    bad = dict(games[0])
    bad["possessions"] = games[0]["possessions"].assign(end_kind="nope").drop(columns=["gap"])
    r = V.validate_game(bad)
    assert any("possessions" in p and "gap" in p for p in r["problems"]) and any("enum" in p for p in r["problems"])
    res = F.main(["--fixtures", "--out", str(tmp_path), "--seed", "3", "--n-games", "2", "--heldout-every", "2", "--n-possessions", "12"])
    written = sorted(p.name for p in (tmp_path / "analytics" / "matchups").rglob("*.parquet"))
    assert len(written) == 2 and any(HELD in n for n in written)
    assert (tmp_path / "analytics" / "matchups" / "heldout" / "2015-16" / (HELD + ".parquet")).exists()
    train_files = sorted((tmp_path / "analytics" / "matchups" / "2015-16").glob("*.parquet"))
    back = pd.read_parquet(train_files[0])
    assert len(train_files) == 1 and list(back.columns) == F.MATCHUP_COLUMNS and len(res["matchup_possessions"]) > 0
    table = V.main(["--fixtures", "--out", str(tmp_path), "--seed", "3", "--n-games", "1", "--n-possessions", "12"])
    assert (table["n_problems"] == 0).all() and (tmp_path / "analytics" / "matchups" / "validate_2015-16.json").exists()


def test_fixture_root_layout_loads(tmp_path):
    """The DATA_ROOT layout written by the fallback generator (the shared analytics.fixtures.write_fixture_root writes
    the same paths; it is not run here because a single 40-possession game takes seconds on a shared machine)."""
    FX._fallback_write_fixture_root(tmp_path, seed=5, n_games=2, heldout_every=2, n_possessions=10)
    games = V.load_games(tmp_path)
    assert len(games) == 2 and sorted(g["heldout"] for g in games) == [False, True]
    held = [g for g in games if g["heldout"]][0]
    assert held["sportvu_game"] == HELD and (tmp_path / "sportvu_contract" / "heldout" / (HELD + ".parquet")).exists()
    assert all(set(("frames", "events", "possessions", "shot_frames", "rosters")) <= set(g) for g in games)
    assert V.validate_games(games)["n_problems"].eq(0).all()


# ── review additions: basket vote, shot join, attribution, gaps, raw model, held-out alignment ───────────
def _frames_1hz(positions, period=1, start_clock=700.0, off_team=1):
    """positions: list per second of {player_id: (x, y)}; offence = player ids below 20 -> a frames table at 1 Hz."""
    rows = []
    for k, pos in enumerate(positions):
        for pid, (x, y) in pos.items():
            rows.append({"game": "g", "period": period, "frame": k, "game_clock_s": start_clock - k, "team_id": off_team if pid < 20 else 2,
                         "player_id": pid, "x": x, "y": y, "ball_x": x, "ball_y": y, "ball_z": 4.0})
    return pd.DataFrame(rows)


def _poss(no, period, sf, ef, start_clock, se=0, ee=0, points=0, off_team=1):
    return {"sportvu_game": "g", "possession": no, "period": period, "start_frame": sf, "end_frame": ef, "start_clock_s": start_clock,
            "end_clock_s": start_clock - (ef - sf), "start_event": se, "end_event": ee, "offense_team_id": off_team,
            "end_kind": "shot_made" if points else "shot_missed", "points": points, "gap": False, "n_frames": ef - sf + 1}


def test_basket_vote_backcourt_start():
    # possession 0: an 8 s walk-up from x = 10 to x = 70, then 3 s of play at x = 72; the centroid over every frame
    # sits in the left half, the last third sits at the right basket
    walk = [{11: (10 + 7.5 * k, 20.0), 12: (10 + 7.5 * k, 30.0), 21: (12 + 7.5 * k, 20.0), 22: (12 + 7.5 * k, 30.0)} for k in range(9)]
    play = [{11: (72.0, 20.0), 12: (72.0, 30.0), 21: (74.0, 20.0), 22: (74.0, 30.0)} for _ in range(3)]
    frames = _frames_1hz(walk + play)
    poss = pd.DataFrame([_poss(0, 1, 0, 11, 700.0)])
    _, info, _ = F.frame_assignments(frames, poss)
    assert info[0]["basket"] == (88.75, 25.0)
    # the majority over the team's possessions in the period overrides a lone possession that ends at the wrong end
    left = [{11: (20.0, 20.0), 12: (20.0, 30.0), 21: (22.0, 20.0), 22: (22.0, 30.0)} for _ in range(3)]
    right = [{11: (75.0, 20.0), 12: (75.0, 30.0), 21: (77.0, 20.0), 22: (77.0, 30.0)} for _ in range(3)]
    frames = _frames_1hz(left + right + right)
    poss = pd.DataFrame([_poss(0, 1, 0, 2, 700.0), _poss(1, 1, 3, 5, 697.0), _poss(2, 1, 6, 8, 694.0)])
    _, info, _ = F.frame_assignments(frames, poss)
    assert [info[i]["basket"] for i in range(3)] == [(88.75, 25.0)] * 3
    # another period votes on its own
    poss2 = poss.copy()
    poss2.loc[0, "period"] = 2
    frames2 = frames.copy()
    frames2.loc[frames2["frame"] <= 2, "period"] = 2
    _, info, _ = F.frame_assignments(frames2, poss2)
    assert info[0]["basket"] == (5.25, 25.0) and info[1]["basket"] == (88.75, 25.0)


def test_shot_value_description_then_distance():
    assert F.shot_value("Curry makes 3-pt jump shot from 26 ft", 10.0) == 3
    assert F.shot_value("Green misses 2-pt layup", 25.0) == 2
    assert F.shot_value(None, 23.0) == 3 and F.shot_value("", 21.9) == 2
    assert F.shot_value("Thompson makes jump shot", float("nan")) == 2 and F.shot_value(None, None) == 2


def _two_period_game():
    play = [{11: (72.0, 20.0), 12: (72.0, 30.0), 21: (74.0, 20.0), 22: (74.0, 30.0)} for _ in range(6)]
    frames = _frames_1hz(play)
    frames.loc[frames["frame"] >= 3, "period"] = 2
    poss = pd.DataFrame([_poss(0, 1, 0, 2, 700.0, se=0, ee=1, points=2), _poss(1, 2, 3, 5, 700.0, se=0, ee=1, points=3)])
    events = pd.DataFrame([{"event": 0, "period": 1, "clock_s": 699.0, "team": "home", "desc": "A misses 2-pt layup", "points": 0, "kind": "shot_missed", "frame": 1},
                           {"event": 1, "period": 1, "clock_s": 698.0, "team": "home", "desc": "B makes 2-pt layup", "points": 2, "kind": "shot_made", "frame": 2},
                           {"event": 0, "period": 2, "clock_s": 699.0, "team": "home", "desc": "A makes 3-pt jump shot", "points": 3, "kind": "shot_made", "frame": 4},
                           {"event": 2, "period": 1, "clock_s": 690.0, "team": "home", "desc": "A misses 2-pt layup", "points": 0, "kind": "shot_missed", "frame": -1}])
    sf_row = lambda ev, per, made, shooter, rel: {"sportvu_game": "g", "event": ev, "period": per, "clock_s": 699.0, "made": made, "pbp_shooter_name": "x",
                                                  "shooter_id": shooter, "release_frame": rel, "apex_frame": rel, "outcome_frame": rel,
                                                  "shooter_at_release_id": shooter, "identity_match": True, "confidence": 0.9, "confidence_flag": "high"}
    sf = pd.DataFrame([sf_row(0, 1, False, 11, 1), sf_row(1, 1, True, 12, 2), sf_row(0, 2, True, 11, 4), sf_row(2, 1, False, 11, -1)])
    return {"sportvu_game": "g", "heldout": False, "frames": frames, "events": events, "possessions": poss, "shot_frames": sf}


def test_shot_join_by_event_and_period_and_points_to_last_shooter():
    g = _two_period_game()
    xfg = pd.DataFrame({"sportvu_game": ["g"] * 3, "event": [0, 1, 0], "p_selected": [0.5, 0.5, 0.5]})
    r = F.build(g, xfg)
    s = r["shots"].sort_values(["possession", "event"])
    assert s["possession"].tolist() == [0, 0, 1]                    # event 1 of period 1 is the boundary (end_event); event 2 joins nothing
    assert s["event"].tolist() == [0, 1, 0] and s["value"].tolist() == [2, 2, 3]
    mp = r["matchup_possessions"]
    p0 = mp[mp["possession"] == 0]
    assert p0.loc[p0["off_player_id"] == 12, "points_allowed_share"].sum() == pytest.approx(2.0)      # the last attempt's shooter
    assert p0.loc[p0["off_player_id"] == 11, "points_allowed_share"].sum() == 0.0
    assert p0.loc[p0["off_player_id"] == 11, "xfg_expected_share"].sum() == pytest.approx(1.0)        # 0.5 * 2 for its own attempt
    p1 = mp[mp["possession"] == 1]
    assert p1.loc[p1["off_player_id"] == 11, "points_allowed_share"].sum() == pytest.approx(3.0)
    # a null shooter on the last attempt attributes nothing
    g["shot_frames"].loc[1, "shooter_id"] = np.nan
    g["shot_frames"]["shooter_id"] = g["shot_frames"]["shooter_id"].astype("Int32")
    mp = F.build(g, xfg)["matchup_possessions"]
    assert mp.loc[mp["possession"] == 0, "points_allowed_share"].sum() == 0.0


def test_sample_frames_across_clock_gap():
    clocks = np.r_[700.0 - np.arange(51) / 25.0, 690.0 - np.arange(26) / 25.0]        # 2 s, an 8 s gap, 1 s
    fl = pd.DataFrame({"frame": np.arange(len(clocks)), "game_clock_s": clocks})
    assert F.sample_frames(fl, 700.0).tolist() == [0, 25, 50, 51, 76]


def _mp(rows):
    return pd.DataFrame(rows, columns=F.MATCHUP_COLUMNS)


def test_raw_model_and_predict_across_games():
    mp = _mp([("A", 0, 1, 11, 1.0, 2.0, 1.0, 0), ("A", 1, 1, 12, 1.0, 0.0, 0.9, 0),
              ("B", 0, 1, 11, 0.5, 1.0, 0.4, 1), ("B", 0, 2, 11, 0.5, 0.0, 0.4, 1)])
    m = T.raw_model(mp)
    assert m["rate"][1] == pytest.approx(3.0 / 2.5) and m["rate"][2] == 0.0 and m["mean_rate"] == pytest.approx(1.0)
    test = _mp([("C", 0, 1, 11, 1.0, 0.0, 0.0, 0), ("C", 1, 9, 11, 1.0, 0.0, 0.0, 0),
                ("C", 2, 1, 11, 0.5, 0.0, 0.0, 0), ("C", 2, 2, 12, 0.5, 0.0, 0.0, 0)])
    pred = T.raw_predict(m, test, [("C", 0), ("C", 1), ("C", 2), ("C", 3)])
    assert pred.tolist() == pytest.approx([1.2, 1.0, 0.6, 0.0])          # seen, unseen (mean rate), mixed, absent possession


def test_score_heldout_aligns_unseen_players():
    rng = np.random.default_rng(0)
    rows = []
    for g in ("A", "B", "C"):
        for p in range(12):
            for d, o in ((1, 11), (2, 12)):
                rows.append((g, p, d, o, 0.5, float(rng.integers(0, 3)) * 0.5, 0.5, 0))
    mp = _mp(rows)
    points = {(g, p): int(mp[(mp["sportvu_game"] == g) & (mp["possession"] == p)]["points_allowed_share"].sum() * 2) for g in "ABC" for p in range(12)}
    cv = T.ridge_cv(mp, points, n_folds=3, seed=0)
    assert len(cv["design"]["columns"]) == 4 and cv["pred"][["p_constant", "p_raw", "p_ridge"]].notna().all().all()
    held_mp = _mp([("H", 0, 1, 11, 0.5, 0.0, 0.5, 0), ("H", 0, 2, 12, 0.5, 0.0, 0.5, 0),
                   ("H", 1, 77, 88, 1.0, 0.0, 0.5, 0)])                                  # possession 1: players unseen in training
    held_games = [{"possessions": pd.DataFrame({"sportvu_game": ["H", "H"], "possession": [0, 1], "points": [2, 0]})}]
    tables = {"matchup_possessions": held_mp, "matchup_frames": pd.DataFrame(columns=F.FRAME_COLUMNS), "shots": pd.DataFrame(columns=F.SHOT_COLUMNS)}
    h = T.score_heldout(tables, held_games, cv, 0, 0)
    assert h["n_possessions"] == 2 and set(h["mse"]) == {"constant", h["candidate"]} and all(np.isfinite(v) for v in h["mse"].values())
    D = T.possession_design(held_mp, T.possession_points(held_games))
    assert D["columns"] == ["D1", "D2", "D77", "O11", "O12", "O88"]
    # the unseen possession is carried by the intercept alone
    X = cv["design"]["X"][:0]
    assert cv["model"].predict(np.zeros((1, X.shape[1])))[0] == pytest.approx(float(cv["model"].intercept_))
