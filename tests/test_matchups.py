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
    # equidistant to both offensive players: the midpoint term picks the one whose basket side the defender is on
    off = np.array([[30.0, 10.0], [30.0, 40.0]])
    dfn = np.array([[24.0, 25.0]])                          # between them, nearer the basket on the y = 25 line: tie on d1
    cost = F.assignment_cost(dfn, off, basket)
    assert cost.shape == (1, 2) and cost[0, 0] == pytest.approx(cost[0, 1])
    dfn = np.array([[24.0, 24.0]])                          # 1 ft toward off 0: both terms favour off 0
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
    assert h["games"] == [HELD] and h["n_possessions"] > 0 and all(np.isfinite(h["mse"][m]) for m in T.MODELS)
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
    res = F.main(["--fixtures", "--out", str(tmp_path), "--seed", "3", "--n-games", "2", "--n-possessions", "12"])
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
