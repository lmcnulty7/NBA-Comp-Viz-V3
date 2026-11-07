"""tests/test_rebounds.py: analytics.rebounds on synthetic games of analytics.fixtures (through the test-only
adapter tests/_fx_rebounds.py). End-to-end fixture run, leakage guards, metric hand checks, input rules. No real
tracking data; every number is fixture output. Sizes are small so the file stays well under 30 s.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fx_rebounds as fx  # noqa: E402
from analytics import schema  # noqa: E402
from analytics.rebounds import eda, features as F, train as T, validate as V  # noqa: E402

HELD = "12.16.2015.PHX.at.GSW"
N_GAMES, N_POSS, N_BOOT = 4, 8, 10


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One end-to-end fixture run of the trainer (4 games, every fourth held out, in-process)."""
    d = tmp_path_factory.mktemp("rebounds")
    ctx = T.main(["--fixtures", "--games", str(N_GAMES), "--possessions", str(N_POSS), "--bootstrap", str(N_BOOT),
                  "--folds", "3", "--report", str(d / "rebounds.md"), "--out", str(d / "out")])
    ctx["dir"] = d
    return ctx


# ── end to end ─────────────────────────────────────────────────────────────────────────────────────
def test_end_to_end_fixture_run_report_and_predictions(run):
    text = (run["dir"] / "rebounds.md").read_text()
    assert "Fixture output, not a result" in text and chr(0x2014) not in text
    for m in T.MODELS:
        assert ("| %s" % m) in text
    assert run["selected"] in T.MODELS
    assert run["train_games"] == N_GAMES - 1 and run["heldout_game_ids"] == [HELD]
    assert set(run["heldout_scores"]) == {run["selected"], "uniform floor"}   # only the selected candidate is scored on held-out rows
    assert run["mode"] == "out-of-fold" and run["folds"] == 3
    pred = run["predictions"]
    assert list(pred.columns) == T.PRED_COLUMNS
    tr = pred[~pred["heldout"]]
    sums = tr.groupby(["sportvu_game", "event"])[["p_prior", "p_logit", "p_selected"]].sum()
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-4)
    hd = pred[pred["heldout"]]
    assert np.allclose(hd.groupby(["sportvu_game", "event"])["p_selected"].sum().to_numpy(), 1.0, atol=1e-4)
    for m in ("prior", "logit"):                                    # non-selected models are never scored on held-out rows
        if m != run["selected"]:
            assert hd["p_" + m].isna().all()
        else:
            assert np.allclose(hd["p_" + m].to_numpy(), hd["p_selected"].to_numpy())
    assert (pred.groupby(["sportvu_game", "event"])["is_rebounder"].sum() == 1).all()
    assert (pred.loc[pred["heldout"], "fold"] == T.HELDOUT_FOLD).all() and (pred.loc[~pred["heldout"], "fold"] >= 0).all()
    assert (pred.loc[pred["heldout"], "sportvu_game"] == HELD).all()
    paths = [Path(p) for p in run["written_paths"]]
    assert len(paths) == N_GAMES and all(p.exists() for p in paths)
    held_paths = [p for p in paths if p.stem == HELD]
    assert held_paths and held_paths[0].parts[-4:-1] == ("rebounds", "heldout", "2015-16")
    back = pd.read_parquet(held_paths[0])
    assert list(back.columns) == T.PRED_COLUMNS and back["heldout"].all()
    assert back["p_selected"].notna().all()
    only = T.fit_candidates(run["train"], 0, only=run["selected"])
    assert set(only) == {run["selected"]}
    for m in T.MODELS:                                             # every model beats or matches the uniform reference on fixture rows
        assert np.isfinite(run["scores"][m]["mlogloss"]["point"])
    assert run["scores"][run["selected"]]["mlogloss"]["point"] <= run["uniform"]["mlogloss"] + 0.05


def test_feature_frame_is_time_forward_and_clean(run):
    train = run["train"]
    schema.assert_time_forward(train, "rebound_frames")
    assert set(F.STATIC_FEATURES) <= set(train.columns) and F.feature_columns()[-1] == F.PRIOR
    assert train["prior_rate"].isna().all()                        # filled per fold, never stored on the frame
    assert not train["heldout"].any()
    pl = train[train["is_none"] == 0]
    assert ((pl["distance_to_rim_ft"] >= 0) & (pl["distance_to_rim_ft"] <= 110)).all()
    assert pl["is_offense"].isin([0, 1]).all() and (pl["n_opponents_closer_to_rim"] <= 5).all()
    assert (train.groupby(["sportvu_game", "event"])["is_none"].sum() == 1).all()


# ── leakage guards ─────────────────────────────────────────────────────────────────────────────────
def test_leakage_guards_refuse_targets_heldout_rows_and_heldout_ids(run):
    train = run["train"]
    bad = train.copy(); bad["rebounder_id"] = 1
    with pytest.raises(ValueError):
        schema.assert_time_forward(bad, "rebound_frames")
    with pytest.raises(ValueError):
        F.assert_no_leak(train.assign(rebound_frame=0).drop(columns=[F.TARGET]))
    with pytest.raises(ValueError):
        F.assert_no_leak(pd.DataFrame(columns=F.feature_columns() + ["pbp_rebounder_id"]))
    leaky = train.copy(); leaky.loc[leaky.index[:11], "heldout"] = True
    with pytest.raises(ValueError):
        T.fit_candidates(leaky)
    renamed = train.copy()
    renamed.loc[renamed["sportvu_game"] == renamed["sportvu_game"].iat[0], "sportvu_game"] = HELD
    with pytest.raises(SystemExit):                                # the guard trips on the id even with heldout False
        T.fit_candidates(renamed)
    cle = train.copy()
    cle.loc[cle["sportvu_game"] == cle["sportvu_game"].iat[0], "sportvu_game"] = "01.01.2016.AAA.at.CLE"
    with pytest.raises(SystemExit):                                # the held-out arena, by id suffix
        T.fit_candidates(cle)
    assert run["train_game_ids"] and HELD not in run["train_game_ids"]
    for c in ("rebounder_id", "pbp_rebounder_id", "pbp_rebounder_name", "rebound_frame", "identity_match", "confidence"):
        assert c not in train.columns and c not in F.feature_columns()


def test_prior_is_out_of_fold(run):
    train = run["train"]
    games = sorted(train["sportvu_game"].unique())
    prior, note = T.oof_prior(train, seed=0)
    assert note.startswith("out of fold")
    g0 = train["sportvu_game"] == games[0]
    expected = T.apply_prior(train[g0], T.player_rates(train[~g0]))
    if len(games) == T.INNER_FOLDS or len(games) < T.INNER_FOLDS:   # leave-one-game-out: the other games are the fit
        assert np.allclose(prior[g0.to_numpy()], expected)
    rates = T.player_rates(train)
    assert 0 < rates["none"] < 1 and all(0 <= v <= 1 for v in rates["player"].values())
    p = T.normalise_within(T.apply_prior(train, rates), T.miss_key(train))
    assert np.allclose(pd.Series(p).groupby(T.miss_key(train)).sum(), 1.0)


# ── metrics by hand ────────────────────────────────────────────────────────────────────────────────
def test_metric_hand_check():
    pred = pd.DataFrame({"sportvu_game": ["g"] * 6 + ["h"] * 3, "event": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                         "candidate_player_id": [1, 2, -1] * 3,
                         F.TARGET: [True, False, False, False, True, False, False, False, True],
                         "p": [0.5, 0.25, 0.25, 0.25, 0.25, 0.5, 0.4, 0.4, 0.2]})
    pm = T.per_miss(pred, "p")
    assert T.mlogloss(pm) == pytest.approx(-(np.log(0.5) + np.log(0.25) + np.log(0.2)) / 3)
    assert pm["hit"].tolist() == [1.0, 0.0, 0.0] and pm["p_top"].tolist() == [0.5, 0.5, 0.4]
    assert pm["top_is_rebounder"].tolist() == [True, False, False]
    assert T.top1(pm) == pytest.approx(1 / 3)
    assert T.uniform_reference(pm) == {"mlogloss": pytest.approx(np.log(3)), "top1": pytest.approx(1 / 3)}
    tie = pred.copy(); tie["p"] = [0.4, 0.4, 0.2] * 3
    assert T.per_miss(tie, "p")["hit"].tolist() == [0.5, 0.5, 0.0]    # expected top-1 under random tie-breaking
    key = T.miss_key(pred)
    assert key.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    sm = T.softmax_within(np.log(pred["p"].to_numpy()), key)
    assert np.allclose(sm, pred["p"].to_numpy())
    assert np.allclose(T.normalise_within([0, 0, 0, 2, 2, 0, 1, 0, 0], key), [1 / 3, 1 / 3, 1 / 3, .5, .5, 0, 1, 0, 0])
    two = pred[pred["event"] < 3].copy(); two[F.TARGET] = [True, True, False, False, True, False]
    with pytest.raises(ValueError):
        T.per_miss(two, "p")
    sc = T.score(pred.assign(p_prior=pred["p"], p_logit=pred["p"], p_hgb=pred["p"]), n_boot=10, seed=0)
    assert sc["hgb"]["mlogloss"]["point"] == pytest.approx(T.mlogloss(pm))
    assert T.select(sc) == "prior"                                   # ties resolve to the first, the simplest model


# ── input rules ────────────────────────────────────────────────────────────────────────────────────
def _rf(rows):
    base = {"sportvu_game": "01.01.2016.AAA.at.BBB", "period": 1, "clock_s": 700.0, "rebound_event": 5, "rim_frame": 10,
            "rebound_frame": 15, "rebounder_id": 1, "pbp_rebounder_name": "A. One", "pbp_rebounder_id": 1,
            "identity_match": True, "confidence": 0.9, "confidence_flag": "high"}
    df = pd.DataFrame([{**base, "event": i, **r} for i, r in enumerate(rows)])
    return schema.cast(df, "rebound_frames")


def test_select_misses_is_symmetric_and_counts_the_rest():
    rf = _rf([{},                                                                                              # 0 player, located
              {"rim_frame": -1, "rebound_frame": -1, "rebounder_id": None, "identity_match": None, "confidence_flag": "unmatched"},  # 1 no rim frame
              {"rebound_event": None, "identity_match": None, "pbp_rebounder_id": None, "pbp_rebounder_name": None, "confidence_flag": "unmatched"},  # 2 no rebound event
              {"rebounder_id": None, "rebound_frame": -1, "pbp_rebounder_id": None, "pbp_rebounder_name": None, "identity_match": None, "confidence_flag": "unmatched"},  # 3 team
              {"pbp_rebounder_id": None, "identity_match": None, "confidence_flag": "unmatched"},               # 4 named, roster join failed
              {"rebounder_id": None, "rebound_frame": -1, "identity_match": None, "confidence_flag": "unmatched"},  # 5 player, frame not located
              {"rebounder_id": 2, "identity_match": False, "confidence_flag": "low"},                           # 6 player, localiser disagrees
              {"confidence_flag": "unmatched"}])                                                                # 7 player, flag only
    kept, c = F.select_misses(rf)
    assert kept["event"].tolist() == [0, 3, 5, 6, 7] and kept["label_id"].tolist() == [1, F.NONE_ID, 1, 1, 1]
    assert c["no_rim_frame"] == 1 and c["no_rebound_event"] == 1 and c["unmatched_named_rebounder"] == 1
    assert c["team_rebounds_kept"] == 1 and c["player_rebounds_kept"] == 4
    assert c["identity_match_null_kept"] == 1 and c["identity_match_false_kept"] == 1 and c["unmatched_flag_kept"] == 1
    assert {"no_rim_frame", "no_rebound_event", "unmatched_named_rebounder"} <= set(c) and "identity_match_null" not in c
    assert set(F.INFO_COUNTS) <= set(c)                              # build_game adds the frame-level drop reasons
    assert "rebounder_id" not in kept.columns and "pbp_rebounder_id" not in kept.columns


# ── hand-built frames: windows and signs ───────────────────────────────────────────────────────────
HB_GAME, HB_RIM, HB_FPS = "01.01.2016.AAA.at.BBB", 50, 25
HB_BASKET = (88.75, 25.0)


def _hand_frames(drop_lookback: bool = False) -> pd.DataFrame:
    """Four players, one period, frames 0..70 at 25 Hz, rim frame 50 with the ball at the home basket. Player 1001
    closes on the rim at 10 ft/s along the court axis, everyone else stands still; the ball rises to 16 ft before
    the rim frame. With drop_lookback the frames 36..49 are missing, so no frame sits within 0.2 s of 0.4 s
    before the rim frame."""
    rows = []
    for f in range(71):
        if drop_lookback and 36 <= f <= 49:
            continue
        t = (f - HB_RIM) / HB_FPS
        ball_z = 16.0 - 60.0 * t * t if f <= HB_RIM else 3.0
        for pid, home, x, y in ((1001, True, 78.75 + 10.0 * t, 25.0), (1002, True, 80.0, 30.0),
                                (2001, False, 84.0, 22.0), (2002, False, 70.0, 40.0)):
            rows.append({"game": HB_GAME, "period": 1, "frame": f, "game_clock_s": 700.0 - f / HB_FPS,
                         "ball_x": 88.75 if f >= 30 else 60.0, "ball_y": 25.0, "ball_z": ball_z, "team_id": 100 if home else 200,
                         "home": home, "player_id": pid, "player": "P %d" % pid, "x": float(x), "y": float(y)})
    return pd.DataFrame(rows)


def _hand_rebound_frames(label_player, team: bool = False) -> pd.DataFrame:
    row = {"rim_frame": HB_RIM, "rebound_frame": HB_RIM + 8, "rebounder_id": label_player, "pbp_rebounder_name": "P",
           "pbp_rebounder_id": label_player, "identity_match": True, "confidence": 0.9, "confidence_flag": "high"}
    if team:
        row.update({"rebound_frame": -1, "rebounder_id": None, "pbp_rebounder_name": None, "pbp_rebounder_id": None,
                    "identity_match": None, "confidence": 0.0, "confidence_flag": "unmatched"})
    return _rf([row])


def _hand_events() -> pd.DataFrame:
    return pd.DataFrame({"game": [HB_GAME, HB_GAME], "event": [0, 5], "period": [1, 1], "clock_s": [698.0, 697.5],
                         "team": ["home", "away"], "desc": ["P misses", "rebound"], "kind": ["shot_missed", "rebound"], "frame": [HB_RIM, HB_RIM + 8]})


def test_features_use_only_frames_at_or_before_the_rim_frame():
    frames, rf, ev = _hand_frames(), _hand_rebound_frames(1001), _hand_events()
    df, c = F.build_game(frames, rf, ev)
    assert c["misses_kept"] == 1 and len(df) == 5
    p1 = df[df["candidate_player_id"] == 1001].iloc[0]
    assert p1["distance_to_rim_ft"] == pytest.approx(10.0) and p1["speed_toward_rim_ftps"] == pytest.approx(10.0, abs=1e-6)
    assert p1["is_offense"] == 1 and p1["is_rebounder"] and p1["angle_to_rim"] == pytest.approx(0.0, abs=1e-6)
    assert p1["ball_apex_ft"] == pytest.approx(16.0) and p1["distance_to_ball_landing_ft"] == pytest.approx(10.0)
    assert p1["nearest_opponent_dist_ft"] == pytest.approx(np.hypot(84.0 - 78.75, 3.0)) and p1["n_opponents_closer_to_rim"] == 1
    p2 = df[df["candidate_player_id"] == 2002].iloc[0]
    assert p2["is_offense"] == 0 and p2["speed_toward_rim_ftps"] == pytest.approx(0.0, abs=1e-6) and p2["n_opponents_closer_to_rim"] == 2
    assert p1["shot_distance_ft"] == pytest.approx(28.75)               # fallback: the ball 1 s before the rim frame (frame 25) sits at x = 60
    # corrupt everything after the rim frame: no static feature may move
    bad = frames.copy()
    after = bad["frame"] > HB_RIM
    bad.loc[after, ["x", "y"]] = 5.0
    bad.loc[after, ["ball_x", "ball_y"]] = 10.0
    bad.loc[after, "ball_z"] = 40.0
    df2, _ = F.build_game(bad, rf, ev)
    pd.testing.assert_frame_equal(df[F.STATIC_FEATURES].reset_index(drop=True), df2[F.STATIC_FEATURES].reset_index(drop=True))
    # no frame within 0.2 s of the 0.4 s look-back: speed is NaN, everything else unchanged
    df3, _ = F.build_game(_hand_frames(drop_lookback=True), rf, ev)
    assert df3[df3["is_none"] == 0]["speed_toward_rim_ftps"].isna().all()
    others = [f for f in F.STATIC_FEATURES if f != "speed_toward_rim_ftps"]
    pd.testing.assert_frame_equal(df[others].reset_index(drop=True), df3[others].reset_index(drop=True))
    # a moved apex before the rim frame does change the feature (the window is live, not constant)
    moved = frames.copy(); moved.loc[moved["frame"] == 40, "ball_z"] = 20.0
    assert F.build_game(moved, rf, ev)[0]["ball_apex_ft"].iloc[0] == pytest.approx(20.0)


def test_team_rebound_puts_the_target_on_the_none_row_and_offense_falls_back_to_possessions():
    frames, rf, ev = _hand_frames(), _hand_rebound_frames(None, team=True), _hand_events()
    df, c = F.build_game(frames, rf, ev)
    assert c["team_rebounds_kept"] == 1 and len(df) == 5
    none = df[df["candidate_player_id"] == F.NONE_ID].iloc[0]
    assert none["is_rebounder"] and none["is_none"] == 1 and df["is_rebounder"].sum() == 1
    assert none["ball_apex_ft"] == pytest.approx(16.0) and none["distance_to_rim_ft"] == 0.0
    # no event team: the possession with the largest start_frame at or before the rim frame gives the offence
    poss = pd.DataFrame({"start_frame": [0, 20, 60], "offense_team_id": [200, 200, 100]})
    df4, _ = F.build_game(frames, rf, ev.drop(columns=["team"]), possessions=poss)
    assert (df4.loc[df4["candidate_player_id"].isin([2001, 2002]), "is_offense"] == 1).all()
    assert (df4.loc[df4["candidate_player_id"].isin([1001, 1002]), "is_offense"] == 0).all()
    _, c5 = F.build_game(frames, rf, ev.drop(columns=["team"]))
    assert c5["offense_unknown"] == 1 and c5["misses_kept"] == 0


def test_validate_range_and_duplicate_checks():
    g = fx.synth_game(3, n_possessions=4)
    frames = g["frames"].copy()
    frames.loc[0, "x"] = 120.0                                        # off the court
    frames.loc[1, "game_clock_s"] = 800.0                             # clock out of range
    frames = pd.concat([frames, frames.iloc[[2]]], ignore_index=True) # duplicate (frame, player_id)
    keep, c = V.check_frames(frames)
    assert c["player_out_of_court"] == 1 and c["clock_out_of_range"] == 1 and c["duplicate_frame_player"] == 1
    assert c["rows_in"] == len(frames) and c["rows_out"] == len(frames) - 3 and keep.sum() == c["rows_out"]
    keep, c = V.check_table(g["rebound_frames"], "rebound_frames")
    assert c["schema_problems"] == [] and c["duplicate_key"] == 0 and c["rows_out"] == len(g["rebound_frames"])
    bad = g["rebound_frames"].copy(); bad["confidence_flag"] = "maybe"
    assert V.check_table(bad, "rebound_frames")[1]["schema_problems"]
    b = {"sportvu_game": str(frames["game"].iat[0]), "heldout": False, "frames": frames, "events": g["events"],
         "rebound_frames": g["rebound_frames"], "shot_frames": g["shot_frames"], "rosters": g["rosters"], "possessions": g["possessions"]}
    cb, counts = V.clean_bundle(b)
    assert len(cb["frames"]) == counts["frames"]["rows_out"]
    md = V.markdown(V.summarise([counts]), fixtures=True)
    assert "Fixture output, not a result" in md and chr(0x2014) not in md


def _write_root(root: Path, games: list) -> None:
    for g in games:
        gid = str(g["frames"]["game"].iat[0])
        held = gid == HELD
        sv = root / "sportvu_contract" / ("heldout" if held else "")
        sv.mkdir(parents=True, exist_ok=True)
        g["frames"].to_parquet(sv / (gid + ".parquet"), index=False)
        g["events"].to_parquet(sv / (gid + "_pbp.parquet"), index=False)
        for t in ("rebound_frames", "shot_frames", "rosters", "possessions"):
            p = schema.table_path(root / "analytics", t, "2015-16", gid, held)
            p.parent.mkdir(parents=True, exist_ok=True)
            g[t].to_parquet(p, index=False)


def test_data_root_layout_is_read_with_heldout_detected(tmp_path):
    games = fx.synth_season(1, 2, 2, 6)
    root = tmp_path / "root"
    _write_root(root, games)
    listed = V.list_root(root)
    assert sorted((g, h) for g, h, _ in listed) == sorted((str(g["frames"]["game"].iat[0]), str(g["frames"]["game"].iat[0]) == HELD) for g in games)
    summary = V.main(["--data", str(root), "--out", str(tmp_path / "out")])
    assert summary["games"] == 2 and summary["heldout_games"] == 1 and (tmp_path / "out" / "analytics" / "rebounds" / "validate.json").exists()
    ctx = T.main(["--data", str(root), "--report", str(tmp_path / "r.md"), "--bootstrap", "5"])
    assert ctx["mode"] == "in-sample" and ctx["train_games"] == 1 and ctx["heldout_game_ids"] == [HELD]
    assert (ctx["predictions"].loc[~ctx["predictions"]["heldout"], "fold"] == T.IN_SAMPLE_FOLD).all()
    assert "Fixture output" not in (tmp_path / "r.md").read_text().split("## Inputs")[0]


def test_features_cli_writes_one_file_per_game(tmp_path):
    res = F.main(["--fixtures", "--fixture-games", "2", "--fixture-possessions", "4", "--out", str(tmp_path)])
    assert len(res["paths"]) == 2 and all(Path(p).exists() for p in res["paths"])
    assert all("/analytics/rebounds/features/" in p for p in res["paths"])
    back = pd.read_parquet(res["paths"][0])
    assert set(F.STATIC_FEATURES) <= set(back.columns) and F.TARGET in back.columns
    schema.assert_time_forward(back, "rebound_frames")
    assert res["counts"]["total"]["rows"] == len(res["candidates"])


# ── eda ────────────────────────────────────────────────────────────────────────────────────────────
def test_eda_tables_and_shuffled_label_check(run):
    train = run["train"]
    side = eda.rebound_share_by_side(train)
    assert side["side"].tolist() == ["offense", "defense", "none"] and side["share"].sum() == pytest.approx(1.0)
    assert side["misses"].sum() == run["train_misses"]
    dist = eda.rebound_share_by_distance_bin(train)
    assert set(dist.columns) == {"distance_bin", "is_offense", "rows", "rebounds", "rebound_rate"} and dist["rows"].sum() == (train["is_none"] == 0).sum()
    d = eda.feature_distributions(train)
    assert list(d.index) == F.STATIC_FEATURES and {"mean_rebounder", "mean_other"} <= set(d.columns)
    c = eda.correlation_table(train)
    assert c.shape == (len(F.STATIC_FEATURES) + 1,) * 2 and np.allclose(np.diag(c.fillna(1.0)), 1.0)
    sh = eda.shuffled_label_check(train, seed=0, n_folds=2)
    assert len(sh) == 2 * len(T.MODELS) and set(sh["labels"]) == {"real", "shuffled"}
    shuffled = sh[sh["labels"] == "shuffled"]
    assert (shuffled["per_miss_log_loss"] >= shuffled["uniform_log_loss"] - 0.15).all()   # nothing to learn from permuted labels
    md = eda.markdown(side)
    assert md.startswith("| side | misses | share |") and chr(0x2014) not in md
