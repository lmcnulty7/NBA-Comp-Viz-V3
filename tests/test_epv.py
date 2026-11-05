"""tests/test_epv.py: analytics/epv on the fixtures of tests/_fx_epv.py (the shared analytics.fixtures when it is
importable, the local fallback otherwise, and the fallback explicitly). No network, no raw data, under 30 s."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics import schema  # noqa: E402
from analytics.epv import eda, features as F, states as S, train as T, validate as V  # noqa: E402
from tests import _fx_epv as fx  # noqa: E402

T.MLP_ITER = 40          # the gated sequence model stays cheap in tests
HELD = "12.16.2015.PHX.at.GSW"


@pytest.fixture(scope="module")
def season():
    """The season as the trainer loads it: 3 training games and 1 held-out, from whichever generator is active."""
    return V.load_fixtures(0, n_games=4, heldout_every=4)


@pytest.fixture(scope="module")
def fallback_game():
    return fx.fallback_synth_game(1, n_possessions=20)


# ── fixtures and the states builder ────────────────────────────────────────────────────────────────
def test_fallback_tables_meet_the_contract(fallback_game):
    g = fallback_game
    for table in ("rosters", "shot_frames", "rebound_frames", "possessions", "states_1hz"):
        assert schema.validate(g[table], table) == [], table
    assert len(g["possessions"]) == 20 and g["states_1hz"]["possession"].nunique() == 20
    assert {"sportvu_game", "player_slug", "loc_x", "loc_y", "made", "heldout"} <= set(g["shots"].columns)
    assert fx.fallback_synth_game(1, n_possessions=20)["states_1hz"].equals(g["states_1hz"])   # deterministic


def _check_states_against_possessions(st, poss):
    p = poss.set_index("possession")
    assert schema.validate(st, "states_1hz") == []
    assert set(st["possession"]) <= set(p.index)
    lo = p.loc[st["possession"], "start_frame"].to_numpy()
    hi = p.loc[st["possession"], "end_frame"].to_numpy()
    assert ((st["frame"].to_numpy() >= lo) & (st["frame"].to_numpy() <= hi)).all()
    el = st["elapsed_in_possession_s"].to_numpy(dtype=float)
    assert np.allclose(el, np.round(el)) and (el >= 0).all()
    clock_el = p.loc[st["possession"], "start_clock_s"].to_numpy(dtype=float) - st["game_clock_s"].to_numpy(dtype=float)
    assert np.abs(clock_el - el).max() <= 0.5 + 1e-3
    first = st.sort_values("elapsed_in_possession_s").drop_duplicates("possession")
    assert (first["points_to_end"].to_numpy() == p.loc[first["possession"], "points"].to_numpy()).all()
    assert (st["possession_end_kind"].astype(str).to_numpy() == p.loc[st["possession"], "end_kind"].astype(str).to_numpy()).all()
    assert (st["offense_team_id"].to_numpy() == p.loc[st["possession"], "offense_team_id"].to_numpy()).all()
    off = st[["off%d_id" % i for i in range(1, 6)]].to_numpy()
    de = st[["def%d_id" % i for i in range(1, 6)]].to_numpy()
    assert (np.diff(off, axis=1) > 0).all() and (np.diff(de, axis=1) > 0).all()
    h = st["handler_id"].dropna().to_numpy()
    assert np.isin(h, off[st["handler_id"].notna().to_numpy()]).all()
    assert not st.duplicated(["possession", "frame"]).any()


def test_states_builder_is_consistent_with_possessions_on_the_fallback(fallback_game):
    g = fallback_game
    st = S.build_states(g["frames"], g["possessions"], g["sportvu_game"], events=g["events"])
    _check_states_against_possessions(st, g["possessions"])
    assert st.attrs["n_states"] == len(st) and st.attrs["n_skipped_split"] == 0
    # every possession has a state at elapsed 0 and one per whole second of its clock span
    dur = (g["possessions"]["start_clock_s"] - g["possessions"]["end_clock_s"]).to_numpy(dtype=float)
    assert st.groupby("possession").size().to_numpy().tolist() == (np.floor(dur + 1e-6) + 1).astype(int).tolist()


def test_states_builder_on_the_active_fixture_frames(season):
    """Rebuilt from the frames and possessions, the states agree with the fixture's own states_1hz on the keys."""
    game, held = season["games"][0]
    s = fx.season_tables(0, n_games=4, heldout_every=4)
    frames = s["frames"][s["frames"]["game"] == game]
    poss = s["possessions"][s["possessions"]["sportvu_game"] == game]
    st = S.build_states(frames, poss, game)
    _check_states_against_possessions(st, poss)
    ref = s["states_1hz"][s["states_1hz"]["sportvu_game"] == game]
    shared = st.merge(ref, on=["possession", "elapsed_in_possession_s"], suffixes=("", "_ref"))
    assert len(shared) >= 0.8 * min(len(st), len(ref))
    assert (shared["points_to_end"].to_numpy() == shared["points_to_end_ref"].to_numpy()).all()
    assert np.abs(shared["frame"].to_numpy() - shared["frame_ref"].to_numpy()).max() <= 13   # within half a second at 25 Hz


def test_points_to_end_subtracts_points_logged_before_the_state(fallback_game):
    g = fallback_game
    poss = g["possessions"].iloc[:1].copy()
    p = poss.iloc[0]
    poss.loc[poss.index[0], "points"] = 3
    ev = pd.DataFrame({"event": [int(p.start_event), int(p.end_event)], "period": [int(p.period)] * 2,
                       "clock_s": [float(p.start_clock_s) - 0.5, float(p.end_clock_s)],
                       "team": ["home", "home"], "points": [2, 1], "kind": ["shot_made", "ft_made"]})
    frames = g["frames"][g["frames"]["frame"].between(p.start_frame, p.end_frame)]
    side = "home" if bool(frames.loc[frames["team_id"] == p.offense_team_id, "home"].iloc[0]) else "away"
    ev["team"] = side
    st = S.build_states(frames, poss, g["sportvu_game"], events=ev)
    pts = st.sort_values("elapsed_in_possession_s")["points_to_end"].to_numpy()
    assert pts[0] == 3 and (pts[1:] == 1).all()          # the and-one's two points were logged before every later state


# ── validation ─────────────────────────────────────────────────────────────────────────────────────
def test_validate_logs_and_drops(fallback_game):
    st = fallback_game["states_1hz"]
    bad = pd.concat([st, st.iloc[:3]], ignore_index=True)
    bad.loc[0, "points_to_end"] = 7
    bad.loc[1, "elapsed_in_possession_s"] = -1.0
    clean, rep = V.check_states(bad)
    assert rep["duplicates_dropped"] == 3 and rep["rows_dropped"] == 2 and rep["out_of_range"] == {"points_to_end": 1, "elapsed_in_possession_s": 1}
    assert len(clean) == len(st) - 2 and rep["rows_out"] == len(clean)
    text = V.format_report({"states_1hz": rep})
    assert "| states_1hz |" in text and "points_to_end 1" in text


def test_loaders_flag_held_out_games(season):
    st = season["states_1hz"]
    assert st.loc[st["sportvu_game"] == HELD, "heldout"].all() and not st.loc[st["sportvu_game"] != HELD, "heldout"].any()
    assert sorted(season["games"]) == sorted(set(zip(st["sportvu_game"], st["heldout"])))
    assert season["shots"].loc[season["shots"]["sportvu_game"] == HELD, "heldout"].all()


# ── features and the leakage guard ─────────────────────────────────────────────────────────────────
def test_features_come_from_the_allowlist_only_and_pass_the_guard(fallback_game):
    st = fallback_game["states_1hz"]
    feats = F.build_features(st)
    assert list(feats.columns) == F.KEYS + F.FEATURES and len(feats) == len(st)
    assert not set(feats.columns) & schema.ALL_EXCLUDED
    # the targets never enter: changing them changes nothing
    alt = st.copy()
    alt["points_to_end"] = 4 - alt["points_to_end"]
    alt["possession_end_kind"] = "turnover"
    assert F.build_features(alt).equals(feats)
    leaked = feats.copy()
    leaked["points_to_end"] = 0
    with pytest.raises(ValueError, match="points_to_end"):
        schema.assert_time_forward(leaked, "states_1hz")
    with pytest.raises(ValueError, match="points_to_end"):
        T.Calibrated("ridge", F.FEATURES, 0).fit(leaked, np.zeros(len(leaked)), st["sportvu_game"].to_numpy())
    # geometry sanity
    assert (feats["off_dist_1"] <= feats["off_dist_5"]).all() and (feats["off_hull_area"] >= 0).all()
    assert feats["handler_zone_value"].isna().all()           # no priors given
    lag = F.lagged(feats, 3)
    assert set(F.lagged_columns(3)) <= set(lag.columns)
    first = feats["elapsed_in_possession_s"] == 0
    assert np.allclose(lag.loc[first, "ball_x_lag1"], feats.loc[first, "ball_x"])


def test_priors_are_out_of_fold_by_game(season):
    st = season["states_1hz"]
    train = st[~st["heldout"]].reset_index(drop=True)
    shots = season["shots"][~season["shots"]["heldout"]].reset_index(drop=True)
    games = sorted(train["sportvu_game"].unique())
    g0, g1 = games[0], games[1]
    pri = F.Priors.from_training(train, shots, season["rosters"])
    feats = F.build_features(train, pri)
    assert feats[F.PRIORS].notna().all().all()
    assert feats["handler_fg_prior"].between(0, 1).all() and feats["lineup_tov_prior"].between(0, 1).all()
    # flipping every shot of g0 changes the prior of other games' rows but not g0's own rows
    flipped = shots.copy()
    m = flipped["sportvu_game"] == g0
    flipped.loc[m, "made"] = ~flipped.loc[m, "made"].astype(bool)
    # same xFG shrinkage target (it is fitted on the whole training fold by design), only the zone statistics differ
    pri2 = F.Priors(F.shooter_stats(flipped, season["rosters"]), pri.lineup, pri.xfg, pri.tov_rate, pri.train_games)
    feats2 = F.build_features(train, pri2)
    own = (train["sportvu_game"] == g0).to_numpy()
    other = (train["sportvu_game"] == g1).to_numpy()
    assert np.allclose(feats.loc[own, "handler_fg_prior"], feats2.loc[own, "handler_fg_prior"])
    assert not np.allclose(feats.loc[other, "handler_fg_prior"], feats2.loc[other, "handler_fg_prior"], atol=1e-3)
    # a game outside the training set (a held-out row) gets the full training statistics
    ho = st[st["heldout"]].reset_index(drop=True)
    fh = F.build_features(ho, pri)
    assert fh[F.PRIORS].notna().all().all()
    schema.assert_time_forward(feats, "states_1hz")


# ── models ─────────────────────────────────────────────────────────────────────────────────────────
def test_calibrated_candidates_learn_a_strong_signal_and_calibrate():
    rng = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame(rng.normal(size=(n, len(F.FEATURES))), columns=F.FEATURES)
    X["sportvu_game"], X["possession"], X["frame"] = ["g%d" % (i % 6) for i in range(n)], np.arange(n), np.arange(n)
    z = 1.5 * X["handler_dist_basket"] - X["elapsed_in_possession_s"]
    y = np.clip(np.round(1.2 + z + rng.normal(scale=0.5, size=n)), 0, 4)
    games = X["sportvu_game"].to_numpy()
    tr, te = np.isin(games, ["g0", "g1", "g2", "g3"]), np.isin(games, ["g4", "g5"])
    out = {}
    for m in T.CANDIDATES:
        c = T.Calibrated(m, F.FEATURES, 0, 60).fit(X.loc[tr], y[tr], games[tr])
        e, p = c.predict(X.loc[te])
        assert ((p >= 0) & (p <= 1)).all() and len(e) == te.sum()
        out[m] = (T.mse(y[te], e), float(np.mean((p - (y[te] > 0)) ** 2)))
    assert out["ridge"][0] < 0.5 * out["constant"][0] and out["hgb"][0] < 0.5 * out["constant"][0]
    assert out["hgb"][1] < out["constant"][1] and out["ridge"][1] < out["constant"][1]
    c = T.Calibrated("mlp", F.lagged_columns(2), 0).fit(F.lagged(X.loc[tr], 2), y[tr], games[tr])
    e, _ = c.predict(F.lagged(X.loc[te], 2))
    assert T.mse(y[te], e) < out["constant"][0]


def test_buckets():
    assert T.elapsed_bucket([0, 3.9, 4, 9.9, 10, 23, 30]).tolist() == ["0-4", "0-4", "4-10", "4-10", "10-24", "10-24", "10-24"]
    assert T.shot_clock_bucket([0, 6.9, 7, 13.9, 14, 24, np.nan]).tolist() == ["0-7", "0-7", "7-14", "7-14", "14-24", "14-24", "none"]


# ── end to end ─────────────────────────────────────────────────────────────────────────────────────
def test_train_end_to_end_writes_report_and_output_tables(season, tmp_path):
    ctx = T.run(season, n_folds=3, seed=0, n_boot=10, hgb_iter=30, fixtures=True, command="test",
                report=tmp_path / "epv.md", out_root=tmp_path / "out")
    text = (tmp_path / "epv.md").read_text()
    assert "fixture output, not a result" in text and "—" not in text
    for section in ("## Inputs", "## Candidates by elapsed-time bucket", "## Calibration", "the gate", "## Shuffled-target check",
                    "## Held-out games, scored once", "python -m analytics.epv.train --data"):
        assert section in text
    assert ctx["selected"] in ctx["models"] and ctx["note"]["folds"] == 3
    pred = ctx["pred"]
    assert len(pred) == ctx["train_rows"] and pred["fold"].between(0, 2).all()
    for m in ctx["models"]:
        assert pred["epv_" + m].notna().all() and pred["p_" + m].between(0, 1).all()
    assert set(ctx["by_elapsed"]["bucket"]) >= {"all", "0-4", "4-10"}
    assert ctx["heldout"] is not None and ctx["heldout_games"] == [HELD]
    # the output layout the decisions and playcalling lanes read
    season_dir = tmp_path / "out" / "analytics" / "epv" / "2015-16"
    train_games = [g for g, h in season["games"] if not h]
    for g in train_games:
        df = pd.read_parquet(season_dir / ("%s.parquet" % g))
        assert {"sportvu_game", "possession", "frame", "elapsed_in_possession_s", "epv_points", "p_score", "points_to_end", "fold", "heldout"} <= set(df.columns)
        assert (df["sportvu_game"] == g).all() and not df["heldout"].any() and (df["fold"] >= 0).all()
        assert df["frame"].is_monotonic_increasing or df.sort_values(["possession", "frame"]).equals(df)
    held = pd.read_parquet(tmp_path / "out" / "analytics" / "epv" / "heldout" / "2015-16" / ("%s.parquet" % HELD))
    assert held["heldout"].all() and (held["fold"] == -1).all() and len(held) == ctx["heldout_rows"]
    po = pd.read_parquet(season_dir / "epv_possessions.parquet")
    assert list(po.columns) == ["sportvu_game", "possession", "epv_at_start", "points"] and set(po["sportvu_game"]) == set(train_games)
    assert len(po) == ctx["train_possessions"]
    hpo = pd.read_parquet(tmp_path / "out" / "analytics" / "epv" / "heldout" / "2015-16" / "epv_possessions.parquet")
    assert set(hpo["sportvu_game"]) == {HELD}
    # epv_states agrees with the possession table at the first state
    st0 = pd.concat([pd.read_parquet(season_dir / ("%s.parquet" % g)) for g in train_games]).sort_values("elapsed_in_possession_s").drop_duplicates(["sportvu_game", "possession"])
    joined = st0.merge(po, on=["sportvu_game", "possession"])
    assert np.allclose(joined["epv_points"], joined["epv_at_start"]) and (joined["points_to_end"] == joined["points"]).all()


def test_shuffled_target_shows_no_lift(season):
    t = eda.shuffled_check(season, folds=3, seed=0, hgb_iter=30)
    shuffled = t.loc[t["target"] == "shuffled"].iloc[0]
    assert shuffled["lift"] <= 0.02, t


def test_trainer_refuses_a_held_out_id_in_the_training_set(season):
    tables = {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in season.items()}
    tables["states_1hz"]["heldout"] = False          # the held-out game disguised as a training game by its flag
    with pytest.raises(SystemExit, match="held-out"):
        T.split_tables(tables)


def test_eda_tables(season):
    st = season["states_1hz"]
    train = st[~st["heldout"]].reset_index(drop=True)
    b = eda.points_by_elapsed_bucket(train)
    assert set(b["bucket_s"]) <= {"0-4", "4-10", "10-24"} and b["states"].sum() == len(train)
    feats = F.build_features(train)
    d = eda.feature_distributions(feats)
    assert list(d["feature"]) == F.FEATURES and d.loc[d["feature"] == "handler_fg_prior", "missing_share"].iloc[0] == 1.0
    c = eda.target_correlation(feats, train["points_to_end"])
    assert set(c["feature"]) == set(F.FEATURES)
    assert eda.markdown(b).startswith("| bucket_s |")
    nb = __import__("json").loads((Path(__file__).resolve().parents[1] / "notebooks" / "eda_epv.ipynb").read_text())
    assert nb["nbformat"] == 4 and any("eda." in "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


# ── CLIs on a temporary DATA_ROOT ──────────────────────────────────────────────────────────────────
def test_clis_on_a_temporary_data_root(tmp_path):
    root = tmp_path / "root"
    games = fx.fallback_write_fixture_root(root, seed=3, n_games=2, heldout_every=2, fallback=True)
    assert len(games) == 2 and sum(h for _, h in games) == 1
    out = tmp_path / "out"
    written = S.main(["--data", str(root), "--out", str(out)])
    assert len(written) == 2 and (out / "analytics" / "states_1hz" / "heldout" / "2015-16" / ("%s.parquet" % HELD)).exists()
    rebuilt = pd.read_parquet(out / "analytics" / "states_1hz" / "2015-16" / ("%s.parquet" % games[0][0]))
    stored = pd.read_parquet(root / "analytics" / "states_1hz" / "2015-16" / ("%s.parquet" % games[0][0]))
    assert len(rebuilt) == len(stored)
    reports = V.main(["--data", str(root), "--out", str(out)])
    assert set(reports) >= {"states_1hz", "possessions", "shots"} and (out / "analytics" / "epv_validate.md").exists()
    feats = F.main(["--data", str(root), "--out", str(out)])
    assert len(feats) == 2 and (out / "analytics" / "epv_features" / "heldout" / "2015-16" / ("%s.parquet" % HELD)).exists()
    loaded = V.load_root(root)
    assert loaded["states_1hz"]["heldout"].sum() == len(stored_held := loaded["states_1hz"][loaded["states_1hz"]["sportvu_game"] == HELD]) and len(stored_held) > 0
