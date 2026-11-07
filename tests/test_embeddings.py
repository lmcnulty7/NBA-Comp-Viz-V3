"""tests/test_embeddings.py: the player style embeddings lane (analytics/embeddings) on analytics.fixtures
(re-exported by tests/_fx_embeddings.py).

Two end-to-end fixture runs (module scope) feed the table, metric and report checks; the feature, metric and gate
tests are hand cases. Every fixture number is fixture output, not a result."""
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from analytics import schema  # noqa: E402
from analytics.eval import is_heldout_game  # noqa: E402
from analytics.embeddings import eda, features as F, train as T, validate as V  # noqa: E402
from tests import _fx_embeddings as fx  # noqa: E402

N_BOOT = 30


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """train.main on a 4-game analytics.fixtures season with every second game held out (the older flag spellings
    --n-games and --heldout-every), tables and report under tmp."""
    out = tmp_path_factory.mktemp("emb_out")
    report = out / "embeddings.md"
    ctx = T.main(["--fixtures", "--report", str(report), "--out", str(out), "--bootstrap", str(N_BOOT),
                  "--n-games", "4", "--heldout-every", "2", "--seed", "0"])
    ctx["_report"] = report.read_text()
    return ctx


@pytest.fixture(scope="module")
def run_six(tmp_path_factory):
    """A 6-game season, every third game held out, three folds (the canonical --fixture-* flags): enough sufficient
    player-seasons for both metrics to have labelled rows."""
    out = tmp_path_factory.mktemp("emb_out_six")
    report = out / "embeddings.md"
    ctx = T.main(["--fixtures", "--report", str(report), "--out", str(out), "--bootstrap", str(N_BOOT),
                  "--fixture-games", "6", "--fixture-heldout-every", "3", "--seed", "0", "--folds", "3"])
    ctx["_report"] = report.read_text()
    return ctx


# ----------------------------------------------------------------------------- fixtures themselves

def test_fixture_generator_shapes_and_archetypes():
    g = fx.synth_game(3, n_possessions=8)
    assert {"frames", "events", "shots", "rosters", "shot_frames", "rebound_frames", "possessions", "states_1hz"} <= set(g)
    assert {"game", "period", "frame", "ts_ms", "game_clock_s", "ball_x", "ball_y", "ball_z", "team_id", "player_id", "x", "y"} <= set(g["frames"].columns)
    assert schema.validate(g["possessions"], "possessions") == []
    assert schema.validate(g["rosters"], "rosters") == []
    assert (g["frames"].groupby("frame").size() == 10).all()
    agg, pos = fx.synth_aggregates(0)
    assert {"player_id", "season", "endpoint", "metric", "value"} <= set(agg.columns)
    assert set(pos["position"]) == {"G", "F", "C"}
    # determinism
    g2 = fx.synth_game(3, n_possessions=8)
    pd.testing.assert_frame_equal(g["frames"], g2["frames"])
    # a league season: the tracked players carry positions and aggregates (this lane joins the three tables by player_id)
    season = fx.synth_season(0, n_games=2, n_possessions=8)
    rosters = pd.concat([d["rosters"] for d in season], ignore_index=True).drop_duplicates("player_id")
    _, pos_l = fx.synth_aggregates(0, players=rosters)
    tracked = set(pd.concat([d["frames"]["player_id"] for d in season]).unique())
    assert tracked <= set(pos_l["player_id"].unique())


# ----------------------------------------------------------------------------- features: hand cases

def _hand_frames():
    """Two players per team, five frames at 1 Hz, team A (ids 1, 2) on offence attacking the left basket.
    Player 1 carries the ball and moves 3 ft per second along x; player 2 stands still at (40, 40)."""
    rows = []
    x1 = [10.0, 13.0, 16.0, 19.0, 22.0]
    for i in range(5):
        base = {"game": "11.01.2015.BKN.at.GSW", "date": "2015-11-01", "period": 1, "frame": i, "ts_ms": 1000 * i,
                "game_clock_s": 700.0 - i, "ball_x": x1[i], "ball_y": 10.0, "ball_z": 3.0}
        rows.append({**base, "team_id": 1, "player_id": 1, "x": x1[i], "y": 10.0})
        rows.append({**base, "team_id": 1, "player_id": 2, "x": 40.0, "y": 40.0})
        rows.append({**base, "team_id": 2, "player_id": 3, "x": 30.0, "y": 20.0})
        rows.append({**base, "team_id": 2, "player_id": 4, "x": 32.0, "y": 20.0})
    frames = pd.DataFrame(rows)
    poss = pd.DataFrame([{"sportvu_game": "11.01.2015.BKN.at.GSW", "possession": 0, "period": 1, "start_frame": 0,
                          "end_frame": 4, "start_clock_s": 700.0, "end_clock_s": 696.0, "start_event": 0, "end_event": 0,
                          "offense_team_id": 1, "end_kind": "shot_made", "points": 2, "gap": False, "n_frames": 5}])
    return frames, schema.cast(poss, "possessions")


def test_tracking_features_hand_case():
    frames, poss = _hand_frames()
    feats = F.combine_partials([F.tracking_partials(frames, poss)]).set_index("player_id")
    assert (feats["season"] == "2015-16").all()
    assert feats.loc[1, "avg_speed_ftps"] == pytest.approx(3.0)
    assert feats.loc[1, "dist_per_poss_ft"] == pytest.approx(12.0)
    assert feats.loc[2, "avg_speed_ftps"] == pytest.approx(0.0)
    assert feats.loc[1, "ball_share"] == pytest.approx(1.0) and feats.loc[2, "ball_share"] == pytest.approx(0.0)
    assert feats.loc[1, "zone_near_left"] == pytest.approx(0.4) and feats.loc[1, "zone_mid_left"] == pytest.approx(0.6)
    assert feats.loc[2, "zone_far_right"] == pytest.approx(1.0)
    assert feats.loc[2, "offball_ball_dist_ft"] == pytest.approx(np.mean([np.hypot(40 - x, 30) for x in [10, 13, 16, 19, 22]]))
    assert feats.loc[3, "nearest_teammate_ft"] == pytest.approx(2.0) and feats.loc[4, "nearest_teammate_ft"] == pytest.approx(2.0)
    assert np.isnan(feats.loc[3, "ball_share"])          # the defence has no offensive frame
    assert (feats["n_possessions"] == 1).all()


def test_tracking_features_mirrored_half_gap_rule_and_glitch():
    frames, poss = _hand_frames()
    # the same play attacking the right basket: ball and players at (94 - x, 50 - y) give the same zone shares
    m = frames.copy()
    for a, b in (("x", "ball_x"),):
        m[a], m[b] = 94.0 - m[a], 94.0 - m[b]
    m["y"], m["ball_y"] = 50.0 - m["y"], 50.0 - m["ball_y"]
    f0 = F.combine_partials([F.tracking_partials(frames, poss)]).set_index("player_id")
    f1 = F.combine_partials([F.tracking_partials(m, poss)]).set_index("player_id")
    for z in F.ZONE_FEATURES + ["ball_share", "avg_speed_ftps"]:
        pd.testing.assert_series_equal(f0[z], f1[z], check_names=False)
    assert f1.loc[1, "zone_near_left"] == pytest.approx(0.4) and f1.loc[2, "zone_far_right"] == pytest.approx(1.0)
    # the gap rule on ts_ms: frame 4 moved 3.5 s after frame 3, so that step is dropped (3 steps at 3 ft/s, 9 ft)
    g = frames.copy()
    g.loc[g["frame"] == 4, "ts_ms"] = 3000 + 3500
    fg = F.combine_partials([F.tracking_partials(g, poss)]).set_index("player_id")
    assert fg.loc[1, "avg_speed_ftps"] == pytest.approx(3.0) and fg.loc[1, "dist_per_poss_ft"] == pytest.approx(9.0)
    # a 40 ft jump in one second is a tracking glitch (over 30 ft/s): dropped from speed and distance
    j = frames.copy()
    j.loc[(j["frame"] == 4) & (j["player_id"] == 1), "x"] = 19.0 + 40.0
    fj = F.combine_partials([F.tracking_partials(j, poss)]).set_index("player_id")
    assert fj.loc[1, "avg_speed_ftps"] == pytest.approx(3.0) and fj.loc[1, "dist_per_poss_ft"] == pytest.approx(9.0)


def test_positions_from_the_collect_positions_file_join_rosters_and_collapse():
    """An I2-shaped positions frame (player_slug, player, season, position, no player_id) resolves through the rosters."""
    rosters = pd.DataFrame({"sportvu_game": ["11.01.2015.BKN.at.GSW"] * 4, "player_id": [1, 2, 3, 4],
                            "bref_slug": ["aa01", "bb01", "cc01", "dd01"]})
    pos = pd.DataFrame({"player_slug": ["aa01", "bb01", "cc01", "dd01", "ee01"], "player": ["A", "B", "C", "D", "E"],
                        "season": ["2015-16"] * 5, "position": ["PG", "SF-SG", "C-PF", "XX", "C"]})
    out = T.normalise_positions(pos, rosters)
    assert out.sort_values("player_id").set_index("player_id")["position"].to_dict() == {1: "G", 2: "F", 3: "C"}
    assert list(out.columns) == ["player_id", "season", "position"]
    assert T.collapse_position("PF") == "F" and T.collapse_position("G-F") == "G" and np.isnan(T.collapse_position(None))
    ready = pd.DataFrame({"player_id": [7, 7], "season": ["2015-16"] * 2, "position": ["SG", "SG"]})
    assert len(T.normalise_positions(ready, None)) == 1
    chk = V.check_positions(pd.DataFrame({"player_id": [1, 2], "season": ["2015-16"] * 2, "position": ["SF-SG", "QB"]}))
    assert chk["counts"] == {"F": 1} and chk["problems"] == ["positions that do not collapse to G, F or C (part before the first hyphen): ['QB']"]


def test_zone_basic_derivation_and_nba_names():
    shots = pd.DataFrame({"court_x": [6.0, 15.0, 20.0, 9.0, 30.0, 60.0], "court_y": [25.0, 30.0, 40.0, 47.0, 25.0, 25.0],
                          "shot_type": ["2PT", "2PT", "2PT", "3PT", "3PT", "3PT"]})
    assert F.derive_zone_basic(shots).tolist() == ["restricted_area", "paint_non_ra", "mid_range", "corner_3", "above_break_3", "backcourt"]
    named = shots.assign(zone_basic=["Restricted Area", "In The Paint (Non-RA)", "Mid-Range", "Left Corner 3", "Above the Break 3", "Backcourt"])
    assert F.derive_zone_basic(named).tolist() == F.derive_zone_basic(shots).tolist()


def test_feature_frame_has_no_leak_and_aggregates_read_one_season():
    assert len(F.FEATURES) == len(set(F.FEATURES)) and not (set(F.FEATURES) & schema.ALL_EXCLUDED)
    assert F.feature_columns(with_aggregates=False) == F.TRACKING_FEATURES + F.SHOT_FEATURES
    agg, _ = fx.synth_aggregates(0)
    both = F.aggregate_features(agg, "2015-16")
    only = F.aggregate_features(agg[agg["season"] == "2015-16"], "2015-16")
    pd.testing.assert_frame_equal(both, only)                      # 2016-17 rows never touch the 2015-16 features
    assert both["agg_drives"].notna().all() and np.allclose(both[F.PLAY_TYPE_FEATURES].sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        F.assert_no_leak(pd.DataFrame({"avg_speed_ftps": [1.0], "points": [2]}))
    with pytest.raises(ValueError):
        F.assert_no_leak(pd.DataFrame({"avg_speed_ftps": [1.0], "position": ["G"]}))


# ----------------------------------------------------------------------------- metrics: hand cases

def test_knn_position_accuracy_hand_case():
    E = np.array([[0, 0.1, 0.2, 0.3, 10, 10.1, 10.2, 10.3, 20, 20.1]]).T
    labels = ["G"] * 4 + ["C"] * 4 + ["F"] * 2
    pred = T.knn_position_predictions(E, labels, k=5)
    assert pred.tolist() == ["G"] * 4 + ["C"] * 4 + ["C", "C"]      # the two F lose the vote to the C cluster
    assert T.knn_position_accuracy(E, labels, k=5) == pytest.approx(0.8)
    assert T.majority_baseline(labels) == pytest.approx(0.4)
    # a tie goes to the nearest of the tied labels
    assert T.knn_position_predictions(np.array([[0.0], [1.0], [2.0]]), ["x", "a", "b"], k=2)[0] == "a"


def test_next_season_r2_against_mean_baseline():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(40, 8))
    y = 2.0 * E[:, 0] - E[:, 1] + rng.normal(scale=0.1, size=40)
    p = T.next_season_predictions(E, y, groups=np.arange(40), seed=0)
    assert p["folds"] == 5 and np.isfinite(p["ridge"]).all()
    assert T.next_season_predictions(E, y, groups=np.arange(40), seed=0, n_folds=3)["folds"] == 3
    assert T.next_season_predictions(E, y, groups=np.arange(40) % 2, seed=0, n_folds=5)["folds"] == 2   # capped by players
    assert T.r2(y, p["ridge"]) > 0.9 > T.r2(y, p["mean"])


def test_pca_or_autoencoder_gate():
    assert T.decide(0.9, None)["chosen"] == "pca" and not T.decide(0.9, None)["ae_trained"]
    assert T.decide(T.AE_GATE, None)["chosen"] == "pca" and not T.decide(T.AE_GATE, None)["ae_trained"]   # the boundary
    assert T.decide(float("nan"), None)["chosen"] == "pca"
    assert T.decide(0.5, float("nan")) == {"chosen": "pca", "ae_trained": True, "reason": T.decide(0.5, float("nan"))["reason"]} and "undefined" in T.decide(0.5, None)["reason"]
    assert T.decide(0.5, 0.7) == {**T.decide(0.5, 0.7), "chosen": "autoencoder", "ae_trained": True}
    assert T.decide(0.5, 0.45)["chosen"] == "pca" and T.decide(0.5, 0.45)["ae_trained"]
    Z = np.random.default_rng(1).normal(size=(40, 12))
    ae = T.fit_autoencoder(Z, seed=0, max_iter=50)
    E = ae["embed"](Z)
    assert E.shape == (40, T.N_DIM) and np.abs(E).max() <= 1.0 and np.isfinite(ae["reconstruction_mse"])


# ----------------------------------------------------------------------------- held-out exclusion

def test_heldout_games_never_reach_an_aggregate(run):
    root = Path(run["fixture_root"])        # the module run's root: 4 games, every second one held out
    held_files = sorted(p.stem for p in (root / "sportvu_contract" / "heldout").glob("*.parquet") if not p.stem.endswith("_pbp"))
    train_files = sorted(p.stem for p in (root / "sportvu_contract").glob("*.parquet") if not p.stem.endswith("_pbp"))
    assert len(held_files) == 2 and fx.HELDOUT_GAME_ID in held_files and len(train_files) == 2
    inputs = T.load_inputs(root)
    ids = [g["sportvu_game"] for g in inputs["games"]]
    assert sorted(ids) == train_files and not any(is_heldout_game(g) for g in ids)
    assert inputs["dropped"]["heldout_frame_files"] == 2
    assert not T.heldout_shot_mask(inputs["shots"]).any()
    assert set(inputs["shots"]["sportvu_game"]) <= set(ids)
    # the possession counts equal the training-game counts, so no held-out frame entered the sums
    feats = T.build_features(inputs, n_min=1)
    expect = {}
    for g in inputs["games"]:
        fr, po = g["frames"](), g["possessions"]
        for pid, grp in fr.groupby("player_id"):
            n = 0
            for r in po.itertuples():
                if ((grp["frame"] >= r.start_frame) & (grp["frame"] <= r.end_frame)).any():
                    n += 1
            expect[int(pid)] = expect.get(int(pid), 0) + n
    got = feats.set_index("player_id")["n_possessions"].to_dict()
    assert got == expect
    # the guard itself
    inputs["games"].append({"sportvu_game": fx.HELDOUT_GAME_ID, "frames": lambda: None, "possessions": None, "rosters": None})
    with pytest.raises(SystemExit):
        T.training_games(inputs)
    inputs["games"][-1]["sportvu_game"] = "11.30.2015.SAC.at.CLE"
    with pytest.raises(SystemExit):
        T.training_games(inputs)


# ----------------------------------------------------------------------------- end to end

def test_end_to_end_fixture_run_tables(run):
    t = run["tables"]
    feats = t["player_season_features"]
    assert list(feats.columns) == F.KEYS + F.FEATURES + ["n_possessions", "insufficient"]
    assert feats["n_possessions"].min() >= 0 and run["n_fit"] >= 10
    schema.assert_time_forward(feats, "possessions")
    emb = t["player_embeddings"]
    assert list(emb.columns) == ["player_id", "season", "method"] + T.EMB_COLS
    assert set(emb["method"]) <= set(T.METHODS) and run["decision"]["chosen"] in T.METHODS
    chosen = emb[emb["method"] == run["decision"]["chosen"]]
    assert len(chosen) == run["n_fit"] and not chosen.duplicated(["player_id", "season"]).any()
    nb = t["embedding_neighbours"]
    assert list(nb.columns) == ["player_id", "season", "method", "neighbour_rank", "neighbour_player_id", "neighbour_season", "distance"]
    assert (nb.groupby(["player_id", "season"])["neighbour_rank"].apply(lambda s: sorted(s) == [1, 2, 3, 4, 5])).all()
    assert (nb["player_id"] != nb["neighbour_player_id"]).all() and (nb["distance"] >= 0).all()
    assert (nb.groupby(["player_id", "season"])["distance"].apply(lambda s: s.is_monotonic_increasing)).all()
    for name in ("player_season_features", "player_embeddings", "embedding_neighbours"):
        assert Path(run["written"][name][0]).exists()


def test_end_to_end_fixture_run_metrics_and_report(run):
    """The fixture's positions and aggregates share the tracked players, so both metrics have labelled rows; the
    report carries the fixture label on every numbered section and no temporary path."""
    ev = run["candidates"]["pca"]["eval"]
    assert ev["knn"]["n"] == run["n_fit"] and ev["next"]["n"] > 0
    assert ev["knn"]["accuracy"]["lo"] <= ev["knn"]["accuracy"]["point"] <= ev["knn"]["accuracy"]["hi"]
    assert len(run["candidates"]["pca"]["explained_variance_ratio"]) == T.N_DIM
    text = run["_report"]
    assert "fixture output, not a result" in text and "## Decision" in text and T.MAC_CMD in text
    assert fx.HELDOUT_GAME_ID not in run["games"] and len(run["games"]) == 2
    body = text.split("\n", 3)[3]                                      # after the title and the command line
    assert "\u2014" not in text and "\u2013" not in text and "/tmp" not in body
    for h in ("## Inputs", "## Candidates", "## Decision", "## Outputs"):
        assert h + " (fixture output, not a result)" in text


def test_six_game_run_beats_both_baselines(run_six):
    """Planted archetypes on league rosters: 5-NN accuracy above the majority class, ridge R squared above the mean."""
    ev = run_six["candidates"]["pca"]["eval"]
    assert ev["knn"]["n"] == run_six["n_fit"] >= 20
    assert ev["knn"]["accuracy"]["point"] > ev["knn"]["majority"]["point"]
    assert ev["knn"]["accuracy"]["lo"] <= ev["knn"]["accuracy"]["point"] <= ev["knn"]["accuracy"]["hi"]
    assert ev["next"]["n"] >= 20 and ev["next"]["r2"]["point"] > ev["next"]["r2_mean"]["point"]
    assert ev["next"]["folds"] == 3 == run_six["folds"] and "3 grouped folds by player_id" in run_six["_report"]
    assert T.parse_args(["--fixtures"]).folds == 5
    assert run_six["decision"]["chosen"] in T.METHODS and len(run_six["games"]) == 4
    assert "fixture output, not a result" in run_six["_report"]


def test_validate_and_eda_on_the_fixture_runs(run, run_six):
    summary = V.validate_inputs(T.load_inputs(run["fixture_root"]))
    assert summary["totals"]["games"] == 2 and summary["totals"]["problems"] == 0
    assert summary["shots"]["unresolved_shooter_ids"] == 0 and summary["positions"]["counts"]
    feats = run_six["tables"]["player_season_features"]
    positions = T.load_inputs(run_six["fixture_root"])["positions"]
    dist = eda.feature_distribution_by_position(feats, positions)
    assert set(dist["position"]) == {"G", "F", "C"} and set(dist["feature"]) == set(F.available_features(feats))
    corr = eda.correlation_table(feats, top=5)
    assert len(corr) == 5 and (corr["r"].abs() <= 1.0 + 1e-9).all()
    ev = eda.explained_variance(feats)
    assert len(ev) == T.N_DIM and ev["cumulative"].iloc[-1] <= 1.0 + 1e-9
    E, labels = eda.embedding_with_labels(run_six, positions)
    sep = eda.archetype_separation(E, labels)
    assert set(sep["label"]) == {"G", "F", "C"} and (sep["ratio"] < 1.0).all()
