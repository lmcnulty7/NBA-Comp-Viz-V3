"""tests/test_embeddings.py: the player style embeddings lane (analytics/embeddings) on the fixtures of tests/_fx_embeddings.py.

One end-to-end fixture run (module scope) feeds the table and report checks; the metric, feature and gate tests are
hand cases. Everything here passes with the local fallback generator alone."""
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from analytics import schema  # noqa: E402
from analytics.embeddings import eda, features as F, train as T, validate as V  # noqa: E402
from tests import _fx_embeddings as fx  # noqa: E402

N_BOOT = 30


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """train.main on a 6-game fixture season with every third game held out, tables and report under tmp."""
    out = tmp_path_factory.mktemp("emb_out")
    report = out / "embeddings.md"
    ctx = T.main(["--fixtures", "--report", str(report), "--out", str(out), "--bootstrap", str(N_BOOT),
                  "--n-games", "6", "--heldout-every", "3", "--seed", "0"])
    ctx["_report"] = report.read_text()
    ctx["_out"] = out
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
    # the tracked players carry positions (this lane joins the three tables by player_id)
    assert set(g["frames"]["player_id"].unique()) <= set(pos["player_id"].unique())
    # determinism
    g2 = fx.synth_game(3, n_possessions=8)
    pd.testing.assert_frame_equal(g["frames"], g2["frames"])


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
    assert T.r2(y, p["ridge"]) > 0.9 > T.r2(y, p["mean"])


def test_pca_or_autoencoder_gate():
    assert T.decide(0.9, None) == {"chosen": "pca", "ae_trained": False, "reason": T.decide(0.9, None)["reason"]}
    assert not T.decide(0.9, None)["ae_trained"]
    assert T.decide(float("nan"), None)["chosen"] == "pca"
    assert T.decide(0.5, 0.7) == {**T.decide(0.5, 0.7), "chosen": "autoencoder", "ae_trained": True}
    assert T.decide(0.5, 0.45)["chosen"] == "pca" and T.decide(0.5, 0.45)["ae_trained"]
    Z = np.random.default_rng(1).normal(size=(40, 12))
    ae = T.fit_autoencoder(Z, seed=0, max_iter=50)
    E = ae["embed"](Z)
    assert E.shape == (40, T.N_DIM) and np.abs(E).max() <= 1.0 and np.isfinite(ae["reconstruction_mse"])


# ----------------------------------------------------------------------------- held-out exclusion

def test_heldout_games_never_reach_an_aggregate(tmp_path):
    root = tmp_path / "root"
    summary = fx.write_fixture_root(root, seed=1, n_games=4, heldout_every=2)
    assert len(summary["heldout_games"]) == 2 and fx.HELDOUT_GAME_ID in summary["heldout_games"]
    inputs = T.load_inputs(root)
    ids = [g["sportvu_game"] for g in inputs["games"]]
    assert sorted(ids) == sorted(summary["train_games"])
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
    ev = run["candidates"]["pca"]["eval"]
    assert ev["knn"]["n"] == run["n_fit"]
    assert ev["knn"]["accuracy"]["point"] > ev["knn"]["majority"]["point"]      # planted archetypes separate
    assert ev["knn"]["accuracy"]["lo"] <= ev["knn"]["accuracy"]["point"] <= ev["knn"]["accuracy"]["hi"]
    assert ev["next"]["n"] >= 10 and ev["next"]["r2"]["point"] > ev["next"]["r2_mean"]["point"]
    assert len(run["candidates"]["pca"]["explained_variance_ratio"]) == T.N_DIM
    text = run["_report"]
    assert "fixture output, not a result" in text and "## Decision" in text and T.MAC_CMD in text
    assert fx.HELDOUT_GAME_ID not in run["games"] and len(run["games"]) == 4
    assert "—" not in text


def test_validate_and_eda_on_the_fixture_run(run, tmp_path):
    summary = V.validate_inputs(T.load_inputs(Path(run["source"].split("under ")[-1])))
    assert summary["totals"]["games"] == 4 and summary["totals"]["problems"] == 0
    assert summary["shots"]["unresolved_shooter_ids"] == 0 and summary["positions"]["counts"]
    feats, pos = run["tables"]["player_season_features"], run["features"]
    positions = pd.read_parquet(Path(run["source"].split("under ")[-1]) / "collect" / "positions.parquet")
    dist = eda.feature_distribution_by_position(feats, positions)
    assert set(dist["position"]) == {"G", "F", "C"} and set(dist["feature"]) == set(F.available_features(feats))
    corr = eda.correlation_table(feats, top=5)
    assert len(corr) == 5 and (corr["r"].abs() <= 1.0 + 1e-9).all()
    ev = eda.explained_variance(feats)
    assert len(ev) == T.N_DIM and ev["cumulative"].iloc[-1] <= 1.0 + 1e-9
    E, labels = eda.embedding_with_labels(run, positions)
    sep = eda.archetype_separation(E, labels)
    assert set(sep["label"]) == {"G", "F", "C"} and (sep["ratio"] < 1.0).all()
