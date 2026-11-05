"""tests/test_rebounds.py: analytics.rebounds on synthetic games (analytics.fixtures when importable, else the
tests._fx_rebounds fallback). End-to-end fixture run, leakage guards, metric hand checks, input rules. No real
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
N_GAMES, N_POSS, N_BOOT = 4, 10, 20


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
    assert set(run["heldout_scores"]) == set(T.MODELS)           # held-out scored once, every model
    assert run["mode"] == "out-of-fold" and run["folds"] == 3
    pred = run["predictions"]
    assert list(pred.columns) == T.PRED_COLUMNS
    sums = pred.groupby(["sportvu_game", "event"])[["p_prior", "p_logit", "p_selected"]].sum()
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-4)
    assert (pred.groupby(["sportvu_game", "event"])["is_rebounder"].sum() == 1).all()
    assert (pred.loc[pred["heldout"], "fold"] == -1).all() and (pred.loc[~pred["heldout"], "fold"] >= 0).all()
    assert (pred.loc[pred["heldout"], "sportvu_game"] == HELD).all()
    paths = [Path(p) for p in run["written_paths"]]
    assert len(paths) == N_GAMES and all(p.exists() for p in paths)
    held_paths = [p for p in paths if p.stem == HELD]
    assert held_paths and held_paths[0].parts[-4:-1] == ("rebounds", "heldout", "2015-16")
    back = pd.read_parquet(held_paths[0])
    assert list(back.columns) == T.PRED_COLUMNS and back["heldout"].all()
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


def test_select_misses_keeps_player_and_team_rebounds_and_counts_the_rest():
    rf = _rf([{}, {"rim_frame": -1, "rebound_frame": -1, "rebounder_id": None, "identity_match": None, "confidence_flag": "unmatched"},
              {"rebound_event": None, "identity_match": None, "pbp_rebounder_id": None, "pbp_rebounder_name": None, "confidence_flag": "unmatched"},
              {"rebounder_id": None, "pbp_rebounder_id": None, "pbp_rebounder_name": None, "identity_match": None, "confidence_flag": "unmatched"},
              {"pbp_rebounder_id": None, "identity_match": None, "confidence_flag": "unmatched"},
              {"rebounder_id": None, "identity_match": None, "confidence_flag": "low"},
              {"rebounder_id": 2, "identity_match": False, "confidence_flag": "low"}])
    kept, c = F.select_misses(rf)
    assert kept["event"].tolist() == [0, 3, 6] and kept["label_id"].tolist() == [1, F.NONE_ID, 1]
    assert c["no_rim_frame"] == 1 and c["no_rebound_event"] == 1 and c["team_rebounds_kept"] == 1
    assert c["unmatched_named_rebounder"] == 1 and c["identity_match_null"] == 1 and c["player_rebounds_kept"] == 2
    assert "rebounder_id" not in kept.columns and "pbp_rebounder_id" not in kept.columns


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
    games = fx.synth_season(1, 2, 2, 6) if fx.SOURCE == "analytics.fixtures" else fx.synth_season(1, 2, 2)
    root = tmp_path / "root"
    _write_root(root, games)
    listed = V.list_root(root)
    assert sorted((g, h) for g, h, _ in listed) == sorted((str(g["frames"]["game"].iat[0]), str(g["frames"]["game"].iat[0]) == HELD) for g in games)
    summary = V.main(["--data", str(root), "--out", str(tmp_path / "out")])
    assert summary["games"] == 2 and summary["heldout_games"] == 1 and (tmp_path / "out" / "analytics" / "rebounds" / "validate.json").exists()
    ctx = T.main(["--data", str(root), "--report", str(tmp_path / "r.md"), "--bootstrap", "5"])
    assert ctx["mode"] == "in-sample" and ctx["train_games"] == 1 and ctx["heldout_game_ids"] == [HELD]
    assert "Fixture output" not in (tmp_path / "r.md").read_text().split("## Inputs")[0]


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
