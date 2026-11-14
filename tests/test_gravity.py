"""tests/test_gravity.py: analytics/gravity on the fixture generators. No network, nothing written to the repo.

Two generators (tests/_fx_gravity.py): the shared analytics.fixtures (synth_season) and the lane's synthetic
planted-pull generator analytics/gravity/fixture_pull.py (synth_pull_season). The structural tests run on the shared
one; the end-to-end run happens on both, and the effect-size assertions (sign test, recovered pull within tolerance
of the planted pull) run on the planted-pull generator, whose truth is known and documented.

Covers: fixture shapes; hull area on a hand-built square; 1 Hz sampling; the leakage guards (a target column
in the feature frame, a held-out row reaching a fit, a held-out id in the training ids); the out-of-fold prior
(flipping a test-fold game's shots leaves that game's prior unchanged); validate's duplicate and null handling;
the EDA tables and the shuffled-y check; the end-to-end runs with the top-decile checks, the sign test, the
held-out scoring, the shipped-ridge rule and the planted-pull tolerance. Under 30 s.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from analytics import schema  # noqa: E402
from analytics.gravity import eda, features as F, train as T, validate as V  # noqa: E402
from tests import _fx_gravity as FX  # noqa: E402

N_GAMES, N_POSS = 4, 24
PULL_TOL_FT = 0.6   # recovered pull (ridge residual) vs the generator's effective pull. The expectation is fitted on
#                     rows that include the planted players (two stars on the floor 60 percent of the time: about a
#                     quarter of the player-frames), so its level sits lower by about that share of the pull, 0.35 ft
#                     of 1.5; the observed gap is 0.1 to 0.2 ft on 6 games and up to 0.45 ft on this 3-game fixture


@pytest.fixture(scope="module")
def season():
    return FX.synth_season(0, n_games=N_GAMES, heldout_every=4, n_possessions=N_POSS)


@pytest.fixture(scope="module")
def train_games(season):
    return [d for d in season if not F.is_heldout(d)]


@pytest.fixture(scope="module")
def built(train_games):
    return F.build(train_games)


# ── fixtures ───────────────────────────────────────────────────────────────────────────────────────
def test_fixture_shapes_and_columns(season):
    assert FX.synth_season.__module__ == "analytics.fixtures"
    d = season[0]
    for k in ("frames", "events", "shots", "rosters", "shot_frames", "rebound_frames", "possessions", "states_1hz"):
        assert k in d, k
    for c in V.FRAME_COLUMNS:
        assert c in d["frames"].columns, c
    assert not schema.validate(d["possessions"], "possessions")
    assert not schema.validate(d["rosters"], "rosters")
    held = [F.game_id(d) for d in season if F.is_heldout(d)]
    assert held == ["12.16.2015.PHX.at.GSW"]       # every fourth game is held out and carries a held-out token
    assert len(season) == N_GAMES
    fb = FX.synth_pull_game(3, n_possessions=8)
    assert fb["high_gravity_ids"] == [1001, 2001] and fb["truth"]["planted_pull_ft"] == FX.HG_PULL_FT
    assert all(0 < v <= FX.HG_PULL_FT for v in fb["truth"]["effective_pull_ft"].values())


# ── spacing ────────────────────────────────────────────────────────────────────────────────────────
def test_hull_area_on_a_square():
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert F.hull_area_ft2(square) == pytest.approx(100.0)
    assert F.hull_area_ft2(square + [(5.0, 5.0)]) == pytest.approx(100.0)     # an interior point adds nothing
    assert F.hull_area_ft2([(2.0, 3.0), (12.0, 3.0), (12.0, 23.0), (2.0, 23.0), (7.0, 13.0)]) == pytest.approx(200.0)
    assert np.isnan(F.hull_area_ft2([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]))  # collinear


# ── sampling and features ──────────────────────────────────────────────────────────────────────────
def test_sampling_one_frame_per_second(train_games):
    d = train_games[0]
    samp = F.sample_frames(d["frames"], d["possessions"])
    assert not samp.duplicated(["possession", "frame"]).any()
    per = samp.groupby("possession")["elapsed_s"].agg(["min", "max", "size"])
    assert (per["min"] == 0).all()
    assert (per["size"] == per["max"] + 1).all()                       # one sample per whole second of elapsed time
    fl = F.frame_level(d["frames"]).set_index("frame")
    poss = d["possessions"].set_index("possession")
    elapsed = poss.loc[samp["possession"], "start_clock_s"].to_numpy() - fl.loc[samp["frame"], "game_clock_s"].to_numpy()
    assert np.abs(elapsed - samp["elapsed_s"].to_numpy()).max() < 0.5
    order = d["possessions"].sort_values(["period", "start_frame"]).reset_index(drop=True)
    nxt = {int(r.possession): int(order.iloc[i + 1]["start_frame"]) for i, r in enumerate(order.itertuples()) if i + 1 < len(order)
           and int(order.iloc[i + 1]["period"]) == int(r.period)}
    starts = poss.loc[samp["possession"], "start_frame"].to_numpy()
    assert (samp["frame"].to_numpy() >= starts).all()
    for p, f in zip(samp["possession"], samp["frame"]):                 # the span stops before the next possession's start
        assert int(p) not in nxt or int(f) < nxt[int(p)]


def test_features_shape_and_time_forward(built, train_games):
    pf, sp, counts = built
    assert list(F.feature_columns()) == F.FEATURES and "zone_make_rate" in F.FEATURES
    assert not any(c.startswith("skill") for c in F.FEATURES)          # nothing player-specific in the features
    assert set(F.KEYS + F.FEATURES + [F.TARGET, "heldout"]) <= set(pf.columns)
    assert not (set(pf.columns) & schema.ALL_EXCLUDED)
    schema.assert_time_forward(pf, "states_1hz")
    assert (pf.groupby(["sportvu_game", "frame"]).size() == 5).all()   # five offensive players per sampled frame
    assert pf["has_ball"].groupby([pf["sportvu_game"], pf["frame"]]).sum().eq(1).all()
    assert (pf[F.TARGET] > 0).all() and pf["dist_basket_ft"].between(0, 100).all()
    assert list(sp.columns) == F.SPACING_COLUMNS and len(sp) == len(pf) // 5 and sp["hull_area_ft2"].notna().all()
    assert sp["possession"].dtype == np.int32 and sp["frame"].dtype == np.int32 and not sp["heldout"].any()
    assert sum(c["kept"] for c in counts) == len(sp)
    known = {p for d in train_games for p in F.known_gravity_ids(d)}
    assert known                                                        # the generator names its planted players
    assert pf[pf["player_id"].isin(known)][F.TARGET].mean() < pf[~pf["player_id"].isin(known)][F.TARGET].mean()


def test_leakage_guards(built, train_games):
    pf, _, _ = built
    ok = F.attach_skill(pf, F.skill_table(T.shots_of(train_games, [F.game_id(d) for d in train_games])))
    with pytest.raises(ValueError, match="never features"):
        F.design(ok.assign(points=0))                                   # a target column in the frame
    with pytest.raises(ValueError, match="never features"):
        F.design(ok.assign(end_frame=1))                                # a post-outcome column
    with pytest.raises(ValueError, match="skill"):
        F.design(pf)                                                    # the prior must be attached per fold first
    with pytest.raises(ValueError, match="held-out"):
        T.fit_models(ok.assign(heldout=True), which=("ridge",))         # a held-out row never reaches a fit
    with pytest.raises(SystemExit):
        T.refuse_heldout(["01.01.2016.AAA.at.HHH", "12.16.2015.PHX.at.GSW"], "test")


def test_prior_is_out_of_fold(built, train_games):
    """Flipping every shot of one test-fold game must not move that game's prior (it comes from the other games'
    shots) and must move the prior of the games whose folds trained on it."""
    pf, _, _ = built
    pred1, note = T.out_of_fold(pf, train_games, n_folds=len(train_games), seed=0)
    assert note["mode"] == "out-of-fold" and pred1["zone_make_rate"].notna().all()
    g = F.game_id(train_games[0])
    flipped = [dict(d) for d in train_games]
    flipped[0]["shots"] = train_games[0]["shots"].assign(made=~train_games[0]["shots"]["made"].astype(bool))
    pred2, _ = T.out_of_fold(pf, flipped, n_folds=len(train_games), seed=0)
    own = (pred1["sportvu_game"] == g).to_numpy()
    assert np.array_equal(pred1.loc[own, "zone_make_rate"].to_numpy(), pred2.loc[own, "zone_make_rate"].to_numpy())
    assert not np.array_equal(pred1.loc[~own, "zone_make_rate"].to_numpy(), pred2.loc[~own, "zone_make_rate"].to_numpy())
    assert np.array_equal(pred1.loc[own, "p_ridge"].to_numpy(), pred2.loc[own, "p_ridge"].to_numpy())
    assert F.zone_of([1.0, 5.0, 12.0, 20.0, 25.0]).tolist() == ["rim", "paint", "short_mid", "long_mid", "three"]
    # fold convention: >= 0 out of fold, -1 held-out, -2 in-sample (one training game), never 0 for in-sample rows
    assert (pred1["fold"] >= 0).all() and T.HELDOUT_FOLD == -1 and T.IN_SAMPLE_FOLD == -2
    one = pf[pf["sportvu_game"] == g]
    pred_one, note_one = T.out_of_fold(one, train_games[:1], n_folds=5, seed=0)
    assert note_one["mode"] == "in-sample" and (pred_one["fold"] == T.IN_SAMPLE_FOLD).all()
    assert (T.gravity_frames(pred_one)["fold"] == T.IN_SAMPLE_FOLD).all()
    table = F.skill_table(train_games[0]["shots"])
    assert set(table["prior_zone"]) == {z[2] for z in F.ZONES} and all(0 <= v <= 1 for v in table["prior_zone"].values())


# ── validate ───────────────────────────────────────────────────────────────────────────────────────
def test_validate_counts_duplicates_and_nulls(train_games, season):
    held = [V.validate_game(g)[1] for g in season if F.is_heldout(g)]
    assert held and all(c["heldout"] is True for c in held)             # a summary never names a held-out game unflagged
    unflagged = {**next(g for g in season if F.is_heldout(g)), "heldout": False, "meta": {}}
    assert V.validate_game(unflagged)[1]["heldout"] is True             # the id alone flags it (analytics.eval.is_heldout_game)
    assert V.validate_game(train_games[0])[1]["heldout"] is False
    d = dict(train_games[0])
    fr = d["frames"]
    dup = pd.concat([fr, fr.iloc[:7]], ignore_index=True)
    dup.loc[dup.index[-1], "x"] = np.nan
    d["frames"] = dup
    d["possessions"] = pd.concat([d["possessions"], d["possessions"].iloc[:2]], ignore_index=True)
    clean, counts = V.validate_game(d)
    assert counts["dropped"]["frames_null_xy"] == 1
    assert counts["dropped"]["frames_duplicate_frame_player"] == 6
    assert counts["dropped"]["possessions_duplicate"] == 2
    assert len(clean["frames"]) == len(fr) and len(clean["possessions"]) == len(train_games[0]["possessions"])
    assert counts["frames"]["x_out_of_range"] == 0 and counts["shots"]["shooter_unmatched"] == 0
    assert all(not v for v in counts["schema"].values())
    with pytest.raises(ValueError, match="lacks column"):
        V.validate_game({**d, "frames": fr.drop(columns=["ball_x"])})


# ── eda ────────────────────────────────────────────────────────────────────────────────────────────
def test_eda_tables_and_shuffled_y(train_games):
    tables = eda.run(train_games, seed=0)
    assert set(tables) == {"counts", "nearest_def_by_distance_bin", "nearest_def_by_has_ball", "hull_area_by_end_kind",
                           "correlation_table", "shuffled_y_check"}
    hb = tables["nearest_def_by_has_ball"].set_index("has_ball")
    assert hb.loc[True, "mean_nearest_def_ft"] < hb.loc[False, "mean_nearest_def_ft"]
    ek = tables["hull_area_by_end_kind"]
    assert set(ek["end_kind"]) <= set(schema.END_KINDS) and ek["n"].sum() == tables["counts"]["kept"].sum()
    corr = tables["correlation_table"].set_index("feature")
    assert corr.loc["has_ball", "corr_with_target"] < 0                # the handler is guarded closer
    sy = tables["shuffled_y_check"].set_index("target")
    assert sy.loc["shuffled", "r2"] < 0.05                              # nothing in the features leaks a permuted target
    assert sy.loc["real", "r2"] > sy.loc["shuffled", "r2"] + 0.05
    assert eda.markdown(ek).startswith("| end_kind |")


# ── end to end ─────────────────────────────────────────────────────────────────────────────────────
def _run(tmp_path, generator: str) -> dict:
    report = tmp_path / ("gravity_%s.md" % generator)
    return T.main(["--fixtures", "--fixture-generator", generator, "--no-pull-section", "--fixture-games", str(N_GAMES),
                   "--fixture-possessions", str(N_POSS), "--fixture-heldout-every", "4", "--report", str(report),
                   "--bootstrap", "30", "--min-possessions", "10", "--out", str(tmp_path / generator)]), report


def _check_common(ctx, report, tmp_path, generator):
    assert ctx["train_games"] == N_GAMES - 1 and ctx["heldout_games"] == 1
    assert ctx["selected"] == "ridge" == T.SHIPPED                      # ridge ships whatever the MSE table says
    s = ctx["scores"]["mse"]
    assert s["ridge"]["point"] < s["constant"]["point"]                 # location explains part of the distance
    assert "paired_hgb_minus_ridge" in ctx["scores"] and "hgb_lower_mse" in ctx["scores"]
    assert ctx["known_ids"]
    gf, players, sp = ctx["gravity_frames"], ctx["gravity_players"], ctx["spacing_frames"]
    assert list(gf.columns) == T.FRAME_COLUMNS
    assert np.allclose(gf["gravity_ft"], gf["expected_ft"] - gf["nearest_def_dist_ft"], atol=1e-5)
    tr = gf[~gf["heldout"]]
    assert np.allclose(tr["expected_ft"], ctx["pred"]["p_ridge"].to_numpy(dtype=np.float32), atol=1e-5)   # the shipped expectation
    assert (gf.loc[gf["heldout"], "fold"] == -1).all() and (tr["fold"] >= 0).all()
    assert gf["heldout"].any() and ctx["heldout_rows"] == int(gf["heldout"].sum())
    assert set(ctx["heldout_scores"]) == {"constant", "ridge"}          # held-out scored by the shipped model and the floor only
    assert list(players.columns) == T.PLAYER_COLUMNS
    assert all(players[c].dtype == np.int32 for c in ("player_id", "n_possessions", "n_frames", "n_games"))
    assert players["n_possessions"].sum() == tr[["player_id", "sportvu_game", "possession"]].drop_duplicates().shape[0]
    assert (players.loc[~players["insufficient"], "n_possessions"] >= 10).all()
    assert list(sp.columns) == F.SPACING_COLUMNS and sp["heldout"].any() and sp["possession"].dtype == np.int32
    st = ctx["sign_test"]
    assert set(st) >= {"n_games", "n_positive", "p_value", "mean_diff", "lo", "hi", "candidates", "per_game"}
    assert st["candidates"] == ctx["known_ids"]
    for name in ("gravity_frames", "gravity_players", "spacing_frames"):
        assert (tmp_path / generator / "analytics" / "gravity" / (name + ".parquet")).exists()
    text = report.read_text()
    assert "fixture output, not a result" in text and "Mac command" in text and "Held-out, scored once" in text
    assert "MSE is not the selection criterion" in text and chr(0x2014) not in text


def test_end_to_end_default_generator(tmp_path):
    ctx, report = _run(tmp_path, "shared")
    _check_common(ctx, report, tmp_path, "shared")
    tdg = ctx["top_decile_by_game"]
    assert tdg["n_games"] >= 1 and tdg["n_passed"] >= tdg["n_games"] - 1    # the planted players lead (almost) every game
    assert ctx["top_decile"]["decile_cut"] >= len([p for p in ctx["known_ids"] if ctx["top_decile"]["ranks"][p]])


def test_end_to_end_pull_generator_recovers_the_planted_pull(tmp_path):
    ctx, report = _run(tmp_path, "pull")
    _check_common(ctx, report, tmp_path, "pull")
    assert ctx["top_decile"]["passed"] is True                          # one roster across the season: the season-wide check
    assert ctx["top_decile_by_game"]["passed"] is True
    st = ctx["sign_test"]
    assert st["n_games"] == N_GAMES - 1 and np.isfinite(st["p_value"])
    assert st["mean_diff"] > 0 and st["n_positive"] >= st["n_games"] - 1    # the planted xFG effect
    truth = ctx["truth_pull"]
    assert set(truth) == set(ctx["known_ids"]) and all(0 < v <= FX.HG_PULL_FT for v in truth.values())
    for pid in ctx["known_ids"]:
        rec = ctx["recovered_pull"][pid]
        assert abs(rec - truth[pid]) <= PULL_TOL_FT, (pid, rec, truth[pid])   # the shipped expectation leaves the pull in the residual
        assert ctx["known_by_model"]["ridge"][pid] == pytest.approx(rec, abs=1e-6)
        assert ctx["known_by_model"]["hgb"][pid] < rec + 0.2            # HGB never recovers more of the pull than ridge here
    assert "Recovered pull under the shipped expectation" in report.read_text()


def test_validate_and_features_cli(tmp_path):
    out = V.main(["--fixtures", "--n-games", "2", "--n-possessions", "16", "--out", str(tmp_path)])
    assert out["games"] == 2 and out["schema_problems"] == 0
    assert (tmp_path / "analytics" / "gravity" / "validate.json").exists()
    summ = F.main(["--fixtures", "--fixture-generator", "fallback", "--n-games", "2", "--n-possessions", "16", "--out", str(tmp_path)])
    assert V.generator_name("fallback") == "pull" and V.generator_name("auto") == "shared"     # the older spellings still parse
    assert summ["player_frames"] == 5 * summ["spacing_frames"] > 0 and "validation" in summ
    sp = pd.read_parquet(tmp_path / "analytics" / "gravity" / "spacing_frames.parquet")
    assert list(sp.columns) == F.SPACING_COLUMNS                        # the same schema train.py writes



def test_out_root_defaults_to_data_root_on_real_data():
    from types import SimpleNamespace as NS
    from analytics.gravity import train as T
    assert T.out_root_for(NS(out="/o", fixtures=False, data="/d")) == "/o"
    assert T.out_root_for(NS(out=None, fixtures=False, data="/d")) == "/d"
    assert T.out_root_for(NS(out=None, fixtures=True, data=None)) is None
