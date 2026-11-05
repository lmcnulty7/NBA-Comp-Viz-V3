"""tests/test_gravity.py: analytics/gravity on the fixture generator (tests/_fx_gravity.py, which uses
analytics.fixtures when it exists and its own fallback otherwise). No network, nothing written to the repo.

Covers: the fixture shapes; the hull area on a hand-built square; 1 Hz sampling; the leakage guards (a target
column in the feature frame, a held-out row reaching a fit, a held-out id in the training ids); the out-of-fold
shooting prior; validate's duplicate and null handling; the EDA tables and the shuffled-y check; and the
end-to-end fixture run of the trainer with the top-decile check, the sign test and the held-out scoring.
Runs in well under 30 s on the fallback generator (six small games).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

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
# The fallback generator plants a known linear pull, the xFG effect, sits its star and keeps one roster across
# the season; the shared generator (analytics.fixtures) plants its own star physics, plays its star every period
# and gives every game a fresh roster, so the assertions on effect sizes are made only on the fallback.
FALLBACK = FX.SOURCE == "tests._fx_gravity"


@pytest.fixture(scope="module")
def season():
    return FX.synth_season(0, n_games=N_GAMES, heldout_every=4, n_possessions=N_POSS)


@pytest.fixture(scope="module")
def train_games(season):
    return [d for d in season if not F.is_heldout(d)]


@pytest.fixture(scope="module")
def built(train_games):
    pf, sp, counts = F.build(train_games)
    return pf, sp, counts


# ── fixtures ───────────────────────────────────────────────────────────────────────────────────────
def test_fixture_shapes_and_columns(season):
    assert FX.SOURCE in ("analytics.fixtures", "tests._fx_gravity")
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


def test_features_shape_and_time_forward(built, train_games):
    pf, sp, counts = built
    assert list(F.feature_columns()) == F.FEATURES
    assert set(F.KEYS + F.FEATURES + [F.TARGET, "heldout"]) <= set(pf.columns)
    assert not (set(pf.columns) & schema.ALL_EXCLUDED)
    schema.assert_time_forward(pf, "states_1hz")
    assert (pf.groupby(["sportvu_game", "frame"]).size() == 5).all()   # five offensive players per sampled frame
    assert pf["has_ball"].groupby([pf["sportvu_game"], pf["frame"]]).sum().eq(1).all()
    assert (pf[F.TARGET] > 0).all() and pf["dist_basket_ft"].between(0, 100).all()
    assert len(sp) == len(pf) // 5 and sp["hull_area_ft2"].notna().all()
    assert sum(c["kept"] for c in counts) == len(sp)
    known = {p for d in train_games for p in F.known_gravity_ids(d)}
    assert known                                                        # the generator names its planted players
    hg = pf[pf["player_id"].isin(known)][F.TARGET].mean()
    rest = pf[~pf["player_id"].isin(known)][F.TARGET].mean()
    assert hg < rest                                                    # the planted pull is in the target


def test_leakage_guards(built, train_games):
    pf, _, _ = built
    table = F.skill_table(T.shots_of(train_games, [F.game_id(d) for d in train_games]), T.rosters_of(train_games))
    ok = F.attach_skill(pf, table)
    with pytest.raises(ValueError, match="never features"):
        F.design(ok.assign(points=0))                                   # a target column in the frame
    with pytest.raises(ValueError, match="never features"):
        F.design(ok.assign(end_frame=1))                                # a post-outcome column
    with pytest.raises(ValueError, match="skill"):
        F.design(pf)                                                    # the prior must be attached per fold first
    with pytest.raises(ValueError, match="held-out"):
        T.fit_models(ok.assign(heldout=True))                           # a held-out row never reaches a fit
    with pytest.raises(SystemExit):
        T.refuse_heldout(["01.01.2016.AAA.at.HHH", "12.16.2015.PHX.at.GSW"], "test")


def test_skill_prior_is_out_of_fold(train_games):
    a, b = train_games[0], train_games[1]
    table = F.skill_table(a["shots"], a["rosters"])
    assert table["n_matched"] == len(a["shots"])                        # bref rows (null player_id) matched by name
    pf_b, _, _ = F.build([b])
    shooters_b = set(F.shot_player_ids(b["shots"], b["rosters"]).dropna().astype(int))
    unseen = shooters_b - {p for p, _ in table["zone"]}
    out = F.attach_skill(pf_b, table)
    assert out[F.SKILL_COLUMNS].notna().all().all()
    zone = F.zone_of(out["dist_basket_ft"].to_numpy())
    for p in list(unseen)[:3]:
        m = out["player_id"] == p
        prior = np.array([table["prior_zone"][z] for z in zone[m.to_numpy()]])
        assert np.allclose(out.loc[m, "skill_zone"].to_numpy(), prior)  # a player with no training-fold shots gets the prior
    assert F.zone_of([1.0, 5.0, 12.0, 20.0, 25.0]).tolist() == ["rim", "paint", "short_mid", "long_mid", "three"]


# ── validate ───────────────────────────────────────────────────────────────────────────────────────
def test_validate_counts_duplicates_and_nulls(train_games):
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
    tables = eda.run(train_games[:3], seed=0)
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
    if FALLBACK:
        assert corr.loc["dist_basket_ft", "corr_with_target"] > 0.3     # the planted linear term in the fallback
        assert sy.loc["real", "r2"] > 0.3
    assert eda.md_table(ek).startswith("| end_kind |")


# ── end to end ─────────────────────────────────────────────────────────────────────────────────────
def test_end_to_end_fixture_run(tmp_path):
    report = tmp_path / "gravity.md"
    ctx = T.main(["--fixtures", "--n-games", str(N_GAMES), "--n-possessions", str(N_POSS), "--report", str(report),
                  "--bootstrap", "30", "--min-possessions", "10", "--out", str(tmp_path)])
    assert ctx["train_games"] == N_GAMES - 1 and ctx["heldout_games"] == 1
    assert ctx["selected"] in ("ridge", "hgb")
    s = ctx["scores"]["mse"]
    assert s["ridge"]["point"] < s["constant"]["point"]                 # location explains part of the distance
    assert ctx["known_ids"] and ctx["top_decile_by_game"]["passed"] is True   # the planted players lead every game
    if FALLBACK:                                                        # one roster across the season: the season-wide check
        assert ctx["top_decile"]["passed"] is True
    st = ctx["sign_test"]
    assert set(st) >= {"n_games", "n_positive", "p_value", "mean_diff", "lo", "hi", "candidates", "per_game"}
    assert st["candidates"] == ctx["known_ids"]
    if FALLBACK:                                                        # the fallback plants the xFG effect and sits its star
        assert st["n_games"] == N_GAMES - 1 and np.isfinite(st["p_value"])
        assert st["mean_diff"] > 0 and st["n_positive"] >= st["n_games"] - 1
    gf, players, sp = ctx["gravity_frames"], ctx["gravity_players"], ctx["spacing_frames"]
    assert list(gf.columns) == ["sportvu_game", "possession", "frame", "player_id", "nearest_def_dist_ft", "expected_ft",
                                "gravity_ft", "has_ball", "fold", "heldout"]
    assert np.allclose(gf["gravity_ft"], gf["expected_ft"] - gf["nearest_def_dist_ft"], atol=1e-5)
    assert (gf.loc[gf["heldout"], "fold"] == -1).all() and (gf.loc[~gf["heldout"], "fold"] >= 0).all()
    assert gf["heldout"].any() and ctx["heldout_rows"] == int(gf["heldout"].sum())
    assert list(players.columns) == ["player_id", "n_possessions", "n_frames", "n_games", "gravity_ft", "lo", "hi", "insufficient"]
    assert players["n_possessions"].sum() == gf[~gf["heldout"]][["player_id", "sportvu_game", "possession"]].drop_duplicates().shape[0]
    assert (players.loc[~players["insufficient"], "n_possessions"] >= 10).all()
    assert {"hull_area_ft2", "mean_nearest_def_ft", "heldout"} <= set(sp.columns)
    for name in ("gravity_frames", "gravity_players", "spacing_frames"):
        assert (tmp_path / "analytics" / "gravity" / (name + ".parquet")).exists()
    text = report.read_text()
    assert "fixture output, not a result" in text and "Mac command" in text and "Held-out, scored once" in text
    assert "—" not in text


def test_validate_and_features_cli(tmp_path, capsys):
    out = V.main(["--fixtures", "--n-games", "2", "--n-possessions", "16", "--out", str(tmp_path)])
    assert out["games"] == 2 and out["schema_problems"] == 0
    assert (tmp_path / "analytics" / "gravity" / "validate.json").exists()
    summ = F.main(["--fixtures", "--n-games", "2", "--n-possessions", "16", "--out", str(tmp_path)])
    assert summ["player_frames"] == 5 * summ["spacing_frames"] > 0
    assert (tmp_path / "analytics" / "gravity" / "player_frames.parquet").exists()
