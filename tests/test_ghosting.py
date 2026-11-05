"""analytics/ghosting on synthetic games (tests/_fx_ghosting.py: analytics.fixtures when importable, else the
local fallback). No network, no raw data. The MLP is shortened (MAX_ITER 60) to keep the file under 30 s;
every fixture number here is fixture output, not a result."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics import schema  # noqa: E402
from analytics.eval import refuse_heldout  # noqa: E402
from analytics.ghosting import eda, features as F, train as T, validate as V  # noqa: E402

HELD = "12.16.2015.PHX.at.GSW"
_CACHE: dict = {}


def games(n_games=3, n_possessions=4, heldout_every=0, seed=0):
    key = (n_games, n_possessions, heldout_every, seed)
    if key not in _CACHE:
        gs = V.fixture_games(seed=seed, n_games=n_games, n_possessions=n_possessions, heldout_every=heldout_every)
        for g in gs:
            g["validation"] = V.validate_game(g)
        _CACHE[key] = gs
    return [dict(g) for g in _CACHE[key]]


# ── validate ───────────────────────────────────────────────────────────────────────────────────────
def test_validate_counts_and_contract_tables():
    gs = games()
    for g in gs:
        v = g["validation"]
        assert v["frames"]["rows"] > 0 and v["frames"]["duplicate_rows_dropped"] == 0
        assert v["possessions"]["schema_problems"] == [] and v["possessions"]["kept"] >= 4
        assert schema.validate(g["possessions"], "possessions") == []
        for c in V.FRAME_COLUMNS:
            assert c in g["frames"].columns
    # a duplicated frame row and a reversed span are dropped with a count
    g = games()[0]
    frames = pd.concat([g["frames"], g["frames"].iloc[:3]], ignore_index=True)
    _, fc = V.check_frames(frames)
    assert fc["duplicate_rows_dropped"] == 3
    poss = g["possessions"].copy()
    poss.loc[0, "start_frame"] = poss.loc[0, "end_frame"] + 5
    _, pc = V.check_possessions(poss, g["frames"])
    assert pc["start_after_end_dropped"] == 1 and pc["kept"] == len(poss) - 1
    with pytest.raises(ValueError):
        V.check_frames(g["frames"].drop(columns=["ball_x"]))


# ── features ───────────────────────────────────────────────────────────────────────────────────────
def test_feature_frame_is_past_only_and_lags_line_up():
    gs = games()[:2]
    panel, frame, counts = F.build(gs)
    assert len(frame) > 0 and all(c["used"] >= 3 for c in counts.values())
    assert F.feature_columns() == list(F.FEATURES) and len(F.window_columns()) == F.N_LAGS * len(F.FEATURES)
    assert not (set(frame.columns) & schema.ALL_EXCLUDED)
    schema.assert_time_forward(frame, "states_1hz")
    pr = frame["panel_row"].to_numpy()
    dx, dy = panel["def_x"].to_numpy(), panel["def_y"].to_numpy()
    # the lag-1 defender position is the position at t minus 1, lag 3 the position at t minus 3
    assert np.allclose(frame["def_x"], dx[pr - 1]) and np.allclose(frame["def_y_lag3"], dy[pr - 3])
    # the target is the displacement from t minus 1 to t
    assert np.allclose(frame["target_dx"], dx[pr] - dx[pr - 1])
    # the velocity is over 0.4 s (two sampled steps) ending at t minus 1
    assert np.allclose(frame["def_vx"], (dx[pr - 1] - dx[pr - 3]) / (2 * F.DT))
    # the offence, the ball and the shot clock come from the panel at t minus 1 too
    assert np.allclose(frame["rel_off1_dx"], panel["off1_x"].to_numpy()[pr - 1] - dx[pr - 1])
    assert np.allclose(frame["rel_ball_dy"], panel["ball_y"].to_numpy()[pr - 1] - dy[pr - 1])
    assert np.allclose(frame["shot_clock_s"], panel["shot_clock_s"].to_numpy()[pr - 1])
    assert np.allclose(frame["elapsed_s_lag2"], panel["elapsed_s"].to_numpy()[pr - 2])
    # relative features are offence minus defender at t minus 1, and the assignment is one of the five
    rel = np.hypot(frame["rel_asg_dx"].to_numpy(), frame["rel_asg_dy"].to_numpy())
    offs = np.stack([np.hypot(frame[f"rel_off{i}_dx"].to_numpy(), frame[f"rel_off{i}_dy"].to_numpy()) for i in range(1, 6)], axis=1)
    assert np.allclose(np.min(np.abs(offs - rel[:, None]), axis=1), 0, atol=1e-6)
    assert set(frame["attack_right"].unique()) <= {0.0, 1.0}
    assert frame["step"].min() == F.MIN_STEPS - 1
    # a target column smuggled into the frame trips the guard
    with pytest.raises(ValueError):
        schema.assert_time_forward(frame.assign(points=0), "states_1hz")
    with pytest.raises(ValueError):
        T._design(frame.assign(end_frame=0), F.feature_columns() + ["end_frame"])


def test_dropped_frames_cut_the_sequence_instead_of_being_sampled_as_a_step():
    g = games()[0]
    poss = g["possessions"].sort_values("possession").reset_index(drop=True)
    r = poss.iloc[0]
    start = int(r.start_frame)
    assert int(r.end_frame) - start > 60 and not bool(r.gap)
    base_panel, base_counts = F.build_panel_game(g["frames"], poss, g["sportvu_game"])
    n0 = int((base_panel["possession"] == r.possession).sum() // 5)
    assert n0 > 7 and base_counts["cut_frame_gap"] == 0
    # (a) a run of frame ids missing inside the possession: the sampled id after the hole is not STRIDE on
    drop = g["frames"]["frame"].between(start + 45, start + 49)
    panel_a, counts_a = F.build_panel_game(g["frames"][~drop], poss, g["sportvu_game"])
    seq = panel_a[(panel_a["possession"] == r.possession) & (panel_a["def_slot"] == 0)]
    assert counts_a["cut_frame_gap"] == 1 and len(seq) == 9 and seq["frame"].max() == start + 40
    # (b) the source dropped frames and the ids were renumbered: contiguous ids, a clock jump of 0.52 s
    fr = g["frames"][~drop].copy()
    fr["frame"] = pd.factorize(fr["frame"], sort=True)[0].astype("int32")
    panel_b, counts_b = F.build_panel_game(fr, poss, g["sportvu_game"])
    seq = panel_b[(panel_b["possession"] == r.possession) & (panel_b["def_slot"] == 0)]
    assert counts_b["cut_frame_gap"] == 1 and len(seq) == 9
    # the cut bounds the roll-forward: no 1.0 s row of that sequence reaches past the hole
    frame_a = F.feature_frame(panel_a)
    models = {c: (lambda W: np.zeros((len(W), 2))) for c in T.CANDIDATES}
    err = T.roll_forward(models, frame_a, panel_a, fold=0)
    e = err[(err["possession"] == r.possession) & (err["horizon_s"] == 1.0)]
    assert len(e) and e["frame"].max() <= start + 40
    # a stopped clock with frames still flowing is a dead ball, not a gap: nothing is cut
    fz = g["frames"].copy()
    m = fz["frame"].between(start + 10, start + 20)
    fz.loc[m, "game_clock_s"] = float(fz.loc[fz["frame"] == start + 10, "game_clock_s"].iloc[0])
    later = fz["frame"].between(start + 21, int(r.end_frame))
    fz.loc[later, "game_clock_s"] = fz.loc[later, "game_clock_s"] + np.float32(11 / F.HZ)   # the clock resumes
    _, counts_c = F.build_panel_game(fz, poss, g["sportvu_game"])
    assert counts_c["cut_frame_gap"] == 0


def test_persistence_roll_forward_matches_actual_displacements():
    gs = games()[:1]
    panel, frame, _ = F.build(gs)
    models = {c: (lambda W: np.zeros((len(W), 2))) for c in T.CANDIDATES}
    err = T.roll_forward(models, frame, panel, fold=0)
    e2 = err[(err["candidate"] == "persistence") & (err["horizon_s"] == 0.2)]
    assert np.allclose(e2["error_ft"], np.hypot(frame["target_dx"], frame["target_dy"]))
    e1 = err[(err["candidate"] == "persistence") & (err["horizon_s"] == 1.0)]
    assert 0 < len(e1) < len(e2)
    pr = frame["panel_row"].to_numpy()
    end = T._sequence_end(panel)
    ok = pr + 4 <= end[pr]
    dxy = panel[["def_x", "def_y"]].to_numpy()
    assert np.allclose(e1["error_ft"], np.linalg.norm(dxy[pr[ok] + 4] - dxy[pr[ok] - 1], axis=1))
    assert set(err["frame"]) <= set(panel["frame"])


# ── leakage ────────────────────────────────────────────────────────────────────────────────────────
def test_heldout_guard_sets_the_game_aside_and_refuses_fits():
    gs = games(n_games=3, heldout_every=3)
    assert any(g["sportvu_game"] == HELD for g in gs)
    train, held = T.split_games(gs)
    assert [g["sportvu_game"] for g in held] == [HELD] and HELD not in [g["sportvu_game"] for g in train]
    # the flag is not what protects it: an unflagged copy with the held-out id is still set aside
    g = dict(next(g for g in gs if g["sportvu_game"] == HELD))
    g["heldout"] = False
    assert T.split_games([g] + train)[1][0]["sportvu_game"] == HELD
    with pytest.raises(SystemExit):
        refuse_heldout([HELD], "test_ghosting")
    _, frame, _ = F.build(train[:1])
    with pytest.raises(ValueError):
        T.fit_candidates(frame.assign(heldout=True), seed=0)


def test_shift_test_error_collapses_when_features_come_from_the_target_frame():
    gs = games()[:2]
    _, frame, _ = F.build(gs)
    cols = F.window_columns()
    a, b = gs[0]["sportvu_game"], gs[1]["sportvu_game"]
    tr, te = frame[frame["sportvu_game"] == a], frame[frame["sportvu_game"] == b]

    def err(fit_frame, test_frame):          # a near-unpenalised linear probe on the window
        pred = T.fit_ridge(fit_frame, cols, alpha=1e-8)(test_frame[cols].to_numpy(dtype=float))
        return float(np.linalg.norm(pred - test_frame[F.TARGETS].to_numpy(), axis=1).mean())

    honest = err(tr, te)
    leaked = err(F.shift_features_forward(tr), F.shift_features_forward(te))
    assert honest > 0.05
    assert leaked < 0.05 * honest and leaked < 0.05


# ── end to end ─────────────────────────────────────────────────────────────────────────────────────
def test_end_to_end_fixture_run(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "MAX_ITER", 60)
    report = tmp_path / "ghosting.md"
    ctx = T.main(["--fixtures", "--games", "3", "--possessions", "4", "--heldout-every", "3", "--folds", "3",
                  "--bootstrap", "20", "--report", str(report), "--out", str(tmp_path / "out")])
    text = report.read_text()
    assert "fixture output, not a result" in text and "## Decision" in text and "Torch GRU" in text
    assert HELD in ctx["heldout_games"] and HELD not in ctx["train_games"] and ctx["note"]["folds"] == 2
    gf, gp = ctx["ghost_frames"], ctx["ghost_possessions"]
    assert list(gf.columns) == T.GHOST_FRAMES_COLUMNS and list(gp.columns) == T.GHOST_POSSESSIONS_COLUMNS
    assert np.allclose(sorted(gf["horizon_s"].unique()), [0.2, 1.0]) and str(gf["horizon_s"].dtype) == "float32"
    assert gf["heldout"].any() and not gf.loc[gf["fold"] >= 0, "heldout"].any()
    assert (gf.loc[gf["sportvu_game"] == HELD, "fold"] == -1).all()
    gap = np.hypot((gf["actual_x"] - gf["ghost_x"]).to_numpy(), (gf["actual_y"] - gf["ghost_y"]).to_numpy())
    assert (gf["deviation_ft"].abs().to_numpy() <= gap + 1e-3).all()       # float32 tables
    assert gp["points"].ge(0).all() and gp.groupby("sportvu_game").size().min() >= 1
    s = ctx["scores"]
    for h in F.HORIZONS:
        for c in T.CANDIDATES:
            assert np.isfinite(s[h][c]["point"]) and s[h][c]["point"] >= 0
    # the paired test and the decision are consistent with the rule
    r = ctx["pairs"]["mlp_vs_ridge"][1.0]
    assert ctx["decision"]["keep_mlp"] == bool(r["diff"] < 0 and r["excludes_zero"])
    assert ctx["decision"]["selected"] in ("ridge", "mlp")
    assert ctx["validation"]["n_pos"] + ctx["validation"]["n_neg"] + ctx["validation"]["n_zero"] <= 2
    assert ctx["heldout_scores"][1.0]["persistence"] > 0
    # tables land per game, held-out under heldout/
    out = tmp_path / "out" / "analytics" / "ghosting"
    assert (out / "heldout" / V.SEASON / ("ghost_frames_%s.parquet" % HELD)).exists()
    assert len(list((out / V.SEASON).glob("ghost_possessions_*.parquet"))) == 2
    back = pd.read_parquet(out / "heldout" / V.SEASON / ("ghost_frames_%s.parquet" % HELD))
    assert str(back["possession"].dtype) == "int32" and str(back["ghost_x"].dtype) == "float32"
    assert str(gp["points"].dtype) == "int16" and str(gf["fold"].dtype) == "int32"
    # the EDA tables run on the same context
    assert len(eda.displacement_distributions(ctx["frame"])) == 5
    assert set(eda.error_by_horizon(ctx["errors"])["candidate"]) == set(T.CANDIDATES)
    poss = pd.concat([g["possessions"] for g in games(n_games=3, heldout_every=3)], ignore_index=True)
    assert eda.deviation_by_end_kind(gp, poss)["possessions"].sum() == len(gp)
    per, st = eda.correlation_table(gp)
    assert len(per) == 3 and "p_value" in st
    assert "|" in eda.markdown(per)


def test_validate_and_features_cli_run_on_fixtures(tmp_path, capsys):
    rep = V.main(["--fixtures", "--games", "2", "--possessions", "4", "--out", str(tmp_path)])
    assert rep["n_games"] == 2 and (tmp_path / "ghosting_validate.json").exists()
    out = F.main(["--fixtures", "--games", "2", "--possessions", "4", "--out", str(tmp_path)])
    assert out["rows"] > 0
    assert len(list((tmp_path / "analytics" / "ghosting" / V.SEASON).glob("ghost_features_*.parquet"))) == 2
    assert "rel_asg_dx" in capsys.readouterr().out
