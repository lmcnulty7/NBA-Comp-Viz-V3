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
    # relative features are offence minus defender at t minus 1, and the assignment is one of the five
    rel = np.hypot(frame["rel_asg_dx"], frame["rel_asg_dy"])
    offs = np.stack([np.hypot(frame[f"rel_off{i}_dx"], frame[f"rel_off{i}_dy"]) for i in range(1, 6)], axis=1)
    assert np.allclose(np.min(np.abs(offs - rel[:, None]), axis=1), 0, atol=1e-6)
    assert set(frame["attack_right"].unique()) <= {0.0, 1.0}
    assert frame["step"].min() == F.MIN_STEPS - 1
    # a target column smuggled into the frame trips the guard
    with pytest.raises(ValueError):
        schema.assert_time_forward(frame.assign(points=0), "states_1hz")
    with pytest.raises(ValueError):
        T._design(frame.assign(end_frame=0), F.feature_columns() + ["end_frame"])


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

    def err(fit_frame, test_frame):
        pred = T.fit_ridge(fit_frame, cols)(test_frame[cols].to_numpy(dtype=float))
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
    assert set(gf["horizon_s"]) == {0.2, 1.0} and gf["heldout"].any() and not gf.loc[gf["fold"] >= 0, "heldout"].any()
    assert (gf.loc[gf["sportvu_game"] == HELD, "fold"] == -1).all()
    assert np.allclose(gf["deviation_ft"].abs() <= np.hypot(gf["actual_x"] - gf["ghost_x"], gf["actual_y"] - gf["ghost_y"]) + 1e-6, True)
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
    out = tmp_path / "out"
    assert (out / "ghost_frames" / "heldout" / V.SEASON / (HELD + ".parquet")).exists()
    assert len(list((out / "ghost_possessions" / V.SEASON).glob("*.parquet"))) == 2
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
    assert out["rows"] > 0 and len(list((tmp_path / "ghost_features" / V.SEASON).glob("*.parquet"))) == 2
    assert "rel_asg_dx" in capsys.readouterr().out
