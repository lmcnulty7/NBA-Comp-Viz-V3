"""analytics/ghosting.train: fit on 1 Hz float32 rows, evaluate at 5 Hz one game at a time.

(a) For fixed models, the streamed evaluation gives the same error rows and the same metrics as the
    season-wide evaluation it replaced (one roll_forward over every game's panel, then the bootstrap over
    every error row). (b) The fit rows are exactly the feature-frame rows whose step is a multiple of
    FIT_EVERY, from training games only, stored as float32, and the fits see nothing else. (c) The
    real-data path (no rows kept, tables written per game) agrees with the in-memory path.
No network, no real data; every number here is fixture output, not a result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics import schema  # noqa: E402
from analytics.eval import bootstrap_by_game, paired_bootstrap_by_game  # noqa: E402
from analytics.fixtures import synth_season  # noqa: E402
from analytics.ghosting import features as F, train as T, validate as V  # noqa: E402

HELD = "12.16.2015.PHX.at.GSW"
KEYS = ["sportvu_game", "possession", "frame", "def_player_id", "horizon_s", "candidate"]


def games(n_games=4, heldout_every=0, n_possessions=6):
    gs = V.fixture_games(seed=0, n_games=n_games, n_possessions=n_possessions, heldout_every=heldout_every)
    for g in gs:
        g["validation"] = V.validate_game(g)
    return gs


def fixed_models(frame):
    """Models fitted once: persistence, a ridge on the lag-1 block and a fixed linear map for the deep slot."""
    nb = len(F.BASE_FEATURES)
    ridge = T.fit_ridge(frame)
    return {"persistence": lambda W: np.zeros((len(W), 2)),
            "ridge": lambda W: ridge(W[:, :nb]),
            "mlp": lambda W: 0.05 * W[:, [nb - 7, nb - 6]] - 0.01 * W[:, [0, 1]]}


def old_score(err, n_boot, seed):
    """The season-wide scoring the streamed path replaced: the bootstrap over every error row."""
    out = {}
    for h in F.HORIZONS:
        w = T._wide(err, h)
        out[h] = {}
        for c in T.CANDIDATES:
            r = bootstrap_by_game(w, "sportvu_game", lambda d, c=c: d[c].mean(), n=n_boot, seed=seed)
            out[h][c] = {**r, "rows": int(len(w))}
    return out


def old_paired(err, n_boot, seed):
    out = {}
    for a, b in (("mlp", "ridge"), ("ridge", "persistence")):
        out[a + "_vs_" + b] = {}
        for h in F.HORIZONS:
            w = T._wide(err, h)
            out[a + "_vs_" + b][h] = paired_bootstrap_by_game(w, "sportvu_game", lambda d, a=a: d[a].mean(),
                                                              lambda d, b=b: d[b].mean(), n=n_boot, seed=seed)
    return out


def _close(a, b):
    return np.allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)


# ── (a) the evaluation is unchanged ─────────────────────────────────────────────────────────────────
def test_streamed_evaluation_matches_the_season_wide_evaluation():
    gs = games()
    panel, frame, _ = F.build([dict(g) for g in gs])          # the old path: one panel and frame over every game
    models = fixed_models(frame)
    ref = T.roll_forward(models, frame, panel, fold=0)

    acc = T._new_acc(True, None)
    T.evaluate(models, gs, 0, acc)                           # the new path: one game at a time
    new = pd.concat(acc["err"], ignore_index=True)
    a = ref.sort_values(KEYS, kind="stable").reset_index(drop=True)
    b = new.sort_values(KEYS, kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)                      # the same error rows, bit for bit

    n_boot, seed = 50, 3
    s_ref, s_new = old_score(ref, n_boot, seed), T.score(T.summaries(acc["summ"]), n_boot, seed)
    for h in F.HORIZONS:
        for c in T.CANDIDATES:
            assert s_new[h][c]["rows"] == s_ref[h][c]["rows"]
            for k in ("point", "lo", "hi"):
                assert _close(s_new[h][c][k], s_ref[h][c][k]), (h, c, k, s_new[h][c][k], s_ref[h][c][k])
    p_ref, p_new = old_paired(ref, n_boot, seed), T.paired(T.summaries(acc["summ"]), n_boot, seed)
    for pair in p_ref:
        for h in F.HORIZONS:
            for k in ("diff", "lo", "hi"):
                assert _close(p_new[pair][h][k], p_ref[pair][h][k]), (pair, h, k)
            assert p_new[pair][h]["excludes_zero"] == p_ref[pair][h]["excludes_zero"]
    # score() on an error frame goes through the same summaries
    assert _close(T.score(new, n_boot, seed)[1.0]["ridge"]["point"], s_ref[1.0]["ridge"]["point"])


def test_streamed_ghost_tables_match_the_season_wide_tables(tmp_path):
    gs = games()
    panel, frame, _ = F.build([dict(g) for g in gs])
    models = fixed_models(frame)
    ref = T.roll_forward(models, frame, panel, fold=0)
    poss = pd.concat([g["possessions"] for g in gs], ignore_index=True)
    acc = T._new_acc(False, tmp_path / "spill")
    T.evaluate(models, gs, 0, acc)
    assert acc["err"] == []                                   # no error rows kept on this path
    for sel in T.SELECTABLE:
        gf_ref, gp_ref = T.ghost_tables(ref, sel, poss)
        pd.testing.assert_frame_equal(T._gp_of(acc, sel), gp_ref.sort_values(["sportvu_game", "possession"]).reset_index(drop=True))
        assert acc["gf_rows"][sel] == len(gf_ref)
        for g in gs:
            gid = g["sportvu_game"]
            got = pd.read_parquet(tmp_path / "spill" / sel / ("%s.parquet" % gid))
            want = gf_ref[gf_ref["sportvu_game"] == gid].sort_values(["horizon_s", "possession", "frame", "def_player_id"], kind="stable")
            got = got.sort_values(["horizon_s", "possession", "frame", "def_player_id"], kind="stable")
            pd.testing.assert_frame_equal(got.reset_index(drop=True), want.reset_index(drop=True))


# ── (b) the fit rows ────────────────────────────────────────────────────────────────────────────────
def test_fit_rows_are_the_1hz_float32_subset_of_the_training_games(monkeypatch):
    monkeypatch.setattr(T, "MAX_ITER", 20)
    gs = games(n_games=3, heldout_every=3)
    assert F.FIT_EVERY == 5 and F.FIT_DTYPE == np.float32
    expected = {}
    for g in gs:
        _, ff, _ = F.build([dict(g, heldout=False)])
        keep = ff[ff["step"] % F.FIT_EVERY == 0]
        assert len(keep) and set(keep["step"] % 5) == {0} and keep["step"].min() == 5
        expected[g["sportvu_game"]] = keep
        fr = F.fit_rows(ff)
        cols = F.window_columns() + F.TARGETS
        assert list(fr.columns) == ["sportvu_game"] + cols + ["heldout"]
        assert all(fr[c].dtype == np.float32 for c in cols)
        assert np.array_equal(fr[cols].to_numpy(), keep[cols].to_numpy(dtype=np.float32))
        assert T._design(fr, F.window_columns()).dtype == np.float32          # the fits read float32
        schema.assert_time_forward(fr, "states_1hz")

    seen = []
    real_fit = T.fit_candidates
    monkeypatch.setattr(T, "fit_candidates", lambda train, *a, **k: seen.append(train) or real_fit(train, *a, **k))
    ctx = T.run([dict(g) for g in gs], n_folds=2, seed=0, n_boot=5)
    assert HELD in ctx["heldout_games"] and len(seen) == 3                     # two folds and the held-out refit
    train_ids = ctx["train_games"]
    every = pd.concat([expected[g] for g in train_ids], ignore_index=True)
    last = seen[-1]                                                             # the refit sees every fit row
    assert HELD not in set(last["sportvu_game"]) and not last["heldout"].any()
    assert len(last) == len(every) == ctx["fit_rows"]
    assert np.array_equal(last[F.window_columns()].to_numpy(), every[F.window_columns()].to_numpy(dtype=np.float32))
    for tr in seen[:-1]:                                                        # a fold fits on its games' fit rows only
        ids = sorted(set(tr["sportvu_game"]))
        assert len(tr) == sum(len(expected[g]) for g in ids)
    # evaluation stays at 5 Hz: every training feature row is scored at 0.2 s
    assert ctx["train_rows"] == sum(F.build([dict(g, heldout=False)])[1].shape[0] for g in gs if g["sportvu_game"] in train_ids)
    assert ctx["scores"][0.2]["persistence"]["rows"] == ctx["train_rows"]
    assert ctx["fit_rows"] < ctx["train_rows"] / 4


# ── (c) the real-data path agrees with the in-memory path ───────────────────────────────────────────
def write_root(root: Path, n_games: int = 4, heldout_every: int = 4) -> None:
    for g in synth_season(0, n_games=n_games, heldout_every=heldout_every, n_possessions=6):
        gid, season, held = g["meta"]["sportvu_game"], g["meta"]["season"], g["meta"]["heldout"]
        d = root / "sportvu_contract" / ("heldout" if held else "")
        d.mkdir(parents=True, exist_ok=True)
        g["frames"].to_parquet(d / (gid + ".parquet"), index=False)
        for table in ("possessions", "rosters"):
            p = schema.table_path(root / "analytics", table, season, gid, held)
            p.parent.mkdir(parents=True, exist_ok=True)
            g[table].to_parquet(p, index=False)


def test_streamed_run_writes_the_same_tables_as_the_in_memory_run(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "MAX_ITER", 20)
    root = tmp_path / "root"
    write_root(root)
    args = T.parse_args(["--data", str(root)])
    mem = T.run(V.load_games(args, on_frames=T.prepare_game), n_folds=2, seed=0, n_boot=10, keep_rows=True)
    out, spill = tmp_path / "out", tmp_path / "spill"
    spill.mkdir()
    loaded = V.load_games(args, on_frames=lambda g, f: T.prepare_game(g, f, spill))
    assert all(isinstance(g["fit_rows"], dict) for g in loaded if g["sportvu_game"] != HELD)    # fit rows on disk
    st = T.run(loaded, n_folds=2, seed=0, n_boot=10, keep_rows=False, out_root=out, spill_dir=spill)
    assert HELD in st["heldout_games"] and st["heldout_scores"]
    assert st["decision"] == mem["decision"] and st["train_rows"] == mem["train_rows"] and st["fit_rows"] == mem["fit_rows"]
    for h in F.HORIZONS:
        for c in T.CANDIDATES:
            assert st["scores"][h][c] == mem["scores"][h][c]
        for c in st["heldout_candidates"]:
            assert _close(st["heldout_scores"][h][c], mem["heldout_scores"][h][c])
    gp_m = mem["ghost_possessions"].sort_values(["sportvu_game", "possession"], kind="stable").reset_index(drop=True)
    gp_s = st["ghost_possessions"].sort_values(["sportvu_game", "possession"], kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(gp_s, gp_m)
    assert st["ghost_frames_rows"] == len(mem["ghost_frames"])
    keys = ["horizon_s", "possession", "frame", "def_player_id"]
    for g, want in mem["ghost_frames"].groupby("sportvu_game"):
        p = V.output_path(out, "ghost_frames", str(g), g in st["heldout_games"], V.SEASON)
        got = pd.read_parquet(p).sort_values(keys, kind="stable").reset_index(drop=True)
        pd.testing.assert_frame_equal(got, want.sort_values(keys, kind="stable").reset_index(drop=True))
    assert len(list(out.rglob("ghost_possessions_*.parquet"))) == len(mem["train_games"]) + len(mem["heldout_games"])
