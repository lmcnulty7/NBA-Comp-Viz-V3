"""tests/test_deep_models.py: the stage 6 torch models behind ``--deep torch`` (analytics/deep_torch.py and the three
lane modules analytics/{ghosting,epv,embeddings}/deep.py) on analytics.fixtures.

Per lane: (a) --deep torch runs on small fixture settings and the report names the torch model; (b) the decision still
comes from the lane's unchanged keep function, forced both ways; (c) the default --deep mlp keeps the sklearn stand-in;
(d) torch missing (simulated by blocking it in sys.modules, never by uninstalling it) falls back to the stand-in and
says so; (e) the torch fit refuses a held-out row; (f) seeded fits reproduce. Tests that need torch skip without it.
Epochs are cut (monkeypatched) to keep the file fast. Every number here is fixture output, not a result.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analytics import deep_torch as DT  # noqa: E402
from analytics.embeddings import deep as ED, train as ET  # noqa: E402
from analytics.epv import deep as PD, features as PF, train as PT, validate as PV  # noqa: E402
from analytics.ghosting import deep as GD, features as GF, train as GT, validate as GV  # noqa: E402

HELD = "12.16.2015.PHX.at.GSW"
NOTICE = "torch not available; MLP stand-in used"
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
_CACHE: dict = {}


def needs_torch():
    return pytest.importorskip("torch")


def block_torch(monkeypatch):
    """Make ``import torch`` fail for the rest of the test, the way a box without torch behaves."""
    monkeypatch.setitem(sys.modules, "torch", None)
    assert not DT.torch_available()


# ── shared ─────────────────────────────────────────────────────────────────────────────────────────
def test_resolve_deep_and_the_inner_split(monkeypatch):
    assert DT.resolve_deep("mlp", "t") == {"requested": "mlp", "used": "mlp", "notice": ""}
    with pytest.raises(ValueError):
        DT.resolve_deep("gru", "t")
    block_torch(monkeypatch)
    assert DT.resolve_deep("torch", "t") == {"requested": "torch", "used": "mlp", "notice": NOTICE}
    with pytest.raises(RuntimeError):
        DT.require_torch("t")
    # by whole groups with enough groups, never splitting one; a seeded row split otherwise
    g = np.repeat(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"], 7)
    tr, va = DT.inner_split(len(g), g, seed=3)
    assert len(va) and not set(g[tr]) & set(g[va]) and len(np.unique(g[va])) == 2
    assert sorted(np.concatenate([tr, va]).tolist()) == list(range(len(g)))
    tr2, va2 = DT.inner_split(len(g), g, seed=3)
    assert np.array_equal(va, va2)
    tr, va = DT.inner_split(40, np.repeat(["a", "b"], 20), seed=0)
    assert len(va) == 8 and len(set(tr) | set(va)) == 40


def test_packages_import_without_torch_and_never_import_it_eagerly():
    code = ("import sys; sys.modules['torch'] = None\n"
            "import analytics.ghosting.train as g, analytics.epv.train as e, analytics.embeddings.train as m\n"
            "import analytics.ghosting.deep, analytics.epv.deep, analytics.embeddings.deep\n"
            "assert g.parse_args(['--fixtures', '--deep', 'torch']).deep == 'torch'\n"
            "assert e.parse_args(['--fixtures']).deep == 'mlp' and m.parse_args(['--fixtures']).deep == 'mlp'\n"
            "print('imported without torch')\n")
    env = {k: v for k, v in os.environ.items() if k != "DATA_ROOT"}
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "imported without torch"


# ── ghosting ───────────────────────────────────────────────────────────────────────────────────────
def ghost_games(n_games=3, n_possessions=4, heldout_every=3):
    key = ("ghost", n_games, n_possessions, heldout_every)
    if key not in _CACHE:
        gs = GV.fixture_games(seed=0, n_games=n_games, n_possessions=n_possessions, heldout_every=heldout_every)
        for g in gs:
            g["validation"] = GV.validate_game(g)
        _CACHE[key] = gs
    return [dict(g) for g in _CACHE[key]]


def ghost_frame():
    if "ghost_frame" not in _CACHE:
        train, _ = GT.split_games(ghost_games(heldout_every=0))
        _CACHE["ghost_frame"] = GF.build(train[:2])[1]
    return _CACHE["ghost_frame"]


def _forced_pairs(real, keep: bool):
    def paired(err, n_boot, seed):
        out = real(err, n_boot, seed)
        out["mlp_vs_ridge"][1.0] = ({"diff": -1.0, "lo": -1.5, "hi": -0.5, "excludes_zero": True} if keep else
                                    {"diff": -1.0, "lo": -1.5, "hi": 0.5, "excludes_zero": False})
        return out
    return paired


@pytest.mark.parametrize("keep", [True, False])
def test_ghosting_torch_gru_goes_through_the_unchanged_keep_rule(keep, monkeypatch, tmp_path):
    needs_torch()
    monkeypatch.setattr(GD, "MAX_EPOCHS", 12)
    calls, decided = [], []
    real_gru, real_decide = GD.fit_gru, GT.decide
    monkeypatch.setattr(GD, "fit_gru", lambda frame, **kw: calls.append(sorted(frame["sportvu_game"].unique())) or real_gru(frame, **kw))
    monkeypatch.setattr(GT, "fit_mlp", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the stand-in ran under --deep torch")))
    monkeypatch.setattr(GT, "decide", lambda pairs: decided.append(pairs) or real_decide(pairs))
    monkeypatch.setattr(GT, "paired", _forced_pairs(GT.paired, keep))
    ctx = GT.run(ghost_games(), n_folds=3, seed=0, n_boot=10, deep="torch")
    assert ctx["deep"] == {"requested": "torch", "used": "torch", "notice": ""}
    assert len(calls) == 3 and all(HELD not in c for c in calls)          # 2 folds + the held-out refit, training games only
    assert len(decided) == 1 and ctx["decision"] == real_decide(ctx["pairs"])
    assert ctx["decision"]["selected"] == ("mlp" if keep else "ridge") and ctx["decision"]["keep_mlp"] is keep
    assert set(ctx["heldout_candidates"]) == {"persistence", ctx["decision"]["selected"]}
    s = ctx["scores"]
    assert all(np.isfinite(s[h]["mlp"]["point"]) for h in GF.HORIZONS)
    # the GRU's ghosts fill the contract table when it is kept, both horizons rolled forward
    gf = ctx["ghost_frames"]
    sel = ctx["errors"][ctx["errors"]["candidate"] == ctx["decision"]["selected"]]
    assert len(gf[gf["fold"] >= 0]) == len(sel) and np.allclose(np.sort(gf.loc[gf["fold"] >= 0, "ghost_x"]), np.sort(sel["ghost_x"]), atol=1e-3)
    assert sorted(gf["horizon_s"].unique().tolist()) == pytest.approx([0.2, 1.0])
    text = GT.write_report(tmp_path / "g.md", ctx, fixtures=True, command="test")
    assert GD.LABEL in text and "Deep candidate (--deep torch): torch GRU" in text
    assert ("**kept**" in text) is keep and "deep (GRU) vs ridge" in text and chr(0x2014) not in text


def test_ghosting_default_is_the_mlp_stand_in(monkeypatch):
    monkeypatch.setattr(GT, "MAX_ITER", 30)
    monkeypatch.setattr(GD, "fit_gru", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the GRU ran under --deep mlp")))
    assert GT.parse_args(["--fixtures"]).deep == "mlp"
    frame = ghost_frame()
    models = GT.fit_candidates(frame, seed=0)
    assert set(models) == set(GT.CANDIDATES) == {"persistence", "ridge", "mlp"}
    W = frame[GF.window_columns()].to_numpy(dtype=float)
    assert np.array_equal(models["mlp"](W), GT.fit_mlp(frame, seed=0)(W))     # byte for byte the sklearn stand-in
    assert "GRU stand-in" in GT.CANDIDATE_LABELS["mlp"]


def test_ghosting_without_torch_falls_back_to_the_stand_in(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(GT, "MAX_ITER", 30)
    base = GT.run(ghost_games(), n_folds=3, seed=0, n_boot=10)
    block_torch(monkeypatch)
    with caplog.at_level(logging.WARNING):
        ctx = GT.run(ghost_games(), n_folds=3, seed=0, n_boot=10, deep="torch")
    assert ctx["deep"] == {"requested": "torch", "used": "mlp", "notice": NOTICE}
    assert any(NOTICE in r.getMessage() for r in caplog.records)
    for h in GF.HORIZONS:
        assert ctx["scores"][h]["mlp"]["point"] == base["scores"][h]["mlp"]["point"]      # the same stand-in, the same numbers
    assert ctx["decision"] == base["decision"]
    text = GT.write_report(tmp_path / "g.md", ctx, fixtures=True, command="test")
    assert NOTICE in text and "MLPRegressor 64x64" in text and GD.LABEL not in text
    default_text = GT.write_report(tmp_path / "d.md", base, fixtures=True, command="test")
    assert "--deep torch" not in default_text and NOTICE not in default_text


def test_ghosting_gru_refuses_a_held_out_row(monkeypatch):
    block_torch(monkeypatch)                        # the guard runs before torch is touched
    frame = ghost_frame()
    with pytest.raises(ValueError):
        GD.fit_gru(frame.assign(heldout=True))
    with pytest.raises(ValueError):
        GD.fit_gru(frame.assign(sportvu_game=HELD, heldout=False))        # the id protects it, not the flag
    with pytest.raises(ValueError):
        GT.fit_candidates(frame.assign(heldout=True), deep="torch")
    with pytest.raises(RuntimeError):
        GD.fit_gru(frame)                           # a clean frame gets past the guard and stops at the missing torch


def test_ghosting_gru_seeded_fit_reproduces():
    needs_torch()
    frame = ghost_frame()
    W = frame[GF.window_columns()].to_numpy(dtype=float)
    a = GD.fit_gru(frame, seed=0, max_epochs=4)(W)
    b = GD.fit_gru(frame, seed=0, max_epochs=4)(W)
    c = GD.fit_gru(frame, seed=1, max_epochs=4)(W)
    assert a.shape == (len(frame), 2) and np.array_equal(a, b) and not np.allclose(a, c)
    # the GRU reads the window as a sequence: reversing the lag order changes the prediction
    m = GD.GRUWindowRegressor(n_features=len(GF.BASE_FEATURES), seed=0, max_epochs=4).fit(W, frame[GF.TARGETS].to_numpy())
    nb = len(GF.BASE_FEATURES)
    W_rev = np.concatenate([W[:, k * nb:(k + 1) * nb] for k in reversed(range(GF.N_LAGS))], axis=1)
    assert not np.allclose(m.predict(W), m.predict(W_rev))


# ── epv ────────────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def season():
    return PV.load_fixtures(0, n_games=4, heldout_every=4, n_possessions=16)


def _lagged_frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(PF.FEATURES))), columns=PF.FEATURES)
    X["sportvu_game"] = ["g%d" % (i % 6) for i in range(n)]
    X["possession"] = np.arange(n) // 6
    X["frame"] = np.arange(n)
    X["elapsed_in_possession_s"] = (np.arange(n) % 6).astype(float)
    y = np.clip(np.round(1.0 + X["handler_dist_basket"] + rng.normal(scale=0.5, size=n)), 0, 4).to_numpy()
    return PF.lagged(X, PT.N_LAGS), y


@pytest.mark.parametrize("keep", [True, False])
def test_epv_torch_sequence_model_goes_through_the_unchanged_selection(keep, season, monkeypatch, tmp_path):
    needs_torch()
    monkeypatch.setattr(PD, "MAX_EPOCHS", 8)
    monkeypatch.setattr(PT, "MLP_ITER", 20)
    monkeypatch.setattr(PT, "MIN_LIFT", 1.0)                 # opens the gate whatever the fixture says
    made, fits, selected = [], [], []
    real_make, real_fit, real_select, real_metrics = PT.make_model, PD.fit_sequence, PT.select_model, PT.metrics_by_bucket
    monkeypatch.setattr(PT, "make_model", lambda name, *a, **k: made.append(name) or real_make(name, *a, **k))
    monkeypatch.setattr(PD, "fit_sequence", lambda X, *a, **k: fits.append(set(X["sportvu_game"].astype(str))) or real_fit(X, *a, **k))
    monkeypatch.setattr(PT, "select_model", lambda overall, paired: selected.append((dict(overall), paired)) or real_select(overall, paired))
    forced = {"diff": 1.0, "lo": 0.5, "hi": 1.5, "excludes_zero": True} if keep else {"diff": 1.0, "lo": -0.5, "hi": 1.5, "excludes_zero": False}
    monkeypatch.setattr(PT, "paired_mse", lambda *a, **k: dict(forced))

    def metrics(pred, models, bucket_col, n_boot, seed):     # the keep side also needs the lower point MSE the rule asks for
        t = real_metrics(pred, models, bucket_col, n_boot, seed)
        if keep and bucket_col == "elapsed_bucket" and "y" in pred and pred["fold"].max() >= 0:
            for i in t.index[(t["bucket"] == "all") & (t["model"] == PT.SEQUENCE)]:
                t.at[i, "mse"] = {"point": 0.0, "lo": 0.0, "hi": 0.0}
        return t
    monkeypatch.setattr(PT, "metrics_by_bucket", metrics)
    ctx = PT.run(season, n_folds=3, seed=0, n_boot=0, hgb_iter=20, fixtures=True, command="test", report=tmp_path / "e.md", deep="torch")
    assert ctx["deep"]["used"] == "torch" and PT.SEQUENCE in ctx["models"] and ctx["gate"]["train_sequence"]
    assert PT.SEQUENCE not in made                           # the stand-in never ran
    # 3 outer folds x (2 inner + 1 full) on 2 training games each, plus the held-out refit (3 inner + 1) when kept
    assert fits and all(HELD not in s for s in fits) and len(fits) == 9 + (4 if keep else 0)
    assert len(selected) == 1 and (ctx["selected"], ctx["selection_reason"], ctx["keep_sequence"]) == real_select(*selected[0])
    assert ctx["selected"] == (PT.SEQUENCE if keep else ctx["base"]) and ctx["keep_sequence"] is keep
    assert ctx["pred"]["epv_mlp"].notna().all() and ctx["pred"]["p_mlp"].between(0, 1).all()
    if keep:      # the held-out game is scored once by the kept sequence model, the constant as the floor
        assert ctx["heldout"]["models"] == [PT.SEQUENCE, "constant"] and ctx["heldout"]["pred"]["epv_mlp"].notna().all()
    text = ctx["text"]
    assert PD.LABEL in text and "Sequence model (--deep torch): a torch GRU" in text and "fixture output, not a result" in text


def test_epv_default_is_the_mlp_stand_in(monkeypatch):
    from sklearn.neural_network import MLPRegressor
    monkeypatch.setattr(PT, "MLP_ITER", 20)
    monkeypatch.setattr(PD, "fit_sequence", lambda *a, **k: (_ for _ in ()).throw(AssertionError("torch ran under --deep mlp")))
    assert PT.parse_args(["--fixtures"]).deep == "mlp"
    X, y = _lagged_frame()
    c = PT.Calibrated(PT.SEQUENCE, PF.lagged_columns(PT.N_LAGS), 0).fit(X, y, X["sportvu_game"].to_numpy())
    assert isinstance(c.model.steps[-1][1], MLPRegressor) and c.deep == "mlp"
    assert PT.CANDIDATES == ("constant", "ridge", "hgb") and PT.SEQUENCE == "mlp"


def test_epv_without_torch_falls_back_to_the_stand_in(season, monkeypatch, tmp_path, caplog):
    block_torch(monkeypatch)
    monkeypatch.setattr(PT, "MLP_ITER", 20)
    monkeypatch.setattr(PT, "MIN_LIFT", 1.0)
    made = []
    real_make = PT.make_model
    monkeypatch.setattr(PT, "make_model", lambda name, *a, **k: made.append(name) or real_make(name, *a, **k))
    with caplog.at_level(logging.WARNING):
        ctx = PT.run(season, n_folds=3, seed=0, n_boot=0, hgb_iter=20, fixtures=True, command="test", report=tmp_path / "e.md", deep="torch")
    assert ctx["deep"] == {"requested": "torch", "used": "mlp", "notice": NOTICE}
    assert PT.SEQUENCE in made and PT.SEQUENCE in ctx["models"]
    assert any(NOTICE in r.getMessage() for r in caplog.records)
    assert ("Sequence model (--deep torch): %s." % NOTICE) in ctx["text"] and "standing in for a GRU" in ctx["text"]


def test_epv_sequence_model_refuses_a_held_out_row(monkeypatch):
    block_torch(monkeypatch)
    X, y = _lagged_frame(120)
    cols = PF.lagged_columns(PT.N_LAGS)
    with pytest.raises(ValueError):
        PD.fit_sequence(X.assign(sportvu_game=HELD), y, cols)
    with pytest.raises(ValueError):
        PD.fit_sequence(X.assign(heldout=True), y, cols)
    with pytest.raises(ValueError):
        PT.Calibrated(PT.SEQUENCE, cols, 0, deep="torch")._fit_raw(X.assign(sportvu_game=HELD), y)
    with pytest.raises(RuntimeError):
        PD.fit_sequence(X, y, cols)


def test_epv_sequence_model_reproduces_and_reads_only_the_past():
    needs_torch()
    X, y = _lagged_frame()
    cols = PF.lagged_columns(PT.N_LAGS)
    A = X[cols].to_numpy(dtype=float)
    a = PD.fit_sequence(X, y, cols, seed=0, max_epochs=5).predict(A)
    b = PD.fit_sequence(X, y, cols, seed=0, max_epochs=5).predict(A)
    c = PD.fit_sequence(X, y, cols, seed=2, max_epochs=5).predict(A)
    assert a.shape == (len(X),) and np.array_equal(a, b) and not np.allclose(a, c)
    # time-forward: changing the possession's last state moves its own prediction and no earlier state's
    m = PD.fit_sequence(X, y, cols, seed=0, max_epochs=5)
    base = X.drop(columns=[c for c in X.columns if "_lag" in c])
    last = base.index[(base["possession"] == 3) & (base["elapsed_in_possession_s"] == 5)]
    moved = base.copy()
    moved.loc[last, PF.FEATURES] += 5.0
    p0 = m.predict(PF.lagged(base, PT.N_LAGS)[cols].to_numpy(dtype=float))
    p1 = m.predict(PF.lagged(moved, PT.N_LAGS)[cols].to_numpy(dtype=float))
    same = (base["possession"] == 3).to_numpy() & (base["elapsed_in_possession_s"] < 5).to_numpy()
    assert np.array_equal(p0[same], p1[same]) and not np.allclose(p0[last], p1[last])
    assert np.array_equal(np.delete(p0, base.index.get_indexer(last)), np.delete(p1, base.index.get_indexer(last)))


# ── embeddings ─────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def emb_inputs(tmp_path_factory):
    return ET.fixture_inputs(seed=0, n_games=4, heldout_every=2, n_possessions=40, root=tmp_path_factory.mktemp("emb_deep"))


def _forced_evaluate(real, ae_acc):
    seen = []

    def evaluate(E, rows, *a, **k):
        out = real(E, rows, *a, **k)
        seen.append(E)
        if len(seen) == 2:          # the second call scores the autoencoder; force its accuracy around PCA's
            out["knn"]["accuracy"] = {"point": ae_acc, "lo": ae_acc, "hi": ae_acc}
        return out
    return evaluate, seen


@pytest.mark.parametrize("keep", [True, False])
def test_embeddings_torch_autoencoder_goes_through_the_unchanged_rule(keep, emb_inputs, monkeypatch, tmp_path):
    needs_torch()
    monkeypatch.setattr(ED, "MAX_EPOCHS", 40)
    monkeypatch.setattr(ET, "AE_GATE", 1.01)                 # the gate opens on any finite PCA accuracy
    fits, decided = [], []
    real_fit, real_decide = ED.fit_autoencoder, ET.decide
    monkeypatch.setattr(ED, "fit_autoencoder", lambda Z, *a, **k: fits.append((Z.shape, k.get("games"))) or real_fit(Z, *a, **k))
    monkeypatch.setattr(ET, "fit_autoencoder", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the stand-in ran under --deep torch")))
    monkeypatch.setattr(ET, "decide", lambda p, a: decided.append((p, a)) or real_decide(p, a))
    evaluate, seen = _forced_evaluate(ET.evaluate, 1.0 if keep else 0.0)
    monkeypatch.setattr(ET, "evaluate", evaluate)
    ctx = ET.run(emb_inputs, seed=0, n_boot=10, n_min=ET.FIXTURE_N_MIN, fixture=True, n_folds=3, deep="torch")
    assert ctx["deep"]["used"] == "torch" and len(fits) == 1 and HELD not in fits[0][1]
    assert len(decided) == 1 and ctx["decision"] == real_decide(*decided[0])
    assert ctx["decision"]["chosen"] == ("autoencoder" if keep else "pca") and ctx["decision"]["ae_trained"]
    c = ctx["candidates"]["autoencoder"]
    assert 1 <= c["n_iter"] <= 40 and np.isfinite(c["reconstruction_mse"])
    E = seen[1]
    assert E.shape == (ctx["n_fit"], ET.N_DIM) and np.abs(E).max() <= 1.0
    emb = ctx["tables"]["player_embeddings"]
    assert emb["method"].iloc[0] == ctx["decision"]["chosen"] and set(emb["method"]) == {"pca", "autoencoder"}
    ae_rows = emb[emb["method"] == "autoencoder"]
    assert np.allclose(ae_rows[ET.EMB_COLS].to_numpy(), E)
    assert set(ctx["tables"]["embedding_neighbours"]["method"]) == {ctx["decision"]["chosen"]}
    text = ET.write_report(tmp_path / "m.md", ctx, "test")
    assert ED.LABEL in text and "Autoencoder (--deep torch)" in text and chr(0x2014) not in text


def test_embeddings_default_is_the_mlp_stand_in(emb_inputs, monkeypatch, tmp_path):
    monkeypatch.setattr(ET, "AE_GATE", 1.01)
    monkeypatch.setattr(ED, "fit_autoencoder", lambda *a, **k: (_ for _ in ()).throw(AssertionError("torch ran under --deep mlp")))
    calls = []
    real = ET.fit_autoencoder
    monkeypatch.setattr(ET, "fit_autoencoder", lambda Z, seed=0, max_iter=400: calls.append(max_iter) or real(Z, seed, max_iter=60))
    assert ET.parse_args(["--fixtures"]).deep == "mlp"
    ctx = ET.run(emb_inputs, seed=0, n_boot=10, n_min=ET.FIXTURE_N_MIN, fixture=True, n_folds=3)
    assert calls == [400] and ctx["deep"]["used"] == "mlp" and set(ctx["candidates"]) == set(ET.METHODS)
    text = ET.write_report(tmp_path / "m.md", ctx, "test")
    assert "--deep torch" not in text and ED.LABEL not in text


def test_embeddings_without_torch_falls_back_to_the_stand_in(emb_inputs, monkeypatch, tmp_path, caplog):
    block_torch(monkeypatch)
    monkeypatch.setattr(ET, "AE_GATE", 1.01)
    calls = []
    real = ET.fit_autoencoder
    monkeypatch.setattr(ET, "fit_autoencoder", lambda Z, seed=0, max_iter=400: calls.append(1) or real(Z, seed, max_iter=60))
    with caplog.at_level(logging.WARNING):
        ctx = ET.run(emb_inputs, seed=0, n_boot=10, n_min=ET.FIXTURE_N_MIN, fixture=True, n_folds=3, deep="torch")
    assert ctx["deep"] == {"requested": "torch", "used": "mlp", "notice": NOTICE} and calls == [1]
    assert any(NOTICE in r.getMessage() for r in caplog.records)
    text = ET.write_report(tmp_path / "m.md", ctx, "test")
    assert ("Autoencoder (--deep torch): %s." % NOTICE) in text


def test_embeddings_autoencoder_refuses_held_out_material(monkeypatch):
    block_torch(monkeypatch)
    Z = np.random.default_rng(0).normal(size=(20, 6))
    rows = pd.DataFrame({"player_id": np.arange(20), "season": "2015-16"})
    with pytest.raises(ValueError):
        ED.fit_autoencoder(Z, rows=rows, games=["03.08.2016.GRN.at.HLX", HELD])
    with pytest.raises(ValueError):
        ED.fit_autoencoder(Z, rows=rows.assign(heldout=[False] * 19 + [True]))
    with pytest.raises(RuntimeError):
        ED.fit_autoencoder(Z, rows=rows, games=["03.08.2016.GRN.at.HLX"])


def test_embeddings_autoencoder_seeded_fit_reproduces():
    needs_torch()
    Z = np.random.default_rng(1).normal(size=(40, 12))
    rows = pd.DataFrame({"player_id": np.arange(40) // 2})
    a = ED.fit_autoencoder(Z, seed=0, rows=rows, max_epochs=30)
    b = ED.fit_autoencoder(Z, seed=0, rows=rows, max_epochs=30)
    c = ED.fit_autoencoder(Z, seed=5, rows=rows, max_epochs=30)
    Ea, Eb, Ec = a["embed"](Z), b["embed"](Z), c["embed"](Z)
    assert Ea.shape == (40, ET.N_DIM) and np.abs(Ea).max() <= 1.0
    assert np.array_equal(Ea, Eb) and a["reconstruction_mse"] == b["reconstruction_mse"] and not np.allclose(Ea, Ec)
    assert set(a) >= {"name", "embed", "reconstruction_mse", "n_iter"} and a["name"] == "autoencoder"
