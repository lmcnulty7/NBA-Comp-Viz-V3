"""ghosting-E1 machinery (docs/research/ghosting.md, the D113 amendment): the training rollout against the frozen
roll_forward on fixture games, the torch feature function against the frozen one, the cached rows, and guards on the
module's source. No real data, no held-out game, no test game."""

import ast
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from analytics.ghosting import features as F
from analytics.ghosting import train as T
from analytics.ghosting import validate as V
from tools import research_ghosting as G

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "tools" / "research_ghosting.py"


def fixture_tables(seed=0):
    g = V.fixture_games(seed=seed, n_games=1, n_possessions=12, heldout_every=0)[0]
    frames, _ = V.check_frames(V.frames_of(g))
    poss, _ = V.check_possessions(g["possessions"], frames)
    panel, _ = F.build_panel_game(frames, poss, g["sportvu_game"])
    panel["heldout"] = False
    panel["panel_row"] = np.arange(len(panel))
    return panel, F.feature_frame(panel)


def small_net(seed=0):
    torch.manual_seed(seed)
    nw = len(F.window_columns())
    net = G.make_net(np.zeros(nw), np.full(nw, 30.0)).double()
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(0.3)
    return net


def test_the_torch_features_equal_the_frozen_features():
    rng = np.random.default_rng(0)
    n = 50
    ctx = rng.uniform(0, 94, (n, len(F.CTX_COLUMNS)))
    ctx[:, F.CTX_COLUMNS.index("ball_y")] = rng.uniform(0, 50, n)
    ctx[:, F.CTX_COLUMNS.index("shot_clock_s")] = rng.uniform(0, 24, n)
    xy, back = rng.uniform(0, 94, (n, 2)), rng.uniform(0, 94, (n, 2))
    want = F.base_features(ctx, xy, back)
    got = G.base_features_torch(torch.from_numpy(ctx), torch.from_numpy(xy), torch.from_numpy(back)).numpy()
    assert np.allclose(got, want, atol=1e-10)


def test_cached_rows_hold_the_fit_rows_and_the_rollout_block():
    panel, frame = fixture_tables()
    rows = G.cache_rows(panel, frame)
    L = G.layout()
    fit = F.fit_rows(frame)
    assert len(rows) == len(fit)
    assert np.allclose(rows[:, :L["tgt"][1]], fit[T.fit_columns()].to_numpy(np.float32))
    act = rows[:, L["act"][0]:L["act"][1]].reshape(-1, G.STEPS, 2)
    hist = rows[:, L["hist"][0]:L["hist"][1]]
    tgt = rows[:, L["tgt"][0]:L["tgt"][1]]
    assert np.allclose(act[:, 0] - hist[:, 2:4], tgt, atol=1e-3)       # step 1's position minus p-1 is the target
    nv = rows[:, L["nvalid"][0]]
    assert nv.min() >= 1 and nv.max() <= G.STEPS


def test_the_training_rollout_is_the_frozen_roll_forward():
    """The objective trains exactly what is scored: the same network rolled forward by the module and by the frozen
    roll_forward from the same origins gives the same ghosts at 0.2 s and 1.0 s."""
    panel, frame = fixture_tables()
    net = small_net()
    keep = (frame["step"].to_numpy() % F.FIT_EVERY) == 0
    rows = G.cache_rows(panel, frame).astype(np.float64)
    path = [p.detach().numpy() for p in G.rollout_path(net, torch.from_numpy(rows), 5)]

    def f(W):
        with torch.no_grad():
            return net(torch.from_numpy(np.asarray(W, dtype=np.float64))).numpy()

    origins = frame[keep].reset_index(drop=True)
    err = T.roll_forward({"mlp": f}, origins, panel, fold=0)
    nv = rows[:, G.layout()["nvalid"][0]]
    for h, step in ((0.2, 0), (1.0, 4)):
        e = err[err["horizon_s"] == h]
        mine = path[step][nv > step]
        assert len(e) == len(mine)
        assert np.allclose(e[["ghost_x", "ghost_y"]].to_numpy(), mine, atol=1e-3)   # float32 cache against float64


def test_a_one_step_rollout_loss_is_the_one_step_loss():
    panel, frame = fixture_tables()
    net = small_net()
    rows = torch.from_numpy(G.cache_rows(panel, frame).astype(np.float64))
    L = G.layout()
    one = ((net(rows[:, L["win"][0]:L["win"][1]]) - rows[:, L["tgt"][0]:L["tgt"][1]]) ** 2).sum(dim=1).mean()
    assert torch.allclose(G.rollout_loss(net, rows, 1), one, atol=1e-6)


def test_the_loss_leaves_out_steps_past_the_sequence_end():
    panel, frame = fixture_tables()
    net = small_net()
    rows = G.cache_rows(panel, frame).astype(np.float64)
    L = G.layout()
    nv = rows[:, L["nvalid"][0]].copy()
    a = G.rollout_loss(net, torch.from_numpy(rows), 5)
    rows2 = rows.copy()
    act = rows2[:, L["act"][0]:L["act"][1]].reshape(-1, G.STEPS, 2)
    for j in range(G.STEPS):
        act[nv <= j, j] = 1e6                       # garbage where the step is invalid
    rows2[:, L["act"][0]:L["act"][1]] = act.reshape(len(rows2), -1)
    b = G.rollout_loss(net, torch.from_numpy(rows2), 5)
    assert torch.allclose(a, b)


def test_the_horizon_schedule_and_budget_are_the_amendments():
    assert G.HORIZON_BY_EPOCH == (1, 1, 2, 2, 3, 3, 4, 4, 5, 5) and G.EPOCHS == 10
    assert (G.LR, G.BATCH, G.SEED) == (1e-3, 1024, 0)
    assert (G.DIFF_1S, G.GUARD_02S, G.SIGN_P, G.GHOST_TOL) == (-0.10, 0.02, 0.05, 1e-3)


# ----------------------------------------------------------------------------- source guards

def _tree():
    return ast.parse(SOURCE.read_text())


def test_the_module_never_runs_the_lane_command_line_nor_reads_held_out_or_test_data():
    text = SOURCE.read_text()
    for banned in ("T.main", "train.main", "iter_data_root", "load_games(", "_run(", "write_tables", "heldout_once",
                   "test_features", "heldout/"):
        assert banned not in text, banned
    assert "dev_games(" in text and "training_games_only(" in text and "refuse_test(" in text


def test_results_are_written_only_to_the_research_results_and_the_work_folder():
    tree = _tree()
    checked = 0
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        bound = {}
        for a in ast.walk(fn):
            if isinstance(a, ast.Assign):
                for x in a.targets:
                    if isinstance(x, ast.Name):
                        bound.setdefault(x.id, []).append(ast.unparse(a.value))
        for c in (c for c in ast.walk(fn) if isinstance(c, ast.Call)):
            name = getattr(c.func, "attr", getattr(c.func, "id", None))
            if name == "write_text":
                dest = c.func.value
            elif name in ("to_parquet", "rename", "save") and c.args:
                dest = c.args[0] if name != "save" or ast.unparse(c.func).startswith("np.") else c.args[1]
            else:
                continue
            checked += 1
            text = ast.unparse(dest).lstrip("(")
            base = text.split(" /")[0]
            if text.startswith("RESULTS /"):
                continue
            if base in ("folder", "out", "wd", "mm_path"):
                chain = bound.get(base, []) or ["<param>"]
                assert all(b.startswith(("work_dir(", "wd /", "folder /")) or b == "<param>" for b in chain), (fn.name, chain)
                continue
            raise AssertionError(f"{fn.name}: {text}")
    assert checked >= 6


def test_the_gate_merge_matches_a_float32_horizon():
    import pandas as pd
    a = pd.DataFrame({"k": [1, 2], "horizon_s": [0.2, 1.0]})
    b = pd.DataFrame({"k": [1, 2], "horizon_s": np.array([0.2, 1.0], dtype=np.float32)})
    assert len(a.merge(b, on=["k", "horizon_s"])) == 1                  # the bug the rounding fixes
    a2 = a.assign(horizon_s=a["horizon_s"].astype(float).round(1))
    b2 = b.assign(horizon_s=b["horizon_s"].astype(float).round(1))
    assert len(a2.merge(b2, on=["k", "horizon_s"])) == 2
    assert "round(1)" in SOURCE.read_text()



# ----------------------------------------------------------------------------- the Set A scorer (D113)

def _calls(node, name):
    return [c for c in ast.walk(node) if isinstance(c, ast.Call)
            and getattr(c.func, "id", getattr(c.func, "attr", None)) == name]


def test_test_data_is_reached_only_inside_the_scorer_given_to_score_once():
    tree = _tree()
    once = _calls(tree, "score_once")
    assert len(once) == 1 and [ast.unparse(a) for a in once[0].args] == ["EXPERIMENT", "TRACKING", "scorer"]
    scorer = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "scorer")
    inside = {id(n) for n in ast.walk(scorer)}
    seen = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                assert not (a.name == "research_data_root" and a.asname), "aliased import"
        name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
        if name == "research_data_root":
            seen += 1
            assert id(node) in inside, f"{ast.unparse(node)} outside the scorer"
        if isinstance(node, ast.Constant) and node.value == "research_data_root":
            raise AssertionError("named as a string")
    assert seen >= 1 and "_SCORING" in ast.unparse(scorer)
    body = ast.unparse(scorer)
    for fit in ("fit_mlp", "train_net", "frame_pass", "run_fold"):
        assert fit not in body, fit                      # nothing is fitted inside the scorer


def test_final_checks_a_clean_tree_first_and_refuses_unless_development_adopted():
    tree = _tree()
    final = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "final")
    assert "clean_head()" in ast.unparse(final.body[0])
    text = ast.unparse(final)
    assert "dev_out['decision']['adopted']" in text and "dev_out['gate']['passed']" in text
    for needed in ("'code_commit': head", "'scored': out", "'setup': {'gate':"):
        assert needed in text, needed


@pytest.mark.parametrize("dev,lo,hi,want", [(-1.3, -1.6, -1.0, "confirms"), (-1.3, -1.6, 0.2, "consistent"),
                                             (-1.3, -0.9, -0.2, "inconsistent"), (-1.3, -2.0, -1.4, "inconsistent")])
def test_the_d113_reading_and_its_order(dev, lo, hi, want):
    assert G.reading(dev, lo, hi) == want


def _fixture_root(tmp_path, monkeypatch, n_games=3):
    import json
    from analytics.fixtures import write_fixture_root
    from tools import research_testset as R
    root = tmp_path / "root"
    write_fixture_root(root, seed=0, n_games=n_games, heldout_every=0, n_possessions=40)
    monkeypatch.setenv("V3_RESEARCH_DATA", str(root))
    games = sorted(p.stem for p in (root / "analytics" / "possessions" / "2015-16").glob("*.parquet"))
    testset = tmp_path / "set.json"
    testset.write_text(json.dumps({"sources": {R.TRACKING: {"games": games, "sha256": R.list_hash(games)}}}))
    return root, games, testset, tmp_path / "ledger.json"


def test_the_scorer_runs_through_score_once_on_a_synthetic_root(tmp_path, monkeypatch):
    """A rehearsal of e1-final's scorer: a fault here would otherwise spend the one score."""
    import json
    from tools import research_testset as R
    root, games, testset, ledger = _fixture_root(tmp_path, monkeypatch)
    net = small_net()

    def f(W):
        with torch.no_grad():
            return net(torch.from_numpy(np.asarray(W, dtype=np.float64))).numpy()

    preds = {"mlp": lambda W: np.zeros((len(W), 2)), "rollout": f}
    out = R.score_once("ghosting-E1-rehearsal", R.TRACKING, G.make_scorer(preds, -1.3, None, {"code_commit": "x"}, n=200),
                       testset_path=testset, ledger_path=ledger)
    json.dumps(out, allow_nan=False)
    assert "verdict" not in out, out.get("verdict")
    assert out["games"] == len(games) and out["reading"] in ("confirms", "consistent", "inconsistent")
    assert out["1.0"]["rows"] > 0 and "pre_score_sensitivity_t" in out["1.0"]
    assert json.loads(ledger.read_text())[0]["status"] == "scored"
    with pytest.raises(SystemExit):
        G.make_scorer(preds, -1.3)(games)                   # outside score_once the test data is refused


def test_the_scorer_refuses_a_test_game_without_possessions(tmp_path, monkeypatch):
    import json
    from tools import research_testset as R
    root, games, testset, ledger = _fixture_root(tmp_path, monkeypatch)
    (root / "analytics" / "possessions" / "2015-16" / f"{games[0]}.parquet").rename(tmp_path / "moved.parquet")
    preds = {"mlp": lambda W: np.zeros((len(W), 2)), "rollout": lambda W: np.zeros((len(W), 2))}
    out = R.score_once("ghosting-E1-rehearsal", R.TRACKING, G.make_scorer(preds, -1.3, None, {}, n=50),
                       testset_path=testset, ledger_path=ledger)
    assert out["verdict"]["refused"] and out["missing"] == {games[0]: ["possessions"]}
    json.dumps(out, allow_nan=False)
