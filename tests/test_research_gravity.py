"""gravity-E1 machinery (docs/research/gravity.md, the D107 amendment): the release-frame teammates, the out-of-fold
gravity, the models and the cross-fitted effect on synthetic data with a known answer, D107's reading, the refusals,
and guards on the module's source. No real data, no held-out game, no test game."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import research_gravity as G

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "tools" / "research_gravity.py"


def test_release_rows_take_the_shooters_team_and_need_four_teammates():
    frames = pd.DataFrame({"frame": [10] * 10 + [20] * 9,
                           "team_id": [1] * 5 + [2] * 5 + [1] * 4 + [2] * 5,
                           "player_id": list(range(1, 11)) + [1, 2, 3, 4, 6, 7, 8, 9, 10]})
    shots = pd.DataFrame({"sportvu_game": "g", "event": [1, 2, 3, 4], "player_id": [1, 7, 5, 2],
                          "release_frame": [10, 10, 20, 30]})
    rows = {r["event"]: r for r in G.release_rows(frames, shots)}
    assert rows[1]["status"] == "ok" and rows[1]["teammates"] == [2, 3, 4, 5]
    assert rows[2]["status"] == "ok" and rows[2]["teammates"] == [6, 8, 9, 10]
    assert rows[3]["status"] == "shooter_not_in_frame"
    assert rows[4]["status"] == "no_frame"
    t = G._release_frame_table(G.release_rows(frames, shots))
    assert list(t.loc[t["event"] == 1, ["tm1", "tm2", "tm3", "tm4"]].iloc[0]) == [2, 3, 4, 5]


def test_teammates_gravity_comes_from_games_outside_the_attempts_fold():
    pgs = pd.DataFrame({"player_id": [2, 2, 3, 4, 5], "sportvu_game": ["a", "b", "a", "a", "a"],
                        "g_sum": [10.0, 30.0, 4.0, 6.0, 8.0], "n": [10, 10, 2, 2, 2], "d_sum": [50.0, 50.0, 10, 10, 10],
                        "fold": [0, 1, 0, 0, 0]})
    assert G.player_means(pgs, 0) == {2: (3.0, 5.0)}
    assert G.player_means(pgs, 1)[2] == (1.0, 5.0)
    att = pd.DataFrame({"sportvu_game": ["b", "a"], "event": [1, 2], "player_id": [1, 1], "fold": [1, 0],
                        "closest_defender_dist_ft": [4.0, 5.0]})
    rel = pd.DataFrame({"sportvu_game": ["b", "a"], "event": [1, 2], "shooter": [1, 1], "status": ["ok", "ok"],
                        "tm1": [2, 2], "tm2": [3, 3], "tm3": [4, 4], "tm4": [5, 5]})
    data, counts = G.attach_teammates(att, rel, lambda f: G.player_means(pgs, f))
    assert counts["used"] == 1 and counts["teammate_without_gravity"] == 1     # fold 0 leaves only player 2
    r = data.iloc[0]
    assert r["sportvu_game"] == "b" and r["g_sum"] == pytest.approx(1.0 + 2 + 3 + 4) and r["g_max"] == 4.0


@pytest.mark.parametrize("dev,lo,hi,want", [(0.3, 0.1, 0.5, "confirms"), (0.3, -0.1, 0.5, "consistent"),
                                             (0.3, 0.4, 0.9, "inconsistent"), (0.3, -0.5, 0.2, "inconsistent"),
                                             (-0.3, -0.5, -0.1, "confirms"), (-0.3, -0.5, 0.1, "consistent")])
def test_the_d107_reading_and_its_order(dev, lo, hi, want):
    assert G.reading(dev, lo, hi) == want


def synthetic(n_games=40, per_game=60, players=30, effect=0.5, seed=0):
    """Attempts with location, a shooter effect and teammates' gravity features; the release distance moves by
    `effect` ft per SD of the gravity sum. The gravity sum is correlated with distance_ft, so only a partial effect
    recovers `effect`."""
    rng = np.random.default_rng(seed)
    n = n_games * per_game
    games = np.repeat([f"g{i:03d}" for i in range(n_games)], per_game)
    fold = np.repeat(np.arange(n_games) % 5, per_game)
    shooter = rng.integers(0, players, n)
    dist = rng.uniform(0, 28, n)
    zs = rng.normal(0, 1, n) + 0.05 * (dist - 14)
    zs = (zs - zs.mean()) / zs.std()
    zm = 0.6 * zs + 0.8 * rng.normal(0, 1, n)
    y = 2 + 0.15 * dist + rng.normal(0, 0.5, players)[shooter] + effect * zs + rng.normal(0, 2.5, n)
    made = (rng.random(n) < 1 / (1 + np.exp(0.05 * (dist - 10)))).astype(int)
    return pd.DataFrame({"sportvu_game": games, "fold": fold, "player_id": shooter, "distance_ft": dist,
                         "angle_rad": rng.uniform(-1.5, 1.5, n), "is_three": (dist > 22).astype(int),
                         "is_corner": (rng.random(n) < 0.1).astype(int), "closest_defender_dist_ft": y, "made": made,
                         "z_g_sum": zs, "z_g_max": zm, "z_d_sum": rng.normal(0, 1, n), "z_d_max": rng.normal(0, 1, n)})


def test_the_cross_fitted_effect_recovers_a_planted_partial_effect_and_the_candidate_wins():
    data = synthetic(effect=0.5)
    o = G.oof_models(data)
    y = o["y"]
    eff = G.effect_bootstrap(data["sportvu_game"], y - o["base"],
                             np.column_stack([data["z_g_sum"] - o["zg_sum"], data["z_g_max"] - o["zg_max"]]), n=200)
    assert abs(eff["sum"]["value"] - 0.5) < 0.12 and eff["sum"]["lo"] < 0.5 < eff["sum"]["hi"]
    assert abs(eff["max"]["value"]) < 0.12
    d = (y - o["cand"]) ** 2 - (y - o["base"]) ** 2
    boot = G.game_bootstrap(data["sportvu_game"], {"d": d}, n=200)
    assert boot["intervals"]["d"]["hi"] < 0
    noise = G.effect_bootstrap(data["sportvu_game"], y - o["base"],
                               np.column_stack([data["z_d_sum"] - o["zd_sum"], data["z_d_max"] - o["zd_max"]]), n=200)
    assert abs(noise["sum"]["value"]) < 0.15                           # unrelated features find no effect of size
    assert np.isfinite(o["p_off"]).all() and np.isfinite(o["p_loc"]).all()


def test_no_effect_gives_no_effect():
    data = synthetic(effect=0.0, seed=3)
    o = G.oof_models(data)
    eff = G.effect_bootstrap(data["sportvu_game"], o["y"] - o["base"],
                             np.column_stack([data["z_g_sum"] - o["zg_sum"], data["z_g_max"] - o["zg_max"]]), n=200)
    assert eff["sum"]["lo"] < 0 < eff["sum"]["hi"] and eff["sum"]["value"] < G.EFFECT_MIN


def test_an_unseen_shooter_gets_no_shooter_indicator():
    data = synthetic(n_games=5)
    D = G.Design(data[data["player_id"] != 0])
    X = D(data[data["player_id"] == 0].head(3))
    assert X.shape[1] == 4 + len(D.shooters) and X[:, 4:].nnz == 0


def test_the_offset_logistic_recovers_its_coefficients():
    rng = np.random.default_rng(1)
    n = 20000
    off, x = rng.normal(0, 1, n), rng.normal(0, 1, (n, 2))
    p = 1 / (1 + np.exp(-(off + 0.2 + 0.5 * x[:, 0] - 0.3 * x[:, 1])))
    y = (rng.random(n) < p).astype(int)
    q = G.logistic_offset(off, x, y, off[:5], x[:5])
    assert np.allclose(q, p[:5], atol=0.03)


def test_the_t_interval_needs_two_games_and_a_non_finite_summary_is_refused():
    assert G.non_finite(G.per_game_t([0.1])) == ["lo", "hi"]
    r = G.refused("why", counts={"a": 1}, bad=float("nan"))
    assert r["verdict"]["refused"] and "bad" not in r
    json.dumps(r, allow_nan=False)


def test_the_test_summary_reads_d107_on_synthetic_attempts():
    dev = synthetic(effect=0.5, seed=4)
    test = synthetic(n_games=10, per_game=90, effect=0.5, seed=5)
    s = G.test_summary(test, dev, dev_effect=0.5, n=200)
    assert s["games"] == 10 and s["reading"] in ("confirms", "consistent", "inconsistent")
    assert not G.non_finite(s)


# ----------------------------------------------------------------------------- source guards

DOOR_NAMES = ("research_data_root", "test_features")


def _tree():
    return ast.parse(SOURCE.read_text())


def _calls(node, name):
    return [c for c in ast.walk(node) if isinstance(c, ast.Call)
            and getattr(c.func, "id", getattr(c.func, "attr", None)) == name]


def test_test_data_is_reached_only_inside_the_scorer_given_to_score_once():
    tree = _tree()
    once = _calls(tree, "score_once")
    assert len(once) == 1 and [ast.unparse(a) for a in once[0].args] == ["EXPERIMENT", "TRACKING", "scorer"]
    scorer = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "scorer")
    inside = {id(n) for n in ast.walk(scorer)}
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                assert not (a.name in DOOR_NAMES and a.asname), "aliased import"
        name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
        if name in DOOR_NAMES:
            seen.add(name)
            assert id(node) in inside, f"{ast.unparse(node)} outside the scorer"
        if isinstance(node, ast.Constant) and node.value in DOOR_NAMES:
            raise AssertionError("named as a string")
    assert seen == set(DOOR_NAMES)
    assert "_SCORING" in ast.unparse(scorer)


def test_final_checks_a_clean_tree_first_and_nests_its_record():
    tree = _tree()
    final = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "final")
    assert "clean_head()" in ast.unparse(final.body[0])
    text = ast.unparse(final)
    for needed in ("'code_commit': head", "'gate_files_sha256': gate_record['files']", "'dev_json_sha256': _sha256(dev_path)",
                   "'scored': out", "'setup': {'gate': gate_record, 'rule': rule}"):
        assert needed in text, needed
    scorer = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "scorer")
    body = ast.unparse(scorer)
    for fit in ("oof_ridge", "oof_models", "load_dev", "gate("):
        assert fit not in body, fit


def test_the_module_never_runs_the_lane_command_line_and_reads_only_development_games():
    text = SOURCE.read_text()
    for banned in ("GT.main", "train.main", "run_pipeline", "write_tables", "heldout_once", "load_data_root(",
                   "skip_heldout=False"):
        assert banned not in text, banned
    assert "dev_games(" in text and "training_games_only(" in text and "refuse_test(" in text
    reads = [c for c in _calls(_tree(), "read_parquet") if c.args and "SHOTS_FILE" in ast.unparse(c.args[0])]
    assert reads and all(any(k.arg == "filters" for k in c.keywords) for c in reads)


def _bound(fn, name):
    return [ast.unparse(a.value) for a in ast.walk(fn) if isinstance(a, ast.Assign)
            and any(isinstance(x, ast.Name) and x.id == name for x in a.targets)]


def test_results_are_written_only_to_the_research_results_and_the_work_folder():
    tree = _tree()
    checked = 0
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        for c in (c for c in ast.walk(fn) if isinstance(c, ast.Call)):
            name = getattr(c.func, "attr", getattr(c.func, "id", None))
            if name == "write_text":
                dest = c.func.value
            elif name in ("to_parquet", "rename"):
                dest = c.args[0]
            else:
                continue
            checked += 1
            text = ast.unparse(dest).lstrip("(")
            if text.startswith("RESULTS /"):
                continue
            base = text.split(" /")[0]
            params = [a.arg for a in fn.args.args]
            if base in params:
                for f in (f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)):
                    for call in _calls(f, fn.name):
                        arg = call.args[params.index(base)]
                        assert all(b.startswith("work_dir(") for b in _bound(f, arg.id)), f.name
                continue
            bound = _bound(fn, base)
            assert bound and all(b.startswith("work_dir(") for b in bound), f"{fn.name}: {text}"
    assert checked >= 8


# ----------------------------------------------------------------------------- rehearsal of the scorer

def _rehearsal(tmp_path, monkeypatch, n_games=8, n_test=2):
    """A synthetic root (analytics.fixtures, the DATA_ROOT layout): its first games stand in for the development
    attempts (frozen xFG features and release-frame teammates from its frames, made-up gravity per player), its last
    games are a temporary test set with a temporary ledger."""
    from analytics.fixtures import write_fixture_root
    from analytics.gravity import validate as GV
    from tools import research_testset as R
    from tools import research_xfg as X
    root = tmp_path / "root"
    write_fixture_root(root, seed=0, n_games=n_games, heldout_every=0, n_possessions=60)
    monkeypatch.setenv("V3_RESEARCH_DATA", str(root))
    games = sorted((p.stem for p in (root / "analytics" / "possessions" / "2015-16").glob("*.parquet")),
                   key=lambda g: (g.split(".")[2], g.split(".")[0], g.split(".")[1]))
    dev, test = games[:-n_test], games[-n_test:]
    feats = X.test_features(dev)
    att = feats[feats["has_contest"].astype(bool)].copy()
    att["sportvu_game"] = att["sportvu_game"].astype(str)
    fold = {g: i % 5 for i, g in enumerate(dev)}
    att["fold"] = att["sportvu_game"].map(fold)
    rows = []
    for g in dev:
        frames, _, _ = GV.clean_frames(pd.read_parquet(root / "sportvu_contract" / f"{g}.parquet"))
        rows += G.release_rows(frames, att[att["sportvu_game"] == g])
    rel = G._release_frame_table(rows)
    players = sorted({int(p) for c in ("tm1", "tm2", "tm3", "tm4") for p in rel[c] if p >= 0} | set(att["player_id"].astype(int)))
    rng = np.random.default_rng(0)
    grav = dict(zip(players, rng.normal(0, 1, len(players))))
    pgs = pd.DataFrame([{"player_id": p, "sportvu_game": g, "g_sum": 100 * grav[p], "n": 100, "d_sum": 500.0,
                         "fold": fold[g]} for p in players for g in dev])
    data, _ = G.attach_teammates(att, rel, lambda f: G.player_means(pgs, f))
    sd = {k: float(data[k].std()) or 1.0 for k in ("g_sum", "g_max", "d_sum", "d_max")}
    mean = {k: float(data[k].mean()) for k in sd}
    for k in sd:
        data["z_" + k] = (data[k] - mean[k]) / sd[k]
    d = {"data": data, "pgs": pgs, "sd": sd, "mean": mean}
    testset = tmp_path / "set.json"
    testset.write_text(json.dumps({"sources": {R.TRACKING: {"games": test, "sha256": R.list_hash(test)}}}))
    return root, d, test, testset, tmp_path / "ledger.json"


def test_the_scorer_runs_through_score_once_on_a_synthetic_root(tmp_path, monkeypatch):
    """A rehearsal of e1-final's scorer: a fault here would otherwise spend the one score."""
    from tools import research_testset as R
    root, d, test, testset, ledger = _rehearsal(tmp_path, monkeypatch)
    prov = {"code_commit": "rehearsal", "gate_files_sha256": {}, "dev_json_sha256": "x"}
    out = R.score_once("gravity-E1-rehearsal", R.TRACKING, G.make_scorer(d, 0.3, None, prov), testset_path=testset,
                       ledger_path=ledger)
    json.dumps(out, allow_nan=False)
    assert "verdict" not in out, out.get("verdict")
    assert out["games"] == len(test) and out["attempts"] > 0 and out["code_commit"] == "rehearsal"
    assert out["reading"] in ("confirms", "consistent", "inconsistent") and out["development_effect"] == 0.3
    assert json.loads(ledger.read_text())[0]["status"] == "scored"
    with pytest.raises(SystemExit):
        G.make_scorer(d, 0.3, None, prov)(test)                # outside score_once the test data is refused


def test_the_scorer_refuses_a_test_game_without_frames(tmp_path, monkeypatch):
    from tools import research_testset as R
    root, d, test, testset, ledger = _rehearsal(tmp_path, monkeypatch)
    (root / "sportvu_contract" / f"{test[0]}.parquet").unlink()
    out = R.score_once("gravity-E1-rehearsal", R.TRACKING, G.make_scorer(d, 0.3, None, {}), testset_path=testset,
                       ledger_path=ledger)
    assert out["verdict"]["refused"] and out["games_without_frames"] == [test[0]]
    json.dumps(out, allow_nan=False)
