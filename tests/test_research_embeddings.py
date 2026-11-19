"""embeddings-E1 machinery (docs/research/embeddings.md, the D103 amendment): the sums-of-partials features against
the frozen feature functions on synthetic games, the retrieval rules, the bootstrap, and guards on the module's
source. No real data, no held-out game, no test game."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics.embeddings import features as F
from tools import research_embeddings as E

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "tools" / "research_embeddings.py"


def synthetic(n_games=8, players=40, seed=0, poss=60):
    """Per-game frozen tracking partials and shot rows of made-up games 'MM.DD.YYYY.AAA.at.BBB' (no real id). Each
    player has a persistent style (rates of every partial sum), so style is recoverable from a few games."""
    rng = np.random.default_rng(seed)
    games = [f"{m:02d}.{d:02d}.{y}.AAA.at.BBB" for (m, d, y) in
             [(11, 1, 2015), (12, 3, 2015), (1, 5, 2016), (2, 7, 2016), (10, 30, 2015), (3, 9, 2016), (1, 1, 2016),
              (12, 31, 2015), (2, 2, 2016), (3, 3, 2016), (11, 20, 2015), (1, 20, 2016)][:n_games]]
    style = rng.random((players, len(F.PARTIAL_SUMS))) + 0.2
    shot_style = rng.dirichlet(np.ones(6), size=players)
    tparts, shots = [], []
    for g in games:
        rows = []
        for p in range(players):
            if rng.random() < 0.1:
                continue                                    # a player misses some games
            frames = rng.integers(500, 3000)
            r = {"player_id": 1000 + p, "season": "2015-16", "sportvu_game": g,
                 "n_possessions": int(rng.integers(poss // 2, poss + 1))}
            for j, c in enumerate(F.PARTIAL_SUMS):
                r[c] = float(rng.poisson(style[p, j] * frames))
            rows.append(r)
            for _ in range(rng.poisson(6)):
                z = rng.choice(6, p=shot_style[p])
                cx, cy, st = [(3.0, 25.0, "2PT"), (12.0, 25.0, "2PT"), (20.0, 15.0, "2PT"), (2.0, 3.0, "3PT"),
                              (28.0, 25.0, "3PT"), (60.0, 25.0, "3PT")][z]
                shots.append({"sportvu_game": g, "player_id": 1000 + p, "court_x": cx, "court_y": cy,
                              "shot_type": st + " Field Goal", "season": "2015-16"})
        tparts.append(pd.DataFrame(rows))
    return tparts, pd.DataFrame(shots), games


def frozen_path(tparts, shots, keep_games):
    t = F.combine_partials([p for p in tparts if p["sportvu_game"].iloc[0] in keep_games])
    s = shots[shots["sportvu_game"].isin(keep_games)]
    return F.player_season_features(t, s, None, None, n_min=E.N_MIN).sort_values("player_id").reset_index(drop=True)


def experiment_path(tparts, shots, games, keep_games):
    P = E.Parts(pd.concat(tparts, ignore_index=True), E.shot_partials(shots, None), games)
    w = np.isin(np.array(P.games)[P.g], keep_games).astype(float)[None, :]
    X, n = E.features_from_sums(P.sums(w))
    out = pd.DataFrame(X[0], columns=E.feature_order())
    out["player_id"], out["n_possessions"] = P.players, n[0]
    return out[n[0] > 0].sort_values("player_id").reset_index(drop=True)


@pytest.mark.parametrize("subset", [slice(None), slice(0, 3), slice(2, 7)])
def test_features_from_partial_sums_equal_the_frozen_feature_functions(subset):
    tparts, shots, games = synthetic()
    keep = games[subset]
    a, b = frozen_path(tparts, shots, keep), experiment_path(tparts, shots, games, keep)
    assert list(a["player_id"]) == list(b["player_id"])
    assert (a["n_possessions"].to_numpy() == b["n_possessions"].to_numpy()).all()
    x, y = a[E.feature_order()].to_numpy(float), b[E.feature_order()].to_numpy(float)
    assert (np.isnan(x) == np.isnan(y)).all()
    assert np.nanmax(np.abs(x - y)) < 1e-12


def test_halves_alternate_each_players_games_by_date():
    tparts, shots, games = synthetic(n_games=6, players=3)
    P = E.Parts(pd.concat(tparts, ignore_index=True), E.shot_partials(shots, None), games)
    half = P.halves()
    assert P.games == sorted(games, key=E.game_date)
    for p in range(P.P):
        rows = np.flatnonzero(P.p == p)
        rows = rows[np.argsort(P.g[rows])]
        assert list(half[rows]) == [i % 2 for i in range(len(rows))]


def test_hits_count_only_strictly_closer_pool_rows():
    C = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    Q = np.array([[0.0], [0.0]])
    assert list(E.hits(Q, C, np.array([4, 5]))) == [True, False]     # 4 strictly closer than row 4; 5 than row 5
    Q = np.array([[3.0]])
    assert list(E.hits(Q, np.array([[2.0], [4.0], [3.0], [1.0], [5.0], [0.0], [6.0]]), np.array([0])))[0]


def test_the_position_retriever_probability_is_exact():
    pool = np.array(["G"] * 10 + ["C"] * 3 + [np.nan], dtype=object)
    p = E.position_probability(np.array(["G", "C", np.nan], dtype=object), pool)
    assert np.allclose(p, [0.5, 1.0, 5 / 14])


def test_cluster_bootstrap_resamples_whole_groups():
    groups = np.array(["a"] * 4 + ["b"] * 4)
    vals = {"x": np.array([1.0] * 4 + [0.0] * 4)}
    out = E.cluster_bootstrap(groups, vals, n=400)
    iv = out["intervals"]["x"]
    assert out["groups"] == 2 and iv["value"] == 0.5 and iv["lo"] == 0.0 and iv["hi"] == 1.0


def _fitted(players=60, n_games=10, seed=1, poss=60):
    tparts, shots, games = synthetic(n_games=n_games, players=players, seed=seed, poss=poss)
    P = E.Parts(pd.concat(tparts, ignore_index=True), E.shot_partials(shots, None), games)
    X, n = E.features_from_sums(P.sums(np.ones((1, P.R))))
    feats = pd.DataFrame(X[0], columns=E.feature_order())
    feats.insert(0, "season", "2015-16")
    feats.insert(0, "player_id", P.players)
    feats["n_possessions"], feats["insufficient"] = n[0], n[0] < E.N_MIN
    return P, feats, E.frozen_fit(feats), tparts, shots, games


def test_the_pca_space_of_represent_is_the_frozen_embedding():
    P, feats, fit, *_ = _fitted()
    rep = E.represent(fit, fit["rows"][E.feature_order()].to_numpy(float))
    assert np.allclose(rep["pca"], fit["fits"]["full"]["E"]) and np.allclose(rep["raw"], fit["fits"]["full"]["Z"])
    assert rep["movement"].shape[1] == 8 and rep["shot"].shape[1] == 8


def test_split_half_and_leave_one_game_out_find_a_persistent_style():
    P, feats, fit, *_ = _fitted(players=60, n_games=12, poss=120)
    pos = pd.DataFrame({"player_id": P.players, "season": "2015-16", "position": ["G", "F", "C"] * 20})
    d = {"P": P, "fit": fit, "feats": feats, "positions": pos, "games": P.games}
    sh = E.split_half(d, n=20, log=None)
    assert sh["players"] > 20
    assert sh["rates_with_player_bootstrap"]["pca"]["value"] > 5 * sh["chance"]
    lo = E.leave_one_game_out(d, n=50, log=None)
    assert lo["queries"] > 0 and set(lo["rates_with_game_bootstrap"]) >= {"pca", "position", "pca_minus_position"}


def test_the_one_game_test_scores_only_queries_with_a_development_embedding(monkeypatch):
    P, feats, fit, tparts, shots, games = _fitted(players=60, n_games=10)
    pos = pd.DataFrame({"player_id": P.players, "season": "2015-16", "position": ["G", "F", "C"] * 20})
    d = {"P": P, "fit": fit, "feats": feats, "positions": pos, "games": P.games}
    test, tshots, tgames = synthetic(n_games=2, players=60, seed=1, poss=120)
    test[0].loc[0, "player_id"] = 99999                     # a player with no development embedding
    P_test = E.Parts(pd.concat(test, ignore_index=True), E.shot_partials(tshots, None), tgames)
    s = E.one_game_summary(P_test, d, n=50)
    assert s["counts"]["no_development_embedding"] >= 1
    assert s["queries"] == s["counts"]["with_50_possessions"] - s["counts"]["no_development_embedding"]
    assert set(s["rule"]) == {"lower_end", "position_upper_end", "threshold", "passes"} and s["rule"]["threshold"] == 0.10


# ----------------------------------------------------------------------------- source guards

DOOR_NAMES = ("research_data_root",)


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
    seen = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assert not (alias.name in DOOR_NAMES and alias.asname), "aliased import"
        if isinstance(node, ast.Name) and node.id in DOOR_NAMES or isinstance(node, ast.Attribute) and node.attr in DOOR_NAMES:
            seen += 1
            assert id(node) in inside, f"{ast.unparse(node)} used outside the scorer"
        if isinstance(node, ast.Constant) and node.value in DOOR_NAMES:
            raise AssertionError("named as a string")
    assert seen >= 1


def test_the_module_never_runs_the_lane_command_line_and_reads_only_development_games():
    text = SOURCE.read_text()
    for banned in ("T.run(", "write_tables", "train.main", "features.main", "heldout_once", "load_inputs("):
        assert banned not in text, banned
    assert "dev_games(" in text and "training_games_only(" in text and "refuse_test(" in text
    shot_reads = [c for c in _calls(_tree(), "read_parquet") if c.args and "SHOTS_FILE" in ast.unparse(c.args[0])]
    assert len(shot_reads) >= 2                                # development and scorer
    for c in shot_reads:                                       # shot rows of other games are never materialised
        f = next((k.value for k in c.keywords if k.arg == "filters"), None)
        assert f is not None and '"sportvu_game", "in"' in ast.unparse(f).replace("'", '"'), ast.unparse(c)


def _bound_to_work_dir(fn, name) -> bool:
    bound = [ast.unparse(a.value) for a in ast.walk(fn) if isinstance(a, ast.Assign)
             and any(isinstance(x, ast.Name) and x.id == name for x in a.targets)]
    return bool(bound) and all(b.startswith("work_dir(") for b in bound if "work_dir" in b or "/" in b) \
        and any(b.startswith("work_dir(") for b in bound)


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
            if isinstance(dest, ast.Name):
                bound = [ast.unparse(a.value) for a in ast.walk(fn) if isinstance(a, ast.Assign)
                         and any(isinstance(x, ast.Name) and x.id == dest.id for x in a.targets)]
                assert len(bound) == 1, f"{fn.name}: {dest.id}"
                text = bound[0].lstrip("(")
            if text.startswith("RESULTS /"):
                continue
            base = text.split(" /")[0]
            assert base in ("out", "folder"), f"{fn.name}: {text}"
            params = [a.arg for a in fn.args.args]
            if base in params:                  # a helper: every caller passes a folder bound to work_dir
                k = params.index(base)
                callers = [(f, c) for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
                           for c in _calls(f, fn.name)]
                assert callers, fn.name
                for f, c in callers:
                    arg = c.args[k]
                    assert isinstance(arg, ast.Name) and _bound_to_work_dir(f, arg.id), f"{f.name} calls {fn.name}"
            else:
                assert _bound_to_work_dir(fn, base), f"{fn.name}: {text}"
    assert checked >= 6
