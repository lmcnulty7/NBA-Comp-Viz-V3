"""matchups-E2 machinery (docs/research/matchups.md, the D112 amendment): the HMM's forward-backward and Viterbi
against brute force, the simplex step, EM recovering known parameters on synthetic chains, shares and switches, and
guards on the module's source. No real data, no held-out game, no test game."""

import ast
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import research_matchups_e2 as H

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "tools" / "research_matchups_e2.py"


def brute(logE, rho):
    """Every state path of short chains: log likelihood, state posteriors, the best path."""
    T = logE.shape[0]
    A = H._log_trans(rho)
    paths, scores = [], []
    for path in itertools.product(range(H.K), repeat=T):
        s = math.log(1 / H.K) + logE[0, path[0]] + sum(A[path[t - 1], path[t]] + logE[t, path[t]] for t in range(1, T))
        paths.append(path)
        scores.append(s)
    scores = np.array(scores)
    ll = np.log(np.exp(scores - scores.max()).sum()) + scores.max()
    post = np.zeros((T, H.K))
    for path, s in zip(paths, scores):
        for t, k in enumerate(path):
            post[t, k] += math.exp(s - ll)
    return ll, post, np.array(paths[int(np.argmax(scores))])


def test_forward_backward_and_viterbi_match_brute_force():
    rng = np.random.default_rng(0)
    logE = rng.normal(size=(3, 4, H.K))
    ll, post, _st = H.forward_backward(logE, 0.8)
    path = H.viterbi(logE, 0.8)
    for i in range(3):
        bl, bp, bpath = brute(logE[i], 0.8)
        assert abs(ll[i] - bl) < 1e-9 and np.allclose(post[i], bp, atol=1e-9)
        assert (path[i] == bpath).all()


def test_simplex_weights_stay_on_the_simplex_and_solve_the_free_case():
    S = np.array([[2.0, 0.3], [0.3, 1.0]])
    g = np.array([0.4, 0.2])
    assert np.allclose(H.simplex_weights(S, S @ g), g)
    for v in (np.array([-1.0, 3.0]), np.array([5.0, 5.0]), np.array([-1.0, -1.0])):
        w = H.simplex_weights(S, v)
        assert w.min() >= -1e-12 and w.sum() <= 1 + 1e-12


def synthetic_rows(n_poss=400, T=12, p=None, seed=1):
    """Cache-shaped rows drawn from the model with known parameters."""
    p = p or {"gO": 0.6, "gB": 0.15, "rho": 0.9, "s2": 2.0}
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_poss):
        offs = sorted(rng.choice(np.arange(100, 200), size=5, replace=False))
        O = rng.normal(0, 12, size=(T, 5, 2))
        B = O[np.arange(T), rng.integers(0, 5, size=T)] + rng.normal(0, 1, size=(T, 2))
        for dfn in range(5):
            k = int(rng.integers(0, 5))
            for t in range(T):
                if t and rng.random() > p["rho"]:
                    k = int(rng.choice([j for j in range(5) if j != k]))
                mu = p["gO"] * O[t, k] + p["gB"] * B[t]
                x = mu + rng.normal(0, math.sqrt(p["s2"]), size=2)
                rows.append(("G", q, t, t, 1000 + dfn, x[0], x[1], B[t, 0], B[t, 1], *offs, *O[t, :, 0], *O[t, :, 1]))
    return pd.DataFrame(rows, columns=H.CACHE_COLUMNS), p


def test_em_recovers_known_parameters():
    rows, p = synthetic_rows()
    fit = H.em(H.Chains(rows), log=lambda *a, **k: None)["params"]
    assert abs(fit["gO"] - p["gO"]) < 0.02 and abs(fit["gB"] - p["gB"]) < 0.02
    assert abs(fit["rho"] - p["rho"]) < 0.02 and abs(fit["s2"] - p["s2"]) / p["s2"] < 0.05


def test_decoding_beats_chance_and_the_hmm_switches_less_than_emission_only():
    rows, p = synthetic_rows(n_poss=150)
    ch = H.Chains(rows)
    hmm, emi = H.decode(ch, p, "hmm"), H.decode(ch, p, "emission")
    assert H.switches(hmm)["mean_switches"] < H.switches(emi)["mean_switches"]
    sh = H.shares_of(hmm, rows)
    tot = sh.groupby(["possession", "def_player_id"])["partial_possession"].sum()
    assert np.allclose(tot, 1.0)                              # every defender's shares sum to one


def test_a_chain_breaks_when_the_attackers_change():
    rows, _p = synthetic_rows(n_poss=2, T=4)
    rows.loc[(rows["possession"] == 0) & (rows["k"] >= 2), "o0"] = 999
    ch = H.Chains(rows)
    assert ch.n_chains == 2 * 5 + 5                            # possession 0 splits each defender's chain in two


def test_the_module_reads_only_development_games_and_never_runs_a_lane_command_line():
    text = SOURCE.read_text()
    for banned in ("T.run(", "train.main", "features.main", "load_games(", "heldout_once", "score_heldout("):
        assert banned not in text, banned
    assert "development_games(" in text and "refuse_test(" in text and "clean_head()" in text


def test_set_a_readings_follow_d112_and_the_development_decision():
    assert H.reading([0.001, 0.003], dev_passed=True).startswith("confirms")
    assert H.reading([-0.003, -0.001], dev_passed=True).startswith("contradiction")
    assert H.reading([-0.001, 0.002], dev_passed=True).startswith("inconclusive at 10 games")
    assert H.reading([-0.003, -0.001], dev_passed=False) == "confirms the development fail (interval wholly below zero)"
    assert H.reading([0.001, 0.003], dev_passed=False).startswith("wholly above zero; does not overturn")
    assert H.reading([-0.001, 0.002], dev_passed=False).startswith("inconclusive at 10 games")
    assert H.reading(None, context=True) == "context, reads no verdict"


DOOR_NAMES = ("research_data_root",)


def test_test_data_is_reached_only_inside_the_scorer_given_to_score_once():
    tree = ast.parse(SOURCE.read_text())
    once = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "score_once"]
    assert len(once) == 1 and [ast.unparse(a) for a in once[0].args] == ["EXPERIMENT", "TRACKING", "scorer"]
    scorer = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "scorer")
    inside = {id(n) for n in ast.walk(scorer)}
    seen = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in DOOR_NAMES or isinstance(node, ast.Attribute) and node.attr in DOOR_NAMES:
            seen += 1
            assert id(node) in inside, f"{ast.unparse(node)} used outside the scorer"
        if isinstance(node, ast.alias) and node.name in DOOR_NAMES:
            seen += 1
            assert id(node) in inside and not node.asname, f"{node.name} imported outside the scorer or aliased"
    assert seen >= 2


def _rehearsal(tmp_path, monkeypatch, n_games=8, n_test=2):
    """A synthetic root (analytics.fixtures), every model fitted on its first games, its last games as a temporary
    test set with a temporary ledger."""
    from analytics.fixtures import write_fixture_root
    from analytics.matchups import train as T
    from tools import research_matchups as E1
    from tools import research_testset as R
    root = tmp_path / "root"
    write_fixture_root(root, seed=0, n_games=n_games, heldout_every=0, n_possessions=90)
    games = sorted(p.stem for p in (root / "analytics" / "possessions" / "2015-16").glob("*.parquet"))
    dev, test = games[:-n_test], games[-n_test:]
    empty = pd.DataFrame(columns=["sportvu_game", "event", "p_selected"])
    mps, shots, zones, mfs, cells = [], [], [], [], []
    for g in E1.game_dicts(root, dev):
        pt = H.game_parts(g)
        gid = str(g["sportvu_game"])
        t = T.matchup_tables({gid: {"mf": pt["mf"], "info": pt["info"], "shots": pt["shots"]}}, empty, 0.45)
        mps.append(t["matchup_possessions"]); shots.append(t["shots"]); zones.append(pt["zones"])
        mfs.append((gid, pt["mf"], pt["shots"])); cells.append(pt["cells"])
    mp, sh, zn = (pd.concat(x, ignore_index=True) for x in (mps, shots, zones))
    rows_all = H.complete_cells(pd.concat(cells, ignore_index=True))
    a_rows, a_shares, _ = E1.part_a_rows(mp, sh, zn)
    pre_rows, pre_shares = E1.pre_shot_shares(iter(mfs), a_rows)
    params = {**H.START, "s2": 20.0}
    pre_cells = H.pre_shot_cut(rows_all, sh)
    keep = lambda t, rows: t.merge(rows[["sportvu_game", "possession", "off_player_id"]], how="inner",
                                   on=["sportvu_game", "possession", "off_player_id"])
    D = {"hungarian": E1.Design("a", a_rows, a_shares),
         "hmm": E1.Design("a", a_rows, keep(H.shares_of(H.decode(H.Chains(rows_all), params, "hmm"), rows_all), a_rows)),
         "emission": E1.Design("a", a_rows, keep(H.shares_of(H.decode(H.Chains(rows_all), params, "emission"),
                                                             rows_all), a_rows)),
         "no_defender": E1.Design("a", a_rows, a_shares, with_defenders=False),
         "pre_hungarian": E1.Design("a", pre_rows, pre_shares),
         "pre_hmm": E1.Design("a", pre_rows, keep(H.shares_of(H.decode(H.Chains(pre_cells), params, "hmm"), pre_cells),
                                                  pre_rows)),
         "pre_no_defender": E1.Design("a", pre_rows, pre_shares, with_defenders=False)}
    models = {"params": params, "default_p": 0.45}
    for m, Dm in D.items():
        models[m] = {"template": Dm, "lambda": 1.0,
                     "beta": E1.fit_logit(Dm.X, Dm.y, Dm.offset, E1._penalty_vector(Dm.shape[1], Dm.n_free, 1.0))}
    testset = tmp_path / "set.json"
    testset.write_text(json.dumps({"sources": {R.TRACKING: {"games": test, "sha256": R.list_hash(test)}}}))
    monkeypatch.setenv("V3_RESEARCH_DATA", str(root))
    return root, models, test, testset, tmp_path / "ledger.json"


def test_the_scorer_runs_through_score_once_on_a_synthetic_root(tmp_path, monkeypatch):
    """A rehearsal of e2-final's scorer: a fault here would otherwise spend the one score."""
    from tools import research_testset as R
    root, models, test, testset, ledger = _rehearsal(tmp_path, monkeypatch)
    scorer = H.make_scorer(models, {"code_commit": "rehearsal"}, dev_passed=False)
    out = R.score_once("matchups-E2-rehearsal", R.TRACKING, scorer, testset_path=testset, ledger_path=ledger)
    json.dumps(out, allow_nan=False)
    assert "verdict" not in out, out.get("verdict")
    assert out["counts"]["games"] == len(test) and out["code_commit"] == "rehearsal"
    assert out["hmm_minus_hungarian"]["reading"].split(" ")[0] in ("confirms", "inconclusive", "wholly")
    assert "t_interval_of_game_means" in out["hmm_minus_hungarian"]
    assert out["pre_shot_context"]["reading"] == out["ablation"]["reading"] == "context, reads no verdict"
    assert json.loads(ledger.read_text())[0]["status"] == "scored"
    with pytest.raises(SystemExit):
        scorer(test)


def test_the_scorer_refuses_a_test_game_without_a_table(tmp_path, monkeypatch):
    from tools import research_testset as R
    root, models, test, testset, ledger = _rehearsal(tmp_path, monkeypatch)
    (root / "analytics" / "rosters" / "2015-16" / f"{test[0]}.parquet").unlink()
    out = R.score_once("matchups-E2-rehearsal", R.TRACKING, H.make_scorer(models, {}), testset_path=testset,
                       ledger_path=ledger)
    assert out["verdict"]["refused"] and out["missing"] == {test[0]: ["rosters"]}


def test_final_refuses_when_the_development_verdict_did_not_pass(tmp_path, monkeypatch):
    """PLAN.md D114: with a failed development verdict the Set A score is refused before anything is read."""
    from tools import research_matchups as E1
    monkeypatch.setattr(E1, "clean_head", lambda: "x")
    monkeypatch.setattr(E1, "require_gate", lambda artifacts=None: {"passed": True})
    monkeypatch.setattr(H, "RESULTS", tmp_path)
    (tmp_path / f"{H.EXPERIMENT}-dev.json").write_text(json.dumps({"verdict": {"passes": False}}))
    with pytest.raises(SystemExit, match="D114"):
        H.final(None)
