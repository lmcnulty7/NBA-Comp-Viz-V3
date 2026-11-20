"""rebounds-E2 machinery (docs/research/rebounds.md, "rebounds-E2"; PLAN.md D124), on synthetic candidate frames only:
the within-miss softmax objective against a brute-force softmax, the contiguity and sum guards, the softmax and binary
arms with the fixed XGBoost settings, and the development statistic. No DATA_ROOT, no test game."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics.rebounds import features as F
from tools import research_rebounds_e2 as E

REPO = Path(__file__).resolve().parents[1]


def synthetic(n_games=12, misses=25, seed=0):
    """Candidate frames with a known within-miss softmax truth: 8 to 11 rows a miss, one of them the none row;
    players recur across games so the frozen prior has something to learn."""
    rng = np.random.default_rng(seed)
    w = {"distance_to_rim_ft": -0.25, "distance_to_ball_landing_ft": -0.35, "is_offense": -0.4,
         "nearest_opponent_dist_ft": 0.1, "n_opponents_closer_to_rim": -0.3, "is_none": 0.5}
    rows = []
    for gi in range(n_games):
        game = f"11.{gi + 1:02d}.2015.AAA.at.BBB"
        for ev in range(misses):
            n = int(rng.integers(8, 12))
            feats = {c: rng.normal(0, 1, n) for c in F.STATIC_FEATURES}
            feats["distance_to_rim_ft"] = rng.uniform(1, 25, n)
            feats["distance_to_ball_landing_ft"] = rng.uniform(0, 20, n)
            feats["is_offense"] = (np.arange(n) % 2).astype(float)
            feats["n_opponents_closer_to_rim"] = rng.integers(0, 5, n).astype(float)
            feats["is_none"] = np.zeros(n)
            feats["is_none"][-1] = 1.0
            score = sum(w.get(c, 0.0) * feats[c] for c in F.STATIC_FEATURES)
            p = np.exp(score - score.max())
            p /= p.sum()
            winner = rng.choice(n, p=p)
            for i in range(n):
                r = {c: float(feats[c][i]) for c in F.STATIC_FEATURES}
                r.update(sportvu_game=game, event=ev, heldout=False, is_rebounder=int(i == winner),
                         candidate_player_id=-1 if i == n - 1 else int(rng.integers(1, 60)))
                rows.append(r)
    return pd.DataFrame(rows)


def brute_softmax(scores, groups):
    out = np.empty(len(scores))
    for g in np.unique(groups):
        m = groups == g
        e = np.exp(scores[m] - scores[m].max())
        out[m] = e / e.sum()
    return out


def test_gradient_and_hessian_match_a_brute_force_softmax_and_finite_differences():
    rng = np.random.default_rng(1)
    sizes = [3, 5, 2, 7]
    groups = np.repeat(np.arange(len(sizes)), sizes)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    s = rng.normal(0, 2, len(groups))
    y = np.zeros(len(groups))
    for st in starts[:-1]:
        y[st + int(rng.integers(0, 2))] = 1.0
    p = brute_softmax(s, groups)
    assert np.allclose(E.softmax_blocks(s, starts), p)
    grad, hess = E.softmax_grad_hess(s, y, starts)
    assert np.allclose(grad, p - y) and np.allclose(hess, p * (1 - p))

    def total(v):                                       # the summed loss: its gradient is the objective's gradient
        return E.softmax_loss(v, y, starts) * len(sizes)
    h = 1e-6
    for i in range(len(s)):
        e = np.zeros(len(s))
        e[i] = h
        num_g = (total(s + e) - total(s - e)) / (2 * h)
        assert abs(num_g - grad[i]) < 1e-6
        g_plus, _ = E.softmax_grad_hess(s + e, y, starts)
        g_minus, _ = E.softmax_grad_hess(s - e, y, starts)
        assert abs((g_plus[i] - g_minus[i]) / (2 * h) - hess[i]) < 1e-6      # the diagonal of the Hessian


def test_the_guard_refuses_split_misses_and_misses_without_one_rebounder():
    f = synthetic(n_games=2, misses=3)
    E.check_blocks(f)
    swapped = pd.concat([f.iloc[[0]], f.iloc[20:21], f.iloc[1:20], f.iloc[21:]]).reset_index(drop=True)
    with pytest.raises(ValueError, match="not contiguous"):
        E.check_blocks(swapped)
    two = f.copy()
    two.loc[two.index[:3], "is_rebounder"] = 1
    with pytest.raises(ValueError, match="exactly one rebounder"):
        E.check_blocks(two)
    starts = E.check_blocks(f)
    with pytest.raises(ValueError, match="sum to 1"):
        E.check_sums(np.full(len(f), 0.5), starts)


def test_the_settings_are_the_pre_registered_ones():
    assert (E.XGB_PARAMS["max_depth"], E.XGB_PARAMS["eta"], E.ROUNDS) == (3, 0.1, 80)
    assert E.THRESHOLD == -0.010 and E.FOLDS == 5 and E.BOOT == 2000


def test_the_softmax_arm_learns_the_within_miss_competition():
    train, test = synthetic(n_games=10, seed=0), synthetic(n_games=4, seed=1)
    p = E.fit_softmax(train)(test)
    starts = E.check_blocks(test)
    E.check_sums(p, starts)
    y = test["is_rebounder"].astype(bool).to_numpy()
    model_loss = -np.log(p[y]).mean()
    uniform_loss = np.log(np.diff(starts)).mean()
    assert model_loss < uniform_loss - 0.1


def test_the_binary_ablation_uses_the_frozen_tau_procedure():
    from analytics.rebounds import train as T
    train, test = synthetic(n_games=8, seed=2), synthetic(n_games=3, seed=3)
    model = E.fit_binary(train)
    assert T.TAU_BOUNDS[0] <= model.tau <= T.TAU_BOUNDS[1]
    E.check_sums(model(test), E.check_blocks(test))


def test_the_development_statistic_runs_and_pairs_the_arms_on_the_same_misses():
    cands = synthetic(n_games=10, misses=15, seed=4).sample(frac=1.0, random_state=0)      # shuffled rows
    cands = cands.sort_values(["sportvu_game", "event"], kind="stable")                     # misses rejoined
    pred, info = E.out_of_fold(cands)
    assert info["folds"] == 5 and len(info["binary_tau_by_fold"]) == 5
    s = E.summarize(pred, n_boot=200)
    assert s["misses"] == 150 and set(s["paired"]) == {"softmax_minus_baseline", "binary_minus_baseline",
                                                       "softmax_minus_binary"}
    a, b = s["arms"]["softmax"]["per_miss_log_loss"]["value"], s["arms"]["baseline"]["per_miss_log_loss"]["value"]
    assert s["paired"]["softmax_minus_baseline"]["value"] == pytest.approx(a - b)
    assert set(s["verdict"]) >= {"met", "reading", "difference", "lo", "hi"}


def test_out_of_fold_refuses_a_frame_where_a_rebounder_row_landed_in_another_miss():
    cands = synthetic(n_games=3, misses=4, seed=5)
    bad = cands.copy()
    first = bad[(bad["sportvu_game"] == bad["sportvu_game"].iloc[0]) & (bad["event"] == 0)]
    bad.loc[first.index[first["is_rebounder"] == 1][0], "event"] = 3     # miss 0 loses its rebounder, miss 3 has two
    with pytest.raises(ValueError, match="exactly one rebounder"):
        E.out_of_fold(bad)


def test_verdict_readings():
    assert E.verdict({"value": -0.02, "lo": -0.03, "hi": -0.01})["met"] is True
    assert "smaller than" in E.verdict({"value": -0.005, "lo": -0.008, "hi": -0.002})["reading"]
    assert "includes zero" in E.verdict({"value": -0.02, "lo": -0.05, "hi": 0.01})["reading"]
    assert "worse" in E.verdict({"value": 0.02, "lo": 0.01, "hi": 0.03})["reading"]


def test_the_module_reads_data_only_in_the_feasibility_check_and_never_test_data():
    tree = ast.parse((REPO / "tools" / "research_rebounds_e2.py").read_text())
    calls = {getattr(n.func, "attr", getattr(n.func, "id", None)) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not calls & {"scoring_data_root"}
    feas = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "feasibility")
    readers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("load_e1", "game_bundle")]
    inside = {id(c) for fn in [feas] + readers for c in ast.walk(fn)}
    reads = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "read_parquet"]
    assert reads and all(id(n) in inside for n in reads)
    assert "refuse_test(" in ast.unparse(feas) and "fold" in ast.unparse(feas)


def test_the_fold_and_dev_stages_reuse_e1s_folds_and_baseline(tmp_path, monkeypatch):
    """A synthetic rebounds-E1 artifacts folder (candidate rows, a passed gate, five fold pieces with the baseline's
    p) -> the two stages: every fold written, the baseline equal to E1's, the arms paired on the same misses."""
    import json
    from analytics.eval import grouped_folds
    from analytics.rebounds import train as T
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data_root"))
    cands = synthetic(n_games=10, misses=12, seed=6).reset_index(drop=True)
    e1 = tmp_path / "e1" / "research" / "rebounds_e1"
    (e1 / "folds").mkdir(parents=True)
    cands.to_parquet(e1 / "candidates_train.parquet", index=False)
    (e1 / "candidates_train.json").write_text(json.dumps({"rows": len(cands), "matches_frozen_record": True}))
    games = sorted(cands["sportvu_game"].unique())
    g = cands["sportvu_game"]
    p_all = np.full(len(cands), np.nan)
    for k, (tr_g, te_g) in enumerate(grouped_folds(games, n_folds=5, seed=0)):
        tr, te = np.flatnonzero(g.isin(tr_g)), np.flatnonzero(g.isin(te_g))
        p = E.frozen_baseline_fit(cands.iloc[tr].reset_index(drop=True))(cands.iloc[te].reset_index(drop=True))
        p_all[te] = p
        np.savez(e1 / "folds" / f"fold_{k}.npz", test_rows=te, train_rows=tr, p_base=p)
    pred = cands.assign(p_hgb=p_all)
    ll = T.mlogloss(T.per_miss(pred, "p_hgb"))
    (e1 / "gate.json").write_text(json.dumps({"passed": True, "rows_mine": len(cands), "per_miss_log_loss": ll}))
    monkeypatch.setattr(E, "RESULTS", tmp_path / "results")
    (tmp_path / "results").mkdir()
    out = E.fold_stage(tmp_path / "e1", tmp_path / "mine", log=lambda *a, **k: None, head="x")
    assert out["left"] == [] and out["done_this_job"] == [0, 1, 2, 3, 4]
    v = E.dev_stage(tmp_path / "e1", tmp_path / "mine", head="x")
    doc = json.loads((tmp_path / "results" / "rebounds-E2-dev.json").read_text())
    assert doc["arms"]["baseline"]["per_miss_log_loss"]["value"] == pytest.approx(ll)
    assert doc["misses"] == 120 and len(doc["binary_tau_by_fold"]) == 5 and set(v) >= {"met", "reading"}
    assert set(doc["projected_set_a"]) >= {"random_10_game_sets_half_width", "dev_half_width_scaled"}


# ── the Set A confirmation (D127), on a synthetic root ──────────────────────────────────────────────────────────

def test_readings_are_d108s():
    assert E.reading(-0.03, -0.001) == "confirms"
    assert E.reading(-0.03, 0.01) == "inconclusive at 10 games"
    assert E.reading(0.001, 0.03) == "contradiction"


def test_the_rehearsal_check_is_exact():
    a = synthetic(n_games=2, misses=3, seed=7).reset_index(drop=True)
    assert E.rehearsal_passed(E.rehearsal_check(a, a.copy()))
    b = a.copy()
    b.loc[4, "distance_to_rim_ft"] += 1e-12
    assert not E.rehearsal_passed(E.rehearsal_check(a, b))
    c = a.copy()
    c.loc[2, "ball_apex_ft"] = np.nan
    assert not E.rehearsal_passed(E.rehearsal_check(a, c))
    assert not E.rehearsal_passed(E.rehearsal_check(a, a.iloc[::-1].reset_index(drop=True)))


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    from analytics.fixtures import write_fixture_root
    root = tmp_path_factory.mktemp("root")
    write_fixture_root(root, seed=0, n_games=8, heldout_every=0, n_possessions=90)
    games = sorted(p.stem for p in (root / "analytics" / "possessions" / "2015-16").glob("*.parquet"))
    return root, games


def test_the_scorer_runs_once_through_score_once_on_a_synthetic_root(fixture_root, tmp_path, monkeypatch):
    import json
    from tools import research_testset as R
    root, games = fixture_root
    dev, test = games[:-2], games[-2:]
    cands = E.build_candidates(root, games)
    assert set(cands["sportvu_game"].astype(str)) == set(games)
    models = E.fit_dev_models(cands, dev)
    testset = tmp_path / "set.json"
    testset.write_text(json.dumps({"sources": {R.TRACKING: {"games": test, "sha256": R.list_hash(test)}}}))
    monkeypatch.setenv("V3_RESEARCH_DATA", str(root))
    out = R.score_once("rebounds-E2", R.TRACKING, E.make_scorer(models, provenance={"code_commit": "x"}),
                       testset, tmp_path / "ledger.json")
    assert out["games"] == 2 and out["reading"] in ("confirms", "inconclusive at 10 games", "contradiction")
    assert set(out["t_interval_softmax_minus_baseline_per_game"]) >= {"lo", "hi", "games"}
    with pytest.raises(SystemExit):
        R.score_once("rebounds-E2", R.TRACKING, E.make_scorer(models), testset, tmp_path / "ledger.json")


def test_the_scorer_refuses_a_missing_table_before_reading_values(fixture_root, tmp_path, monkeypatch):
    import json
    import shutil
    from tools import research_testset as R
    root, games = fixture_root
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    (copy / "analytics" / "rosters" / "2015-16" / f"{games[-1]}.parquet").unlink()
    testset = tmp_path / "set.json"
    testset.write_text(json.dumps({"sources": {R.TRACKING: {"games": games[-2:], "sha256": R.list_hash(games[-2:])}}}))
    monkeypatch.setenv("V3_RESEARCH_DATA", str(copy))
    out = R.score_once("rebounds-E2", R.TRACKING, E.make_scorer({}), testset, tmp_path / "ledger.json")
    assert out["refused"] is True and games[-1] in out["missing"]


def test_set_a_is_reached_only_inside_the_scorer_and_final_checks_the_tree_first():
    tree = ast.parse((REPO / "tools" / "research_rebounds_e2.py").read_text())
    scorer = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "scorer")
    inside = {id(c) for c in ast.walk(scorer)}
    reads = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", None)) == "research_data_root"]
    assert reads and all(id(n) in inside for n in reads)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "score_once"]
    assert len(calls) == 1
    final = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "final")
    first = next(s for s in final.body if not isinstance(s, (ast.Import, ast.ImportFrom, ast.Expr)))
    assert "clean_head()" in ast.unparse(first)
    assert "check_rehearsal(" in ast.unparse(final) and '["verdict"]["met"]' in ast.unparse(final).replace("'", '"')


# ── the pre-score review of 807bdc2c ────────────────────────────────────────────────────────────────────────────

def test_check_rehearsal_refuses_another_commit_and_a_failed_record(tmp_path):
    import json
    ok = {"code_commit": "abc", "rows_match_cache": True, "shared_columns_match_cache": {"x": True}}
    p = tmp_path / "r.json"
    p.write_text(json.dumps(ok))
    assert E.check_rehearsal(p, "abc")["code_commit"] == "abc"
    with pytest.raises(SystemExit, match="not at this HEAD"):
        E.check_rehearsal(p, "def")
    p.write_text(json.dumps({**ok, "shared_columns_match_cache": {"x": True, "y": False}}))
    with pytest.raises(SystemExit, match="did not pass"):
        E.check_rehearsal(p, "abc")
    with pytest.raises(SystemExit, match="rehearse the scorer first"):
        E.check_rehearsal(tmp_path / "missing.json", "abc")


def test_final_refuses_a_dirty_tree_before_anything_else(monkeypatch):
    import subprocess
    real = subprocess.run

    def fake(args, **kw):
        if args[:3] == ["git", "status", "--porcelain"]:
            return subprocess.CompletedProcess(args, 0, stdout=" M tools/research_rebounds_e2.py\n", stderr="")
        return real(args, **kw)
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(E, "load_e1", lambda *a, **k: pytest.fail("final read data before the tree check"))
    with pytest.raises(SystemExit, match="not clean"):
        E.final(None)


def test_the_baseline_is_fitted_on_the_rows_in_build_order(monkeypatch):
    cands = synthetic(n_games=4, misses=5, seed=8)
    shuffled_games = cands.sample(frac=1.0, random_state=1).drop_duplicates(["sportvu_game", "event"])
    build_order = pd.concat([cands[(cands["sportvu_game"] == g) & (cands["event"] == e)]
                             for g, e in zip(shuffled_games["sportvu_game"], shuffled_games["event"])],
                            ignore_index=True)                       # misses contiguous, not sorted
    seen = {}
    monkeypatch.setattr(E, "frozen_baseline_fit", lambda train: seen.setdefault("base", train.copy()) is None)
    monkeypatch.setattr(E, "fit_softmax", lambda train: seen.setdefault("soft", train.copy()) is None)
    monkeypatch.setattr(E, "fit_binary", lambda train: seen.setdefault("bin", train.copy()) is None)
    E.fit_dev_models(build_order, sorted(build_order["sportvu_game"].unique()))
    assert E.key_tuples(seen["base"]) == E.key_tuples(build_order)
    assert E.key_tuples(seen["soft"]) == E.key_tuples(build_order.iloc[E._ordered(build_order)])


def test_the_scorer_records_build_counts_and_refuses_on_a_guard_failure(fixture_root, tmp_path, monkeypatch):
    import json
    from tools import research_testset as R
    root, games = fixture_root
    testset = tmp_path / "set.json"
    testset.write_text(json.dumps({"sources": {R.TRACKING: {"games": games[-2:], "sha256": R.list_hash(games[-2:])}}}))
    monkeypatch.setenv("V3_RESEARCH_DATA", str(root))

    def broken(frame):
        raise ValueError(f"{E.WHO}: a miss's probabilities do not sum to 1")
    out = R.score_once("rebounds-E2", R.TRACKING, E.make_scorer({"baseline": broken, "softmax": broken,
                                                                  "binary": broken}), testset, tmp_path / "ledger.json")
    assert out["refused"] is True and "guard failed" in out["reading"]
    counts = out["build_counts_by_game"]
    assert set(counts) == set(games[-2:]) and "unmatched_named_rebounder" in counts[games[-1]]
    entries = json.loads((tmp_path / "ledger.json").read_text())
    assert entries[-1]["status"] == "scored"                     # the refusal is on record, the score is spent
