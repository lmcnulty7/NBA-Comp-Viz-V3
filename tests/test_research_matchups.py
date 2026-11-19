"""matchups-E1 machinery (docs/research/matchups.md, the D108 amendment): the penalized logistic fit, the penalty
grid, the two parts' rows, the pre-shot shares, the Set A readings, the door guards, and a rehearsal of the scorer
through score_once on a synthetic root. No real data, no held-out game, no test game."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import research_matchups as M

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "tools" / "research_matchups.py"
DOOR_NAMES = ("research_data_root", "test_features")


def test_fit_logit_matches_sklearn_with_a_free_intercept():
    import scipy.sparse as sp
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    n, k = 3000, 8
    Z = rng.normal(size=(n, k))
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.3 + Z @ rng.normal(size=k))))).astype(float)
    X = sp.csr_matrix(np.c_[np.ones(n), Z])
    lam = 5.0
    b = M.fit_logit(X, y, np.zeros(n), M._penalty_vector(k + 1, 1, lam))
    sk = LogisticRegression(C=1 / lam, tol=1e-10, max_iter=10000).fit(Z, y)
    assert abs(b[0] - sk.intercept_[0]) < 1e-4 and np.allclose(b[1:], sk.coef_[0], atol=1e-4)


def test_an_offset_shifts_the_linear_predictor_exactly():
    import scipy.sparse as sp
    rng = np.random.default_rng(1)
    n = 4000
    off = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-off))).astype(float)
    b = M.fit_logit(sp.csr_matrix(np.ones((n, 1))), y, off, np.zeros(1))
    assert abs(b[0]) < 0.1                              # the offset already explains the outcome


class _D:
    """A minimal design for choose_lambda."""

    def __init__(self, X, y, groups, n_free=1):
        import scipy.sparse as sp
        self.X, self.y, self.groups, self.n_free = sp.csr_matrix(X), y, groups, n_free
        self.offset = np.zeros(len(y))
        self.shape = self.X.shape


@pytest.mark.parametrize("target, expect", [(30000.0, 27000.0), (0.01, 1 / 81)])
def test_the_penalty_grid_grows_past_whichever_end_wins(monkeypatch, target, expect):
    """A fake fit whose cross-validated loss is minimal at `target`: the search must leave the grid in that direction
    and stop at the first interior minimum."""
    import math
    monkeypatch.setattr(M, "fit_logit", lambda X, y, off, pen, beta0=None: np.array([pen[-1]]))
    monkeypatch.setattr(M, "row_loss", lambda y, z: np.full(len(y), 0.0))

    class Fake(_D):
        pass
    n = 300
    D = Fake(np.ones((n, 2)), np.zeros(n), np.array([f"g{i % 10}" for i in range(n)]), n_free=0)
    calls = []

    def loss(y, z):
        lam = float(np.asarray(z).ravel()[0]) if np.size(z) else 1.0
        calls.append(lam)
        return np.full(len(y), (math.log(lam) - math.log(target)) ** 2)
    monkeypatch.setattr(M, "row_loss", loss)
    D.X = __import__("scipy.sparse", fromlist=["csr_matrix"]).csr_matrix(np.ones((n, 1)))
    lam, curve = M.choose_lambda(D, np.arange(n))
    assert math.isclose(lam, expect, rel_tol=1e-9)
    assert any(float(k) > max(M.LAMBDA_GRID) for k in curve) or any(float(k) < min(M.LAMBDA_GRID) for k in curve)


def _toy_tables():
    zones = pd.DataFrame({"sportvu_game": "G", "possession": [1] * 5 + [2] * 5,
                          "off_player_id": [1, 2, 3, 4, 5] * 2,
                          "zone_rim": 0.1, "zone_paint": 0.2, "zone_mid": 0.3, "zone_three": 0.4})
    shots = pd.DataFrame({"sportvu_game": "G", "possession": [1, 1, 2], "event": [10, 12, 20],
                          "shooter_id": pd.array([2, 3, 9], dtype="Int64"), "assigned_def_id": pd.array([11, 12, pd.NA], dtype="Int64"),
                          "made": [True, False, True], "p": 0.45, "value": 2, "k": [3, 5, 4]})
    mp = pd.DataFrame({"sportvu_game": "G", "possession": [1, 1, 1, 2], "def_player_id": [11, 12, 13, 11],
                       "off_player_id": [3, 3, 2, 1], "partial_possession": [0.6, 0.4, 1.0, 1.0]})
    return mp, shots, zones


def test_part_a_has_one_shooter_per_kept_possession_and_drops_an_untracked_shooter():
    mp, shots, zones = _toy_tables()
    rows, shares, counts = M.part_a_rows(mp, shots, zones)
    assert counts["possessions_with_known_last_shooter"] == 2 and counts["possessions_kept"] == 1
    assert rows.groupby("possession")["y"].sum().to_dict() == {1: 1.0}
    assert rows.loc[rows["y"] == 1, "off_player_id"].tolist() == [3]      # the last attempt's shooter
    assert set(shares["def_player_id"]) == {11, 12, 13}


def test_part_b_uses_the_xfg_offset_and_one_defender_column():
    mp, shots, zones = _toy_tables()
    rows, counts = M.part_b_rows(shots, {("G", 10): 0.6, ("G", 12): 0.3})
    assert counts == {"attempts": 3, "with_assigned_defender": 2, "without_xfg": 0, "rows": 2}
    D = M.Design("b", rows)
    assert np.allclose(D.offset, np.log([0.6 / 0.4, 0.3 / 0.7]))
    assert D.X.toarray().tolist() == [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
    other = rows.assign(assigned_def_id=[11, 99])
    assert D.like(other).X.toarray().tolist() == [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]       # unseen: no column


def test_pre_shot_shares_keep_only_frames_two_seconds_before_the_release():
    mf = pd.DataFrame({"possession": [1] * 6, "k": [0, 1, 2, 3, 4, 5], "frame": range(6),
                       "def_player_id": [11, 11, 12, 12, 12, 12], "off_player_id": [3] * 6, "dist_ft": 1.0})
    shots = pd.DataFrame({"possession": [1], "event": [10], "k": [4]})
    rows = pd.DataFrame({"sportvu_game": ["G"] * 2, "possession": [1, 1], "off_player_id": [3, 2], "y": [1.0, 0.0]})
    r, sh = M.pre_shot_shares(iter([("G", mf, shots)]), rows)
    got = sh.set_index("def_player_id")["partial_possession"].to_dict()
    assert got == {11: 2 / 3, 12: 1 / 3} and len(r) == 2          # frames k = 0, 1, 2 only


def test_set_a_readings_follow_d108():
    assert M.reading([-0.003, -0.001]).startswith("confirms")
    assert M.reading([-0.001, 0.002]).startswith("inconclusive at 10 games")
    assert M.reading([0.001, 0.003]).startswith("contradiction")


def _tree():
    return ast.parse(SOURCE.read_text())


def test_test_data_is_reached_only_inside_the_scorer_given_to_score_once():
    tree = _tree()
    once = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "score_once"]
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
        if isinstance(node, ast.alias) and node.name in DOOR_NAMES:
            seen += 1
            assert id(node) in inside, f"{node.name} imported outside the scorer"
    assert seen >= 2


def test_the_module_never_runs_a_lane_command_line_and_lists_no_heldout_directory():
    text = SOURCE.read_text()
    for banned in ("T.run(", "train.main", "features.main", "load_games(", "heldout_once", "score_heldout("):
        assert banned not in text, banned
    assert "dev_games(" in text and "refuse_test(" in text and "clean_head()" in text


def _rehearsal(tmp_path, monkeypatch, n_games=8, n_test=2):
    """A synthetic root (analytics.fixtures), models fitted on its first games, its last games as a temporary test set."""
    from analytics.fixtures import write_fixture_root
    from analytics.matchups import train as T
    from tools import research_testset as R
    root = tmp_path / "root"
    write_fixture_root(root, seed=0, n_games=n_games, heldout_every=0, n_possessions=90)
    games = sorted(p.stem for p in (root / "analytics" / "possessions" / "2015-16").glob("*.parquet"))
    dev, test = games[:-n_test], games[-n_test:]

    def fake_p(gs):
        out = {}
        for g in gs:
            sf = pd.read_parquet(root / "analytics" / "shot_frames" / "2015-16" / f"{g}.parquet")
            out.update({(g, int(e)): 0.45 for e in sf["event"]})
        return out

    pmap = fake_p(dev)
    xdf = pd.DataFrame([{"sportvu_game": g, "event": e, "p_selected": p} for (g, e), p in pmap.items()])
    mps, shots, zones, mfs = [], [], [], []
    for g in M.game_dicts(root, dev):
        pt = M.assign_game(g)
        gid = str(g["sportvu_game"])
        t = T.matchup_tables({gid: {"mf": pt["mf"], "info": M.info_dict(pt["info"]), "shots": pt["shots"]}}, xdf, 0.45)
        mps.append(t["matchup_possessions"]); shots.append(t["shots"]); zones.append(pt["zones"])
        mfs.append((gid, pt["mf"], pt["shots"]))
    mp, sh, zn = (pd.concat(x, ignore_index=True) for x in (mps, shots, zones))
    a_rows, a_shares, _ = M.part_a_rows(mp, sh, zn)
    b_rows, _ = M.part_b_rows(sh, pmap)
    pre_rows, pre_shares = M.pre_shot_shares(iter(mfs), a_rows)
    designs = {"a": M.Design("a", a_rows, a_shares), "a0": M.Design("a", a_rows, a_shares, with_defenders=False),
               "b": M.Design("b", b_rows), "b0": M.Design("b", b_rows, with_defenders=False),
               "p": M.Design("a", pre_rows, pre_shares), "p0": M.Design("a", pre_rows, pre_shares, with_defenders=False)}
    models = {"default_p": 0.45}
    for k, D in designs.items():
        models[k] = {"template": D, "lambda": 10.0,
                     "beta": M.fit_logit(D.X, D.y, D.offset, M._penalty_vector(D.shape[1], D.n_free, 10.0))}
    testset = tmp_path / "set.json"
    testset.write_text(json.dumps({"sources": {R.TRACKING: {"games": test, "sha256": R.list_hash(test)}}}))
    monkeypatch.setenv("V3_RESEARCH_DATA", str(root))
    return root, models, test, testset, tmp_path / "ledger.json", fake_p


def test_the_scorer_runs_through_score_once_on_a_synthetic_root(tmp_path, monkeypatch):
    """A rehearsal of e1-final's scorer: a fault here would otherwise spend the one score."""
    from tools import research_testset as R
    root, models, test, testset, ledger, fake_p = _rehearsal(tmp_path, monkeypatch)
    scorer = M.make_scorer(models, None, {"code_commit": "rehearsal"}, xfg_p=fake_p)
    out = R.score_once("matchups-E1-rehearsal", R.TRACKING, scorer, testset_path=testset, ledger_path=ledger)
    json.dumps(out, allow_nan=False)
    assert "verdict" not in out, out.get("verdict")
    assert out["code_commit"] == "rehearsal" and out["counts"]["games"] == len(test)
    for part in ("a", "b"):
        r = out["parts"][part]
        assert r["games"] == len(test) and r["reading"].split(" ")[0] in ("confirms", "inconclusive", "contradiction")
        assert "t_interval_of_game_means" in r
    assert json.loads(ledger.read_text())[0]["status"] == "scored"
    with pytest.raises(SystemExit):
        scorer(test)                                     # outside score_once the test data is refused


def test_the_scorer_refuses_a_test_game_without_a_table_before_reading_values(tmp_path, monkeypatch):
    from tools import research_testset as R
    root, models, test, testset, ledger, fake_p = _rehearsal(tmp_path, monkeypatch)
    (root / "analytics" / "rosters" / "2015-16" / f"{test[0]}.parquet").unlink()
    out = R.score_once("matchups-E1-rehearsal", R.TRACKING, M.make_scorer(models, None, {}, xfg_p=fake_p),
                       testset_path=testset, ledger_path=ledger)
    assert out["verdict"]["refused"] and out["missing"] == {test[0]: ["rosters"]}


def test_non_finite_values_are_found():
    assert M.non_finite({"a": 1.0, "b": [0.5, float("nan")], "c": {"d": float("inf")}}) == ["b[1]", "c.d"]
