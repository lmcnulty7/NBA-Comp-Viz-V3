"""embeddings-E2 machinery (docs/research/embeddings.md, embeddings-E2 and its amendment, PLAN.md D118): the histogram
pass against the frozen zone counts, the fixed-basis KL transform, the arms' features, the retrieval statistics on a
synthetic root, the verdict rule, and guards on the module. Synthetic data only; no DATA_ROOT, no test game."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from analytics.embeddings import features as F
from tools import research_embeddings as E1
from tools import research_embeddings_e2 as E2

REPO = Path(__file__).resolve().parents[1]


def test_cells_are_one_foot_on_the_half_court_with_edge_clipping():
    assert E2.cell_index([0.0, 46.99, 47.5, -0.4], [0.0, 49.99, 51.0, -1.5]).tolist() == \
        [0, 46 * 50 + 49, 46 * 50 + 49, 0]
    assert E2.cell_index([10.2], [25.7]).tolist() == [10 * 50 + 25]


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    from analytics.fixtures import write_fixture_root
    root = tmp_path_factory.mktemp("root")
    write_fixture_root(root, seed=0, n_games=6, heldout_every=0, n_possessions=90)
    games = sorted((p.stem for p in (root / "analytics" / "possessions" / "2015-16").glob("*.parquet")),
                   key=E1.game_date)
    return root, games


def test_the_histogram_pass_recounts_the_frozen_zone_counts_exactly(fixture_root):
    root, games = fixture_root
    zcols = [f"{z}_n" for z in F.ZONE_FEATURES]
    for g in games[:3]:
        frames = pd.read_parquet(root / "sportvu_contract" / f"{g}.parquet")
        poss = pd.read_parquet(root / "analytics" / "possessions" / "2015-16" / f"{g}.parquet")
        frozen = F.tracking_partials(frames, poss).set_index("player_id")[zcols].sort_index()
        hist, zones = E2.court_histogram(frames, poss)
        mine = zones.set_index("player_id")[zcols].sort_index()
        assert frozen.index.equals(mine.index) and (frozen.to_numpy() == mine.to_numpy()).all()
        tot = hist.groupby("player_id")["n"].sum().reindex(mine.index).fillna(0)
        assert (tot.to_numpy() == mine.sum(axis=1).to_numpy()).all()
        assert hist["cell"].between(0, E2.CELLS - 1).all()


def basis(seed=0, k=E2.MAPS):
    rng = np.random.default_rng(seed)
    H = rng.gamma(0.3, 1.0, (k, E2.CELLS))
    return H / H.sum(axis=1, keepdims=True)


def sample_rows(H, n, total, seed=1):
    rng = np.random.default_rng(seed)
    W = rng.gamma(1.0, 1.0, (n, H.shape[0]))
    lam = W @ H
    lam = lam / lam.sum(axis=1, keepdims=True)
    return sparse.csr_matrix(np.vstack([rng.multinomial(total, p) for p in lam]).astype(float))


def test_the_fixed_basis_transform_matches_sklearn_and_does_not_depend_on_the_batch():
    from sklearn.decomposition import NMF
    H = basis()
    X = E2.rates(sample_rows(H, 30, 2000))
    W, info = E2.kl_transform(X, H)
    assert info["not_converged"] == 0
    m = NMF(n_components=E2.MAPS, solver="mu", beta_loss="kullback-leibler", max_iter=3000, tol=1e-8)
    m.components_ = H
    m.n_components_ = E2.MAPS
    m._n_components = E2.MAPS
    theirs = m.transform(X)
    assert np.nanmax(np.abs(E2.shares(W) - E2.shares(theirs))) < 0.02
    alone, _ = E2.kl_transform(X[3], H)
    assert np.allclose(alone[0], W[3])                    # a row's weights do not depend on the other rows


def test_an_empty_row_has_missing_shares_like_a_frozen_share_without_attempts():
    H = basis()
    X = sparse.csr_matrix((2, E2.CELLS))
    X[1, 5] = 1.0
    W, _ = E2.kl_transform(E2.rates(X), H)
    s = E2.shares(W)
    assert np.isnan(s[0]).all() and np.isclose(np.nansum(s[1]), 1.0)


def test_the_arms_replace_exactly_the_zone_shares():
    cols = E2.arm_columns()
    nmf_keep, nmf_new = cols["nmf"]
    shot_keep, shot_new = cols["nmf_shot"]
    order = E1.feature_order()
    shot_shares = [f"shot_share_{z}" for z in F.SHOT_ZONES]
    assert len(nmf_keep) + len(nmf_new) == 27 and len(shot_keep) + len(shot_new) == 23
    assert set(order) - set(nmf_keep) == set(F.ZONE_FEATURES) | set(shot_shares)
    assert set(order) - set(shot_keep) == set(shot_shares)
    X19 = np.arange(19.0)[None, :]
    out = E2.arm_features(X19, np.full((1, 10), 0.1), np.full((1, 10), 0.2))
    names, X = out["nmf"]
    assert X.shape == (1, 27) and X[0, names.index("court_basis_0")] == 0.1 and X[0, names.index("shot_basis_9")] == 0.2


def test_verdict_readings():
    def split(v, lo, hi):
        return {"rates_with_player_bootstrap": {"nmf_minus_frozen": {"value": v, "lo": lo, "hi": hi}}}
    assert E2.verdict(split(0.06, 0.01, 0.1))["adopt"] is True
    v = E2.verdict(split(0.03, 0.01, 0.05))
    assert v["adopt"] is False and "smaller than" in v["reading"] and v["set_a"] == "not scored"
    assert "does not lie above zero" in E2.verdict(split(0.06, -0.01, 0.1))["reading"]


def test_split_half_and_leave_one_game_out_run_on_a_synthetic_root(fixture_root, monkeypatch):
    root, games = fixture_root
    parts, rosters, court = [], [], []
    for g in games:
        frames = pd.read_parquet(root / "sportvu_contract" / f"{g}.parquet")
        poss = pd.read_parquet(root / "analytics" / "possessions" / "2015-16" / f"{g}.parquet")
        parts.append(F.tracking_partials(frames, poss))
        court.append(E2.court_histogram(frames, poss)[0])
        rosters.append(pd.read_parquet(root / "analytics" / "rosters" / "2015-16" / f"{g}.parquet"))
    shots = pd.read_parquet(root / E1.SHOTS_FILE, filters=[("sportvu_game", "in", games)])
    rosters = pd.concat(rosters, ignore_index=True)
    P = E1.Parts(pd.concat(parts, ignore_index=True), E1.shot_partials(shots, rosters), games)
    shot_h = E2.shot_histogram(shots, rosters)
    shot_h = shot_h[shot_h["player_id"].isin(set(P.players.tolist()))]
    sp = E1.shot_partials(shots, rosters)
    tot = shot_h.groupby(["sportvu_game", "player_id"])["n"].sum()
    assert (tot.reindex(sp.set_index(["sportvu_game", "player_id"]).index).dropna().astype(int)
            == sp.set_index(["sportvu_game", "player_id"])["attempts"].loc[tot.index].astype(int)).all()
    monkeypatch.setattr(E1, "N_MIN", 40)                        # the fixture games are short; fixture output only
    monkeypatch.setattr(F, "N_MIN", 40)
    monkeypatch.setattr(E1, "ONE_GAME_MIN", 10)
    X, n = E1.features_from_sums(P.sums(np.ones((1, P.R))))
    feats = pd.DataFrame(X[0], columns=E1.feature_order())
    feats.insert(0, "season", "2015-16")
    feats.insert(0, "player_id", P.players)
    feats["n_possessions"], feats["insufficient"] = n[0], n[0] < 40
    fit = E1.frozen_fit(feats)
    pos = pd.DataFrame({"player_id": P.players, "season": "2015-16",
                        "position": [("G", "F", "C")[i % 3] for i in range(P.P)]})
    d = {"P": P, "fit": fit, "feats": feats, "positions": pos, "games": games}
    model = E2.Model(d, E2.Hists(P, pd.concat(court, ignore_index=True), shot_h))
    s = E2.split_half(model, n=50)
    for k in ("frozen", "nmf", "nmf_shot", "nmf_minus_frozen", "position"):
        assert k in s["rates_with_player_bootstrap"]
    logo = E2.leave_one_game_out(model, n=50, log=None)
    assert logo["queries"] > 0 and "nmf_minus_frozen" in logo["rates_with_game_bootstrap"]
    assert set(E2.verdict(s)) >= {"adopt", "reading", "set_a"}


def test_the_module_reads_no_test_data_and_never_calls_score_once():
    text = (REPO / "tools" / "research_embeddings_e2.py").read_text()
    tree = ast.parse(text)
    calls = {getattr(n.func, "attr", getattr(n.func, "id", None)) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "score_once" not in calls and "scoring_data_root" not in calls and "research_data_root" not in calls
    for bad in ("testset.json", "_research-data", "V3_RESEARCH_DATA", "made"):
        assert f'"{bad}"' not in text and f"'{bad}'" not in text, bad


def test_a_row_with_mass_where_every_basis_map_is_zero_is_refused_not_imputed():
    H = basis()
    H[:, 7] = 0.0
    X = sparse.csr_matrix((2, E2.CELLS))
    X[0, 3] = 1.0
    E2.kl_transform(E2.rates(X), H)                      # a row away from the dead cell is fine
    X[1, 7] = 1.0
    with pytest.raises(ValueError, match="every basis map is zero"):
        E2.kl_transform(E2.rates(X), H)


def test_future_runs_record_convergence_and_the_code_commit():
    text = (REPO / "tools" / "research_embeddings_e2.py").read_text()
    assert '"code_commit": head' in text and 'out["transform_record"] = model.transform_record' in text
    assert 'out["transform_record_last_run"] = model.transform_record' in text
