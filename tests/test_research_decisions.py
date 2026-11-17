"""decisions-E1 machinery (docs/research/decisions.md, section 6): the statistics against direct computations on
synthetic data, and guards on the module's source. No data, no held-out game, no test game."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import research_decisions as D

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "tools" / "research_decisions.py"


def synthetic(n_games=8, players=6, per_game=40, seed=0, kind_effect=True, player_effect=0.0):
    """Events and on-court ratings of made-up games 'MM.DD.YYYY.AAA.at.BBB' (no real id). Each player has a fixed
    kind mix; regret depends on kind (and zone) and, with player_effect, on the player."""
    rng = np.random.default_rng(seed)
    games = [f"{m:02d}.{d:02d}.{y}.AAA.at.BBB" for (m, d, y) in
             [(11, 1, 2015), (12, 3, 2015), (1, 5, 2016), (2, 7, 2016), (10, 30, 2015), (3, 9, 2016), (1, 1, 2016),
              (12, 31, 2015), (2, 2, 2016), (3, 3, 2016)][:n_games]]
    kind_q = {"shot": -0.1, "pass": -0.35, "drive": -0.5}
    mixes = rng.dirichlet(np.ones(3), size=players)
    rows, ortg = [], []
    for g in games:
        for p in range(players):
            kinds = rng.choice(D.KINDS, size=per_game, p=mixes[p])
            dist = rng.uniform(0, 30, size=per_game)
            q = np.array([kind_q[k] if kind_effect else -0.3 for k in kinds]) + player_effect * p \
                - 0.01 * dist + rng.normal(0, 0.05, per_game)
            for k, d, x in zip(kinds, dist, q):
                rows.append({"sportvu_game": g, "player_id": 100 + p, "decision_kind": k, "decision_quality": min(x, 0.0),
                             "dist_to_basket_ft": d})
            ortg.append({"sportvu_game": g, "player_id": 100 + p, "ortg": float(rng.normal(105, 20))})
    ev = pd.DataFrame(rows)
    ev["zone"] = D.zone_index(ev["dist_to_basket_ft"])
    return ev, pd.DataFrame(ortg), games


def test_halves_sort_by_date_across_the_new_year_and_alternate():
    games = ["01.02.2016.X.at.Y", "11.01.2015.X.at.Y", "12.31.2015.X.at.Y", "01.01.2016.X.at.Y", "10.28.2015.X.at.Y"]
    h = D.halves(games)
    order = sorted(games, key=D.game_date)
    assert order == ["10.28.2015.X.at.Y", "11.01.2015.X.at.Y", "12.31.2015.X.at.Y", "01.01.2016.X.at.Y",
                     "01.02.2016.X.at.Y"]
    assert [h[g][0] for g in order] == [0, 1, 0, 1, 0]
    assert [h[g][1] for g in order] == [0, 0, 0, 1, 1]


def test_zones_are_the_frozen_epv_zones_with_a_missing_distance_apart():
    z = D.zone_index([0.0, 3.9, 4.0, 13.9, 14.0, 21.9, 22.0, 40.0, np.nan])
    assert [D.ZONES[i] for i in z] == ["rim", "rim", "paint", "paint", "mid", "mid", "three", "three", "none"]


def test_player_values_match_direct_group_means():
    ev, ortg, games = synthetic()
    T = D.Table(ev, ortg, games)
    one = np.ones((1, T.G))
    mu, mu_k = D.cell_means(T, one)
    v = D.player_values(T, one, mu[0], mu_k[0])
    by = ev.groupby("player_id")["decision_quality"].mean()
    assert np.allclose(v["frozen"][0], by.loc[T.players].to_numpy())
    cell = ev.groupby(["decision_kind", "zone"])["decision_quality"].transform("mean")
    kind = ev.groupby("decision_kind")["decision_quality"].transform("mean")
    adj = (ev["decision_quality"] - cell).groupby(ev["player_id"]).mean()
    adjk = (ev["decision_quality"] - kind).groupby(ev["player_id"]).mean()
    assert np.allclose(v["adjusted"][0], adj.loc[T.players].to_numpy())
    assert np.allclose(v["adjusted_kind"][0], adjk.loc[T.players].to_numpy())
    share = ev.assign(s=ev["decision_kind"] == "shot").groupby("player_id")["s"].mean()
    assert np.allclose(v["shares"][0][0], share.loc[T.players].to_numpy())
    assert np.allclose(v["target"][0], ortg.groupby("player_id")["ortg"].mean().loc[T.players].to_numpy())


def test_target_uses_only_player_games_with_at_least_three_decisions():
    ev, ortg, games = synthetic(n_games=2, players=3)
    ev = ev[~((ev["sportvu_game"] == games[0]) & (ev["player_id"] == 100))]
    ev = pd.concat([ev, pd.DataFrame([{"sportvu_game": games[0], "player_id": 100, "decision_kind": "shot",
                                       "decision_quality": -0.1, "dist_to_basket_ft": 5.0, "zone": 1}] * 2)])
    T = D.Table(ev, ortg, games)
    v = D.player_values(T, np.ones((1, T.G)), np.zeros(T.C), np.zeros(3))
    expect = ortg[(ortg["player_id"] == 100) & (ortg["sportvu_game"] == games[1])]["ortg"].iloc[0]
    assert np.isclose(v["target"][0][0], expect)


def test_weights_resample_games_and_keep_stratum_sizes():
    rng = np.random.default_rng(0)
    strata = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    W = D.resample(rng, np.ones(10), strata, 500)
    assert np.allclose(W, np.round(W))
    for s in np.unique(strata):
        assert (W[:, strata == s].sum(axis=1) == (strata == s).sum()).all()
    ev, ortg, games = synthetic()
    T = D.Table(ev, ortg, games)
    w = np.zeros(T.G)
    w[0] = 2.0
    v = D.player_values(T, w[None, :], np.zeros(T.C), np.zeros(3))
    first = ev[ev["sportvu_game"] == T.games[0]].groupby("player_id")["decision_quality"].mean()
    assert np.allclose(v["frozen"][0], first.loc[T.players].to_numpy())


def test_franks_meta_metrics_at_their_limits():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert D.discrimination(x, np.zeros(4)) == 1.0
    assert np.isclose(D.discrimination(x, np.full(4, np.var(x))), 0.0)
    assert D.stability(x, x, np.zeros(4), np.zeros(4)) == 1.0
    rng = np.random.default_rng(1)
    noise = 0.3
    x0, x1 = rng.normal(0, noise, 4000), rng.normal(0, noise, 4000)
    bv = np.full(4000, noise ** 2)
    # no persistent player level: each half's true value drifts independently, so the true stability is 0
    d0, d1 = rng.normal(0, 0.5, 4000), rng.normal(0, 0.5, 4000)
    null = D.stability(d0 + x0, d1 + x1, bv, bv)
    assert np.isfinite(null) and abs(null) < 0.1
    # and with no between-player variance beyond noise the statistic is undefined, not a pass
    assert np.isnan(D.stability(x0, x1, bv * 1.2, bv * 1.2))
    mu = rng.normal(0, 1, 4000)
    assert abs(D.stability(mu + x0, mu + x1, bv, bv) - 1.0) < 0.05


def test_r_squared_matches_numpy_and_a_kind_only_metric_is_explained_by_shares():
    rng = np.random.default_rng(2)
    X = rng.random((50, 2))
    y = 1 + X @ np.array([0.5, -1.0]) + rng.normal(0, 0.1, 50)
    D_ = np.column_stack([np.ones(50), X])
    beta = np.linalg.lstsq(D_, y, rcond=None)[0]
    expect = 1 - ((y - D_ @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert np.isclose(D.r_squared(y, X), expect)
    ev, ortg, games = synthetic(players=30, per_game=60)
    T = D.Table(ev, ortg, games)
    one = np.ones((1, T.G))
    mu, mu_k = D.cell_means(T, one)
    v = D.player_values(T, one, mu[0], mu_k[0])
    S = np.column_stack([v["shares"][0][0], v["shares"][1][0]])
    assert D.r_squared(v["frozen"][0], S) > 0.8
    assert D.r_squared(v["adjusted_kind"][0], S) < 0.2


def test_statistics_and_summary_run_and_stay_finite_on_synthetic_games(monkeypatch):
    monkeypatch.setattr(D, "N_MIN", 50)
    ev, ortg, games = synthetic(n_games=10, players=12, per_game=30, player_effect=0.02)
    T = D.Table(ev, ortg, games)
    out = D.summarize(T, n=30, inner=20, log=None)
    st = out["statistics"]
    assert out["players"]["all"] == 12
    for k in ("frozen.stability", "adjusted.stability", "target.split_half_r", "frozen.r2_on_kind_shares"):
        assert np.isfinite(st[k]["value"]) and st[k]["lo"] <= st[k]["hi"]
    assert st["frozen.discrimination"]["value"] > 0.5        # a real player effect is discriminated
    assert set(D.verdict(st)) == {"H1", "H2", "H3", "adoption", "protocol"}


def _stat(v, lo, hi, finite=2000):
    return {"value": v, "lo": lo, "hi": hi, "finite_resamples": finite}


def _stats():
    return {"target.split_half_r": _stat(0.1, 0.0, 0.25), "frozen.stability": _stat(0.6, 0.4, 0.8),
            "frozen.r2_on_kind_shares": _stat(0.4, 0.3, 0.6), "adjusted.stability": _stat(0.55, 0.31, 0.7),
            "adjusted.r2_on_kind_shares": _stat(0.1, 0.0, 0.2)}


def test_verdict_reads_the_pre_registered_thresholds():
    stats = _stats()
    v = D.verdict(stats)
    assert v["H1"]["result"] == "holds" and v["H1"]["interval_crosses_threshold_reads_invalid_interval"]
    assert v["H1"]["interval_95_valid_as_confidence_interval"] is False
    assert v["H2"]["result"] == "holds" and v["H3"]["result"] == "fails" and v["adoption"]["adopted"]
    stats["adjusted.stability"] = _stat(0.55, 0.29, 0.7)
    assert not D.verdict(stats)["adoption"]["adopted"]
    stats["target.split_half_r"] = _stat(0.2, 0.1, 0.3)
    v = D.verdict(stats)
    assert v["H1"]["result"] == "fails" and "evidence against the metric" in v["protocol"]


@pytest.mark.parametrize("key", ["target.split_half_r", "frozen.stability", "frozen.r2_on_kind_shares"])
def test_a_statistic_that_is_not_a_number_is_undefined_never_holds_or_fails(key):
    stats = _stats()
    stats[key] = _stat(float("nan"), float("nan"), float("nan"), 0)
    v = D.verdict(stats)
    name = {"target.split_half_r": "H1", "frozen.stability": "H2", "frozen.r2_on_kind_shares": "H3"}[key]
    assert v[name]["result"] == "undefined" and v[name]["holds"] is None
    assert v[name]["interval_crosses_threshold_reads_invalid_interval"] is None
    if name == "H1":
        assert "undefined" in v["protocol"]


@pytest.mark.parametrize("key,stat", [("adjusted.stability", _stat(float("nan"), 0.4, 0.9)),
                                      ("adjusted.stability", _stat(0.9, float("nan"), 0.95)),
                                      ("adjusted.stability", _stat(0.9, 0.6, 0.95, finite=1800)),
                                      ("adjusted.r2_on_kind_shares", _stat(float("nan"), 0.0, 0.1))])
def test_adoption_cannot_pass_on_an_undefined_statistic_or_interval(key, stat):
    stats = _stats()
    stats[key] = stat
    a = D.verdict(stats)["adoption"]
    assert a["adopted"] is False and a["result"].startswith("undefined")


def test_an_interval_with_too_few_finite_draws_is_undefined():
    draws = np.concatenate([np.linspace(0, 1, 94), np.full(6, np.nan)])
    iv = D.percentile_interval(0.5, draws)
    assert iv["finite_share"] == 0.94 and not iv["interval_defined"] and np.isnan(iv["lo"]) and np.isnan(iv["hi"])
    iv = D.percentile_interval(0.5, np.concatenate([np.linspace(0, 1, 95), np.full(5, np.nan)]))
    assert iv["interval_defined"] and iv["interval_valid_as_confidence_interval"] is False


def test_basic_interval_reflects_the_draws_about_the_estimate():
    draws = np.linspace(0.80, 0.90, 1001)
    iv = D.basic_interval(0.93, draws)
    assert np.isclose(iv["lo"], 2 * 0.93 - np.percentile(draws, 97.5))
    assert np.isclose(iv["hi"], 2 * 0.93 - np.percentile(draws, 2.5))


def test_player_bootstrap_draws_the_same_players_for_every_measure_and_is_centred():
    ev, ortg, games = synthetic(n_games=10, players=40, per_game=30, player_effect=0.01)
    T = D.Table(ev, ortg, games)
    sets = {"all": np.ones(T.P, bool), "oddeven": np.ones(T.P, bool), "chrono": np.ones(T.P, bool)}
    parts = D.statistic_parts(T, np.ones(T.G), sets, np.random.default_rng(0), 50)
    rs = D.resample_players(parts, np.random.default_rng(1))
    # the same draw for every measure: the frozen and adjusted values move together
    for key in ("Xa", "x0", "X0", "bv0"):
        f, a = parts["frozen"][key], parts["adjusted"][key]
        idx = [int(np.flatnonzero(np.isclose(parts["frozen"][key], x))[0]) for x in rs["frozen"][key]]
        assert np.allclose(rs["adjusted"][key], a[idx])
    point = D._flat(D.stats_from_parts(parts))
    draws = D.player_draws(parts, 300, [0])
    r = draws["frozen.split_half_r"]
    assert np.percentile(r, 2.5) <= point["frozen.split_half_r"] <= np.percentile(r, 97.5)


def test_the_synthetic_season_has_the_answer_it_claims():
    rng = np.random.default_rng(5)
    cfg = dict(players=400, games=40, per_game=20, sd_player=0.05, sd_drift=0.0135, sd_noise=0.28)
    events, ortg, ids, truth = D.simulate(rng, **cfg)
    T = D.Table(events, ortg, ids)
    sets = D.player_sets(T)
    assert sets["all"].sum() == sets["oddeven"].sum() == sets["chrono"].sum() == 400
    s = D._flat(D.statistics(T, np.ones(T.G), sets, np.random.default_rng(0), 200))
    assert abs(s["frozen.split_half_r"] - truth["frozen.split_half_r"]) < 0.03
    assert abs(s["frozen.discrimination"] - truth["frozen.discrimination"]) < 0.03
    assert abs(s["frozen.stability"] - truth["frozen.stability"]) < 0.06


def test_e1_final_refuses_set_a_before_any_scoring_and_writes_no_ledger_entry(monkeypatch):
    called = []
    monkeypatch.setattr(D, "score_once", lambda *a, **k: called.append(a))
    monkeypatch.setattr(D, "_gate_tables", lambda *a, **k: called.append("data read"))
    with pytest.raises(SystemExit) as err:
        D.final()
    assert "one date" in str(err.value) and "No ledger entry" in str(err.value)
    assert called == []
    assert D.SET_A_STATISTICS_POSSIBLE is False


def test_the_printed_validation_of_the_frozen_lane_report_is_parsed_and_compared_to_half_a_digit():
    printed = D.printed_validation((REPO / "reports" / "decisions.md").read_text())
    assert set(printed) == {"location", "xfg_selected"}
    assert printed["xfg_selected"]["mean_rho"] == "0.012" and printed["xfg_selected"]["signed"] == "603"
    summary = {"n_games": 604, "mean_rho": 0.01249, "lo": -0.008, "hi": 0.029, "n_positive": 317, "n_negative": 286,
               "sign_p": 0.222}
    assert D.compare_printed(printed["xfg_selected"], summary, {"mean_rho": -0.013})["passed"]
    summary["mean_rho"] = 0.0126
    assert not D.compare_printed(printed["xfg_selected"], summary, {"mean_rho": -0.013})["passed"]


# ----------------------------------------------------------------------------- source guards

def _tree():
    return ast.parse(SOURCE.read_text())


def _calls(node, name):
    return [c for c in ast.walk(node) if isinstance(c, ast.Call)
            and getattr(c.func, "id", getattr(c.func, "attr", None)) == name]


DOOR_NAMES = ("research_data_root", "test_features")


def _function(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_test_data_is_reached_only_inside_the_scorer_given_to_score_once():
    tree = _tree()
    once = _calls(tree, "score_once")
    assert len(once) == 1
    assert [ast.unparse(a) for a in once[0].args] == ["EXPERIMENT", "TRACKING", "scorer"]
    inside = {id(n) for n in ast.walk(_function(tree, "scorer"))}
    seen = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                assert not (alias.name.split(".")[-1] in DOOR_NAMES and alias.asname), f"aliased import {alias.name}"
                assert alias.name not in ("tools.research_xfg",) or alias.asname is None or isinstance(node, ast.ImportFrom)
        if isinstance(node, ast.Name) and node.id in DOOR_NAMES or isinstance(node, ast.Attribute) and node.attr in DOOR_NAMES:
            seen.append(node)
            assert id(node) in inside, f"{ast.unparse(node)} used outside the scorer"
        if isinstance(node, ast.Constant) and node.value in DOOR_NAMES:
            raise AssertionError(f"{node.value} named as a string (getattr)")
    assert {getattr(n, "id", getattr(n, "attr", None)) for n in seen} == set(DOOR_NAMES)


def _train_bindings(tree):
    """Names bound to analytics.decisions.train (modules) and to its run and main (functions)."""
    modules, functions = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "analytics.decisions":
            modules |= {a.asname or a.name for a in node.names if a.name == "train"}
        if isinstance(node, ast.Import):
            modules |= {a.asname for a in node.names if a.name == "analytics.decisions.train" and a.asname}
        if isinstance(node, ast.ImportFrom) and node.module == "analytics.decisions.train":
            functions |= {a.asname or a.name for a in node.names if a.name in ("run", "main")}
    return modules, functions


def test_the_baseline_never_runs_the_lane_command_line_or_scores_held_out_games():
    text = SOURCE.read_text()
    tree = _tree()
    for banned in ("write_outputs", "write_report", "heldout_once"):
        assert banned not in text, banned
    modules, functions = _train_bindings(tree)
    assert modules, "the frozen trainer is imported as a module"
    run_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "skip_heldout":
            assert isinstance(node.value, ast.Constant) and node.value.value is True, ast.unparse(node.value)
        if isinstance(node, ast.Name) and node.id in functions:
            raise AssertionError(f"{node.id} imported from the frozen trainer by name")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in modules:
            assert node.attr not in ("main",), "the lane's command line"
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "getattr":
            assert not (node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in modules)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "run" \
                and isinstance(node.func.value, ast.Name) and node.func.value.id in modules:
            run_calls.append(node)
    run_refs = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "run"
                and isinstance(n.value, ast.Name) and n.value.id in modules]
    assert len(run_refs) == len(run_calls) >= 1, "the frozen run is only ever called, never aliased"
    for c in run_calls:
        assert not any(k.arg is None for k in c.keywords), "no **kwargs into the frozen run"
        assert any(k.arg == "skip_heldout" and isinstance(k.value, ast.Constant) and k.value.value is True
                   for k in c.keywords)
    assert "dev_games(" in text and "training_games_only(" in text


def test_the_module_fits_nothing_but_the_descriptive_least_squares():
    text = SOURCE.read_text()
    assert ".fit(" not in text and "fit_value_model" not in text
    assert "lstsq" in text


ALLOWED_DESTINATIONS = ("RESULTS /", "(RESULTS /", "folder /", "(folder /", "gate_dir(artifacts) /")


def _destination_ok(expr: ast.AST, fn: ast.FunctionDef) -> bool:
    text = ast.unparse(expr)
    if text.startswith(ALLOWED_DESTINATIONS):
        return True
    if isinstance(expr, ast.Name):
        bound = [a.value for a in ast.walk(fn) if isinstance(a, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == expr.id for t in a.targets)]
        return bool(bound) and all(ast.unparse(b).startswith(ALLOWED_DESTINATIONS) for b in bound)
    return False


def test_results_are_written_only_to_the_research_results_and_the_artifacts_folder():
    tree = _tree()
    checked = 0
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        for c in (c for c in ast.walk(fn) if isinstance(c, ast.Call)):
            name = getattr(c.func, "attr", getattr(c.func, "id", None))
            if name == "write_text":
                dest = c.func.value
            elif name == "to_parquet":
                dest = c.args[0] if c.args else next(k.value for k in c.keywords if k.arg == "path")
            elif name == "open" and len(c.args) > 1 and "w" in ast.unparse(c.args[1]):
                dest = c.args[0]
            else:
                continue
            checked += 1
            assert _destination_ok(dest, fn), f"{fn.name}: {ast.unparse(dest)}"
    assert checked >= 8
