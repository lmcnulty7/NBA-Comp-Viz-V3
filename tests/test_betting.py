"""analytics/betting on the fixtures of tests/_fx_betting.py (the shared analytics.fixtures when present, else the
local fallback with the same signatures). No network, no real odds. Under 30 s."""
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from analytics.betting import eda, features as fx, train as tr, validate as va  # noqa: E402
from tests._fx_betting import team_names  # noqa: E402

N_GAMES = 24      # games per fixture season: enough for the join, the folds and the form features, cheap to generate


@pytest.fixture(scope="module")
def data():
    """One generation of the fixtures per process (validate.fixture_data caches the synth_stints call, which the
    shared analytics.fixtures generator makes expensive); the same frames serve the end-to-end run."""
    d = va.fixture_data(0, n_games=N_GAMES, seasons=("2020-21", "2021-22"))
    assert set(d["games"].columns) >= {"game_id", "season", "game_date", "home_team_id", "away_team_id", "home_score",
                                       "away_score", "home_win", "spread_home"}
    return d


@pytest.fixture(scope="module")
def joined(data):
    return va.validate(data["odds"], data["games"], team_names(data["games"]))


# ── vig removal and prices ─────────────────────────────────────────────────────────────────────────
def test_vig_removal_hand_cases():
    assert np.allclose(va.american_to_prob([-110, +130, -150]), [110 / 210, 100 / 230, 150 / 250])
    even = va.remove_vig(va.american_to_prob([-110, -110]))
    assert np.allclose(even, [0.5, 0.5])
    fair = va.remove_vig(va.american_to_prob([-150, +130]))
    assert np.isclose(fair.sum(), 1.0)
    assert np.allclose(fair, [0.5798, 0.4202], atol=5e-4)         # 0.6 / 1.0348, 0.4348 / 1.0348
    with pytest.raises(ValueError):
        va.remove_vig([0.0, 0.0])


def test_team_key_normalises_names_ids_and_abbreviations():
    assert va.team_key("Los Angeles Lakers") == va.team_key("LA Lakers") == va.team_key(1610612747) == va.team_key("LAL") == "lakers"
    assert va.team_key("Portland Trail Blazers") == va.team_key(1610612757) == "trailblazers"
    assert va.team_key("Golden State Warriors") == va.team_key("GSW") == "warriors"
    assert va.team_key(7) == va.team_key("Team 7") == "team7"


# ── closing lines and the join ─────────────────────────────────────────────────────────────────────
def test_closing_line_is_the_is_close_snapshot_nearest_tip_and_duplicates_are_counted():
    base = {"game_key": "2021-01-02_MiamiHeat_BostonCeltics", "commence_time": "2021-01-02T00:00:00Z", "home_team": "Boston Celtics",
            "away_team": "Miami Heat", "book": "b", "source": "x", "endpoint": "y"}
    rows = []
    for at, close, sp in (("2021-01-01T04:00:00Z", False, -1.5), ("2021-01-01T23:25:00Z", True, -2.5), ("2021-01-01T23:55:00Z", True, -3.5),
                          ("2021-01-02T01:00:00Z", True, -9.5)):   # the last is after tip: never the close
        for market, side, point, price in (("spreads", "Boston Celtics", sp, -110), ("spreads", "Miami Heat", -sp, -110),
                                           ("totals", "Over", 220.5, -110), ("totals", "Under", 220.5, -110),
                                           ("h2h", "Boston Celtics", None, -150), ("h2h", "Miami Heat", None, 130)):
            rows.append({**base, "snapshot_at": at, "is_close": close, "market": market, "side": side, "point": point, "price": price})
    odds = pd.DataFrame(rows)
    assert va.check_schema(odds) == []
    closing, counts = va.closing_lines(odds)
    assert len(closing) == 1 and counts["duplicate_snapshots"] == 1 and counts["fallback_preclose"] == 0
    r = closing.iloc[0]
    assert r["closing_spread_home"] == -3.5 and r["closing_total"] == 220.5
    assert r["n_close_snapshots"] == 1 and r["snapshot_at"] == "2021-01-01T23:55:00Z"   # one book: one chosen instant; its earlier close is the counted duplicate
    assert np.isclose(r["implied_home_prob"], 0.5798, atol=5e-4)
    # no is_close row: the latest pre-tip snapshot stands in and is counted
    closing2, counts2 = va.closing_lines(odds.assign(is_close=False))
    assert counts2["fallback_preclose"] == 1 and closing2.iloc[0]["closing_spread_home"] == -3.5 and closing2.iloc[0]["used_fallback"]
    bad = odds.copy()
    bad.loc[0, "market"] = "props"
    assert any("unknown market" in p for p in va.check_schema(bad))


def test_join_coverage_handles_utc_dates_and_counts_unmatched(joined):
    jn = joined["counts"]["join"]
    assert jn["matched"] >= 0.9 * jn["games"]                 # a 7pm ET tip is the next UTC day; the join must still land
    assert jn["games_without_odds"] + jn["matched"] == jn["games"]
    j = joined["joined"]
    m = j[j["game_key"].notna()]
    # the joined spread is the fixture's own spread up to the fixture's closing noise (median over books)
    both = m[m["spread_home"].notna()]
    diff = both["closing_spread_home"].astype(float) - both["spread_home"].astype(float)
    assert diff.abs().median() <= 1.0 and np.corrcoef(both["closing_spread_home"].astype(float), both["spread_home"].astype(float))[0, 1] > 0.9
    assert j["closing_total"].dropna().between(150, 300).all()


def test_unknown_team_ids_are_inferred_from_date_cooccurrence(data):
    """Without any id -> name map the join must still land: a date with one game on each side votes the pairs."""
    res = va.validate(data["odds"], data["games"])
    jn = res["counts"]["join"]
    assert jn["team_names_given"] == 0
    names = team_names(data["games"])
    if names:                                       # the shared fixtures' synthetic ids: every one inferred
        assert jn["team_names_inferred"] >= 6 and jn["team_ids_unresolved"] == 0
    else:                                           # real ids: nothing to infer
        assert jn["team_names_inferred"] == 0
    assert jn["matched"] >= 0.9 * jn["games"]
    inferred = va.infer_team_names(res["closing"], data["games"])
    for tid, name in inferred.items():
        assert names.get(tid) == name


# ── features: time-forward and leak guards ─────────────────────────────────────────────────────────
def test_features_use_previous_games_only(joined):
    bg = fx.build_games(joined["joined"], "2021-22")
    assert list(bg.columns[:len(fx.GAMES_COLUMNS)]) == list(fx.GAMES_COLUMNS)
    assert bg["heldout"].sum() == (bg["season"] == "2021-22").sum() > 0
    # mutate a mid-season game's own score: its OWN row keeps every feature (an unshifted window would change
    # net10, net30, pf_std, pa_std), every earlier row is unchanged, and later rows of both its teams change
    j = joined["joined"].copy()
    season = sorted(j["season"].unique())[0]
    s = j[j["season"] == season].sort_values(["game_date", "game_id"])
    target = None
    for pos in range(len(s) // 3, len(s)):
        r = s.iloc[pos]
        later = s.iloc[pos + 1:]
        if (later["home_team_id"].eq(r["home_team_id"]) | later["away_team_id"].eq(r["home_team_id"])).any() and \
           (later["home_team_id"].eq(r["away_team_id"]) | later["away_team_id"].eq(r["away_team_id"])).any():
            target = s.index[pos]
            break
    assert target is not None
    gid, date = j.loc[target, "game_id"], j.loc[target, "game_date"]
    j.loc[target, "home_score"] = int(j.loc[target, "home_score"]) + 40
    bg2 = fx.build_games(j, "2021-22")
    own, own2 = bg[bg["game_id"] == gid].iloc[0], bg2[bg2["game_id"] == gid].iloc[0]
    for c in fx.FEATURES:
        assert own[c] == own2[c], c
    earlier = (bg["season"] == season) & (bg["game_date"] < date)
    pd.testing.assert_frame_equal(bg.loc[earlier, list(fx.FEATURES)].reset_index(drop=True),
                                  bg2.loc[earlier, list(fx.FEATURES)].reset_index(drop=True))
    for tid in (own["home_team_id"], own["away_team_id"]):
        nxt = bg[(bg["season"] == season) & (bg["game_date"] > date) & ((bg["home_team_id"] == tid) | (bg["away_team_id"] == tid))]
        nxt = nxt.sort_values(["game_date", "game_id"]).iloc[0]
        side = "home" if nxt["home_team_id"] == tid else "away"
        nxt2 = bg2[bg2["game_id"] == nxt["game_id"]].iloc[0]
        assert nxt[side + "_net10"] != nxt2[side + "_net10"] and nxt[side + "_pf_std"] != nxt2[side + "_pf_std"] or \
               nxt[side + "_pa_std"] != nxt2[side + "_pa_std"]
    # the first game of every team in a season (home or away) carries no form and the capped rest
    srt = bg.sort_values(["game_date", "game_id"])
    seen = set()
    for r in srt.itertuples(index=False):
        for side, tid in (("home", r.home_team_id), ("away", r.away_team_id)):
            if (r.season, tid) not in seen:
                seen.add((r.season, tid))
                assert getattr(r, side + "_rest_days") == fx.REST_CAP
                for c in ("net10", "net30", "pf_std", "pa_std"):
                    assert getattr(r, "%s_%s" % (side, c)) == 0.0, (side, c)
    fx.assert_time_forward(fx.design(bg))
    cover = bg["home_margin"] + bg["closing_spread_home"]
    assert ((bg["home_cover"] == 1) == (cover > 0)).loc[bg["home_cover"].notna()].all()


def test_a_feature_built_from_the_final_margin_is_caught(joined):
    bg = fx.build_games(joined["joined"], "2021-22")
    leaky = bg.copy()
    leaky["momentum"] = 0.5 * leaky["home_margin"] + np.random.default_rng(0).normal(0, 0.1, len(leaky))
    with pytest.raises(ValueError, match="built from the outcome"):
        fx.assert_no_leak(leaky[list(fx.FEATURES) + ["momentum"]], leaky["home_margin"])
    for col in ("home_margin", "closing_spread_home", "implied_home_prob", "home_cover"):
        with pytest.raises(ValueError):
            fx.assert_time_forward(bg[list(fx.FEATURES) + [col]])
    with pytest.raises(ValueError):                                 # a Phase 0 label name is refused too
        fx.assert_time_forward(bg[list(fx.FEATURES)].assign(points=1))
    # the trainer refuses a held-out row at the fit
    with pytest.raises(ValueError, match="held-out"):
        tr.fit_models(bg)


# ── the staking rule ───────────────────────────────────────────────────────────────────────────────
def test_staking_rule_and_pnl():
    assert tr.STAKE_THRESHOLD == 0.55 and tr.VIG == 0.045 and np.isclose(tr.WIN_RETURN, 0.91)
    st = tr.stakes([0.7, 0.3, 0.5, 0.56, 0.44])
    assert st.tolist() == [1, -1, 0, 1, -1]
    out = tr.pnl([1, -1, 1, 1, 0, 1], home_margin=[5, 5, -3, 2, 9, 4], closing_spread_home=[-3, -3, 1, -2, -1, np.nan])
    assert np.allclose(out, [0.91, -1, -1, 0, 0, 0])                # win, loss, loss, push, no bet, no spread
    priced = tr.pnl([1, 1], home_margin=[5, 5], closing_spread_home=[-3, -3], price=[-120, 110])
    assert np.allclose(priced, [100 / 120, 1.1])


# ── end to end ─────────────────────────────────────────────────────────────────────────────────────
def test_end_to_end_fixture_run(tmp_path):
    report = tmp_path / "betting.md"
    ctx = tr.main(["--fixtures", "--out", str(tmp_path), "--report", str(report), "--bootstrap", "10", "--fixture-games", str(N_GAMES)])
    assert ctx["heldout_season"] == "2021-22" and ctx["heldout_games"] > 0
    assert ctx["chosen"]["cover_model"] in tr.CANDIDATES and ctx["chosen"]["margin_model"] in ("market", "ridge")
    for season in ("2020-21", "2021-22"):
        d = tmp_path / "analytics" / "betting" / season
        g = pd.read_parquet(d / "betting_games.parquet")
        p = pd.read_parquet(d / "betting_predictions.parquet")
        s = pd.read_parquet(d / "betting_summary.parquet")
        assert set(fx.GAMES_COLUMNS) <= set(g.columns)
        assert {"game_id", "season", "p_cover", "margin_forecast", "clv", "stake", "pnl", "fold", "heldout"} <= set(p.columns)
        assert {"season", "n_games", "log_loss", "brier", "roi", "lo", "hi"} <= set(s.columns)
        assert len(p) == len(g) and p["p_cover"].between(0, 1).all()
        assert (p["heldout"] == (season == "2021-22")).all() and ((p["fold"] == -1) == p["heldout"]).all()
        assert s["mode"].iloc[0] == ("heldout" if season == "2021-22" else "cv")
    text = report.read_text()
    assert "fixture output, not a result" in text.lower() and "scored ONCE" in text
    assert chr(8212) not in text
    # the market baseline sits at the coin-flip log loss and never bets
    assert np.isclose(ctx["scores"]["market"]["log_loss"]["point"], np.log(2), atol=1e-6)
    assert ctx["scores"]["market"]["n_bets"] == 0
    # grouped folds: every training game scored out of fold exactly once
    oof = ctx["predictions"][~ctx["predictions"]["heldout"]]
    assert oof["fold"].between(0, 4).all() and oof["game_id"].is_unique


def test_eda_tables_and_shuffled_labels(joined):
    bg = fx.build_games(joined["joined"], "2021-22")
    sd = eda.spread_distribution(bg)
    assert sd["n"].iloc[0] == bg["closing_spread_home"].notna().sum()
    train_rows = bg[~bg["heldout"]]
    cr = eda.cover_rate_by_spread_bin(bg)
    assert cr["n"].sum() == train_rows["home_cover"].notna().sum() < bg["home_cover"].notna().sum()
    md_ = eda.margin_distribution(bg)
    assert md_["n"].iloc[0] == len(train_rows)
    md = eda.render_markdown(eda.join_coverage(joined["counts"]))
    assert md.startswith("| count | value |") and "join.coverage" in md
    sh = eda.shuffled_label_check(bg, seed=0, folds=3)
    assert set(sh["labels"]) == {"true", "shuffled"} and len(sh) == 6
    market = sh[sh["candidate"] == "market"]["log_loss"]
    assert np.allclose(market, np.log(2))


def test_staggered_books_each_keep_their_own_closing_snapshot():
    """Book A closes 5 minutes before tip, book B 35 minutes before (and has an earlier duplicate): both books are kept
    at their own instant, the duplicate is counted, and a book with no is_close row is dropped and counted."""
    base = {"game_key": "2021-01-02_MiamiHeat_BostonCeltics", "commence_time": "2021-01-02T00:00:00Z", "home_team": "Boston Celtics",
            "away_team": "Miami Heat", "source": "x", "endpoint": "y"}
    rows = []
    for book, at, close, sp in (("a", "2021-01-01T23:55:00Z", True, -3.5), ("b", "2021-01-01T23:25:00Z", True, -5.5),
                                ("b", "2021-01-01T22:00:00Z", True, -9.5), ("c", "2021-01-01T12:00:00Z", False, -1.5)):
        for market, side, point, price in (("spreads", "Boston Celtics", sp, -110), ("spreads", "Miami Heat", -sp, -110),
                                           ("totals", "Over", 220.5, -110), ("totals", "Under", 220.5, -110),
                                           ("h2h", "Boston Celtics", None, -150), ("h2h", "Miami Heat", None, 130)):
            rows.append({**base, "book": book, "snapshot_at": at, "is_close": close, "market": market, "side": side, "point": point, "price": price})
    closing, counts = va.closing_lines(pd.DataFrame(rows))
    r = closing.iloc[0]
    assert r["n_books"] == 2 and r["closing_spread_home"] == -4.5 and r["n_close_snapshots"] == 2
    assert counts["duplicate_snapshots"] == 1 and counts["books_without_close"] == 1 and counts["multi_book_games"] == 1
    assert r["snapshot_at"] == "2021-01-01T23:55:00Z"


def test_collector_heldout_game_is_dropped_before_the_join(data):
    games = data["games"].copy()
    games["heldout"] = False
    games.loc[games.index[3], "heldout"] = True
    gid = games.loc[games.index[3], "game_id"]
    res = va.validate(data["odds"], games, team_names(games))
    assert gid not in set(res["joined"]["game_id"].astype(str)) and res["counts"]["games_heldout_dropped"] == 1
    dup = pd.concat([games, games.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate game_id"):
        va.validate(data["odds"], dup, team_names(games))


def test_inferred_names_need_votes_and_cannot_collide(data):
    closing = va.closing_lines(data["odds"])[0]
    games = data["games"].copy()
    names = team_names(games)
    if not names:
        pytest.skip("fallback fixtures use real ids: nothing to infer")
    used = set(games["home_team_id"].astype(int)) | set(games["away_team_id"].astype(int))
    nba = {va.team_key(v[1]) for v in va.TEAM_NAMES.values()}
    if any(va.team_key(n) in nba for t, n in names.items() if t in used):
        pytest.skip("a fixture game uses a real NBA name: inference would refuse it as a collision by design")
    # every unknown id is resolved with at least MIN_VOTES votes and a majority on the full fixture
    inferred = va.infer_team_names(closing, games)
    assert len(inferred) >= 6
    # a single date gives one vote per id: below MIN_VOTES, nothing is inferred
    one = games.sort_values("game_date").iloc[[0]]
    assert va.infer_team_names(closing, one) == {}
    # a known name on the odds side can never be taken by an unknown id: nothing is inferred, the ids stay unresolved
    # and counted (the ValueError collision guard behind the vote is the backstop)
    collide = closing.copy()
    collide["home_team"] = "Boston Celtics"
    collide["away_team"] = "Miami Heat"
    assert va.infer_team_names(collide, games) == {}
    _, counts = va.join_games(collide, games)
    assert counts["team_names_inferred"] == 0 and counts["team_ids_unresolved"] == len(used) and counts["matched"] == 0


def test_trainer_never_fits_or_scores_a_held_out_row(tmp_path, monkeypatch):
    seen = {"fit": [], "score": []}
    real_fit, real_score = tr.fit_models, tr.score_candidates
    monkeypatch.setattr(tr, "fit_models", lambda train: (seen["fit"].append(train), real_fit(train))[1])
    monkeypatch.setattr(tr, "score_candidates", lambda pred, n, seed: (seen["score"].append(pred), real_score(pred, n, seed))[1])
    before = (tr.STAKE_THRESHOLD, tr.VIG, tr.WIN_RETURN, tr.UNIT)
    ctx = tr.main(["--fixtures", "--out", str(tmp_path), "--report", str(tmp_path / "r.md"), "--bootstrap", "5", "--fixture-games", str(N_GAMES)])
    assert seen["fit"] and seen["score"]
    assert not any(f["heldout"].astype(bool).any() for f in seen["fit"])
    assert not any(f["heldout"].astype(bool).any() for f in seen["score"])
    assert ctx["heldout_games"] > 0 and (tr.STAKE_THRESHOLD, tr.VIG, tr.WIN_RETURN, tr.UNIT) == before == (0.55, 0.045, 0.91, 1.0)
    assert np.isclose(tr.win_return([-110])[0], 100 / 110) and tr.win_return([np.nan])[0] == tr.WIN_RETURN and tr.win_return(None, 2).tolist() == [0.91, 0.91]
    assert "a temporary directory" not in (tmp_path / "r.md").read_text()      # --out was given


def test_rapm_features_use_values_dated_before_the_game(tmp_path):
    """A player_rapm table with as_of: null for a game on or before as_of, the team sum after it. The lineups lane's
    current table carries no as_of, so on real data the two columns stay null (documented)."""
    games = pd.DataFrame({"game_id": ["g1", "g2", "g3"], "season": ["2020-21"] * 3,
                          "game_date": ["2021-01-01", "2021-01-10", "2021-01-20"],
                          "home_team_id": [1, 1, 2], "away_team_id": [2, 2, 1]})
    d = tmp_path / "analytics" / "lineups" / "2020-21"
    d.mkdir(parents=True)
    pd.DataFrame({"player_id": [11, 12, 21], "team_id": [1, 1, 2], "rapm": [1.5, -0.5, 2.0],
                  "as_of": ["2021-01-10"] * 3}).to_parquet(d / "player_rapm.parquet", index=False)
    r = fx.rapm_features(games, tmp_path).set_index("game_id")
    assert r.loc["g1", list(fx.RAPM_SUMS)].isna().all() and r.loc["g2", list(fx.RAPM_SUMS)].isna().all()
    assert r.loc["g1", list(fx.RAPM_FLAGS)].eq(0).all() and r.loc["g3", list(fx.RAPM_FLAGS)].eq(1).all()
    assert r.loc["g3", "home_rapm_sum"] == 2.0 and r.loc["g3", "away_rapm_sum"] == 1.0
    assert fx.rapm_features(games, None)[list(fx.RAPM_SUMS)].isna().all().all()
    # in_sample True rows are ignored when out-of-sample rows exist (the prior file of the lineups lane)
    pd.DataFrame({"player_id": [11, 12, 21, 11], "team_id": [1, 1, 2, 1], "rapm": [1.5, -0.5, 2.0, 9.0],
                  "as_of": ["2021-01-10"] * 4, "in_sample": [False, False, False, True]}).to_parquet(d / "player_rapm.parquet", index=False)
    assert fx.rapm_features(games, tmp_path).set_index("game_id").loc["g3", "away_rapm_sum"] == 1.0
    pd.DataFrame({"player_id": [11], "team_id": [1], "rapm": [1.0]}).to_parquet(d / "player_rapm.parquet", index=False)
    assert fx.rapm_features(games, tmp_path)[list(fx.RAPM_SUMS)].isna().all().all()      # no as_of: stays null


def test_partial_rapm_join_is_imputed_and_the_trainer_runs(tmp_path, data):
    """An OUT_ROOT that already holds a lineups player_rapm table covering one team from mid-season on (the I6
    integration case): the RAPM pair joins the feature list, missing values are imputed to 0 with the has_rapm flags,
    no design matrix holds NaN and the trainer completes."""
    games = data["games"]
    season = sorted(games["season"].unique())[0]
    s = games[games["season"] == season].sort_values("game_date")
    mid = s["game_date"].iloc[len(s) // 2]
    tid = int(s["home_team_id"].iloc[0])
    d = tmp_path / "analytics" / "lineups" / season
    d.mkdir(parents=True)
    pd.DataFrame({"player_id": [1, 2], "team_id": [tid, tid], "rapm": [1.0, 0.5], "as_of": [mid, mid],
                  "in_sample": [False, False]}).to_parquet(d / "player_rapm_prior.parquet", index=False)
    ctx = tr.main(["--fixtures", "--out", str(tmp_path), "--report", str(tmp_path / "r.md"), "--bootstrap", "5", "--fixture-games", str(N_GAMES)])
    assert set(fx.RAPM_FEATURES) <= set(ctx["features"])
    bg = ctx["games"]
    assert bg["home_rapm_sum"].isna().any() and bg["home_rapm_sum"].notna().any()        # a partial join
    assert ((bg["home_has_rapm"] == 1) == bg["home_rapm_sum"].notna()).all()
    X = fx.design(bg)
    assert not X.isna().any().any() and (X.loc[bg["home_rapm_sum"].isna(), "home_rapm_sum"] == 0).all()
    assert (tmp_path / "analytics" / "betting" / season / "betting_predictions.parquet").exists()


def test_no_em_dash_in_lane_files():
    root = Path(__file__).resolve().parents[1]
    files = [root / "analytics" / "betting" / f for f in ("__init__.py", "validate.py", "eda.py", "features.py", "train.py")]
    files += [root / "tests" / "test_betting.py", root / "tests" / "_fx_betting.py", root / "docs" / "contract" / "betting.md",
              root / "notebooks" / "eda_betting.ipynb", root / "reports" / "betting.md"]
    for f in files:
        if f.exists():
            assert chr(8212) not in f.read_text(), f
