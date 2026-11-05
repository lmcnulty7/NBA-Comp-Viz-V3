"""analytics/betting on the fixtures of tests/_fx_betting.py (the shared analytics.fixtures when present, else the
local fallback with the same signatures). No network, no real odds. Under 30 s."""
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from analytics.betting import eda, features as fx, train as tr, validate as va  # noqa: E402
from tests._fx_betting import synth_odds, synth_stints  # noqa: E402


@pytest.fixture(scope="module")
def data():
    d = synth_stints(0, n_games=40, seasons=("2020-21", "2021-22"))
    return {"games": d["games"], "odds": synth_odds(0, d["games"])}


@pytest.fixture(scope="module")
def joined(data):
    return va.validate(data["odds"], data["games"])


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
    assert r["closing_spread_home"] == -3.5 and r["closing_total"] == 220.5 and r["n_close_snapshots"] == 2
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
    # the joined spread is the fixture's own spread (the close snapshot carries it without noise)
    both = m[m["spread_home"].notna()]
    assert np.allclose(both["closing_spread_home"].astype(float), both["spread_home"].astype(float))
    assert j["closing_total"].dropna().between(150, 300).all()


# ── features: time-forward and leak guards ─────────────────────────────────────────────────────────
def test_features_use_previous_games_only(joined):
    bg = fx.build_games(joined["joined"], "2021-22")
    assert list(bg.columns[:len(fx.GAMES_COLUMNS)]) == list(fx.GAMES_COLUMNS)
    assert bg["heldout"].sum() == (bg["season"] == "2021-22").sum() > 0
    # changing a later game's score must not change any earlier row's features
    j2 = joined["joined"].copy()
    last = j2.sort_values(["game_date", "game_id"]).index[-1]
    j2.loc[last, "home_score"] = 300
    bg2 = fx.build_games(j2, "2021-22")
    earlier = bg["game_id"] != j2.loc[last, "game_id"]
    pd.testing.assert_frame_equal(bg.loc[earlier, list(fx.FEATURES)].reset_index(drop=True),
                                  bg2.loc[earlier, list(fx.FEATURES)].reset_index(drop=True))
    # the first game of every team in a season (home or away) carries no form and the capped rest
    srt = bg.sort_values(["game_date", "game_id"])
    seen = set()
    for r in srt.itertuples(index=False):
        for side, tid in (("home", r.home_team_id), ("away", r.away_team_id)):
            if (r.season, tid) not in seen:
                seen.add((r.season, tid))
                assert getattr(r, side + "_rest_days") == fx.REST_CAP and getattr(r, side + "_net10") == 0.0
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


# ── end to end ─────────────────────────────────────────────────────────────────────────────────────
def test_end_to_end_fixture_run(tmp_path):
    report = tmp_path / "betting.md"
    ctx = tr.main(["--fixtures", "--out", str(tmp_path), "--report", str(report), "--bootstrap", "20", "--fixture-games", "40"])
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
    cr = eda.cover_rate_by_spread_bin(bg)
    assert cr["n"].sum() == bg["home_cover"].notna().sum()
    md = eda.render_markdown(eda.join_coverage(joined["counts"]))
    assert md.startswith("| count | value |") and "join.coverage" in md
    sh = eda.shuffled_label_check(bg, seed=0, folds=3)
    assert set(sh["labels"]) == {"true", "shuffled"} and len(sh) == 6
    market = sh[sh["candidate"] == "market"]["log_loss"]
    assert np.allclose(market, np.log(2))


def test_no_em_dash_in_lane_files():
    root = Path(__file__).resolve().parents[1]
    files = [root / "analytics" / "betting" / f for f in ("__init__.py", "validate.py", "eda.py", "features.py", "train.py")]
    files += [root / "tests" / "test_betting.py", root / "tests" / "_fx_betting.py", root / "docs" / "contract" / "betting.md",
              root / "notebooks" / "eda_betting.ipynb", root / "reports" / "betting.md"]
    for f in files:
        if f.exists():
            assert chr(8212) not in f.read_text(), f
