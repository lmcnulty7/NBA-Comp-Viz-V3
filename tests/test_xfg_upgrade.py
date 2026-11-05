"""analytics/xfg (lane XFG): the contest-feature xFG upgrade on the synthetic season of tests/_fx_xfg.py.

One small season (4 games of 12 possessions, the 4th held out as 12.16.2015.PHX.at.GSW) is generated once per
module and shared by every test; the end-to-end run goes through train.main() with the season patched in so no
test regenerates frames. No network, no real data."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from analytics import schema  # noqa: E402
from analytics.eval import brier, calibration_slope, ece, log_loss  # noqa: E402
from analytics.xfg import eda as E, features as F, train as T, validate as V  # noqa: E402
from tests import _fx_xfg as FX  # noqa: E402

HELD = "12.16.2015.PHX.at.GSW"
N_GAMES, N_POSS = 4, 12


@pytest.fixture(scope="module")
def season():
    return FX.synth_season(0, n_games=N_GAMES, heldout_every=4, n_possessions=N_POSS)


@pytest.fixture(scope="module")
def tables(season):
    t, summaries = V.clean(V.tables_from_season(season))
    t["summaries"] = summaries
    return t


@pytest.fixture(scope="module")
def frames(tables):
    train, heldout = T.assemble(tables)
    return train, heldout


# ── fixtures and validation ────────────────────────────────────────────────────────────────────────
def test_fixture_season_conforms_to_the_contract(season, tables):
    assert len(season) == N_GAMES
    held = [g for g in season if g["heldout"]]
    assert [g["sportvu_game"] for g in held] == [HELD]
    for table in ("shot_frames", "rosters", "possessions"):
        assert schema.validate(tables[table], table) == [], table
    assert {"sportvu_game", "event", "player_id", "period", "clock_s", "made", "loc_x", "loc_y", "heldout"} <= set(tables["shots"].columns)
    assert tables["shots"]["heldout"].sum() > 0
    assert set(tables["frames"]) == {g["sportvu_game"] for g in season}
    same = FX.synth_season(0, n_games=1, heldout_every=0, n_possessions=N_POSS)[0]["shots"]
    again = FX.synth_season(0, n_games=1, heldout_every=0, n_possessions=N_POSS)[0]["shots"]
    pd.testing.assert_frame_equal(same, again)


def test_validate_drops_out_of_range_duplicates_and_missing_rows():
    base = {"game": "g", "period": 1, "frame": 0, "game_clock_s": 700.0, "shot_clock_s": 20.0, "ball_x": 10.0, "ball_y": 10.0,
            "ball_z": 5.0, "team_id": 1, "player_id": 1, "x": 10.0, "y": 10.0}
    rows = [dict(base), dict(base, player_id=2, x=100.0), dict(base, player_id=3, game_clock_s=800.0), dict(base),
            dict(base, player_id=4, x=np.nan), dict(base, player_id=5, shot_clock_s=np.nan), dict(base, frame=1, ball_x=np.nan)]
    out, s = V.clean_frames(pd.DataFrame(rows))
    assert (s["rows_in"], s["rows_out"], s["dropped_missing"], s["dropped_range"], s["dropped_duplicate"]) == (7, 3, 1, 2, 1)
    assert len(out) == 3
    shots = pd.DataFrame({"sportvu_game": ["g"] * 3, "event": [0, 1, 2], "period": [1, 1, 1], "clock_s": [700.0, 650.0, 600.0],
                          "made": [True, False, True], "loc_x": [0.0, 300.0, 10.0], "loc_y": [50.0, 50.0, 50.0]})
    out, s = V.clean_shots(shots)
    assert s["dropped_range"] == 1 and out["event"].tolist() == [0, 2]


# ── features ───────────────────────────────────────────────────────────────────────────────────────
def test_feature_frame_is_time_forward_and_follows_the_join_rule(tables):
    shots, sf = tables["shots"], tables["shot_frames"].copy()
    # break the join for the first two attempts: one with no release frame, one flagged unmatched
    sf.loc[sf.index[0], ["release_frame", "apex_frame", "outcome_frame"]] = -1
    sf.loc[sf.index[1], "confidence_flag"] = "unmatched"
    feat = F.build(shots, sf, tables["frames"])
    assert len(feat) == len(shots) and list(feat["event"]) == list(shots["event"])
    assert set(F.FEATURES) <= set(feat.columns) and "made" not in feat.columns
    schema.assert_time_forward(feat, "shot_frames")
    for c in ("apex_frame", "outcome_frame", "made", "points", "confidence", "pbp_shooter_name"):
        assert c not in feat.columns
    broken = feat.merge(sf.iloc[:2][["sportvu_game", "event"]].assign(sportvu_game=lambda d: d["sportvu_game"].astype(str)),
                        on=["sportvu_game", "event"])
    assert len(broken) == 2 and (broken["has_contest"] == 0).all()
    assert broken[F.CONTEST_FEATURES].isna().all().all()
    assert broken[F.LOCATION_FEATURES].notna().all().all()
    ok = feat[feat["has_contest"] == 1]
    assert len(ok) > 0
    assert ok["closest_defender_dist_ft"].between(0, 60).all()
    assert ok["touch_time_s"].between(0, 10).all() and ok["n_defenders_within_6ft"].between(0, 5).all()
    assert set(feat["zone"].unique()) <= set(F.ZONES)


def test_forbidden_columns_raise():
    X = pd.DataFrame({"distance_ft": [1.0], "made": [1]})
    with pytest.raises(ValueError, match="forbidden"):
        F.assert_no_leak(X)
    for bad in ("outcome_frame", "apex_frame", "points", "score", "confidence_flag"):
        with pytest.raises(ValueError):
            F.assert_no_leak(pd.DataFrame({bad: [1.0]}))
    with pytest.raises(ValueError):
        schema.assert_time_forward(pd.DataFrame({"distance_ft": [1.0], "made": [True]}), "shot_frames")
    F.assert_no_leak(pd.DataFrame(columns=F.feature_columns(with_prior=True)))


def test_contest_features_on_a_hand_made_window():
    """Shooter 1 (team A) at (10, 10) at release frame 40, moving from (9, 10) ten frames earlier (2.5 ft/s).
    Defender 2 (team B) at (16, 10) at release, (18, 10) ten frames earlier: the shooter-defender distance goes
    from 9 to 6 ft in 0.4 s, closing at 7.5 ft/s; 1 defender within 6 ft. Defender 3 far away. Ball within 2.5 ft of the shooter for frames 11 to 40 (30 frames = 1.2 s)
    with two dips under 2 ft (dribbles) inside, far away before frame 11."""
    rows = []
    for fr in range(0, 45):
        lag = max(40 - fr, 0)
        sx = 10.0 - 2.5 * min(lag, 10) / 25.0
        dx = 16.0 + 5.0 * min(lag, 10) / 25.0
        if 11 <= fr <= 40:
            bx, by, bz = sx + 1.0, 10.0, 4.0
            if fr in (20, 30):
                bz = 0.5
        else:
            bx, by, bz = 40.0, 40.0, 4.0
        for pid, team, x, y in ((1, 1, sx, 10.0), (2, 2, dx, 10.0), (3, 2, 60.0, 40.0)):
            rows.append({"game": "g", "period": 1, "frame": fr, "game_clock_s": 700.0 - fr / 25.0, "shot_clock_s": 20.0 - fr / 25.0,
                         "ball_x": bx, "ball_y": by, "ball_z": bz, "team_id": team, "player_id": pid, "x": x, "y": y})
    G = F.GameFrames(pd.DataFrame(rows))
    c = G.contest(40, 1)
    assert c["has_contest"] == 1
    assert c["closest_defender_dist_ft"] == pytest.approx(6.0)
    assert c["closest_defender_closing_speed_ftps"] == pytest.approx(7.5)
    assert c["n_defenders_within_6ft"] == 1
    assert c["shooter_speed_ftps"] == pytest.approx(2.5)
    assert c["touch_time_s"] == pytest.approx(30 / 25.0)
    assert c["dribbles_proxy"] == 2
    assert c["shot_clock_s"] == pytest.approx(20.0 - 40 / 25.0)
    assert G.contest(40, 99)["has_contest"] == 0 and G.contest(1000, 1)["has_contest"] == 0
    assert F.count_dribbles([4, 0.5, 4, 0.5, 0.5, 4]) == 2 and F.count_dribbles([0.5, 4, 0.5]) == 0


# ── shooter prior and metrics on hand cases ────────────────────────────────────────────────────────
def test_shooter_prior_is_smoothed_and_never_sees_the_row_itself():
    rows = pd.DataFrame({"sportvu_game": ["A"] * 4 + ["B"] * 4, "player_id": [7] * 8, "zone": ["mid"] * 8,
                         "made": [True, True, True, True, False, False, False, False]})
    pt = T.prior_table(rows, m=4.0)
    assert pt["zone_rate"]["mid"] == 0.5 and pt["table"][(7, "mid")] == pytest.approx((4 + 4 * 0.5) / (8 + 4))
    assert T.apply_prior(pt, pd.DataFrame({"player_id": [7, 8], "zone": ["mid", "rim"]})).tolist() == pytest.approx([0.5, 0.5])
    oof = T.oof_prior(rows, n_folds=2, seed=0)
    # game A rows see only game B (all misses): (0 + m * 0) / (4 + m) = 0; game B rows see only A: 1
    assert oof[:4].tolist() == pytest.approx([0.0] * 4) and oof[4:].tolist() == pytest.approx([1.0] * 4)


def test_metrics_and_selection_on_a_hand_case():
    y = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2, 0.6, 0.7, 0.3, 0.4])
    assert log_loss(y, p) == pytest.approx(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    assert brier(y, p) == pytest.approx(np.mean((p - y) ** 2))
    assert ece(y, p) == pytest.approx(np.mean(np.abs(p - y)))     # one attempt per bin: the bin gap is |p - y|
    q = np.array([0.6, 0.4, 0.7, 0.3, 0.4, 0.55, 0.45, 0.6])       # not separable, so the slope is finite
    assert T.fast_slope(y, q) == pytest.approx(calibration_slope(y, q), abs=0.02)
    assert np.isnan(T.fast_slope(np.ones(4), p[:4]))
    scores = {"location_logistic": {"log_loss": {"point": 0.60}}, "contest_logistic": {"log_loss": {"point": 0.58}},
              "hgb_contest_prior": {"log_loss": {"point": 0.58}}}
    assert T.select(scores) == "contest_logistic"
    ch = E.chance_levels(y)
    assert ch["log_loss_chance"] == pytest.approx(np.log(2)) and ch["brier_chance"] == pytest.approx(0.25)


# ── leakage guard ──────────────────────────────────────────────────────────────────────────────────
def test_held_out_game_id_never_reaches_a_fit(tables, monkeypatch):
    """A held-out id whose heldout flag is false everywhere is still set aside by id, and the refit guard
    refuses it if it ever arrives at fit_candidate."""
    t = {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in tables.items() if k != "summaries"}
    t["shots"]["heldout"] = False
    t["shots"]["split"] = None
    seen = []
    real = T.fit_candidate

    def spy(name, train, seed=0):
        seen.extend(train["sportvu_game"].astype(str).unique().tolist())
        return real(name, train, seed)
    monkeypatch.setattr(T, "fit_candidate", spy)
    train, heldout = T.assemble(t)
    assert HELD not in set(train["sportvu_game"]) and set(heldout["sportvu_game"]) == {HELD}
    pred, _ = T.out_of_fold(train, 2, 0, candidates=("location_logistic",))
    assert seen and HELD not in set(seen)
    bad = train.copy()
    bad.loc[bad.index[:5], "sportvu_game"] = HELD
    with pytest.raises(SystemExit):
        real("contest_logistic", bad, 0)
    flagged = train.copy()
    flagged.loc[flagged.index[:5], "heldout"] = True
    with pytest.raises(ValueError):
        real("contest_logistic", flagged, 0)


def test_shuffled_labels_score_at_chance(frames):
    train, _ = frames
    t = E.shuffle_check(train, n_folds=2, seed=1, tolerance=0.05)
    assert list(t["candidate"]) == list(T.CANDIDATES)
    assert t["at_chance"].all(), t.to_string()


# ── end to end ─────────────────────────────────────────────────────────────────────────────────────
def test_end_to_end_fixture_run(season, tmp_path, monkeypatch):
    monkeypatch.setattr(V, "fixture_season", lambda *a, **k: season)
    report = tmp_path / "xfg.md"
    ctx = T.main(["--fixtures", "--report", str(report), "--out", str(tmp_path), "--folds", "3", "--bootstrap", "10",
                  "--fixture-games", str(N_GAMES), "--fixture-possessions", str(N_POSS)])
    text = report.read_text()
    assert "fixture output, not a result" in text and "## Candidate table" in text and "## Held-out score" in text
    assert "## The real run on the Mac" in text and "analytics.xfg.train --data" in text
    assert ctx["selected"] in T.CANDIDATES and set(ctx["scores"]) == set(T.CANDIDATES)
    for name in T.CANDIDATES:
        for m in ("log_loss", "brier", "ece", "calibration_slope"):
            assert set(ctx["scores"][name][m]) == {"point", "lo", "hi"}
    assert set(ctx["paired"]) == {"hgb_contest_prior vs location_logistic", "contest_logistic vs location_logistic"}
    out = pd.read_parquet(tmp_path / "analytics" / "xfg" / "2015-16" / "xfg_predictions.parquet")
    assert list(out.columns) == T.PREDICTION_COLUMNS
    assert out["heldout"].sum() == ctx["heldout_rows"] > 0 and (out.loc[out["heldout"], "fold"] == -1).all()
    assert set(out.loc[out["heldout"], "sportvu_game"]) == {HELD} and HELD not in set(out.loc[~out["heldout"], "sportvu_game"])
    assert (out.loc[~out["heldout"], "fold"] >= 0).all() and out.loc[~out["heldout"], "fold"].nunique() == 3
    for c in ("p_location", "p_contest", "p_selected"):
        assert out[c].between(0, 1).all()
    assert ctx["heldout_scores"]["log_loss"] > 0
    assert "\u2014" not in text                                    # no em dashes in the report


def test_data_root_layout_round_trip(tmp_path):
    root = FX.write_fixture_root(tmp_path / "root", seed=3, n_games=3, heldout_every=3, n_possessions=6)
    root = Path(root) if not isinstance(root, dict) else tmp_path / "root"
    assert (root / "collect" / "bref_shots" / "shotchart" / "2015-16.parquet").exists()
    assert (root / "collect" / "bref_shots" / "heldout" / "shotchart" / "2015-16.parquet").exists()
    assert list((root / "analytics" / "shot_frames" / "heldout" / "2015-16").glob("*.parquet"))
    t = V.load_data_root(root)
    assert callable(t["frames"]) and t["shots"]["heldout"].sum() > 0
    t, summaries = V.clean(t)
    assert {s["table"] for s in summaries} >= {"shots", "shot_frames", "rosters", "possessions"}
    train, heldout = T.assemble(t)
    assert set(heldout["sportvu_game"]) == {HELD} and HELD not in set(train["sportvu_game"])
    assert train["has_contest"].sum() > 0
    summ = V.main(["--data", str(root), "--out", str(tmp_path / "out")])
    assert (tmp_path / "out" / "analytics" / "xfg" / "2015-16" / "validate.json").exists() and summ
