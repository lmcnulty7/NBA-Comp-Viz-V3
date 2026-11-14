"""analytics/features_xfg.py and analytics/train_xfg.py on synthetic shots tables (section 10a columns) and, when
the saved basketball-reference pages are present, on the rows collect.bref_shots builds from them. No network."""
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from analytics import features_xfg as fx, train_xfg as tx  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "bref"
HELD = "12.16.2015.PHX.at.GSW"      # the SportVU held-out game of sportvu/splits.json


def shots_frame(n_games=8, per_game=300, seed=0, heldout_game=None) -> pd.DataFrame:
    """A synthetic 10a-shaped table: positions drawn over the half court, makes from a known logistic relation in
    the four location features (so the logistic model must beat the constant and come out calibrated)."""
    rng = np.random.default_rng(seed)
    rows = []
    games = ["01.%02d.2016.AAA.at.BBB" % (d + 1) for d in range(n_games)]
    if heldout_game:
        games.append(heldout_game)
    for g in games:
        for i in range(per_game):
            r = rng.uniform(0, 30)
            th = rng.uniform(-np.pi / 2, np.pi / 2)
            x, y = r * np.sin(th) * 10, max(r * np.cos(th) * 10 - 10, -5)
            dist = np.hypot(x, y) / 10
            corner = abs(x) > 220 and y < 90
            three = dist >= 22
            z = 1.4 - 0.075 * dist + 0.4 * corner - 0.3 * three + 0.2 * np.sin(np.arctan2(x, y))
            rows.append({"game": g, "sportvu_game": g, "period": int(rng.integers(1, 5)), "clock_s": float(rng.uniform(0, 720)),
                         "made": bool(rng.uniform() < 1 / (1 + np.exp(-z))), "loc_x": round(x, 1), "loc_y": round(y, 1),
                         "shot_type": "3PT" if three else "2PT", "shot_distance_ft": float(round(dist)),
                         "court_x": 5.25 + y / 10, "court_y": 25 + x / 10, "split": "heldout_game" if g == heldout_game else None,
                         "heldout": g == heldout_game, "source": "bref", "release_frame": None, "closest_defender_dist_ft": None})
    return pd.DataFrame(rows)


# ── features ───────────────────────────────────────────────────────────────────────────────────────
def test_features_distance_angle_corner_and_columns():
    df = pd.DataFrame({"sportvu_game": ["g"] * 4, "period": [1, 2, 3, 4], "clock_s": [700.0, 10.5, 0.0, 300.0],
                       "made": [True, False, True, False],
                       "loc_x": [0.0, 30.0, 230.0, -150.0], "loc_y": [40.0, 40.0, 50.0, 200.0],
                       "shot_distance_ft": [4.0, np.nan, 23.0, np.nan], "shot_type": ["2PT", None, "3PT", None],
                       "heldout": [False, False, False, True], "closest_defender_dist_ft": [1.0, 2.0, 3.0, 4.0]})
    f = fx.features(df)
    assert list(f.columns) == ["sportvu_game", "event", "made", "heldout", "distance_ft", "angle_rad", "is_three", "is_corner", "period", "clock_s"]
    assert f["distance_ft"].tolist() == pytest.approx([4.0, 5.0, 23.0, 25.0])      # stated feet, else pixels / 10
    assert f["angle_rad"].iloc[0] == 0.0 and f["angle_rad"].iloc[1] > 0 and f["angle_rad"].iloc[3] < 0
    assert f["angle_rad"].iloc[1] == pytest.approx(np.arctan2(30, 40))
    assert f["is_three"].tolist() == [False, False, True, True]                   # shot_type when stated, else >= 22 ft
    assert f["is_corner"].tolist() == [False, False, True, False]
    assert f["heldout"].tolist() == [False, False, False, True] and f["event"].tolist() == [0, 1, 2, 3]
    assert "closest_defender_dist_ft" not in f.columns
    assert fx.feature_columns() == ["distance_ft", "angle_rad", "is_three", "is_corner"]


def test_features_game_id_event_id_and_split_fallback():
    df = pd.DataFrame({"sportvu_game": ["g", "h"], "game_id": ["0021500001", "0021500002"], "event_id": [7, 9],
                       "period": [1, 1], "clock_s": [1.0, 2.0], "made": [1, 0], "loc_x": [0.0, 0.0], "loc_y": [0.0, 250.0],
                       "split": [None, "heldout_arena"]})
    f = fx.features(df)
    assert f["game_id"].tolist() == ["0021500001", "0021500002"] and f["event"].tolist() == [7, 9]
    assert f["heldout"].tolist() == [False, True] and f["is_three"].tolist() == [False, True]
    with pytest.raises(ValueError):
        fx.features(df.drop(columns=["loc_y"]))


def test_assert_no_leak_raises():
    fx.assert_no_leak(pd.DataFrame(columns=fx.feature_columns()))
    for bad in sorted(fx.FORBIDDEN):
        with pytest.raises(ValueError, match=bad):
            fx.assert_no_leak(pd.DataFrame(columns=fx.feature_columns() + [bad]))
    with pytest.raises(ValueError):
        fx.assert_no_leak(["distance_ft", "closest_defender_dist_ft"])


# ── the trainer ────────────────────────────────────────────────────────────────────────────────────
def test_training_frame_sets_heldout_rows_aside(monkeypatch):
    df = shots_frame(n_games=2, per_game=20, heldout_game=HELD)
    df.loc[df.index[-1], "heldout"] = False            # a held-out game id slips through with heldout=false: still set aside
    calls = []
    real_guard = tx.refuse_heldout
    monkeypatch.setattr(tx, "refuse_heldout", lambda items, who: (calls.append((list(items), who)), real_guard(items, who))[1])
    train, held = tx.training_frame(df)
    assert len(train) == 40 and len(held) == 20 and not train["heldout"].any()
    assert HELD not in set(train["sportvu_game"]) and set(held["sportvu_game"]) == {HELD}
    assert calls == [(sorted(train["sportvu_game"].unique()), "train_xfg")]
    # the guard itself (sportvu.splits.refuse_heldout) stops a run whose training ids carry a held-out game
    with pytest.raises(SystemExit, match="train_xfg refuses"):
        real_guard(["01.01.2016.AAA.at.BBB", HELD], "train_xfg")
    with pytest.raises(SystemExit):
        real_guard(["gsw_phx_2016"], "train_xfg")


def test_main_synthetic_eight_games(tmp_path, monkeypatch):
    df = shots_frame(n_games=8, per_game=300, heldout_game=HELD)
    src = tmp_path / "shots.parquet"
    df.to_parquet(src)
    seen = []
    real_fit = tx.fit_models
    monkeypatch.setattr(tx, "fit_models", lambda train: (seen.append(train), real_fit(train))[1])
    report = tmp_path / "xfg.md"
    ctx = tx.main(["--shots", str(src), "--report", str(report), "--folds", "5", "--seed", "0", "--bootstrap", "50"])
    assert report.exists() and "## Metrics (out-of-fold" in report.read_text() and chr(0x2014) not in report.read_text()
    assert ctx["train_rows"] == 2400 and ctx["train_games"] == 8 and ctx["heldout_rows"] == 300 and ctx["folds"] == 5
    s = ctx["scores"]
    assert s["logistic"]["log_loss"]["point"] < s["constant"]["log_loss"]["point"]
    assert s["logistic"]["brier"]["point"] < s["constant"]["brier"]["point"]
    assert 0.7 < s["logistic"]["calibration_slope"]["point"] < 1.3
    assert s["logistic"]["log_loss"]["lo"] <= s["logistic"]["log_loss"]["point"] <= s["logistic"]["log_loss"]["hi"]
    assert np.isnan(s["constant"]["calibration_slope"]["point"])
    assert ctx["calibration"]["n"].sum() == 2400
    assert set(ctx["heldout_scores"]) == {"constant", "logistic"} and ctx["heldout_scores"]["logistic"]["log_loss"] < 0.75
    # every frame that reached a fit: 5 folds plus the refit for the held-out rows, none with a held-out row or game
    assert len(seen) == 6
    for fr in seen:
        assert not fr["heldout"].any() and HELD not in set(fr["sportvu_game"])
    assert all(len(fr) < 2400 for fr in seen[:5]) and len(seen[5]) == 2400
    with pytest.raises(ValueError):
        real_fit(tx.features(df))                       # the fit itself refuses a frame with a held-out row


def test_main_fewer_games_than_folds_and_single_game(tmp_path):
    df = shots_frame(n_games=3, per_game=60)
    src = tmp_path / "three.parquet"
    df.to_parquet(src)
    ctx = tx.main(["--shots", str(src), "--report", str(tmp_path / "r3.md"), "--folds", "5", "--bootstrap", "10"])
    assert ctx["folds"] == 3 and "leave-one-game-out" in ctx["protocol"] and "leave-one-game-out" in (tmp_path / "r3.md").read_text()
    one = tmp_path / "one.csv"
    df[df["sportvu_game"] == df["sportvu_game"].iloc[0]].to_csv(one, index=False)
    ctx = tx.main(["--shots", str(one), "--report", str(tmp_path / "r1.md"), "--bootstrap", "10"])
    txt = (tmp_path / "r1.md").read_text()
    assert ctx["mode"] == "in-sample" and ctx["train_rows"] == 60 and "## Metrics (in-sample, 60 attempts, 1 game(s))" in txt
    assert np.isnan(ctx["scores"]["logistic"]["log_loss"]["lo"])


REAL = [p for p in sorted(FIX.glob("*.html")) if not p.name.endswith(".synthetic.html")]


@pytest.mark.skipif(len(REAL) < 2, reason="the saved basketball-reference pages are not in tests/fixtures/bref")
def test_smoke_run_on_the_saved_bref_pages(tmp_path):
    pytest.importorskip("bs4")
    from collect import bref_shots as bs
    from sportvu import colab_contract as cc, splits
    sp = splits.load()
    frames = []
    for code, game in (("201602270OKC", {"game": "02.27.2016.GSW.at.OKC", "date": "2016-02-27", "away": "GSW", "home": "OKC"}),
                       ("201512160GSW", {"game": HELD, "date": "2015-12-16", "away": "PHX", "home": "GSW"})):
        shots = bs.parse_shot_chart((FIX / (code + ".html")).read_text())
        frames.append(bs.rows_for_game(game, shots, "2026-11-05T00:00:00Z", cc.split_of(game, sp)))
    df = pd.concat(frames, ignore_index=True)
    assert df["heldout"].sum() == (df["sportvu_game"] == HELD).sum() > 0
    src = tmp_path / "bref.parquet"
    df.to_parquet(src)
    ctx = tx.main(["--shots", str(src), "--report", str(tmp_path / "smoke.md"), "--bootstrap", "10", "--smoke"])
    assert ctx["mode"] == "in-sample" and ctx["train_games"] == 1 and ctx["heldout_games"] == 1
    assert ctx["train_rows"] == int((df["sportvu_game"] != HELD).sum()) and ctx["heldout_rows"] == int(df["heldout"].sum())
    assert np.isfinite(ctx["scores"]["logistic"]["log_loss"]["point"]) and "smoke run" in (tmp_path / "smoke.md").read_text()
