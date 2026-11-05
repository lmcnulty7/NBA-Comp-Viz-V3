"""analytics/load on hand-built frames, the local fixture generator (planted schedule and speed decline) and,
when analytics.fixtures imports, the shared season. No network. Fixture numbers are never results."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import _fx_load as fx                                           # noqa: E402
from analytics import schema                                    # noqa: E402
from analytics.load import eda, features as F, train as T, validate as V   # noqa: E402

HELD = "12.16.2015.PHX.at.GSW"      # the SportVU held-out game of sportvu/splits.json


# ── hand-built frames ──────────────────────────────────────────────────────────────────────────────
def walk(player_id, frames, period, clock0, x0, speed_ftps, team="AAA", team_id=1, ts0=1_000_000):
    """One player walking along x at a constant speed over consecutive 25 Hz frames."""
    k = np.arange(len(frames), dtype=float)
    return pd.DataFrame({"game": "11.01.2015.AAA.at.BBB", "period": period, "frame": np.asarray(frames),
                         "ts_ms": ts0 + np.asarray(frames) * 40, "game_clock_s": clock0 - k / 25.0,
                         "team_id": team_id, "team": team, "home": True, "player_id": player_id,
                         "x": x0 + k * speed_ftps / 25.0, "y": 25.0})


def test_stints_from_a_hand_built_on_off_sequence():
    a = pd.concat([walk(7, range(0, 100), 1, 720.0, 10.0, 5.0),            # on: frames 0..99, 4 s, 5 ft/s
                   walk(7, range(300, 400), 1, 600.0, 30.0, 5.0),           # off 200 frames, back at clock 600
                   walk(7, range(400, 450), 2, 720.0, 10.0, 14.0)],         # period 2: a 14 ft/s run
                  ignore_index=True)
    b = walk(8, range(0, 100), 1, 720.0, 10.0, 3.0)                         # a second player, one stint
    st = F.player_stints(pd.concat([a, b], ignore_index=True), "11.01.2015.AAA.at.BBB", heldout=False)
    assert list(st.columns) == F.STINT_COLUMNS
    s7 = st[st["player_id"] == 7].sort_values("stint")
    assert s7["stint"].tolist() == [0, 1, 2]
    assert s7["start_frame"].tolist() == [0, 300, 400] and s7["end_frame"].tolist() == [99, 399, 449]
    assert s7["minutes"].to_numpy() == pytest.approx([99 / 25 / 60, 99 / 25 / 60, 49 / 25 / 60], abs=1e-6)
    assert s7["avg_speed_ftps"].to_numpy() == pytest.approx([5.0, 5.0, 14.0], abs=1e-6)
    assert s7["distance_ft"].to_numpy() == pytest.approx([99 * 0.2, 99 * 0.2, 49 * 14 / 25], abs=1e-6)
    assert s7["high_intensity_share"].tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert s7["accelerations"].tolist() == [0, 0, 0]                          # constant speed: no acceleration episode
    s8 = st[st["player_id"] == 8]
    assert len(s8) == 1 and s8["avg_speed_ftps"].iloc[0] == pytest.approx(3.0, abs=1e-6)
    assert (st["heldout"] == False).all()                                     # noqa: E712


def test_validate_drops_duplicates_offcourt_and_teleports_with_counts():
    f = walk(7, range(0, 50), 1, 720.0, 10.0, 5.0)
    f = pd.concat([f, f.iloc[[10]]], ignore_index=True)                       # duplicate (frame, player)
    f.loc[20, "x"] = 200.0                                                    # off the court margin
    f.loc[30, "x"] = f.loc[29, "x"] + 2.0                                     # a 2 ft jump in 40 ms: 50 ft/s teleport
    clean, log = V.validate_frames(f)
    assert log["rows_in"] == 51 and log["rows_duplicate_frame_player"] == 1 and log["rows_outside_court"] == 1
    assert log["rows_speed_over_30_ftps"] == 1 and log["rows_out"] == 48
    assert set(clean["frame"]) == set(range(50)) - {20, 30}
    with pytest.raises(ValueError, match="lacks column"):
        V.validate_frames(f.drop(columns=["x"]))


# ── schedule and history ───────────────────────────────────────────────────────────────────────────
def test_schedule_features_from_manifest_dates():
    man = {"games": {"11.01.2015.AAA.at.BBB": {"date": "2015-11-01", "away": "AAA", "home": "BBB", "status": "ok"},
                     "11.02.2015.CCC.at.AAA": {"date": "2015-11-02", "away": "CCC", "home": "AAA", "status": "failed"},
                     "11.05.2015.AAA.at.CCC": {"date": "2015-11-05", "away": "AAA", "home": "CCC", "status": "ok"},
                     "11.14.2015.BBB.at.AAA": {"date": "2015-11-14", "away": "BBB", "home": "AAA", "status": "ok"}}}
    s = F.schedule_table(man)
    a = s[s["team"] == "AAA"].sort_values("game_date")
    assert a["days_rest"].tolist() == [7.0, 1.0, 3.0, 7.0]                   # first game takes the cap; 9 days capped at 7
    assert a["back_to_back"].tolist() == [False, True, False, False]
    assert a["games_last_7"].tolist() == [0, 1, 2, 0]
    assert a["prev_game"].tolist()[1] == "11.01.2015.AAA.at.BBB" and a["next_game"].tolist()[0] == "11.02.2015.CCC.at.AAA"
    assert a["next_game"].tolist()[-1] is None or pd.isna(a["next_game"].tolist()[-1])


@pytest.fixture(scope="module")
def local_tables():
    inp = fx.season_inputs(seed=1, n_games=6, heldout_every=3, n_possessions=16, prefer_shared=False)
    return inp, F.build_tables(inp, n_boot=30, seed=0)


def test_player_games_history_reads_previous_games_only_and_absence_from_rosters(local_tables):
    inp, t = local_tables
    pg, st = t["player_games"], t["player_stints"]
    assert list(pg.columns) == F.GAME_COLUMNS and list(st.columns) == F.STINT_COLUMNS
    assert len(pg) == len(pg.drop_duplicates(["sportvu_game", "player_id"]))
    games = {g["sportvu_game"]: g for g in inp["games"]}
    sch = t["schedule"]
    checked_hist = checked_abs = 0
    for r in pg.itertuples():
        row = sch[(sch["team"] == r.team) & (sch["sportvu_game"] == r.sportvu_game)].iloc[0]
        prev, nxt = row["prev_game"], row["next_game"]
        if isinstance(prev, str) and not games[prev]["heldout"]:
            mine = pg[(pg["sportvu_game"] == prev) & (pg["player_id"] == r.player_id)]
            exp = float(mine["minutes"].iloc[0]) if len(mine) else 0.0
            chain = [prev]
            g2 = sch[(sch["team"] == r.team) & (sch["sportvu_game"] == prev)].iloc[0]["prev_game"]
            if isinstance(g2, str):
                chain.append(g2)
            if all(not games[c]["heldout"] for c in chain) and len(chain) == 1:
                assert r.minutes_last_3 == pytest.approx(exp)
                checked_hist += 1
            if len(mine):
                assert r.prev_avg_speed_ftps == pytest.approx(float(mine["avg_speed_ftps"].iloc[0]))
        elif isinstance(prev, str):
            assert pd.isna(r.minutes_last_3) and pd.isna(r.prev_decline_ftps)   # a held-out previous game is never read
        if isinstance(nxt, str) and not games[nxt]["heldout"]:
            on_next = r.player_id in set(games[nxt]["rosters"]["player_id"].astype(int))
            assert bool(r.absent_next) == (not on_next)
            checked_abs += 1
        else:
            assert pd.isna(r.absent_next)
    assert checked_hist > 0 and checked_abs > 0
    assert pg["absent_next"].dropna().astype(bool).any()                      # the fixture plants absences
    # the planted within-game decline is negative on average
    assert pg.loc[~pg["heldout"], "decline_ftps"].mean() < 0
    sm = t["load_summary"]
    assert list(sm.columns) == F.SUMMARY_COLUMNS and (sm["insufficient"] == (sm["n_stints"] < F.MIN_STINTS_RANK)).all()


# ── leakage guards ─────────────────────────────────────────────────────────────────────────────────
def test_leakage_guards_refuse_targets_same_game_totals_and_held_out_rows(local_tables):
    _inp, t = local_tables
    pg = t["player_games"]
    for bad in ("decline_ftps", "absent_next", "minutes", "distance_ft", "end_frame"):
        with pytest.raises(ValueError, match=bad):
            F.assert_no_leak(F.FEATURES + [bad])
    with pytest.raises(ValueError):                                           # a schema target column through assert_time_forward
        schema.assert_time_forward(pg[F.FEATURES].assign(points=0), F.GUARD_TABLE)
    X = F.design(pg)
    assert list(X.columns) == F.FEATURES and not (set(X.columns) & F.FORBIDDEN)
    train, held = T.training_frames(pg)
    assert not train["heldout"].any() and held["heldout"].all() and len(held) > 0
    with pytest.raises(ValueError, match="held-out"):
        T.fit_regression(pg)                                                  # a held-out row must never reach a fit
    with pytest.raises(ValueError, match="held-out"):
        T.fit_absence(pg)
    # a game that carries the real held-out id is set aside even when its flag says otherwise
    pg2 = pg.copy()
    pg2.loc[pg2.index[:3], "sportvu_game"] = HELD
    pg2["heldout"] = False
    train2, held2 = T.training_frames(pg2)
    assert set(held2["sportvu_game"]) == {HELD} and HELD not in set(train2["sportvu_game"])
    # the file-level tripwires test_splits.py looks for
    src = (ROOT / "analytics" / "load" / "train.py").read_text()
    assert "refuse_heldout" in src and "assert_time_forward" in src


# ── end to end ─────────────────────────────────────────────────────────────────────────────────────
def test_train_end_to_end_on_the_local_fixture(tmp_path, local_tables):
    inp, t = local_tables
    ctx = T.run(inp, n_folds=5, seed=0, n_boot=20, fixture=True, tables=t)
    assert ctx["train_games"] == 4 and ctx["heldout_games"] == 2
    assert np.isfinite(ctx["reg"]["mean"]["point"]) and np.isfinite(ctx["reg"]["ridge"]["point"])
    assert ctx["reg_note"]["mode"] == "out-of-fold" and ctx["reg_note"]["folds"] == 4
    assert ctx["abs"] is not None and 0 < ctx["abs"]["base_rate"] < 1 and np.isfinite(ctx["abs"]["logistic"]["log_loss"]["point"])
    assert ctx["heldout"]["rows"] > 0 and np.isfinite(ctx["heldout"]["mse_ridge"])
    assert ctx["aggregates"]["players_matched"] > 0
    assert set(ctx["rest_table"]["bucket"]) == {"back_to_back", "one_day", "two_plus"}
    ctx["command"] = "test"
    text = T.write_report(tmp_path / "load.md", ctx)
    assert "fixture output, not a result" in text and "refuse_heldout" in text and "Absence is not injury" in text
    assert "—" not in text
    written = F.write_tables(t, tmp_path, "2015-16")
    assert (tmp_path / "analytics" / "load" / "2015-16" / "player_games.parquet").exists()
    assert (tmp_path / "analytics" / "load" / "heldout" / "2015-16" / "player_stints.parquet").exists()
    back = pd.read_parquet(tmp_path / "analytics" / "load" / "2015-16" / "player_games.parquet")
    assert not back["heldout"].any() and len(written) == 5


def test_eda_tables(local_tables):
    _inp, t = local_tables
    st, pg, sm = t["player_stints"], t["player_games"], t["load_summary"]
    sp = eda.speed_by_stint_index(st)
    assert {"stint", "n", "mean_speed_ftps"} <= set(sp.columns) and sp["n"].sum() == int((~st["heldout"]).sum())
    rb = eda.decline_by_rest_bucket(pg)
    assert rb["bucket"].tolist() == ["back_to_back", "one_day", "two_plus"]
    dd = eda.distance_distribution(pg, st)
    assert "stint_distance_ft" in set(dd["quantity"]) and (dd["q50"].dropna() >= 0).all()
    corr = eda.correlation_table(pg)
    assert "decline_ftps" in corr.columns and "minutes" in corr.index
    sh = eda.shuffled_check(pg, n_shuffles=10)
    assert sh["feature"].tolist() == F.FEATURES and "outside_band" in sh.columns
    md = eda.markdown(eda.stint_gate(sm))
    assert md.startswith("| players |") and "—" not in md


def test_data_root_layout_round_trip(tmp_path):
    """A tiny DATA_ROOT written from the local generator (frames parquet, manifest, rosters): the --data path."""
    inp = fx.season_inputs(seed=2, n_games=3, heldout_every=3, n_possessions=8, prefer_shared=False)
    root = tmp_path / "root"
    (root / "sportvu_contract" / "heldout").mkdir(parents=True)
    for g in inp["games"]:
        sub = "heldout/" if g["heldout"] else ""
        g["frames"].to_parquet(root / "sportvu_contract" / sub / ("%s.parquet" % g["sportvu_game"]), index=False)
        rp = root / "analytics" / "rosters" / sub / "2015-16"
        rp.mkdir(parents=True, exist_ok=True)
        g["rosters"].to_parquet(rp / ("%s.parquet" % g["sportvu_game"]), index=False)
    (root / "sportvu_contract" / "manifest.json").write_text(json.dumps(inp["manifest"]))
    loaded = V.load_data_root(root, "2015-16")
    assert len(loaded["games"]) == 3 and sum(g["heldout"] for g in loaded["games"]) == 1
    assert loaded["aggregates"] is None and "never overlaps" in V.injuries_note(loaded["injuries"])
    t = F.build_tables(loaded, n_boot=10)
    assert len(t["player_games"]) == sum(len(g["rosters"]) for g in inp["games"])
    rep = V.validate_inputs(loaded)
    assert rep["n_games"] == 3 and rep["totals"]["rows_out"] > 0
    out = V.main(["--data", str(root), "--out", str(tmp_path / "out")])
    assert (tmp_path / "out" / "analytics" / "load" / "2015-16" / "validation.json").exists() and out["n_heldout"] == 1


@pytest.mark.skipif(not fx.SHARED, reason="analytics.fixtures (lane I1) not importable: the local generator covers the tests")
def test_shared_fixtures_run_end_to_end(tmp_path):
    """The shared season: distinct teams and players per game, so the schedule history is empty and absent_next
    unknown; the stints, the decline target, the guards and the held-out path must still work."""
    inp = V.load_fixtures(seed=0, n_games=3, heldout_every=3, prefer_shared=True)
    assert "shared" in inp["source"] and sum(g["heldout"] for g in inp["games"]) == 1
    t = F.build_tables(inp, n_boot=10)
    assert len(t["player_stints"]) > 0 and t["player_games"]["decline_ftps"].notna().any()
    ctx = T.run(inp, n_folds=5, seed=0, n_boot=10, fixture=True, tables=t)
    assert ctx["heldout_games"] == 1 and ctx["train_games"] == 2
    assert np.isfinite(ctx["reg"]["ridge"]["point"])
    ctx["command"] = "test"
    assert "fixture output, not a result" in T.write_report(tmp_path / "load.md", ctx)
