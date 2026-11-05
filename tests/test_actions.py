"""tests/test_actions.py: analytics.actions (validate, features, train, eda) on the shared fixtures of
analytics/fixtures.py (tests/_fx_actions.py falls back to its own scripted generator only when that import fails),
on the fallback generator's hand-scripted screens, drives and isolations, on a hand-built screen, and on a
DATA_ROOT on disk with tombstoned and pass-2 labels. Everything here runs on synthetic motion: fixture output, not
a result. No network."""
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
import _fx_actions as fx  # noqa: E402
from analytics import schema  # noqa: E402
from analytics.actions import CLASSES, eda, features as F, train as T, validate as V  # noqa: E402

HELD = "12.16.2015.PHX.at.GSW"
OWNED = ["analytics/actions/__init__.py", "analytics/actions/validate.py", "analytics/actions/features.py",
         "analytics/actions/train.py", "analytics/actions/eda.py", "tests/test_actions.py", "tests/_fx_actions.py",
         "docs/contract/actions.md", "reports/actions.md", "notebooks/eda_actions.ipynb"]


@pytest.fixture(scope="module")
def games():
    """Six games from the fixtures in use (analytics.fixtures when importable), every third held out (two held-out,
    four training), 24 possessions each."""
    return fx.season_games(seed=0, n_games=6, heldout_every=3, n_possessions=24)


@pytest.fixture(scope="module")
def fallback_games():
    """Four training games from the fallback generator, whose scripted motion the assertions know exactly."""
    return fx.fallback_season_games(seed=0, n_games=4, heldout_every=0, n_possessions=24)


# ── hand-built frames ──────────────────────────────────────────────────────────────────────────────
def hand_frames(players: dict, seconds: float = 4.0, hz: int = 25) -> tuple:
    """players: {player_id: (team_id, [(t, x, y), ...])}; the ball rides 0.8 ft beside player 1 (the handler).
    -> (frames, possessions) for one possession spanning every frame."""
    n = int(seconds * hz) + 1
    t = np.arange(n) / hz
    rows = []
    for pid, (tid, keys) in players.items():
        kt, kx, ky = (np.array([k[i] for k in keys], dtype=float) for i in range(3))
        rows.append(pd.DataFrame({"frame": np.arange(n), "period": 1, "game_clock_s": 700.0 - t, "shot_clock_s": 20.0 - t,
                                  "team_id": tid, "home": tid == 1, "player_id": pid,
                                  "x": np.interp(t, kt, kx), "y": np.interp(t, kt, ky)}))
    fr = pd.concat(rows, ignore_index=True).sort_values(["frame", "player_id"], kind="stable").reset_index(drop=True)
    h = fr[fr["player_id"] == 1].set_index("frame")
    fr["ball_x"] = fr["frame"].map(h["x"]) + 0.8
    fr["ball_y"] = fr["frame"].map(h["y"])
    fr["ball_z"] = 3.0
    poss = schema.cast(pd.DataFrame([{"sportvu_game": "01.02.2016.AAA.at.BBB", "possession": 0, "period": 1, "start_frame": 0,
                                      "end_frame": n - 1, "start_clock_s": 700.0, "end_clock_s": 700.0 - seconds, "start_event": 0,
                                      "end_event": 0, "offense_team_id": 1, "end_kind": "shot_missed", "points": 0, "gap": False,
                                      "n_frames": n}]), "possessions")
    return fr, poss


def base_players() -> dict:
    """Offense (team 1) attacking the left basket (5.25, 25): handler 1 at 24 ft, defender 11 three feet inside him,
    the other four offensive players and their defenders far away and still."""
    p = {1: (1, [(0, 29.25, 25.0)]), 11: (2, [(0, 26.25, 25.0)]),
         2: (1, [(0, 22.0, 42.0)]), 12: (2, [(0, 20.0, 40.0)]),
         3: (1, [(0, 27.0, 5.0)]), 13: (2, [(0, 24.0, 7.0)]),
         4: (1, [(0, 10.0, 46.0)]), 14: (2, [(0, 9.0, 43.0)]),
         5: (1, [(0, 10.0, 4.0)]), 15: (2, [(0, 9.0, 7.0)])}
    return p


def test_candidate_generator_on_a_hand_built_screen():
    p = base_players()
    # player 2 walks to the defender's shoulder between 0.5 s and 1.0 s and stands there; the handler stays within 12 ft
    p[2] = (1, [(0, 22.0, 42.0), (0.5, 22.0, 42.0), (1.0, 27.0, 27.0), (4.0, 27.0, 27.0)])
    fr, poss = hand_frames(p)
    cands, feats = F.build(fr, poss)
    screens = cands[cands["rule_class"] == "ball_screen"]
    assert len(screens) == 1, cands
    s = screens.iloc[0]
    assert s["ball_handler_id"] == 1 and s["screener_id"] == 2
    assert 20 <= s["start_frame"] <= 30 and s["end_frame"] >= 95       # arrives at 1.0 s, stands until the end
    f = feats.iloc[int(screens.index[0])]
    assert f["scr_def_dist_mean"] < 4 and f["scr_speed_mean"] < 1 and f["rule_is_screen"] == 1
    assert feats.columns.tolist() == F.ID_COLUMNS + F.FEATURES
    # the same player passing at 5 ft per s (never still) is not a screen
    p[2] = (1, [(0, 22.0, 42.0), (0.5, 22.0, 42.0), (4.0, 32.0, 10.0)])
    fr, poss = hand_frames(p)
    cands, _ = F.build(fr, poss)
    assert (cands["rule_class"] != "ball_screen").all()


def test_candidate_generator_on_a_hand_built_drive_and_isolation():
    p = base_players()
    p[1] = (1, [(0, 29.25, 25.0), (1.0, 29.25, 25.0), (2.0, 15.25, 25.0), (4.0, 15.25, 25.0)])     # 14 ft per s at the rim
    p[11] = (2, [(0, 26.25, 25.0), (1.1, 26.25, 25.0), (2.1, 13.0, 25.0), (4.0, 13.0, 25.0)])
    fr, poss = hand_frames(p)
    cands, feats = F.build(fr, poss)
    drives = cands[cands["rule_class"] == "drive"]
    assert len(drives) == 1 and drives.iloc[0]["ball_handler_id"] == 1
    assert 20 <= drives.iloc[0]["start_frame"] <= 30 and 45 <= drives.iloc[0]["end_frame"] <= 55
    assert feats.loc[drives.index[0], "handler_radial_speed_mean"] > 8
    # isolation: nobody within 10 ft of the handler for the 4 s (every teammate of base_players is over 10 ft away)
    fr, poss = hand_frames(base_players())
    cands, feats = F.build(fr, poss)
    iso = cands[cands["rule_class"] == "isolation"]
    assert len(iso) == 1 and iso.iloc[0]["start_frame"] == 0 and iso.iloc[0]["end_frame"] >= 95
    assert feats.loc[iso.index[0], "teammate_min_dist_mean"] > 10
    assert (cands["rule_class"] != "drive").all()


def test_iou_label_matching_and_recall():
    assert F.iou_frames(0, 9, 0, 9) == 1.0 and F.iou_frames(0, 9, 10, 19) == 0.0
    assert F.iou_frames(0, 9, 5, 14) == pytest.approx(5 / 15)
    cands = pd.DataFrame({"sportvu_game": ["g"] * 3, "possession": [0, 0, 1], "candidate": [0, 1, 0],
                          "start_frame": [10, 40, 0], "end_frame": [30, 60, 20], "rule_class": ["ball_screen", "drive", "isolation"],
                          "ball_handler_id": [1, 1, 2], "screener_id": [2, None, None]})
    actions = pd.DataFrame({"sportvu_game": ["g", "g", "g"], "possession": [0, 0, 1], "action_type": ["ball_screen", "cut", "isolation"],
                            "start_frame": [12, 41, 50], "end_frame": [32, 58, 80]})
    m = F.match_labels(cands, actions)
    assert m["label_class"].tolist() == ["ball_screen", "none", "none"]        # cut is not a detected class; iso label does not overlap
    assert m["label_action_type"].tolist() == ["ball_screen", "cut", None]
    r = F.candidate_recall(cands, actions).set_index("class")
    assert r.loc["ball_screen", "recall_any"] == 1.0 and r.loc["isolation", "recall_any"] == 0.0 and r.loc["drive", "labels"] == 0


def test_fallback_candidates_recover_every_scripted_label(fallback_games):
    cands, feats, recall, skipped = F.build_season(fallback_games)
    assert skipped == 0 and len(cands) == len(feats)
    r = recall.set_index("class")
    assert (r["labels"] > 0).all()
    assert (r["recall_same_class"] == 1.0).all(), recall
    assert (cands["label_class"] == "none").sum() > 0                   # the slow cutter makes false screen candidates
    assert set(cands["label_class"]) <= set(CLASSES)
    assert cands.columns.tolist()[:8] == F.CANDIDATE_COLUMNS


def test_fixture_candidates_recover_the_labels(games):
    """On the fixtures in use: candidate recall at IoU >= 0.5 for every detected class, with hard negatives present."""
    train = [g for g in games if not g["heldout"]]
    cands, feats, recall, skipped = F.build_season(train)
    assert len(cands) == len(feats)
    r = recall.set_index("class")
    assert (r["labels"] > 0).all()
    assert (r["recall_any"] >= 0.85).all(), recall
    assert (cands["label_class"] == "none").sum() > 0
    assert cands["label_iou"].between(0, 1).all()


def test_leakage_guard(games):
    train = [g for g in games if not g["heldout"]]
    cands, feats, _r, _s = F.build_season(train)
    schema.assert_time_forward(feats, "possessions")
    F.assert_no_leak(feats.drop(columns=F.ID_COLUMNS))
    assert F.feature_columns() == F.FEATURES
    assert not (set(F.FEATURES) & (F.LABEL_COLUMNS | set(schema.ALL_EXCLUDED)))
    for bad in ("label_class", "end_frame", "points", "pred_class", "p_ball_screen", "rule_class", "action_type", "made"):
        with pytest.raises(ValueError):
            F.assert_no_leak(feats[F.FEATURES].assign(**{bad: 0.0}))
    with pytest.raises(ValueError):
        schema.assert_time_forward(feats.assign(end_frame=0), "possessions")
    # a held-out row never reaches a fit; a training game that carries a held-out id stops the trainer
    c = cands.copy()
    c.loc[0, "heldout"] = True
    with pytest.raises(ValueError, match="held-out"):
        T.fit_models(c, feats, seed=0)
    c = cands.copy()
    c["heldout"] = False
    c.loc[c["sportvu_game"] == c["sportvu_game"].iloc[0], "sportvu_game"] = HELD
    with pytest.raises(SystemExit):
        T.training_split(c, feats)


def test_end_to_end_fixture_run(games, tmp_path):
    ctx = T.run([dict(g) for g in games], n_folds=4, seed=0, n_boot=30, fixtures=True, out_root=tmp_path,
                report=tmp_path / "actions.md", command="test")
    text = (tmp_path / "actions.md").read_text()
    assert "fixture output, not a result" in text and "—" not in text
    assert ctx["mode"] == "out-of-fold" and ctx["folds"] == 4 and ctx["train_games"] == 4 and ctx["heldout_games"] == 2
    assert HELD in ctx["heldout_game_ids"] and HELD not in ctx["train_game_ids"]
    s = ctx["scores"]
    assert s["hgb"]["f1_ball_screen"]["point"] >= 0.8                       # the plan's bar, met here on scripted motion only
    assert s["hgb"]["f1_ball_screen"]["point"] >= s["rule"]["f1_ball_screen"]["point"]
    assert s["rule"]["per_class_f1"]["none"] == 0.0 or np.isnan(s["rule"]["per_class_f1"]["none"])   # the rule never says none
    assert list(s["hgb"]["confusion"].index) == list(CLASSES)
    assert ctx["selected"] in T.MODELS and "Selected model: **%s**" % ctx["selected"] in text
    assert "## Held-out games" in text and set(ctx["heldout_scores"]) == set(T.MODELS)
    tab = ctx["table"]
    cols = ["sportvu_game", "possession", "candidate", "start_frame", "end_frame", "rule_class", "ball_handler_id", "screener_id",
            "label_class", "label_iou", "label_action_type", "pred_class", "p_ball_screen", "p_selected_class", "fold", "heldout"]
    assert tab.columns.tolist() == cols
    assert (tab.loc[tab["heldout"], "fold"] == -1).all() and (tab.loc[~tab["heldout"], "fold"] >= 0).all()
    assert tab["p_ball_screen"].between(0, 1).all() and (tab["p_selected_class"] >= tab["p_ball_screen"]).all()
    det = ctx["detected"]
    assert det.columns.tolist() == ["sportvu_game", "possession", "action_type", "start_frame", "end_frame", "ball_handler_id",
                                    "screener_id", "label_source", "labeller_pass", "confidence"]
    assert (det["label_source"] == "model").all() and (det["confidence"] > 0.5).all() and (det["action_type"] != "none").all()
    base = tmp_path / "analytics" / "actions"
    for g in games:
        d = base / ("heldout/2015-16" if g["heldout"] else "2015-16")
        assert (d / "action_candidates" / (g["sportvu_game"] + ".parquet")).exists()
        assert (d / "actions_detected" / (g["sportvu_game"] + ".parquet")).exists()
        assert (d / "actions_detected" / (g["sportvu_game"] + ".jsonl")).exists()
    lab = V.read_labels_jsonl(base / "2015-16" / "actions_detected" / (ctx["train_game_ids"][0] + ".jsonl"))
    assert (lab["actions"]["label_source"] == "model").all()


def test_cli_on_a_data_root(tmp_path):
    root = tmp_path / "data"
    man = fx._fallback_write_fixture_root(root, seed=1, n_games=3, heldout_every=3, n_possessions=10)
    assert len(man["games"]) == 3 and (root / "labels" / "actions").exists() and (root / "labels" / "screens").exists()
    assert (root / "sportvu_contract" / "heldout" / (HELD + ".parquet")).exists()
    v = V.main(["--data", str(root), "--out", str(tmp_path / "out")])
    assert v["summary"]["games"] == 3 and v["summary"]["ok"] == 3 and v["summary"]["heldout_games"] == 1
    assert json.loads((tmp_path / "out" / "analytics" / "actions" / "validation.json").read_text())["summary"]["ok"] == 3
    f = F.main(["--data", str(root), "--out", str(tmp_path / "out")])
    assert f["candidates"] > 0 and (tmp_path / "out" / "analytics" / "actions" / "2015-16" / "features").exists()
    ctx = T.main(["--data", str(root), "--report", str(tmp_path / "r.md"), "--folds", "2", "--bootstrap", "5"])
    assert ctx["train_games"] == 2 and ctx["heldout_games"] == 1 and (tmp_path / "r.md").exists()
    assert "fixture output" not in ctx["report_text"].split("## Inputs")[0]      # a --data run is not labelled a fixture run


def test_validate_flags_label_problems(games):
    g = [x for x in games if not x["heldout"]][0]
    a = g["labels"]["actions"].copy()
    s = g["labels"]["screens"].copy()
    rep = V.validate_labels(a, s, g["possessions"], g["rosters"])
    assert rep["problems"] == [] and rep["counts"]["actions"] == len(a) and rep["counts"]["overlapping_labels"] == 0
    bad = a.copy()
    bad.loc[0, "action_type"] = "pick_and_pop"
    assert any("action_type" in p for p in V.validate_labels(bad, s)["problems"])
    bad = a.copy()
    bad.loc[0, "end_frame"] = bad.loc[0, "start_frame"] - 1
    assert any("end_frame before start_frame" in p for p in V.validate_labels(bad, s)["problems"])
    dup = pd.concat([a, a.iloc[[0]]], ignore_index=True)
    rep = V.validate_labels(dup, s, g["possessions"])
    assert rep["counts"]["duplicate_labels"] == 1 and rep["counts"]["overlapping_labels"] == 1 and rep["problems"] == []
    bad_s = s.copy()
    bad_s.loc[0, "action_id"] = 10 ** 6
    assert any("no actions row" in p for p in V.validate_labels(a, bad_s)["problems"])
    bad_s = s.copy()
    bad_s.loc[0, "coverage"] = "zone"
    rep = V.validate_labels(a, bad_s)
    assert rep["problems"] == [] and any("coverage" in w for w in rep["warnings"])
    fr = g["frames"]
    dupf = pd.concat([fr, fr.iloc[[0]]], ignore_index=True)
    vf = V.validate_frames(dupf)
    assert vf["counts"]["duplicate_rows"] == 1 and len(vf["frames"]) == len(fr)


def test_eda_tables_and_shuffled_label_check(games):
    train = [dict(g) for g in games if not g["heldout"]]
    res = eda.run(train, seed=0, n_folds=4)
    counts = res["counts"].set_index("rule_class")
    assert counts.loc["all", "candidates"] > 0 and counts.loc["all", "possessions_with_0"] >= 0
    bal = res["balance"]["classes"]
    assert bal["label_class"].tolist() == list(CLASSES) and bal["share"].sum() == pytest.approx(1.0)
    dist = res["distributions"]
    assert set(dist["feature"]) == set(F.FEATURES) and set(dist["label_class"]) <= set(CLASSES)
    sh = res["shuffled"].set_index("labels")
    assert sh.loc["real", "f1_ball_screen"] > sh.loc["shuffled_0", "f1_ball_screen"] + 0.2
    md = eda.to_markdown(res, fixtures=True)
    assert "fixture output, not a result" in md and "—" not in md


def test_fallback_generator_end_to_end(fallback_games):
    """The fallback generator (used only when analytics.fixtures is not importable) also runs the whole protocol."""
    ctx = T.run([dict(g) for g in fallback_games[:3]], n_folds=3, seed=0, n_boot=5, fixtures=True, report=None, command="test")
    assert ctx["selected"] in T.MODELS and ctx["scores"]["hgb"]["f1_ball_screen"]["point"] >= 0.8
    assert "fixture" in fx.fixture_source()


def test_hand_labels_drop_tombstones_and_pass_two(tmp_path):
    """Hand labels come through analytics.labels.read_labels(labeller_pass=1): a tombstoned label and a pass-2
    relabel in the file never reach the trainer."""
    from analytics import labels as L
    root = tmp_path / "data"
    fx._fallback_write_fixture_root(root, seed=2, n_games=1, heldout_every=0, n_possessions=8)
    game = V.discover_games(root)[0][0]
    before = V.load_hand_labels(root, game)
    n0 = len(before["actions"])
    assert n0 > 0 and (before["actions"]["labeller_pass"] == 1).all() and (before["actions"]["label_source"] == "hand").all()
    gone = int(before["actions"]["action_id"].iloc[0])
    L.append_tombstone(root, game, "actions", gone)
    relabel = before["actions"].iloc[[1]].copy()
    relabel["action_id"] = 10 ** 5
    relabel["labeller_pass"] = 2
    L.write_labels(root, game, relabel, None)
    raw = [json.loads(l) for l in (root / "labels" / "actions" / (game + ".jsonl")).read_text().splitlines() if l.strip()]
    assert any(r.get("deleted") for r in raw) and any(r.get("labeller_pass") == 2 for r in raw)
    after = V.load_hand_labels(root, game)
    assert len(after["actions"]) == n0 - 1
    assert gone not in set(after["actions"]["action_id"]) and 10 ** 5 not in set(after["actions"]["action_id"])
    assert (after["actions"]["labeller_pass"] == 1).all()
    g = V.load_game(root, game, False)
    assert len(g["labels"]["actions"]) == n0 - 1 and V.validate_game(g)["ok"]


def test_owned_files_have_no_em_dash_and_notebook_is_json():
    for rel in OWNED:
        p = ROOT / rel
        if p.exists():
            assert "—" not in p.read_text(), rel
    nb = json.loads((ROOT / "notebooks" / "eda_actions.ipynb").read_text())
    assert nb["nbformat"] == 4 and any("eda" in "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
