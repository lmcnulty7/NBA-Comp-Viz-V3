"""analytics/coverage on the fixture season (analytics.fixtures when it imports, else the local scripted generator
of tests/_fx_coverage.py): end-to-end run, leakage guards, feature geometry on hand-built screens, validator
counts, the DATA_ROOT reader and the EDA tables. No network, no raw data; every number is fixture output."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from analytics import schema  # noqa: E402
from analytics.coverage import CLASSES, eda, features as F, train as T, validate as V  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EM_DASH = chr(0x2014)
FX = V._fixture_module()


@pytest.fixture(scope="module")
def bundle():
    """The generator every lane uses (shared when importable): 6 games, every third held out."""
    return V.fixture_bundle(seed=0, n_games=6, heldout_every=3, n_possessions=12)


@pytest.fixture(scope="module")
def scripted():
    """The local scripted generator (pinned, whatever the shared one does): 6 games, none held out."""
    return V.fixture_bundle(seed=1, n_games=6, heldout_every=0, local=True, n_possessions=12)


@pytest.fixture(scope="module")
def scripted_feats(scripted):
    cand, counts = V.validate_bundle(scripted)
    return F.features(cand, scripted["games"], counts)


# ── end to end ────────────────────────────────────────────────────────────────────────────────────
def test_fixture_end_to_end(bundle, tmp_path):
    report = tmp_path / "coverage.md"
    ctx = T.run(bundle, n_folds=3, seed=0, n_boot=10, out_root=tmp_path, report=report, command="test", fixtures=True)
    pred = ctx["predictions"]
    assert list(pred.columns) == T.PREDICTION_COLUMNS
    assert np.allclose(pred[T.PROB_COLUMNS].sum(axis=1), 1.0, atol=1e-5)
    assert set(pred["predicted"].unique()) <= set(CLASSES)
    train_rows = pred[~pred["heldout"] & pred["coverage_label"].notna()]
    assert len(train_rows) == ctx["train_rows"] and (train_rows["fold"] >= 0).all()
    held = pred[pred["heldout"]]
    assert len(held) == ctx["heldout_rows"] > 0 and (held["fold"] == -1).all()
    assert set(held["sportvu_game"]).isdisjoint(set(train_rows["sportvu_game"]))
    assert ctx["heldout_games"] == 2 and ctx["train_games"] == 4
    s = ctx["scores"]
    assert s[ctx["best"]]["macro_f1"]["point"] >= s["majority"]["macro_f1"]["point"]
    assert s[ctx["best"]]["confusion"].to_numpy().sum() == ctx["train_rows"]
    rates = ctx["rates"]
    assert list(rates.columns) == T.TEAM_RATE_COLUMNS and len(rates) >= 1 and rates["insufficient"].all()
    assert np.allclose(rates[["share_" + c for c in CLASSES]].sum(axis=1), 1.0)
    text = report.read_text()
    assert "fixture output, not a result" in text and "refuse_heldout" in text and EM_DASH not in text
    written = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.parquet"))
    assert any(p.startswith("analytics/coverage/heldout/") and p.endswith("coverage_predictions.parquet") for p in written)
    assert any(p.endswith("coverage_team_rates.parquet") and "/heldout/" not in p for p in written)
    back = pd.read_parquet([p for p in tmp_path.rglob("coverage_predictions.parquet") if "heldout" not in p.parts][0])
    assert back["coverage_label"].notna().sum() == ctx["train_rows"]


def test_scripted_coverages_beat_majority(scripted_feats):
    feats = scripted_feats
    train, _so, held = T.training_frame(feats)
    assert len(held) == 0 and train["coverage_label"].nunique() == 7
    _fold, probs, note = T.out_of_fold(train, 3, 0)
    y = train["coverage_label"].to_numpy()
    maj = T.macro_f1(y, T.predicted(probs["majority"]))
    best = max(T.macro_f1(y, T.predicted(probs[m])) for m in ("logistic", "hgb"))
    assert note["mode"] == "out-of-fold" and best > maj + 0.4, (maj, best)


# ── leakage guards ────────────────────────────────────────────────────────────────────────────────
def test_label_and_post_outcome_columns_are_refused(scripted_feats):
    feats = scripted_feats
    schema.assert_time_forward(feats, "states_1hz")
    F.assert_no_leak(F.design(feats))
    for bad in ("coverage", "coverage_label", "confidence", "label_source", "end_frame", "points", "fold", "p_switch"):
        with pytest.raises(ValueError):
            F.assert_no_leak(feats[F.feature_columns()].assign(**{bad: 0}))
    with pytest.raises(ValueError):
        schema.assert_time_forward(feats.assign(end_frame=0), "states_1hz")
    train, _so, _h = T.training_frame(feats)
    with pytest.raises(ValueError):
        T.fit_models(train.assign(heldout=True), 0)
    with pytest.raises(ValueError):
        T.fit_models(train.assign(coverage_label=None), 0)
    with pytest.raises(SystemExit):
        T.training_frame(feats.assign(sportvu_game="12.16.2015.PHX.at.GSW", heldout=False))


def test_frames_beyond_the_declared_window_never_enter(scripted):
    gid, game = next(iter(scripted["games"].items()))
    cand, counts = V.validate_bundle(scripted)
    cand = cand[cand["sportvu_game"] == gid].head(1)
    base = F.features(cand, {gid: game}, {})
    f1 = int(cand["end_frame"].iloc[0]) + F.POST_FRAMES
    frames = game["frames"].copy()
    beyond = frames["frame"].to_numpy() > f1          # every later frame of the game
    edge = frames["frame"].to_numpy() == f1           # the last frame of the declared window
    frames.loc[beyond, ["x", "y"]] = frames.loc[beyond, ["x", "y"]].to_numpy() + 30.0
    same = F.features(cand, {gid: {**game, "frames": frames}}, {})
    pd.testing.assert_frame_equal(base, same)
    frames.loc[edge, ["x", "y"]] = frames.loc[edge, ["x", "y"]].to_numpy() + 30.0
    moved = F.features(cand, {gid: {**game, "frames": frames}}, {})
    assert not np.allclose(base[F.feature_columns()].to_numpy(), moved[F.feature_columns()].to_numpy(), equal_nan=True)


# ── feature geometry on hand-built screens ────────────────────────────────────────────────────────
def _hand_built(d1_pass_depth: float, switch: bool, n: int = 60):
    """Basket (5.25, 25); screen point P = (25, 30); u the unit vector from P to the basket, v perpendicular and
    pointing to the near sideline (y up). H starts at depth -4, lat 3 and drives to depth 6, lat -5; S stays; D1
    starts at depth -2, lat 2 and passes the screener at depth d1_pass_depth (negative: over the top; positive:
    under), then either stays with S (switch) or trails H; D2 starts at depth 4 and either picks up H (switch)
    or holds its spot."""
    P = np.array([25.0, 30.0])
    u = np.array([5.25, 25.0]) - P
    u = u / np.linalg.norm(u)
    v = np.array([-u[1], u[0]])
    v = v if v[1] > 0 else -v
    at = lambda d, l: P + d * u + l * v  # noqa: E731
    t = np.linspace(0, 1, n)[:, None]
    H = at(-4, 3) + t * (at(6, -5) - at(-4, 3))
    S = np.tile(P, (n, 1))
    half = n // 2
    D1a = at(-2, 2) + np.linspace(0, 1, half)[:, None] * (at(d1_pass_depth, 0) - at(-2, 2))
    D1_end = at(1.5, 0) if switch else H[-1] + 2.0 * (-u)
    D1b = at(d1_pass_depth, 0) + np.linspace(0, 1, n - half)[:, None] * (D1_end - at(d1_pass_depth, 0))
    D1 = np.vstack([D1a, D1b])
    D2 = at(4, 0) + t * ((H[-1] + 2.0 * u) - at(4, 0)) if switch else np.tile(at(4, 0), (n, 1))
    return H, S, D1, D2


def test_feature_geometry_on_a_hand_built_switch():
    H, S, D1, D2 = _hand_built(d1_pass_depth=-1.5, switch=True)
    g = F.screen_geometry(H, S, D1, D2, end_offset=15)
    assert g["screen_basket_dist_ft"] == pytest.approx(np.hypot(25 - 5.25, 30 - 25), abs=1e-6)
    assert g["handler_depth_t0_ft"] == pytest.approx(-4) and g["handler_lat_t0_ft"] == pytest.approx(3)
    assert g["d1_depth_t0_ft"] == pytest.approx(-2) and g["d1_lat_t0_ft"] == pytest.approx(2)
    assert g["d2_depth_t0_ft"] == pytest.approx(4) and g["d2_lat_t0_ft"] == pytest.approx(0)
    assert g["d2_basket_dist_t0_ft"] == pytest.approx(g["screen_basket_dist_ft"] - 4)
    assert g["d2_nearest_end"] == 1.0 and g["d2_handler_dist_end_ft"] == pytest.approx(2.0, abs=1e-6)
    assert g["def_dist_diff_end_ft"] > 3 and g["d1_handler_dist_end_ft"] == pytest.approx(np.hypot(4.5, 5), abs=1e-6)
    assert g["handler_depth_end_ft"] == pytest.approx(6) and g["handler_lat_end_ft"] == pytest.approx(-5)
    assert g["d2_depth_end_ft"] == pytest.approx(8)       # D2 ends 2 ft inside H, at depth 8, lat -5
    assert g["d2_basket_dist_min_ft"] == pytest.approx(np.hypot(g["screen_basket_dist_ft"] - 8, 5), abs=1e-6)
    assert g["d1_depth_rel_screener_pass_ft"] < 0            # D1 passed over the top
    assert 0 < g["both_within_6ft_frac"] < 1 and g["both_within_6ft_any"] == 1.0
    speed = np.hypot(10, 8) / ((len(H) - 1) / F.HZ)
    assert g["handler_speed_mean_ftps"] == pytest.approx(speed, rel=1e-6) and g["handler_speed_max_ftps"] == pytest.approx(speed, rel=1e-6)
    assert g["n_window_frames"] == len(H)
    assert set(g) - {"_end_offset"} == set(F.feature_columns())


def test_pass_side_and_trailing_defender_on_hand_built_over_and_under():
    over = F.screen_geometry(*_hand_built(d1_pass_depth=-1.5, switch=False), end_offset=15)
    under = F.screen_geometry(*_hand_built(d1_pass_depth=2.5, switch=False), end_offset=15)
    assert over["d1_depth_rel_screener_pass_ft"] < 0 < under["d1_depth_rel_screener_pass_ft"]
    for g in (over, under):
        assert g["d2_nearest_end"] == 0.0 and g["d1_handler_dist_end_ft"] == pytest.approx(2.0, abs=1e-6)
        assert g["d2_screen_dist_max_ft"] == pytest.approx(4.0, abs=1e-6)
    assert F.screen_geometry(np.full((1, 2), np.nan), *_hand_built(-1.5, True)[1:], end_offset=1) is None


def test_defenders_are_inferred_for_detected_screens(scripted):
    gid, game = next(iter(scripted["games"].items()))
    cand, _ = V.validate_bundle(scripted)
    row = cand[cand["sportvu_game"] == gid].iloc[0]
    tracks = F.GameTracks(game["frames"])
    d1, d2 = F.infer_defenders(tracks, int(row.ball_handler_id), int(row.screener_id), int(row.start_frame))
    assert {d1, d2} <= {p for p, t in tracks.team.items() if t != tracks.team[int(row.ball_handler_id)]}
    detected = cand[cand["sportvu_game"] == gid].head(2).assign(on_ball_defender_id=pd.NA, screener_defender_id=pd.NA,
                                                                 coverage=pd.NA, label_source="model")
    feats = F.features(detected, {gid: game}, {})
    assert len(feats) == 2 and feats["coverage_label"].isna().all() and feats[F.feature_columns()].notna().all().all()


# ── validator ─────────────────────────────────────────────────────────────────────────────────────
def test_validator_drops_and_counts_bad_rows(scripted):
    gid = next(iter(scripted["games"]))
    actions = scripted["actions"]
    screens = scripted["screens"]
    good = screens[screens["sportvu_game"] == gid].iloc[0]
    non_screen = actions[(actions["sportvu_game"] == gid) & (actions["action_type"] != "ball_screen")]
    extra = [dict(good, coverage="zone"), dict(good, confidence="maybe"), dict(good),
             dict(good, action_id=999), dict(good, possession=int(good["possession"]) + 1, action_id=int(good["action_id"])),
             dict(good, on_ball_defender_id=int(good["screener_defender_id"]))]
    if len(non_screen):
        extra.append(dict(good, action_id=int(non_screen.iloc[0]["action_id"])))
    extra = pd.DataFrame(extra)
    extra.loc[len(extra) - 1, "coverage"] = None
    counts = {}
    a = V.check_actions(actions, counts)
    s = V.check_screens(pd.concat([screens, extra], ignore_index=True), a, counts)
    assert counts["screens_unknown_coverage"] == 1 and counts["screens_unknown_confidence"] == 1
    assert counts["screens_missing_value"] == 1 and counts["screens_duplicate_id"] >= 1
    assert counts["screens_without_ball_screen_action"] >= 1 and counts["screens_bad_defender_ids"] == 0
    assert counts["screens_kept"] == len(screens) and (s["possession"] == s["possession"].astype(int)).all()
    with pytest.raises(ValueError):
        V.check_screens(screens.drop(columns=["coverage"]), a, {})
    assert V.season_of("12.16.2015.PHX.at.GSW") == "2015-16" and V.season_of("02.07.2016.A.at.B") == "2015-16"
    assert V.season_of("nonsense") == "unknown"
    assert V.heldout_mask(["12.16.2015.PHX.at.GSW", "01.01.2016.AAA.at.CLE", "01.01.2016.AAA.at.BBB"]).tolist() == [True, True, False]


def test_data_root_layout_is_read_by_the_cli(tmp_path):
    import inspect
    root = tmp_path / "data_root"
    small = {"n_possessions": 8} if "n_possessions" in inspect.signature(FX.write_fixture_root).parameters else {}
    FX.write_fixture_root(root, 0, 2, 2, **small)
    assert sorted((root / "labels" / "screens").glob("*.jsonl"))
    counts = V.main(["--data", str(root)])
    assert counts["games"] == 2 and counts["games_heldout"] == 1 and counts["screens_kept"] > 0
    feats = F.main(["--data", str(root), "--out", str(tmp_path / "out")])
    assert 0 < len(feats) <= counts["candidates"] and feats["heldout"].sum() > 0
    assert list((tmp_path / "out").rglob("coverage_features.parquet"))
    # the detector's table at OUT_ROOT/analytics/actions/<season>/actions_detected/<game>.parquet (lane ACTIONS layout)
    screens = pd.concat([V._read_jsonl(p, "screens") for p in sorted((root / "labels" / "screens").glob("*.jsonl"))])
    actions = pd.concat([V._read_jsonl(p, "actions") for p in sorted((root / "labels" / "actions").glob("*.jsonl"))])
    train_game = [g for g in sorted(screens["sportvu_game"].unique()) if not V.heldout_mask([g]).iloc[0]][0]
    src = actions[(actions["sportvu_game"] == train_game) & (actions["action_type"] == "ball_screen")].head(2)
    detected = src.drop(columns=[c for c in ("action_id", "labelled_at") if c in src.columns]).copy()
    detected["start_frame"] = detected["start_frame"] + 60       # a minute later than the hand label: not a duplicate
    detected["end_frame"] = detected["end_frame"] + 60
    detected["label_source"] = "model"
    out = tmp_path / "out"
    ddir = out / "analytics" / "actions" / V.season_of(train_game) / "actions_detected"
    ddir.mkdir(parents=True)
    detected.to_parquet(ddir / (train_game + ".parquet"), index=False)
    assert len(V.load_detected(out)) == 2
    bundle = V.load_data_root(root, out)
    cand, counts2 = V.validate_bundle(bundle)
    assert counts2["detected_screens"] == 2 and counts2["candidates"] == counts["candidates"] + 2
    model_rows = cand[cand["label_source"] == "model"]
    assert len(model_rows) == 2 and model_rows["coverage"].isna().all() and model_rows["on_ball_defender_id"].isna().all()


def test_label_files_are_resolved_like_the_label_tool_writes_them(tmp_path):
    """Append-only JSONL: a tombstone deletes, the last line about an id wins, pass-2 relabels never reach training,
    and label_source "hand" is a known source. Written in the label tool's line shape (analytics/labels.py)."""
    import json
    game = "01.09.2016.AAA.at.BBB"
    stamp = "2026-01-01T00:00:00Z"
    acts = [dict(sportvu_game=game, possession=0, action_id=1, action_type="ball_screen", start_frame=100, end_frame=140,
                 ball_handler_id=101, screener_id=102, label_source="hand", labeller_pass=1, labelled_at=stamp),
            dict(sportvu_game=game, possession=1, action_id=2, action_type="ball_screen", start_frame=300, end_frame=340,
                 ball_handler_id=101, screener_id=103, label_source="hand", labeller_pass=1, labelled_at=stamp),
            dict(sportvu_game=game, possession=1, action_id=2, action_type="ball_screen", start_frame=305, end_frame=345,
                 ball_handler_id=101, screener_id=103, label_source="hand", labeller_pass=1, labelled_at=stamp),
            dict(sportvu_game=game, possession=2, action_id=3, action_type="isolation", start_frame=500, end_frame=540,
                 ball_handler_id=101, screener_id=None, label_source="hand", labeller_pass=1, labelled_at=stamp),
            dict(action_id=3, deleted=True, labelled_at=stamp),
            dict(sportvu_game=game, possession=0, action_id=4, action_type="ball_screen", start_frame=100, end_frame=140,
                 ball_handler_id=101, screener_id=102, label_source="hand", labeller_pass=2, labelled_at=stamp)]
    scr = [dict(sportvu_game=game, possession=0, screen_id=1, action_id=1, coverage="drop", on_ball_defender_id=201,
                screener_defender_id=202, label_source="hand", confidence="sure", labeller_pass=1, labelled_at=stamp),
           dict(sportvu_game=game, possession=1, screen_id=2, action_id=2, coverage="ice", on_ball_defender_id=201,
                screener_defender_id=203, label_source="hand", confidence="unsure", labeller_pass=1, labelled_at=stamp),
           dict(sportvu_game=game, possession=1, screen_id=2, action_id=2, coverage="over", on_ball_defender_id=201,
                screener_defender_id=203, label_source="hand", confidence="sure", labeller_pass=1, labelled_at=stamp),
           dict(sportvu_game=game, possession=0, screen_id=3, action_id=1, coverage="switch", on_ball_defender_id=201,
                screener_defender_id=202, label_source="hand", confidence="sure", labeller_pass=2, labelled_at=stamp),
           dict(screen_id=1, deleted=True, labelled_at=stamp)]
    for kind, rows in (("actions", acts), ("screens", scr)):
        d = tmp_path / "labels" / kind
        d.mkdir(parents=True)
        (d / (game + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = {}
    labels = V.load_labels(tmp_path, counts)
    a, s = labels["actions"], labels["screens"]
    assert counts["label_reader"] in ("analytics.labels", "raw")
    assert counts["actions_other_pass_dropped"] == 1 and counts["screens_other_pass_dropped"] == 1
    assert sorted(a["action_id"].astype(int)) == [1, 2]                        # 3 tombstoned, 4 is pass 2
    assert int(a.loc[a["action_id"] == 2, "start_frame"].iloc[0]) == 305        # last line wins
    assert sorted(s["screen_id"].astype(int)) == [2] and s["coverage"].iloc[0] == "over"
    checked = {}
    acts_ok = V.check_actions(a, checked)
    scr_ok = V.check_screens(s, acts_ok, checked)
    assert checked["actions_unknown_label_source"] == 0 and checked["screens_unknown_label_source"] == 0
    assert checked["screens_kept"] == 1 and checked["label_over"] == 1
    assert "hand" in V.LABEL_SOURCES and "human" not in V.LABEL_SOURCES


# ── EDA ───────────────────────────────────────────────────────────────────────────────────────────
def test_eda_tables_and_shuffled_label_check(scripted_feats):
    feats = scripted_feats
    bal = eda.class_balance(feats)
    assert list(bal["class"]) == list(CLASSES) and bal["n"].sum() == feats["coverage_label"].notna().sum()
    dist = eda.feature_distributions_by_class(feats)
    assert set(dist["feature"]) == set(F.feature_columns()) and dist["n"].min() >= 1
    corr = eda.correlation_table(feats)
    assert corr.shape == (len(F.feature_columns()),) * 2 and np.allclose(np.diag(corr.fillna(1.0)), 1.0)
    sep = eda.class_separation(feats)
    assert sep.iloc[0]["F"] > 1
    chk = eda.shuffled_label_check(feats, seed=0, n_folds=2)
    true_lr = chk[(chk["labels"] == "true labels") & (chk["model"] == "logistic")]["macro_f1"].iloc[0]
    perm_lr = chk[(chk["labels"] == "permuted labels") & (chk["model"] == "logistic")]["macro_f1"].iloc[0]
    assert true_lr > perm_lr + 0.3 and perm_lr < 0.4


def test_owned_files_carry_no_em_dash_and_notebook_is_valid_json():
    import json
    owned = [ROOT / "analytics" / "coverage" / n for n in ("__init__.py", "validate.py", "features.py", "train.py", "eda.py")]
    owned += [ROOT / "tests" / "test_coverage.py", ROOT / "tests" / "_fx_coverage.py", ROOT / "docs" / "contract" / "coverage.md",
              ROOT / "reports" / "coverage.md", ROOT / "notebooks" / "eda_coverage.ipynb"]
    for p in owned:
        assert EM_DASH not in p.read_text(), p
    nb = json.loads((ROOT / "notebooks" / "eda_coverage.ipynb").read_text())
    assert nb["nbformat"] == 4 and any("eda" in "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    report = (ROOT / "reports" / "coverage.md").read_text()
    assert "fixture output, not a result" in report and "No hand labels exist yet" in report
