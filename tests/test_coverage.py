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
FX = V._fixture_module()


@pytest.fixture(scope="module")
def bundle():
    """The generator every lane uses (shared when importable): 6 games, every third held out."""
    return V.fixture_bundle(seed=0, n_games=6, heldout_every=3, n_possessions=16)


@pytest.fixture(scope="module")
def scripted():
    """The local scripted generator (pinned, whatever the shared one does): 6 games, none held out."""
    return V.fixture_bundle(seed=1, n_games=6, heldout_every=0, local=True, n_possessions=12)


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
    assert "fixture output, not a result" in text and "refuse_heldout" in text and "—" not in text
    written = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.parquet"))
    assert any(p.startswith("analytics/coverage/heldout/") and p.endswith("coverage_predictions.parquet") for p in written)
    assert any(p.endswith("coverage_team_rates.parquet") and "/heldout/" not in p for p in written)
    back = pd.read_parquet([p for p in tmp_path.rglob("coverage_predictions.parquet") if "heldout" not in p.parts][0])
    assert back["coverage_label"].notna().sum() == ctx["train_rows"]


def test_scripted_coverages_beat_majority(scripted):
    cand, counts = V.validate_bundle(scripted)
    feats = F.features(cand, scripted["games"], counts)
    train, _so, held = T.training_frame(feats)
    assert len(held) == 0 and train["coverage_label"].nunique() == 7
    _fold, probs, note = T.out_of_fold(train, 3, 0)
    y = train["coverage_label"].to_numpy()
    maj = T.macro_f1(y, T.predicted(probs["majority"]))
    best = max(T.macro_f1(y, T.predicted(probs[m])) for m in ("logistic", "hgb"))
    assert note["mode"] == "out-of-fold" and best > maj + 0.4, (maj, best)


# ── leakage guards ────────────────────────────────────────────────────────────────────────────────
def test_label_and_post_outcome_columns_are_refused(scripted):
    cand, counts = V.validate_bundle(scripted)
    feats = F.features(cand, scripted["games"], counts)
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
    cand = cand[cand["sportvu_game"] == gid].head(3)
    base = F.features(cand, {gid: game}, {})
    last_in = cand["end_frame"].to_numpy() + F.POST_FRAMES
    frames = game["frames"].copy()
    beyond = np.zeros(len(frames), dtype=bool)
    edge = np.zeros(len(frames), dtype=bool)
    for f1 in last_in:
        beyond |= frames["frame"].to_numpy() > f1
        edge |= frames["frame"].to_numpy() == f1
    frames.loc[beyond, ["x", "y"]] = frames.loc[beyond, ["x", "y"]].to_numpy() + 30.0
    same = F.features(cand, {gid: {**game, "frames": frames}}, {})
    pd.testing.assert_frame_equal(base, same)
    frames.loc[edge, ["x", "y"]] = frames.loc[edge, ["x", "y"]].to_numpy() + 30.0
    moved = F.features(cand, {gid: {**game, "frames": frames}}, {})
    assert not np.allclose(base[F.feature_columns()].to_numpy(), moved[F.feature_columns()].to_numpy(), equal_nan=True)


# ── feature geometry on hand-built screens ────────────────────────────────────────────────────────
def _hand_built(d1_pass_depth: float, switch: bool, n: int = 60):
    """Basket (5.25, 25); screen point P = (25, 30); u = (-1, 0) toward the basket, v = (0, 1) toward the near
    sideline. H starts at depth -4, lat 3 and drives to depth 6, lat -5; S stays; D1 starts at depth -2, lat 2
    and passes the screener at depth d1_pass_depth (negative: over the top; positive: under), then either stays
    with S (switch) or trails H; D2 starts at depth 4 and either picks up H (switch) or holds its spot."""
    P, u, v = np.array([25.0, 30.0]), np.array([-1.0, 0.0]), np.array([0.0, 1.0])
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
    assert g["d2_depth_end_ft"] == pytest.approx(8) and g["d2_basket_dist_min_ft"] == pytest.approx(g["screen_basket_dist_ft"] - 8)
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
    root = tmp_path / "data_root"
    FX.write_fixture_root(root, 0, 3, 3)
    assert sorted((root / "labels" / "screens").glob("*.jsonl"))
    counts = V.main(["--data", str(root)])
    assert counts["games"] == 3 and counts["games_heldout"] == 1 and counts["screens_kept"] > 0
    feats = F.main(["--data", str(root), "--out", str(tmp_path / "out")])
    assert len(feats) == counts["features_rows"] > 0 and feats["heldout"].sum() > 0
    assert list((tmp_path / "out").rglob("coverage_features.parquet"))


# ── EDA ───────────────────────────────────────────────────────────────────────────────────────────
def test_eda_tables_and_shuffled_label_check(scripted):
    cand, counts = V.validate_bundle(scripted)
    feats = F.features(cand, scripted["games"], counts)
    bal = eda.class_balance(feats)
    assert list(bal["class"]) == list(CLASSES) and bal["n"].sum() == feats["coverage_label"].notna().sum()
    dist = eda.feature_distributions_by_class(feats)
    assert set(dist["feature"]) == set(F.feature_columns()) and dist["n"].min() >= 1
    corr = eda.correlation_table(feats)
    assert corr.shape == (len(F.feature_columns()),) * 2 and np.allclose(np.diag(corr.fillna(1.0)), 1.0)
    sep = eda.class_separation(feats)
    assert sep.iloc[0]["F"] > 1
    chk = eda.shuffled_label_check(feats, seed=0, n_folds=3)
    true_lr = chk[(chk["labels"] == "true labels") & (chk["model"] == "logistic")]["macro_f1"].iloc[0]
    perm_lr = chk[(chk["labels"] == "permuted labels") & (chk["model"] == "logistic")]["macro_f1"].iloc[0]
    assert true_lr > perm_lr + 0.3 and perm_lr < 0.4


def test_owned_files_carry_no_em_dash_and_notebook_is_valid_json():
    import json
    owned = [ROOT / "analytics" / "coverage" / n for n in ("__init__.py", "validate.py", "features.py", "train.py", "eda.py")]
    owned += [ROOT / "tests" / "test_coverage.py", ROOT / "tests" / "_fx_coverage.py", ROOT / "docs" / "contract" / "coverage.md",
              ROOT / "reports" / "coverage.md", ROOT / "notebooks" / "eda_coverage.ipynb"]
    for p in owned:
        assert "—" not in p.read_text(), p
    nb = json.loads((ROOT / "notebooks" / "eda_coverage.ipynb").read_text())
    assert nb["nbformat"] == 4 and any("eda" in "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    report = (ROOT / "reports" / "coverage.md").read_text()
    assert "fixture output, not a result" in report and "No hand labels exist yet" in report
