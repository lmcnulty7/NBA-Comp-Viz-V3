"""analytics.playcalling on the shared fixtures (analytics.fixtures, through tests/_fx_playcalling.py; the local
generator there stands in only when the import fails, and is checked once on its own): the end-to-end fixture run,
the leakage guards, the gate on a hand case, validation counts, the data-root CLI and the EDA tables. No network;
every number is fixture output, not a result."""
import json
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from analytics import schema  # noqa: E402
from analytics.playcalling import N_MIN, eda, features as F, train as T, validate as V  # noqa: E402
from tests import _fx_playcalling as fx  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HELD = "12.16.2015.PHX.at.GSW"


@pytest.fixture(scope="module")
def inputs():
    inp, _ = V.validate_inputs(fx.season_inputs(0, n_games=4, n_possessions=10, heldout_every=4))
    return inp


@pytest.fixture(scope="module")
def table(inputs):
    t, log = F.build_action_table(inputs["frames"], inputs["possessions"], inputs["actions"], inputs["screens"],
                                  inputs["coverage_predictions"], inputs["epv_states"], inputs["heldout_games"])
    assert log["actions_without_possession"] == 0 and log["null_lineups"] == 0
    return t


# ── end to end ─────────────────────────────────────────────────────────────────────────────────────
def test_fixture_run_end_to_end_writes_report_and_tables(tmp_path):
    report = tmp_path / "playcalling.md"
    ctx = T.main(["--fixtures", "--report", str(report), "--out", str(tmp_path / "out"),
                  "--bootstrap", "10", "--fixture-games", "4", "--fixture-possessions", "8"])
    text = report.read_text()
    first = text.splitlines()[0]
    assert "Association only, not causal" in first and "causal" in first
    assert "fixture output, not a result" in text and chr(0x2014) not in text
    assert ctx["heldout_rows"] > 0 and ctx["train_rows"] > 0
    assert set(ctx["heldout_scores"]) == {"mean", "by_action", "ridge"}
    out = tmp_path / "out" / "analytics" / "playcalling" / "2015-16"
    ao = pd.read_parquet(out / "action_outcomes.parquet")
    assert list(ao.columns) == list(F.ACTION_OUTCOMES_COLUMNS) and not ao["heldout"].any()
    held = pd.read_parquet(tmp_path / "out" / "analytics" / "playcalling" / "heldout" / "2015-16" / "action_outcomes.parquet")
    assert held["heldout"].all() and len(held) == ctx["heldout_rows"]
    ppp = pd.read_parquet(out / "ppp_by_action.parquet")
    assert list(ppp.columns) == ["action_type", "coverage", "n", "ppp", "lo", "hi", "insufficient"]
    assert ppp["coverage"].isna().any() and ppp["coverage"].notna().any()
    assert ppp["insufficient"].all()            # 32 fixture possessions: every group is under the 300 gate
    adj = pd.read_parquet(out / "playcalling_adjusted.parquet")
    assert list(adj.columns) == ["coverage", "action_type", "adjusted_ppp", "lo", "hi", "n", "insufficient", "recommended"]
    assert (adj["n"] > 0).all() and not adj["recommended"].any()
    assert "No coverage has a cell with at least 300 rows" in text
    assert "## 2b. EPV gained" in text and ctx["epv_rows"] == ctx["train_rows"]


def test_trainer_recovers_the_planted_association_of_the_local_generator():
    """The local generator plants PPP on action type and coverage (the shared physics fixture does not), so on it the
    by-action table must order isolation under cut and the nested out-of-fold ridge must stay close to the mean."""
    inp, _ = V.validate_inputs(fx.season_inputs(0, n_games=8, n_possessions=40, heldout_every=4, shared=False))
    table, _ = F.build_action_table(inp["frames"], inp["possessions"], inp["actions"], inp["screens"],
                                    inp["coverage_predictions"], inp["epv_states"], inp["heldout_games"])
    ctx = T.run(table, n_folds=3, seed=0, n_boot=5, n_min=N_MIN)
    t1 = ctx["ppp_by_action"].set_index("action_type")["ppp"]
    assert t1["ball_screen"] > t1["isolation"]          # planted: 1.06 against 0.84 expected points
    assert set(ctx["adjusted"]["coverage"]) <= set(table["coverage"])
    s = ctx["scores"]["mse"]
    assert s["ridge"]["point"] < s["mean"]["point"] + 0.05
    assert 0.3 <= ctx["alpha"] <= 100


# ── leakage guards ─────────────────────────────────────────────────────────────────────────────────
def test_feature_frame_carries_no_outcome_and_the_guard_trips(table):
    f = F.features(table)
    assert list(f.columns) == ["sportvu_game", "possession", "action_id"] + F.feature_columns() + ["heldout"]
    assert not ({"ppp", "epv_gain"} & set(f.columns))
    for bad in ("points", "end_kind", "end_frame", "points_to_end"):
        with pytest.raises(ValueError):
            schema.assert_time_forward(f.assign(**{bad: 0}), "possessions")
    # a FEATURES tuple that named an outcome would be refused the moment features() runs
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(F, "FEATURES", F.FEATURES + ("ppp",))
        with pytest.raises(ValueError):
            F.features(table)
        mp.setattr(F, "FEATURES", F.FEATURES[:-1] + ("end_frame",))
        with pytest.raises(ValueError):
            F.features(table)


def test_held_out_rows_never_reach_a_fit(table):
    train = table[~table["heldout"]].reset_index(drop=True)
    leaked = train.copy()
    leaked.loc[0, "heldout"] = True
    with pytest.raises(ValueError):
        T.fit_ridge(leaked, 1.0)
    with pytest.raises(ValueError):
        T.fit_baselines(leaked)
    # a held-out game id with the flag off trips refuse_heldout before any fit
    bad = train.copy()
    bad.loc[bad["sportvu_game"] == bad["sportvu_game"].iloc[0], "sportvu_game"] = HELD
    with pytest.raises(SystemExit):
        T.training_split(bad)
    # a bare analytics/playcalling/train.py keeps the two tripwire names the repo-wide guard test looks for
    text = (ROOT / "analytics" / "playcalling" / "train.py").read_text()
    assert "refuse_heldout" in text and "assert_time_forward" in (ROOT / "analytics" / "playcalling" / "features.py").read_text()


def test_score_margin_uses_only_earlier_possessions_and_lineups_come_from_the_first_frame():
    poss = pd.DataFrame({"sportvu_game": ["g"] * 4, "possession": [0, 1, 2, 3], "period": [1] * 4, "start_frame": [0, 10, 20, 30],
                         "end_frame": [9, 19, 29, 39], "start_clock_s": [700.0, 650, 600, 550], "end_clock_s": [690.0, 640, 590, 540],
                         "start_event": [0, 2, 4, 6], "end_event": [1, 3, 5, 7], "offense_team_id": [1, 2, 1, 2],
                         "end_kind": ["shot_made", "shot_missed", "shot_made", "shot_made"], "points": [3, 0, 2, 2], "gap": [False] * 4, "n_frames": [10] * 4})
    assert F.score_margin_before(poss).tolist() == [0.0, -3.0, 3.0, -5.0]
    rows = []
    for f in (0, 10, 20, 30):
        for pid in range(10):
            rows.append({"game": "g", "frame": f, "team_id": 1 if pid < 5 else 2, "player_id": 100 + pid + (f if f == 20 else 0),
                         "shot_clock_s": 20.0 - f / 10})
    rows = [r for r in rows if not (r["frame"] == 30 and r["player_id"] == 109)]        # possession 3: nine players
    frames = pd.DataFrame(rows)
    lus, log = F.lineups_at_first_frame(frames, poss)
    assert lus["off_lineup"].tolist()[0] == "100-101-102-103-104" and lus["def_lineup"].tolist()[0] == "105-106-107-108-109"
    assert lus["off_lineup"].tolist()[1] == "105-106-107-108-109"          # team 2 on offence: sides swap
    assert lus["off_lineup"].tolist()[2] == "120-121-122-123-124"          # read from frame 20, not frame 0
    assert lus["off_lineup"].isna().tolist()[3] and log["not_five_v_five"] == 1     # team 2 (offence) is one short


# ── the gate on a hand case ────────────────────────────────────────────────────────────────────────
def hand_table(n_a=320, n_b=40, n_games=4, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_a + n_b):
        at = "ball_screen" if i < n_a else "post_up"
        rows.append({"sportvu_game": "01.%02d.2016.AAA.at.BBB" % (i % n_games + 1), "possession": i, "action_id": 0, "action_type": at,
                     "coverage": "drop" if i % 2 == 0 else "switch", "off_lineup": "1-2-3-4-%d" % (5 + i % 2), "def_lineup": "6-7-8-9-10",
                     "ppp": float(rng.choice([0, 2, 3], p=[0.5, 0.4, 0.1] if at == "ball_screen" else [0.7, 0.25, 0.05])),
                     "epv_gain": np.nan, "heldout": False, "score_margin": 0.0, "shot_clock_s": 15.0, "period": 1, "label_source": "hand",
                     "coverage_source": "hand", "start_frame": i, "end_frame": i})
    return pd.DataFrame(rows, columns=list(F.ACTION_OUTCOMES_COLUMNS))


def test_gate_marks_small_groups_insufficient_and_the_recommendation_respects_it():
    t = hand_table()
    one = T.ppp_by_action(t, by_coverage=False, n_boot=30, seed=0)
    by = one.set_index("action_type")
    assert not by.loc["ball_screen", "insufficient"] and by.loc["post_up", "insufficient"]
    assert by.loc["ball_screen", "n"] == 320 and by.loc["post_up", "n"] == 40
    assert by.loc["ball_screen", "lo"] <= by.loc["ball_screen", "ppp"] <= by.loc["ball_screen", "hi"]
    two = T.ppp_by_action(t, by_coverage=True, n_boot=30, seed=0)
    assert two["insufficient"].all()                   # 160 per cell at most
    assert T.ppp_by_action(t, by_coverage=False, n_boot=30, seed=0, n_min=30)["insufficient"].tolist() == [False, False]
    adj = T.adjusted_table(t, alpha=1.0, n_boot=20, seed=0, n_min=100)
    rec = adj[adj["recommended"]]
    assert set(rec["coverage"]) == {"drop", "switch"} and (rec["action_type"] == "ball_screen").all()
    assert adj[adj["action_type"] == "post_up"]["insufficient"].all()
    assert (adj["lo"] <= adj["adjusted_ppp"]).all() and (adj["adjusted_ppp"] <= adj["hi"]).all()
    assert not T.adjusted_table(t, alpha=1.0, n_boot=5, seed=0, n_min=N_MIN)["recommended"].any()
    assert T.ppp_by_action(t[t["sportvu_game"] == t["sportvu_game"].iloc[0]], n_boot=5)["lo"].isna().all()   # one game: no interval


# ── validation ─────────────────────────────────────────────────────────────────────────────────────
def test_validate_counts_duplicates_unknown_enums_and_missing_columns(inputs):
    a = inputs["actions"]
    dup = pd.concat([a, a.head(3)], ignore_index=True)
    dup.loc[0, "action_type"] = "floppy"
    out, log = V.validate_actions(dup, inputs["possessions"])
    assert log["duplicates_dropped"] == 3 and len(out) == len(a)
    assert log["unknown_action_type"] == {"floppy": 1} and (out["action_type"] == "floppy").sum() == 1
    with pytest.raises(V.ValidationError):
        V.validate_actions(a.drop(columns=["action_type"]), inputs["possessions"])
    with pytest.raises(V.ValidationError):
        V.validate_possessions(inputs["possessions"].head(0))
    off = a.copy()
    off.loc[0, "end_frame"] = 10_000
    assert V.validate_actions(off, inputs["possessions"])[1]["outside_possession_span"] == 1
    p = inputs["possessions"].copy()
    p.loc[0, "points"] = 12
    assert V.validate_possessions(p)[1]["points_out_of_range"] == 1
    cp = inputs["coverage_predictions"].copy()
    cp.loc[0, "p_drop"] = 5.0
    lg = V.validate_coverage_predictions(cp)[1]
    assert lg["p_out_of_range"] == 1 and lg["p_not_summing_to_one"] == 1
    e = inputs["epv_states"].copy()
    e.loc[0, "epv_points"] = np.inf
    assert V.validate_epv_states(e)[1]["epv_not_finite"] == 1
    assert V.is_heldout_game(HELD) and V.is_heldout_game("01.02.2016.AAA.at.CLE") and not V.is_heldout_game("01.02.2016.AAA.at.BBB")


def test_action_ids_are_the_row_position_within_the_game():
    a = pd.DataFrame({"sportvu_game": ["g1", "g1", "g2", "g1"], "possession": [0, 0, 0, 1], "start_frame": [5, 1, 0, 9]})
    assert F.assign_action_ids(a)["action_id"].tolist() == [0, 1, 0, 2]
    assert F.assign_action_ids(a.assign(action_id=[7, 8, 9, 10]))["action_id"].tolist() == [7, 8, 9, 10]


def test_shared_fixture_labels_join_and_build(inputs, table):
    """The shared generator's labels (no action_id on actions, game-wide ids on screens, label_source synthetic)
    join and build. With the import missing, the same checks run on the local generator."""
    inp, t = inputs, table
    assert inp["heldout_games"] == [HELD]
    assert V.validate_inputs(inp)[1]["actions"]["unknown_action_type"] == {}
    assert len(t) == len(inp["actions"])
    assert set(t.loc[t["action_type"] == "ball_screen", "coverage_source"]) <= {"hand"}
    assert (t.loc[t["action_type"] != "ball_screen", "coverage"] == "none").all()
    assert t["off_lineup"].notna().all() and t["heldout"].sum() > 0
    F.features(t)


def test_coverage_resolution_prefers_hand_labels_then_predictions_then_none(inputs):
    a = F.assign_action_ids(inputs["actions"]).head(40).reset_index(drop=True)
    cov, src = F.resolve_coverage(a, inputs["screens"], inputs["coverage_predictions"])
    assert set(src) <= {"hand", "predicted", "none"} and (cov[src == "none"] == "none").all()
    cov2, src2 = F.resolve_coverage(a, None, inputs["coverage_predictions"])
    assert set(src2) <= {"predicted", "none"} and (src2 == "predicted").sum() == (src == "hand").sum()
    cov3, src3 = F.resolve_coverage(a, None, None)
    assert (cov3 == "none").all() and (src3 == "none").all()
    t, log = F.build_action_table(inputs["frames"], inputs["possessions"], inputs["actions"], None, None, None, [])
    assert t["epv_gain"].isna().all() and (t["coverage"] == "none").all() and not t["heldout"].any()


def test_data_root_layout_round_trip(tmp_path):
    root = fx.write_fixture_root(tmp_path / "data", seed=3, n_games=3, heldout_every=3, n_possessions=4, shared=True)
    assert (root / "sportvu_contract" / "heldout" / (HELD + ".parquet")).exists()
    inp = V.load_data_root(root, root)
    assert inp["heldout_games"] == [HELD] and inp["actions"]["label_source"].eq("synthetic").all()
    assert len(inp["screens"]) and len(inp["coverage_predictions"]) and len(inp["epv_states"])
    assert "team_id" not in inp["coverage_predictions"].columns               # the coverage_team_rates decoys are not read
    assert inp["files"]["coverage_predictions"]["rows"] == len(inp["coverage_predictions"]) > 0
    assert inp["files"]["actions_detected"]["skipped_files"] == 0             # actions_detected preferred over action_candidates
    assert inp["files"]["actions_detected"]["rows"] > 0
    # a root with detected actions only (no hand labels): they become the actions input
    import shutil
    shutil.rmtree(root / "labels" / "actions")
    det = V.load_data_root(root, root)
    assert det["actions"]["label_source"].eq("model").all() and "action_id" in det["actions"].columns
    res = F.main(["--data", str(root), "--out", str(tmp_path / "out")])
    assert res["table"]["heldout"].sum() > 0
    assert (tmp_path / "out" / "analytics" / "playcalling" / "heldout" / "2015-16" / "action_outcomes.parquet").exists()
    assert V.main(["--data", str(root)])["heldout_games"] == [HELD]
    assert V.main(["--fixtures"])["actions"]["rows"] > 0


# ── eda and notebook ───────────────────────────────────────────────────────────────────────────────
def test_eda_tables_and_the_shuffled_check(table):
    train = table[~table["heldout"]]
    ac = eda.action_counts(train)
    assert ac["actions"].sum() == len(train) and set(ac["label_source"]) == {"synthetic"}
    cc = eda.coverage_counts(train)
    assert cc["actions"].sum() == len(train)
    dist = eda.ppp_distribution(train)
    assert np.allclose(dist[["share_0", "share_1", "share_2", "share_3plus"]].sum(axis=1), 1.0)
    lu = eda.lineup_counts(train)
    assert lu["side"].tolist() == ["off_lineup", "def_lineup"] and (lu["distinct_lineups"] > 0).all()
    chk = eda.shuffled_action_check(train, n_shuffles=20, seed=0)
    assert chk["observed"].iloc[0] > 0 and 0 <= chk["share_shuffled_ge_observed"].iloc[0] <= 1
    md = eda.to_markdown(dist)
    assert md.startswith("| action_type |") and chr(0x2014) not in md


def test_notebook_is_valid_nbformat_4_json_and_imports_the_eda_functions():
    nb = json.loads((ROOT / "notebooks" / "eda_playcalling.ipynb").read_text())
    assert nb["nbformat"] == 4 and nb["cells"]
    src = "\n".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"])
    assert "from analytics.playcalling import validate, features, eda" in src
    for fn in ("action_counts", "coverage_counts", "ppp_distribution", "lineup_counts", "shuffled_action_check"):
        assert "eda.%s(" % fn in src
    assert "fixture output, not a result" in src


def test_no_em_dash_in_the_lane_files():
    files = [ROOT / "reports" / "playcalling.md", ROOT / "docs" / "contract" / "playcalling.md", ROOT / "notebooks" / "eda_playcalling.ipynb",
             ROOT / "tests" / "test_playcalling.py", ROOT / "tests" / "_fx_playcalling.py"]
    files += sorted((ROOT / "analytics" / "playcalling").glob("*.py"))
    for p in files:
        if p.exists():
            assert chr(0x2014) not in p.read_text(), p
