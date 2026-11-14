"""tools/frozen_epv.py and tools/frozen_ghosting.py: the rebuilt EPV and ghosting model objects (PLAN.md D45, D49).

Everything here runs on the synthetic season of analytics.fixtures written to a temporary data root. The "frozen
record" of that root is made by the frozen trainers themselves (the EPV trainer's own held-out refit, the ghosting
trainer's fit and roll-forward), so the tests ask the real question in miniature: does the rebuild reproduce what
the frozen code stored, and can a later process predict from the saved parts without fitting and without reading
an outcome. Every number is fixture output, not a result. Nothing reads the real data root.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics import deep_torch as DT  # noqa: E402
from analytics import fixtures  # noqa: E402
from analytics.epv import deep as epv_deep, features as epv_features, states as epv_states  # noqa: E402
from analytics.epv import train as epv_train, validate as epv_validate  # noqa: E402
from analytics.eval import fit_threads  # noqa: E402
from analytics.ghosting import train as gh_train, validate as gh_validate  # noqa: E402
from export import heavy, schemas  # noqa: E402
from tools import frozen_common as C  # noqa: E402
from tools import frozen_epv as E  # noqa: E402
from tools import frozen_ghosting as G  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

REPO = Path(__file__).resolve().parents[1]
HELD = fixtures.HELDOUT_GAME
SEASON = "2015-16"
EPV_COLUMNS = ["sportvu_game", "possession", "frame", "epv_points"]
GHOST_COLUMNS = ["sportvu_game", "possession", "frame", "def_player_id", "ghost_x", "ghost_y", "horizon_s"]
CHECK_FIELDS = ("name", "compared", "n", "n_expected", "max_abs_diff", "tolerance", "passed", "unit")
OUTCOMES = ("points_to_end", "possession_end_kind", "points", "end_kind")


# ------------------------------------------------------------------------------------------- the fixture world

def _states_from_the_frozen_builder(root: Path) -> None:
    """The fixture generator writes its own states table; the real root holds the frozen builder's. Rewrite it."""
    for game, season, held, frames_path, pbp_path, poss_path in epv_states.game_files(root):
        events = pd.read_parquet(pbp_path) if pbp_path else None
        states = epv_states.build_states(pd.read_parquet(frames_path), pd.read_parquet(poss_path), game, events=events)
        epv_states.write_states(states, root, season, game, held)


def _store_epv_record(root: Path) -> None:
    """What the frozen EPV trainer stores for the held-out games: its own refit and scoring, then its own tables."""
    d = epv_train.split_tables(epv_validate.load_root(root))
    st, ho = d["states_train"], d["states_heldout"]
    out = epv_train.heldout_once(st, epv_features.geometry(st), ho, epv_features.geometry(ho), d["shots_train"],
                                 d["rosters"], 0, 100, [epv_train.SEQUENCE], 0, deep="torch")
    epv_train.write_tables(root, *epv_train.epv_tables(out["pred"], epv_train.SEQUENCE, heldout=True))


def _store_ghost_record(root: Path, spill: Path) -> None:
    """What the frozen ghosting trainer stores for the held-out games: its fit on every fit row, its roll-forward
    with fold -1, its ghost table at its path."""
    games = gh_validate.iter_data_root(root)
    train_games, held_games = gh_train.split_games(games)
    spill.mkdir()
    for g in train_games:
        g["validation"] = gh_validate.validate_game(g, on_frames=lambda game, frames: gh_train.prepare_game(game, frames, spill))
    with fit_threads(gh_train.THREADS):
        models = gh_train.fit_candidates(gh_train.fit_frame(train_games, spill), 0, gh_train.THREADS, "mlp")
    for g in held_games:
        g["validation"] = gh_validate.validate_game(g)
        panel, rows = gh_train.game_rows(g)
        err = gh_train.roll_forward({c: models[c] for c in ("persistence", "mlp")}, rows, panel, fold=gh_train.HELDOUT_FOLD)
        ghost_frames, _ = gh_train.ghost_tables(err, "mlp", g["possessions"])
        path = gh_validate.output_path(root, "ghost_frames", g["sportvu_game"], True)
        path.parent.mkdir(parents=True, exist_ok=True)
        ghost_frames.to_parquet(path, index=False)


def _snapshot(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A synthetic data root with the frozen record of its held-out game, and both rebuilds run against it."""
    base = tmp_path_factory.mktemp("frozen_models")
    root, artifacts = base / "root", base / "artifacts"
    patch = pytest.MonkeyPatch()
    patch.setenv("DATA_ROOT", str(root))
    patch.delenv("V3_ARTIFACTS", raising=False)
    try:
        fixtures.write_fixture_root(root, seed=0, n_games=5, heldout_every=5, n_possessions=12, n_stint_games=2)
        _states_from_the_frozen_builder(root)
        _store_epv_record(root)
        _store_ghost_record(root, base / "spill")
        before = _snapshot(root)
        epv = E.rebuild(root, artifacts, expect=None, require_projection=False)
        ghosting = G.rebuild(root, artifacts, expect=None, require_projection=False)
        after = _snapshot(root)
        frames = pd.read_parquet(root / "sportvu_contract" / "heldout" / f"{HELD}.parquet")
        possessions = pd.read_parquet(root / "analytics" / "possessions" / "heldout" / SEASON / f"{HELD}.parquet")
        yield {"base": base, "root": root, "artifacts": artifacts, "epv": epv, "ghosting": ghosting,
               "changed": sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k)),
               "frames": frames, "possessions": possessions,
               "stored_epv": pd.read_parquet(root / "analytics" / "epv" / "heldout" / SEASON / f"{HELD}.parquet"),
               "stored_ghosts": pd.read_parquet(gh_validate.output_path(root, "ghost_frames", HELD, True))}
    finally:
        patch.undo()


def _copy(world, tmp_path):
    """A private copy of the data root and the artifacts, for a test that changes them."""
    root, artifacts = tmp_path / "root", tmp_path / "artifacts"
    shutil.copytree(world["root"], root)
    shutil.copytree(world["artifacts"], artifacts)
    return root, artifacts


def _boom(*args, **kwargs):
    raise AssertionError("a fit was called")


def _block_every_fit(monkeypatch) -> None:
    """Make every way of fitting anything raise, so that a test proves a path predicts only."""
    from sklearn.impute import SimpleImputer
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    for owner, name in ((epv_train.Calibrated, "fit"), (epv_deep, "fit_sequence"), (epv_deep.GRUSequenceRegressor, "fit"),
                        (DT, "train_net"), (epv_features.Priors, "from_training"), (epv_features, "fit_xfg"),
                        (gh_train, "fit_candidates"), (gh_train, "fit_mlp"), (gh_train, "fit_ridge"), (gh_train, "_scaled_fit"),
                        (StandardScaler, "fit"), (StandardScaler, "partial_fit"), (SimpleImputer, "fit"),
                        (IsotonicRegression, "fit"), (MLPRegressor, "fit"), (LogisticRegression, "fit"), (Ridge, "fit")):
        monkeypatch.setattr(owner, name, _boom)


# ------------------------------------------------------------------------- the check logic on synthetic values

def _table(n=6, shift=0.0):
    keys = pd.DataFrame({"sportvu_game": ["g"] * n, "possession": np.arange(n) // 3, "frame": np.arange(n) * 25})
    return keys.assign(value=np.linspace(0.5, 1.5, n) + shift)


def test_compare_passes_only_when_every_stored_row_is_reproduced():
    keys, stored = ["sportvu_game", "possession", "frame"], _table()
    same = E.compare(_table(), stored, keys, ["value"], 1e-5)
    assert same["passed"] and same["n"] == 6 and same["max_abs_diff"] == 0.0 and same["values_exact"] == 6
    near = E.compare(_table(shift=5e-6), stored, keys, ["value"], 1e-5)
    assert near["passed"] and 0 < near["max_abs_diff"] <= 1e-5 and near["values_exact"] == 0
    off = _table()
    off.loc[4, "value"] += 3e-5
    far = E.compare(off, stored, keys, ["value"], 1e-5)
    assert not far["passed"] and far["max_abs_diff"] == pytest.approx(3e-5) and far["worst"][0]["frame"] == 100
    short = E.compare(_table().iloc[:5], stored, keys, ["value"], 1e-5)
    assert not short["passed"] and short["rows_missing"] == 1 and short["n"] == 5 and short["max_abs_diff"] == 0.0
    long = E.compare(pd.concat([_table(), _table().assign(frame=999).iloc[:1]]), stored, keys, ["value"], 1e-5)
    assert not long["passed"] and long["rows_extra"] == 1 and long["n"] == 6
    twice = E.compare(pd.concat([_table(), _table().iloc[:1]]), stored, keys, ["value"], 1e-5)
    assert not twice["passed"] and twice["duplicate_keys"] == 1
    hole = _table()
    hole.loc[2, "value"] = np.nan
    blank = E.compare(hole, stored, keys, ["value"], 1e-5)
    assert not blank["passed"] and blank["not_finite"] == 1
    nothing = E.compare(_table().iloc[:0], stored.iloc[:0], keys, ["value"], 1e-5)
    assert not nothing["passed"] and nothing["n"] == 0 and nothing["max_abs_diff"] is None


def test_compare_counts_rows_once_when_a_row_holds_two_values():
    stored = _table().assign(other=1.0)
    moved = stored.copy()
    moved.loc[0, "other"] += 2e-3
    result = E.compare(moved, stored, ["sportvu_game", "possession", "frame"], ["value", "other"], 1e-3)
    assert result["n"] == 6 and result["values"] == 12 and not result["passed"]
    assert result["max_abs_diff"] == pytest.approx(2e-3) and result["values_exact"] == 11


def test_combine_adds_the_games_and_fails_when_one_game_fails():
    keys = ["sportvu_game", "possession", "frame"]
    good = E.compare(_table(), _table(), keys, ["value"], 1e-5)
    bad = E.compare(_table(shift=1e-3), _table(), keys, ["value"], 1e-5)
    both = E.combine({"a": good, "b": bad})
    assert both["n"] == 12 and both["games"] == 2 and not both["passed"]
    assert both["max_abs_diff"] == pytest.approx(1e-3) and both["per_game"]["a"]["passed"] and not both["per_game"]["b"]["passed"]
    assert E.combine({"a": good})["passed"] and not E.combine({})["passed"]


def test_a_compare_result_becomes_a_passing_record_only_through_the_shared_rule(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data_root"))
    folder = tmp_path / "artifacts" / "epv"
    folder.mkdir(parents=True)
    (folder / "model.bin").write_bytes(b"weights")
    keys = ["sportvu_game", "possession", "frame"]
    for predicted, expected, passes in ((_table(), 6, True), (_table(shift=1e-4), 6, False), (_table().iloc[:5], 6, False)):
        result = E.compare(predicted, _table(), keys, ["value"], E.TOLERANCE)
        C.write_check(folder, {"name": "epv", "unit": "points", "compared": "synthetic", "n": result["n"], "n_expected": expected,
                               "max_abs_diff": result["max_abs_diff"], "tolerance": E.TOLERANCE, "passed": result["passed"]},
                      files=["model.bin"])
        assert C.read_check(folder)["passed"] is passes
    E.write_stopped(folder, "epv", E.TOLERANCE, "ended early")
    stopped = C.read_check(folder)
    assert stopped["passed"] is False and stopped["status"] == "ended early" and stopped["reasons"]


def test_stopped_reason_keeps_its_own_messages_and_no_unexpected_text():
    local_path = "/" + "Users" + "/someone/data.parquet"
    assert "someone" not in E.stopped_reason(FileNotFoundError(local_path))
    assert schemas.find_private(E.stopped_reason(OSError(local_path))) == []
    assert "above the guard" in E.stopped_reason(MemoryError("resident memory reached 12.0 GB, above the guard"))
    assert "not rebuilt" in E.stopped_reason(SystemExit("lane code differs; not rebuilt"))


# ------------------------------------------------------------------------------------------------- the guards

def _wait_for(done, timeout_s=5.0):
    end = time.monotonic() + timeout_s
    while not done() and time.monotonic() < end:
        time.sleep(0.01)


def test_memory_watch_applies_the_guard_rule():
    seen = []
    with pytest.raises(MemoryError) as stop:
        with E.MemoryWatch(limit_gb=1e-6, where="tiny limit", every_s=0.01, on_trip=seen.append):
            _wait_for(lambda: seen)
    assert len(seen) == 1 and seen[0].tripped > 0
    assert "guard" in str(stop.value) and "tiny limit" in str(stop.value)
    with E.MemoryWatch(limit_gb=1e6, every_s=0.01, on_trip=seen.append) as watch:
        time.sleep(0.05)
    summary = watch.summary()
    assert watch.tripped is None and len(seen) == 1
    assert summary["samples"] >= 2 and summary["peak_rss_gb"] > 0
    assert E.MemoryWatch().on_trip is E.stop_process and E.MemoryWatch().limit_gb == C.MemoryGuard().limit_gb


def test_memory_watch_ends_a_run_whose_footprint_surges_past_the_guard(monkeypatch):
    monkeypatch.setattr(E.C, "rss_gb", lambda: 0.1)
    reading = {"rss": 0.1, "footprint": 53.0}
    monkeypatch.setattr(E, "memory_now", lambda: dict(reading))
    seen = []
    with E.MemoryWatch(limit_gb=50.0, surge_gb=4.0, every_s=0.01, on_trip=seen.append):
        time.sleep(0.05)                       # above the guard, inside the surge: resident memory decides, and it is low
    assert seen == []
    reading["footprint"] = 54.5
    with pytest.raises(MemoryError) as stop:
        with E.MemoryWatch(limit_gb=50.0, surge_gb=4.0, every_s=0.01, on_trip=seen.append):
            _wait_for(lambda: seen)
    assert len(seen) == 1 and "footprint" in str(stop.value)
    reading.update(rss=51.0, footprint=None)   # a system that reports no footprint: resident memory alone
    with pytest.raises(MemoryError) as stop:
        with E.MemoryWatch(limit_gb=50.0, every_s=0.01, on_trip=seen.append):
            _wait_for(lambda: len(seen) == 2)
    assert "resident" in str(stop.value)


def test_a_tripped_watch_ends_the_process_by_default():
    code = ("import time\nfrom tools import frozen_epv as E\n"
            "with E.MemoryWatch(limit_gb=1e-6, every_s=0.01):\n    time.sleep(30)\n")
    done = subprocess.run([sys.executable, "-B", "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert done.returncode == E.MEMORY_EXIT and "memory watch" in done.stderr


def test_stage_records_its_cost_and_applies_the_guard_before_the_next_stage():
    stages = {}
    with E.stage("quiet", stages, limit_gb=1e6, every_s=0.01):
        time.sleep(0.03)
    assert set(stages["quiet"]) >= {"seconds", "samples", "peak_rss_gb", "peak_footprint_gb", "held_footprint_gb"}
    with pytest.raises(MemoryError):
        with E.stage("over", stages, limit_gb=1e-6, every_s=60.0, on_trip=lambda watch: None):
            pass                                 # the watch never sampled: the guard between stages stops the run
    assert "over" in stages


def test_held_level_leaves_out_a_short_peak():
    times = np.arange(0, 60, 0.5)
    values = np.where((times >= 20) & (times < 22), 9.0, 4.0)        # two seconds at 9, the rest at 4
    values = np.where(times >= 40, 6.0, values)                       # then twenty seconds at 6
    assert E.held_level(times, values, 5.0) == 6.0
    assert E.held_level(times, values, 1.0) == 9.0
    assert E.held_level(times[:4], values[:4], 5.0) == 4.0 and E.held_level([], [], 5.0) is None


def _run(rows, rss, footprint, seconds=10.0, load=1.0):
    fit = {"seconds": seconds, "peak_rss_gb": rss, "peak_footprint_gb": footprint, "held_footprint_gb": footprint - 0.5}
    before = {"seconds": 5.0, "peak_rss_gb": load, "peak_footprint_gb": load, "held_footprint_gb": load}
    return {"games": rows // 1000, "rows": rows, "stages": {"load": before, "fit": fit}}


def test_extrapolate_follows_the_line_through_the_runs_and_gates_on_the_guard():
    runs = [_run(100_000, 1.0, 1.2), _run(300_000, 2.0, 2.6, seconds=30.0)]
    record = E.extrapolate(runs, 1_700_000, limit_gb=11.5, fixed=("load",), scaled=("fit",))
    assert record["projected_rss_gb"] == pytest.approx(0.5 + 5.0 * 1.7, abs=0.01)             # 0.5 GB plus 5 GB per million
    assert record["projected_footprint_gb"] == pytest.approx(0.5 + 7.0 * 1.7, abs=0.01)
    assert record["upper_rss_gb"] == pytest.approx(2.0 * 1.7 / 0.3, abs=0.01) and record["upper_rss_gb"] > record["projected_rss_gb"]
    assert record["projected_seconds"] == round(5.0 + 0.0 + 100.0 * 1.7) and record["ok"] is True
    assert E.extrapolate(runs, 1_700_000, limit_gb=8.0, fixed=("load",))["ok"] is False        # resident above the guard
    steep = [_run(100_000, 1.0, 1.2), _run(300_000, 2.0, 4.4)]
    assert E.extrapolate(steep, 1_700_000, limit_gb=11.5, fixed=("load",))["ok"] is False     # footprint above guard plus surge
    one = E.extrapolate(runs[:1], 1_700_000, limit_gb=11.5, fixed=("load",))
    assert one["lines"]["peak_rss_gb"]["method"] == "proportional" and one["projected_rss_gb"] == pytest.approx(17.0)
    heavy_load = E.extrapolate([_run(100_000, 1.0, 1.2, load=12.0), _run(300_000, 2.0, 2.6, load=12.0)], 1_700_000, 11.5, fixed=("load",))
    assert heavy_load["projected_rss_gb"] == 12.0 and heavy_load["ok"] is False                # a stage that costs the same at any size


def test_rebuild_does_not_start_without_a_projection_under_the_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data_root"))
    folder = tmp_path / "artifacts" / "epv"
    with pytest.raises(SystemExit, match="no memory projection"):
        E.need_projection(folder, 11.5, "test")
    runs = [_run(100_000, 1.0, 1.2), _run(300_000, 2.0, 2.6)]
    E.write_projection(folder, E.extrapolate(runs, 1_700_000, 11.5, fixed=("load",)))
    assert E.projection_path(folder).parent.name == "projection"      # never a file of the artifact folder itself
    assert E.need_projection(folder, 11.5, "test")["ok"] is True
    with pytest.raises(SystemExit, match="above the 8.0 GB guard"):
        E.need_projection(folder, 8.0, "test")
    E.write_projection(folder, E.extrapolate(runs, 5_000_000, 11.5, fixed=("load",)))
    with pytest.raises(SystemExit, match="not started"):
        E.need_projection(folder, 11.5, "test")
    assert schemas.find_private(E.projection_path(folder).read_text()) == []


def test_load_refuses_a_check_made_under_other_library_versions(monkeypatch):
    record = {"versions": E.versions()}
    E.same_versions(record, "epv")
    monkeypatch.setattr(E.np, "__version__", "0.0.0")
    with pytest.raises(RuntimeError, match="numpy"):
        E.same_versions(record, "epv")


def test_possession_tables_reach_the_frozen_builders_without_an_outcome_column():
    table = pd.DataFrame({"sportvu_game": ["a", "a", "b"], "possession": [0, 1, 0], "offense_team_id": [1, 2, 1],
                          "start_frame": [0, 100, 0], "end_frame": [100, 200, 90], "start_clock_s": [720.0, 700.0, 720.0],
                          "gap": [False, False, True], "points": [3, 2, 4], "end_kind": ["shot_made"] * 3, "n_frames": [101, 101, 91]})
    for_epv = E._possessions_for(table, "a")
    assert len(for_epv) == 2 and set(for_epv["points"]) == {0} and set(for_epv["end_kind"]) == {"other"}
    assert list(for_epv.columns) == list(E.POSSESSION_COLUMNS) + ["points", "end_kind"]
    for_ghosts = G._possessions_for(table, "b")
    assert len(for_ghosts) == 1 and list(for_ghosts.columns) == list(G.POSSESSION_COLUMNS)
    assert not set(OUTCOMES) & set(G.POSSESSION_COLUMNS) and not set(OUTCOMES) & set(E.POSSESSION_COLUMNS)
    assert len(E._possessions_for(table[table["sportvu_game"] == "a"], "a#broadcast")) == 2     # one game under another id
    for module in (E, G):
        with pytest.raises(ValueError, match="several games"):
            module._possessions_for(table, "c")
        with pytest.raises(ValueError, match="lack"):
            module._possessions_for(table.drop(columns=["start_clock_s"]), "a")


def test_the_modules_hold_no_dash_no_private_string_and_no_call_that_scores_heldout():
    for path in (REPO / "tools" / "frozen_epv.py", REPO / "tools" / "frozen_ghosting.py", Path(__file__)):
        text = path.read_text(encoding="utf-8")
        assert not any(dash in text for dash in (chr(0x2013), chr(0x2014))), path.name
        assert schemas.find_private(text) == [], path.name
        if path.parent.name == "tools":
            assert not any(call in text for call in C.FORBIDDEN_CALLS), path.name
            assert E.TOLERANCE == C.TOLERANCE_CAP["epv"] and G.TOLERANCE == C.TOLERANCE_CAP["ghosting"]


# ------------------------------------------------------------------------------------------------------- EPV

def test_epv_rebuild_reproduces_the_frozen_record_of_the_fixture_bit_for_bit(world):
    record, stored = world["epv"], world["stored_epv"]
    assert record["passed"] is True and record["reasons"] == []
    assert record["n"] == record["n_expected"] == len(stored) and record["max_abs_diff"] == 0.0
    assert record["tolerance"] == 1e-5 and record["unit"] == "points" and "no outcome column read" in record["compared"]
    for block in ("stored_states", "from_frames", "p_score"):
        assert record[block]["passed"] and record[block]["n"] == len(stored) and record[block]["max_abs_diff"] == 0.0
        assert record[block]["rows_missing"] == record[block]["rows_extra"] == 0
    assert set(record["files"]) == set(E.FILES) and record["fit"]["train"]["games"] == 4
    assert record["fit"]["final_fit"]["refit"] is True and record["fit"]["final_fit"]["best_epoch"] >= 1
    assert C.require_passed(world["artifacts"] / "epv", "epv")["passed"] is True
    schemas.RebuildCheck(**{k: record[k] for k in CHECK_FIELDS})
    meta = json.loads((world["artifacts"] / "epv" / "meta.json").read_text())
    assert meta["saved_equals_fitted"] and meta["xfg_prior_equals_frozen_closure"] and meta["settings"]["seed"] == 0
    assert meta["settings"]["deep"] == "torch" and len(meta["calls"]) == 7
    assert schemas.find_private((world["artifacts"] / "epv" / "check.json").read_text() + json.dumps(meta)) == []


def test_rebuilds_write_nothing_under_the_data_root(world):
    assert world["changed"] == []
    assert sorted(p.name for p in world["artifacts"].iterdir()) == ["epv", "ghosting"]


def test_epv_predictor_gives_the_stored_values_in_the_agreed_shape(world):
    predictor = E.load(world["artifacts"])
    out = predictor.predict_states(world["frames"], world["possessions"], HELD)
    stored = world["stored_epv"]
    assert list(out.columns) == EPV_COLUMNS
    assert out["possession"].dtype == np.int32 and out["frame"].dtype == np.int32 and out["epv_points"].dtype == np.float64
    assert set(out["sportvu_game"]) == {HELD} and not out.duplicated(["possession", "frame"]).any()
    assert out[["possession", "frame"]].to_numpy().tolist() == stored[["possession", "frame"]].to_numpy().tolist()
    assert np.array_equal(out["epv_points"].to_numpy(), stored["epv_points"].to_numpy())
    assert out.attrs["states"] == len(stored) and np.isfinite(out["epv_points"]).all()
    more = predictor.predict_states(world["frames"], world["possessions"], HELD, extra=True)
    assert list(more.columns) == EPV_COLUMNS + ["elapsed_in_possession_s", "p_score", "from_start"]
    assert np.array_equal(more["p_score"].to_numpy(), stored["p_score"].to_numpy()) and more["from_start"].all()
    assert np.array_equal(more["elapsed_in_possession_s"].to_numpy(), stored["elapsed_in_possession_s"].to_numpy())


def test_epv_prediction_cannot_depend_on_an_outcome(world):
    predictor = E.load(world["artifacts"])
    base = predictor.predict_states(world["frames"], world["possessions"], HELD)
    scrambled = world["possessions"].assign(points=4, end_kind="turnover")
    bare = world["possessions"][list(E.POSSESSION_COLUMNS)]
    frames = world["frames"][list(E.FRAME_COLUMNS)]
    for possessions in (scrambled, bare):
        again = predictor.predict_states(frames, possessions, HELD)
        assert again.equals(base)
    renamed = predictor.predict_states(frames, bare, HELD + "#broadcast")      # not a training id: the same priors
    assert np.array_equal(renamed["epv_points"].to_numpy(), base["epv_points"].to_numpy())


def test_epv_predictor_refuses_a_training_game_and_flags_a_possession_cut_at_its_start(world):
    predictor = E.load(world["artifacts"])
    with pytest.raises(ValueError, match="in-sample"):
        predictor.predict_states(world["frames"], world["possessions"], sorted(predictor.train_games)[0])
    possessions = world["possessions"]
    second = possessions.sort_values("possession").iloc[1]
    late = world["frames"][world["frames"]["frame"] >= int(second["start_frame"]) + 60]
    out = predictor.predict_states(late, possessions, HELD, extra=True)
    cut = out[out["possession"] == int(second["possession"])]
    assert len(cut) and not cut["from_start"].any()                       # values exist, flagged as not from the start
    assert out[out["possession"] > int(second["possession"])]["from_start"].all()
    empty = predictor.predict_states(world["frames"].iloc[:0], possessions, HELD, extra=True)
    assert len(empty) == 0 and list(empty.columns) == EPV_COLUMNS + ["elapsed_in_possession_s", "p_score", "from_start"]


def test_epv_xfg_prior_stand_in_equals_the_frozen_closure_in_both_its_forms(world):
    shots = pd.read_parquet(world["root"] / "collect" / "bref_shots" / "shotchart" / f"{SEASON}.parquet")
    rng = np.random.default_rng(1)
    loc_x, loc_y = rng.uniform(-250, 250, 500), rng.uniform(-50, 900, 500)
    for frozen in (epv_features.fit_xfg(shots), epv_features.fit_xfg(shots.iloc[:10]), epv_features.fit_xfg(None)):
        parts = E._xfg_parts(frozen)
        assert (parts["pipeline"] is None) != (parts["rate"] is None)
        assert np.array_equal(E.XfgPrior(**parts)(loc_x, loc_y), frozen(loc_x, loc_y))
    assert E._xfg_parts(epv_features.fit_xfg(shots))["pipeline"] is not None
    with pytest.raises(RuntimeError):
        E._xfg_parts(lambda loc_x, loc_y, m="not a model": None)


def test_epv_load_refuses_a_model_that_was_not_checked_or_changed_after_its_check(world, tmp_path):
    with pytest.raises(RuntimeError, match="no rebuilt artifact"):
        E.load(tmp_path / "nothing")
    root, artifacts = _copy(world, tmp_path)
    assert E.load(artifacts).check["passed"] is True
    with open(artifacts / "epv" / E.PARTS_FILE, "ab") as handle:
        handle.write(b"x")
    with pytest.raises(RuntimeError, match="changed after its check"):
        E.load(artifacts)


def test_epv_check_fails_when_the_stored_predictions_are_not_reproduced(world, tmp_path):
    root, artifacts = _copy(world, tmp_path)
    path = root / "analytics" / "epv" / "heldout" / SEASON / f"{HELD}.parquet"
    stored = pd.read_parquet(path)
    moved = stored.copy()
    moved.loc[3, "epv_points"] += 1e-3
    moved.to_parquet(path, index=False)
    record = E.check(root, artifacts, expect=None)
    assert record["passed"] is False and record["max_abs_diff"] == pytest.approx(1e-3, rel=1e-6)
    assert any("above the tolerance" in reason for reason in record["reasons"])
    assert record["stored_states"]["worst"][0]["frame"] == int(stored.loc[3, "frame"]) and record["n"] == len(stored)
    with pytest.raises(RuntimeError, match="did not reproduce"):
        E.load(artifacts)
    stored.iloc[:-1].to_parquet(path, index=False)                     # one stored row gone: a prediction with no stored row
    record = E.check(root, artifacts, expect=None)
    assert record["passed"] is False and record["stored_states"]["rows_extra"] == 1 and record["max_abs_diff"] == 0.0
    stored.to_parquet(path, index=False)                                # the record as the frozen run wrote it passes again
    assert E.check(root, artifacts, expect=None)["passed"] is True
    wrong_count = E.check(root, artifacts, expect={"heldout_games": 1, "heldout_states": len(stored) + 1})
    assert wrong_count["passed"] is False and wrong_count["matches_frozen_record"] is False


def test_epv_rebuild_refuses_bad_outputs_inputs_and_code(world, tmp_path, monkeypatch):
    root = world["root"]
    with pytest.raises(SystemExit, match="refusing to write"):
        E.rebuild(root, root / "analytics" / "elsewhere", expect=None, require_projection=False)
    with pytest.raises(SystemExit, match="no memory projection"):
        E.rebuild(root, tmp_path / "a", expect=None)
    monkeypatch.setattr(E, "_fit_calibrated", _boom)
    with pytest.raises(SystemExit, match="not the frozen record"):
        E.rebuild(root, tmp_path / "b", require_projection=False)       # the fixture is not the 604-game record
    status = C.read_check(tmp_path / "b" / "epv")
    assert status["passed"] is False and "not the frozen record" in status["status"]
    assert not (tmp_path / "b" / "epv" / E.STATE_FILE).exists()
    real = E._load_training

    def with_a_heldout_id(data_root):
        box = real(data_root)
        states = box["states"].copy()
        first = str(states["sportvu_game"].iloc[0])
        states["sportvu_game"] = states["sportvu_game"].astype(str).where(states["sportvu_game"].astype(str) != first, "01.02.2016.ORL.at.CLE")
        box["states"] = states
        return box

    monkeypatch.setattr(E, "_load_training", with_a_heldout_id)
    with pytest.raises(SystemExit):
        E.rebuild(root, tmp_path / "c", expect=None, require_projection=False)   # a held-out id among the fit games
    monkeypatch.setattr(E, "_load_training", real)
    state = {"head": "abc", "differs_from_freeze_v1": ["analytics/results.py", "analytics/epv/deep.py"], "uncommitted": []}
    monkeypatch.setattr(E, "frozen_code_state", lambda: state)
    with pytest.raises(SystemExit, match="differs from freeze-v1"):
        E.rebuild(root, tmp_path / "d", expect=None, require_projection=False)
    assert not (tmp_path / "d").exists()
    state.update(differs_from_freeze_v1=["analytics/results.py"], uncommitted=None)
    assert E.require_frozen_code("test")["head"] == "abc"


# -------------------------------------------------------------------------------------------------- ghosting

def test_ghosting_rebuild_reproduces_the_frozen_record_of_the_fixture_bit_for_bit(world):
    record, stored = world["ghosting"], world["stored_ghosts"]
    assert record["passed"] is True and record["reasons"] == []
    assert record["n"] == record["n_expected"] == len(stored) and record["max_abs_diff"] == 0.0
    assert record["tolerance"] == 1e-3 and record["unit"] == "ft" and "no outcome column read" in record["compared"]
    assert record["positions"]["values"] == 2 * len(stored) and record["positions"]["values_exact"] == 2 * len(stored)
    assert set(record["by_horizon"]) == {"0.2 s", "1.0 s"} and all(h["passed"] and h["n"] > 0 for h in record["by_horizon"].values())
    assert sum(h["n"] for h in record["by_horizon"].values()) == len(stored)
    assert set(record["files"]) == set(G.FILES) and record["fit"]["train"]["games"] == 4
    assert record["fit"]["final_fit"]["weights"] == 14274 and record["fit"]["final_fit"]["weights_dtype"] == "float32"
    assert C.require_passed(world["artifacts"] / "ghosting", "ghosting")["passed"] is True
    schemas.RebuildCheck(**{k: record[k] for k in CHECK_FIELDS})
    meta_text = (world["artifacts"] / "ghosting" / "meta.json").read_text()
    assert json.loads(meta_text)["saved_equals_fitted"] and json.loads(meta_text)["settings"]["fit_dtype"] == "float32"
    assert schemas.find_private((world["artifacts"] / "ghosting" / "check.json").read_text() + meta_text) == []


def test_ghost_predictor_gives_the_stored_positions_in_the_agreed_shape(world):
    predictor = G.load(world["artifacts"])
    out = predictor.predict_ghosts(world["frames"], world["possessions"], HELD)
    stored = world["stored_ghosts"]
    assert list(out.columns) == GHOST_COLUMNS and len(out) == len(stored) == out.attrs["rows"]
    assert {c: str(out[c].dtype) for c in GHOST_COLUMNS[1:]} == {"possession": "int32", "frame": "int32", "def_player_id": "int32",
                                                                "ghost_x": "float32", "ghost_y": "float32", "horizon_s": "float32"}
    assert set(out["sportvu_game"]) == {HELD}
    assert out[GHOST_COLUMNS].equals(stored[GHOST_COLUMNS])                 # the same rows in the same order, bit for bit
    one_second = out[out["horizon_s"] > 0.5]
    assert len(one_second) and sorted(round(float(h), 1) for h in out["horizon_s"].unique()) == [0.2, 1.0]
    keys = ["possession", "frame", "horizon_s"]
    assert (out.groupby(keys)["def_player_id"].nunique() == 5).all() and not out.duplicated(keys + ["def_player_id"]).any()
    starts = world["possessions"].set_index("possession")["start_frame"]
    first = one_second.groupby("possession")["frame"].min()
    assert ((first - starts.loc[first.index]) == 35).all()                # no 1.0 s ghost in the first 1.4 s


def test_ghost_prediction_cannot_depend_on_an_outcome_and_takes_bare_tables(world):
    predictor = G.load(world["artifacts"])
    base = predictor.predict_ghosts(world["frames"], world["possessions"], HELD)
    frames = world["frames"][list(G.MODEL_FRAME_COLUMNS)]                 # no game and no n_players column
    for possessions in (world["possessions"].assign(points=4, end_kind="turnover"), world["possessions"][list(G.POSSESSION_COLUMNS)]):
        assert predictor.predict_ghosts(frames, possessions, HELD).equals(base)
    with pytest.raises(ValueError, match="lack"):
        predictor.predict_ghosts(frames.drop(columns=["ball_x"]), world["possessions"], HELD)


def test_ghost_predictor_refuses_a_training_game_and_keeps_whole_possessions_only(world):
    predictor = G.load(world["artifacts"])
    with pytest.raises(ValueError, match="in-sample"):
        predictor.predict_ghosts(world["frames"], world["possessions"], sorted(predictor.train_games)[0])
    frames, possessions = world["frames"], world["possessions"]
    whole = predictor.predict_ghosts(frames, possessions, HELD)
    half = predictor.predict_ghosts(frames[frames["frame"] <= int(frames["frame"].max()) // 2], possessions, HELD)
    assert 0 < len(half) < len(whole) and half.attrs["possessions_outside_frames"] > 0
    kept = whole[whole["possession"].isin(half["possession"].unique())].reset_index(drop=True)
    keys = ["possession", "frame", "def_player_id", "horizon_s"]
    assert half[keys].equals(kept[keys])                                    # a kept possession has the stored rows
    assert np.allclose(half[["ghost_x", "ghost_y"]].to_numpy(), kept[["ghost_x", "ghost_y"]].to_numpy(), atol=1e-4)
    empty = predictor.predict_ghosts(frames.iloc[:0], possessions, HELD)
    assert len(empty) == 0 and list(empty.columns) == GHOST_COLUMNS and str(empty["ghost_x"].dtype) == "float32"


def test_ghosting_load_and_check_refuse_what_was_not_reproduced(world, tmp_path):
    with pytest.raises(RuntimeError, match="no rebuilt artifact"):
        G.load(tmp_path / "nothing")
    root, artifacts = _copy(world, tmp_path)
    path = gh_validate.output_path(root, "ghost_frames", HELD, True)
    stored = pd.read_parquet(path)
    moved = stored.copy()
    moved.loc[7, "ghost_y"] += np.float32(0.01)
    moved.to_parquet(path, index=False)
    record = G.check(root, artifacts, expect=None)
    assert record["passed"] is False and record["max_abs_diff"] == pytest.approx(0.01, rel=1e-3) and record["n"] == len(stored)
    assert record["positions"]["worst"][0]["frame"] == int(stored.loc[7, "frame"])
    with pytest.raises(RuntimeError, match="did not reproduce"):
        G.load(artifacts)
    stored.iloc[5:].to_parquet(path, index=False)                       # stored rows gone: predictions with no stored row
    record = G.check(root, artifacts, expect=None)
    assert record["passed"] is False and record["positions"]["rows_extra"] == 5
    stored.to_parquet(path, index=False)
    assert G.check(root, artifacts, expect=None)["passed"] is True
    with open(artifacts / "ghosting" / G.MODEL_FILE, "ab") as handle:
        handle.write(b"x")
    with pytest.raises(RuntimeError, match="changed after its check"):
        G.load(artifacts)


def test_ghosting_rebuild_refuses_bad_outputs_inputs_and_heldout_games(world, tmp_path, monkeypatch):
    root = world["root"]
    with pytest.raises(SystemExit, match="refusing to write"):
        G.rebuild(root, root / "analytics" / "elsewhere", expect=None, require_projection=False)
    with pytest.raises(SystemExit, match="no memory projection"):
        G.rebuild(root, tmp_path / "a", expect=None)
    monkeypatch.setattr(G, "_fit_models", _boom)
    with pytest.raises(SystemExit, match="the frozen record has 604"):
        G.rebuild(root, tmp_path / "b", require_projection=False)
    few_rows = dict(G.FROZEN_RECORD, train_games=4)
    with pytest.raises(SystemExit, match="fit rows found"):
        G.rebuild(root, tmp_path / "c", expect=few_rows, require_projection=False)
    status = C.read_check(tmp_path / "c" / "ghosting")
    assert status["passed"] is False and "fit rows found" in status["status"]
    monkeypatch.setattr(G, "_prepare", _boom)
    monkeypatch.setattr(G.T, "split_games", lambda games: (list(games), []))      # nothing set aside
    with pytest.raises(SystemExit):
        G.rebuild(root, tmp_path / "d", expect=None, require_projection=False)     # the guard on the fit ids says no
    with pytest.raises(RuntimeError):
        G._parts_of(lambda W, m=(lambda W: W): m(W))


# ------------------------------------------------------- a later process: no fit, no outcome, no training data

def test_a_later_process_checks_and_predicts_without_fitting_or_reading_an_outcome(world, tmp_path, monkeypatch):
    root, artifacts = _copy(world, tmp_path)
    analytics = root / "analytics"
    for folder in (analytics / "states_1hz" / SEASON, analytics / "possessions" / SEASON, analytics / "rosters",
                   analytics / "shot_frames", analytics / "rebound_frames", root / "collect", root / "labels"):
        shutil.rmtree(folder)                                             # no training table left, no shots, no rosters
    for path in (root / "sportvu_contract").glob("*.parquet"):
        path.unlink()                                                     # no training frames
    for path in (root / "sportvu_contract" / "heldout").glob("*_pbp.parquet"):
        path.unlink()                                                     # no play-by-play
    for path in list(analytics.rglob("*.parquet")):
        if path.name == "epv_possessions.parquet":
            path.write_bytes(b"never opened")                             # it holds an outcome column and is not a game
            continue
        table = pd.read_parquet(path)
        table.drop(columns=[c for c in OUTCOMES + ("actual_x", "actual_y", "deviation_ft") if c in table.columns]).to_parquet(path, index=False)
    assert not any(set(OUTCOMES) & set(pd.read_parquet(p).columns) for p in analytics.rglob("*.parquet") if p.name != "epv_possessions.parquet")
    _block_every_fit(monkeypatch)
    epv = E.check(root, artifacts, expect=None)
    ghosting = G.check(root, artifacts, expect=None)
    assert epv["passed"] and epv["max_abs_diff"] == 0.0 and epv["from_frames"]["max_abs_diff"] == 0.0
    assert ghosting["passed"] and ghosting["max_abs_diff"] == 0.0
    frames = pd.read_parquet(root / "sportvu_contract" / "heldout" / f"{HELD}.parquet")
    possessions = pd.read_parquet(analytics / "possessions" / "heldout" / SEASON / f"{HELD}.parquet")
    assert not set(OUTCOMES) & set(possessions.columns)
    states = E.load(artifacts).predict_states(frames, possessions, HELD)
    ghosts = G.load(artifacts).predict_ghosts(frames, possessions, HELD)
    assert np.array_equal(states["epv_points"].to_numpy(), world["stored_epv"]["epv_points"].to_numpy())
    assert np.array_equal(ghosts[["ghost_x", "ghost_y"]].to_numpy(), world["stored_ghosts"][["ghost_x", "ghost_y"]].to_numpy())


def test_command_line_checks_and_refuses_a_rebuild_without_a_projection(world, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("V3_HEAVY_LOCK", str(tmp_path / "heavy.lock"))
    monkeypatch.delenv(heavy.HELD_ENV, raising=False)
    root, artifacts = _copy(world, tmp_path)
    for module in (E, G):                                                # the real record is the 604-game season
        assert module.main(["check", "--data", str(root), "--artifacts", str(artifacts)]) == 1
        assert json.loads(capsys.readouterr().out)["matches_frozen_record"] is False
    monkeypatch.setattr(E, "FROZEN_RECORD", {"train_games": 4, "train_states": world["epv"]["fit"]["train"]["states"],
                                             "heldout_games": 1, "heldout_states": len(world["stored_epv"])})
    monkeypatch.setattr(G, "FROZEN_RECORD", {"train_games": 4, "fit_rows": world["ghosting"]["fit"]["train"]["fit_rows"],
                                             "heldout_games": 1, "ghost_rows": len(world["stored_ghosts"])})
    for module, name in ((E, "epv"), (G, "ghosting")):
        assert module.main(["check", "--data", str(root), "--artifacts", str(artifacts)]) == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed["name"] == name and printed["passed"] is True
        with pytest.raises(SystemExit, match="no memory projection"):
            module.main(["rebuild", "--data", str(root), "--artifacts", str(tmp_path / "fresh")])
        with pytest.raises(SystemExit, match="no rebuilt model"):
            module.main(["check", "--data", str(root), "--artifacts", str(tmp_path / "fresh")])
    assert (tmp_path / "heavy.lock").read_text() == ""                     # the lock was taken and given back


@pytest.mark.slow
def test_projection_runs_the_same_code_on_subsets_and_opens_the_way_for_a_rebuild(world, tmp_path, monkeypatch):
    monkeypatch.setenv("V3_HEAVY_LOCK", str(tmp_path / "heavy.lock"))
    monkeypatch.delenv(heavy.HELD_ENV, raising=False)
    root = world["root"]
    artifacts = tmp_path / "artifacts"
    record = E.project(root, artifacts, games=(2, 4))
    assert [run["games"] for run in record["runs"]] == [2, 4] and record["rows_full"] == world["epv"]["fit"]["train"]["states"]
    assert record["ok"] is True and set(record["runs"][0]["stages"]) == {"load", "fit"}
    assert E.read_projection(artifacts / "epv")["projected_rss_gb"] == record["projected_rss_gb"]
    rebuilt = E.rebuild(root, artifacts, expect=None)                       # the projection is there: the rebuild starts
    assert rebuilt["passed"] is True and rebuilt["max_abs_diff"] == 0.0
    assert E.load(artifacts).check["passed"] is True                       # the projection file does not undo the check
    ghost_record = G.project(root, artifacts, games=(2, 4))
    assert [run["games"] for run in ghost_record["runs"]] == [2, 4] and ghost_record["rows_full"] == G.FROZEN_RECORD["fit_rows"]
    assert set(ghost_record["runs"][0]["stages"]) == {"prepare", "fit"} and E.projection_path(artifacts / "ghosting").exists()
