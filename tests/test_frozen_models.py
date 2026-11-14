"""tools/frozen_models.py (PLAN.md D45): the guards, the check logic on synthetic values, the report parsers on the
committed reports, and the rebuilds on fixtures. Every number produced here is fixture output, never a result.

No test reads the real DATA_ROOT or waits for the machine-wide heavy-job lock: each one gets an empty folder as
DATA_ROOT and its own lock file."""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from analytics import results as R  # noqa: E402
from export import heavy  # noqa: E402
from export import schemas as S  # noqa: E402
from tools import frozen_common as C  # noqa: E402
from tools import frozen_models as M  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HELD = "12.16.2015.PHX.at.GSW"              # the held-out game of sportvu/splits.json
HELD_ARENA = "01.18.2016.GSW.at.CLE"        # every game at CLE is held out
TRAIN_GAMES = ["11.11.2015.NYK.at.CHA", "11.12.2015.GSW.at.MIN", "11.13.2015.BKN.at.SAC", "11.14.2015.DAL.at.HOU"]
REBUILDS = [M.rebuild_absence, M.rebuild_embeddings, M.rebuild_xfg]


@pytest.fixture(autouse=True)
def own_root(tmp_path, monkeypatch):
    """An empty folder as DATA_ROOT and a private heavy-job lock for every test."""
    monkeypatch.setenv("V3_HEAVY_LOCK", str(tmp_path / "heavy.lock"))
    monkeypatch.delenv(heavy.HELD_ENV, raising=False)
    monkeypatch.delenv("V3_ARTIFACTS", raising=False)
    root = tmp_path / "data_root"
    root.mkdir()
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


def as_site_check(record: dict):
    """A check record as the site contract carries it (export.schemas.RebuildCheck)."""
    keys = ("name", "compared", "n", "n_expected", "max_abs_diff", "tolerance", "passed", "unit")
    return S.RebuildCheck(**{key: record.get(key) for key in keys})


def player_games(games, n_players=8, seed=0):
    """A small player_games table with the columns the frozen load trainer reads (random values, both classes)."""
    from analytics.load.features import GAME_COLUMNS
    rng = np.random.default_rng(seed)
    rows = []
    for g, game in enumerate(games):
        for p in range(n_players):
            rest = float(rng.integers(1, 4))
            rows.append({"sportvu_game": game, "game_date": "2015-11-%02d" % (g + 1), "player_id": 100 + p, "team": "AAA",
                         "minutes": 20.0, "distance_ft": 8000.0, "avg_speed_ftps": 6.0, "decline_ftps": float(rng.normal()),
                         "slope_ftps_per_stint": 0.0, "days_rest": rest, "back_to_back": rest <= 1,
                         "games_last_7": float(rng.integers(0, 4)), "minutes_last_3": float(rng.uniform(0, 90)),
                         "distance_last_3_ft": float(rng.uniform(0, 30000)), "prev_decline_ftps": float(rng.normal()),
                         "prev_avg_speed_ftps": float(rng.uniform(4, 8)), "absent_next": bool(rng.random() < 0.3),
                         "heldout": False})
    table = pd.DataFrame(rows)[GAME_COLUMNS]
    table["absent_next"] = table["absent_next"].astype("boolean")
    table["player_id"] = table["player_id"].astype("int32")
    return table


# ----------------------------------------------------------------------------- settings and the frozen code

def test_settings_are_the_frozen_command_line_defaults():
    from analytics.embeddings import train as E
    from analytics.load import train as L
    from analytics.xfg import train as T
    for parse in (L.parse_args, T.parse_args, E.parse_args):
        args = parse(["--fixtures"])
        assert (args.seed, args.folds, args.bootstrap) == (M.SEED, M.FOLDS, M.BOOTSTRAP)
    assert L.parse_args(["--fixtures"]).season == T.parse_args(["--fixtures"]).season == M.SEASON
    # the committed runs named the seed and left folds and bootstrap at those defaults
    for lane in ("load", "embeddings", "xfg"):
        record = R.parse_report(lane, (REPO / "reports" / f"{lane}.md").read_text(encoding="utf-8"), f"reports/{lane}.md")
        assert f"--seed {M.SEED}" in record.command
        assert "--folds" not in record.command and "--bootstrap" not in record.command
    xfg = R.parse_report("xfg", (REPO / "reports" / "xfg.md").read_text(encoding="utf-8"), "reports/xfg.md")
    assert xfg.selected == M.XFG_CANDIDATE and M.XFG_CANDIDATE in T.CANDIDATES


def test_frozen_code_is_held_against_the_tag(monkeypatch, tmp_path):
    state = M.frozen_code()
    if state["verified"]:
        assert state["unexpected"] == [] and set(state["changed"]) <= set(M.CHANGED_SINCE_FREEZE)
    changed = {"tag": M.FROZEN_TAG, "verified": True, "changed": ["analytics/eval.py"], "unexpected": ["analytics/eval.py"]}
    monkeypatch.setattr(M, "frozen_code", lambda: changed)
    for rebuild in REBUILDS:
        with pytest.raises(SystemExit, match="frozen code differs"):
            rebuild(artifacts=tmp_path / "artifacts")
    assert not (tmp_path / "artifacts").exists()


# ----------------------------------------------------------------------------- the output guard

@pytest.mark.parametrize("rebuild", REBUILDS)
def test_an_output_under_data_root_or_reports_is_refused(rebuild, own_root, tmp_path):
    for bad in (own_root / "artifacts", own_root, REPO / "reports" / "frozen_models_test"):
        with pytest.raises(SystemExit, match="refusing to write"):
            rebuild(artifacts=bad)
    assert list(own_root.iterdir()) == []
    assert not (REPO / "reports" / "frozen_models_test").exists()
    assert not (tmp_path / "heavy.lock").exists()            # refused before the lock was taken


@pytest.mark.parametrize("rebuild", REBUILDS)
def test_nothing_runs_without_data_root(rebuild, monkeypatch, tmp_path):
    monkeypatch.delenv("DATA_ROOT")
    with pytest.raises(SystemExit, match="DATA_ROOT"):
        rebuild(artifacts=tmp_path / "artifacts")
    assert not (tmp_path / "artifacts").exists()


def test_embeddings_command_always_names_an_output_root_and_a_report_inside_the_folder(own_root, tmp_path):
    folder = (tmp_path / "artifacts" / "embeddings").resolve()
    command = M.embeddings_command(folder, own_root)
    assert command[:3] == [sys.executable, "-m", "analytics.embeddings.train"]
    assert dict(zip(command[3::2], command[4::2])) == {"--data": str(own_root), "--seed": str(M.SEED), "--out": str(folder),
                                                       "--report": str(folder / M.REPORT_NAME)}
    for bad in (own_root / "embeddings", REPO / "reports"):
        with pytest.raises(SystemExit, match="refusing to write"):
            M.embeddings_command(bad, own_root)


# ----------------------------------------------------------------------------- the held-out guard

def test_absence_refuses_a_held_out_game_id_or_flag(monkeypatch):
    from analytics.load import train as L
    pred, note = M.absence_oof(player_games(TRAIN_GAMES))
    assert note["mode"] == "out-of-fold" and note["folds"] == len(TRAIN_GAMES) and pred["p_logistic"].between(0, 1).all()
    monkeypatch.setattr(L, "fit_absence", lambda rows: pytest.fail("a fit ran on a table with a held-out row"))
    for held in (HELD, HELD_ARENA):
        with pytest.raises(SystemExit, match="held-out"):
            M.absence_oof(player_games(TRAIN_GAMES + [held]))
    flagged = player_games(TRAIN_GAMES)
    flagged.loc[3, "heldout"] = True
    with pytest.raises(SystemExit, match="flagged held out"):
        M.absence_oof(flagged)
    with pytest.raises(SystemExit, match="no game id"):
        M.absence_oof(player_games(TRAIN_GAMES).iloc[0:0])          # nothing to check is refused too


def test_absence_rebuild_stops_on_a_held_out_row_and_leaves_nothing_usable(own_root, tmp_path):
    path = own_root / "analytics" / "load" / M.SEASON / "player_games.parquet"
    path.parent.mkdir(parents=True)
    player_games(TRAIN_GAMES + [HELD_ARENA]).to_parquet(path, index=False)
    artifacts = tmp_path / "artifacts"
    with pytest.raises(SystemExit, match="held-out"):
        M.rebuild_absence(artifacts=artifacts)
    check = C.read_check(artifacts / "absence")
    assert check["passed"] is False and check["status"] == "stopped" and check["error"].startswith("SystemExit")
    assert not (artifacts / "absence" / M.ABSENCE_FILE).exists()
    with pytest.raises(RuntimeError):
        M.load_absence(artifacts)
    assert [row["passed"] for row in M.status(artifacts)] == [False, False, False]


# ----------------------------------------------------------------------------- check logic on synthetic values

def test_printed_numbers_agree_within_half_of_the_last_digit():
    for value, agrees in ((0.27644999, True), (0.27635001, True), (0.2764, True), (0.27651, False), (0.27629, False),
                          (float("nan"), False), (float("inf"), False), (None, False), ("text", False)):
        assert M.compare_printed("x", "0.2764", value)["agrees"] is agrees, value
    assert M.compare_printed("x", "-0.0481", -0.04812)["agrees"] and not M.compare_printed("x", "-0.0481", 0.0481)["agrees"]
    assert M.compare_printed("x", "12239", 12239)["agrees"] and not M.compare_printed("x", "12239", 12240)["agrees"]
    for missing in ("n/a", None):               # a number the report does not give agrees only with no number
        assert M.compare_printed("x", missing, float("nan"))["agrees"] and not M.compare_printed("x", missing, 0.5)["agrees"]
    row = M.compare_printed("x", "0.2764", 0.27646, required=True)
    assert row["required"] and row["half_unit"] == 5e-05 and row["abs_diff"] == pytest.approx(6e-05)


def test_a_missing_or_wrong_number_never_derives_a_pass(tmp_path):
    good = [M.compare_printed("a", "0.5000", 0.50001), M.compare_printed("b", "n/a", float("nan"))]
    bad = good + [M.compare_printed("c", "0.5000", float("nan"))]
    wrong = good + [M.compare_printed("d", "0.5000", 0.5002)]
    loose = [M.compare_printed("e", "0.50", 0.501)]           # agrees as printed, but two decimals are below the bar
    assert M._numbers(good) == {"n": 2, "max_abs_diff": pytest.approx(1e-05), "tolerance": C.TOLERANCE_CAP["absence"]}
    assert M._numbers(bad)["max_abs_diff"] is None and all(row["agrees"] for row in loose)
    for case, rows, expected in (("good", good, True), ("bad", bad, False), ("wrong", wrong, False), ("loose", loose, False)):
        folder = tmp_path / case
        folder.mkdir()
        (folder / M.ABSENCE_FILE).write_bytes(b"rows")
        C.write_check(folder, dict(M._numbers(rows), name="absence", compared="synthetic"), files=[M.ABSENCE_FILE])
        assert C.read_check(folder)["passed"] is expected, case
    # two reports that print the same text agree, also when both print nothing; anything else does not
    assert M.compare_same("x", "0.768", "0.768")["agrees"] and M.compare_same("x", None, None)["agrees"]
    assert not M.compare_same("x", "0.768", "0.769")["agrees"] and not M.compare_same("x", "0.768", None)["agrees"]
    assert M.compare_same("x", "0.768", "0.769")["abs_diff"] == pytest.approx(0.001)


def stored_and_rebuilt(n=40):
    rng = np.random.default_rng(1)
    players = pd.array([None if i % 7 == 0 else 200 + i for i in range(n)], dtype="Int64")
    stored = pd.DataFrame({"sportvu_game": [TRAIN_GAMES[i % 2] for i in range(n)], "event": np.arange(n) - 2,
                           "player_id": players, "p_selected": rng.uniform(0.02, 0.98, n).astype("float32")})
    rebuilt = stored[["sportvu_game", "event", "player_id"]].copy()
    rebuilt["has_contest"] = (np.arange(n) % 3 == 0).astype(int)
    return stored, rebuilt, stored["p_selected"].to_numpy(dtype=float)


def test_predictions_are_compared_row_for_row():
    stored, rebuilt, p = stored_and_rebuilt()
    same = M.check_predictions(stored, rebuilt, p + 9e-7)
    assert same["passed"] and same["n"] == same["n_expected"] == len(stored) and same["rows_over_tolerance"] == 0
    assert same["max_abs_diff"] == pytest.approx(9e-7, rel=1e-3) and same["max_abs_diff"] <= same["tolerance"] == M.XFG_TOLERANCE
    assert same["by_group"]["has_contest yes"]["n"] + same["by_group"]["has_contest no"]["n"] == len(stored)
    assert same["by_group"]["tracked_event no"]["n"] == 2 and "worst_rows" not in same

    off = p.copy()
    off[5] += 2e-6
    moved = M.check_predictions(stored, rebuilt, off)
    assert not moved["passed"] and moved["rows_over_tolerance"] == 1 and moved["max_abs_diff"] == pytest.approx(2e-6, rel=1e-3)
    assert moved["worst_rows"][0]["event"] == int(stored["event"].iloc[5]) and moved["games_over_tolerance"] == 1

    nan = p.copy()
    nan[0] = np.nan
    broken = M.check_predictions(stored, rebuilt, nan)
    assert not broken["passed"] and broken["max_abs_diff"] is None and broken["not_finite"] == 1
    assert broken["rows_over_tolerance"] == 1 and broken["worst_rows"][0]["rebuilt"] is None


def test_predictions_are_not_compared_when_the_rows_do_not_line_up():
    stored, rebuilt, p = stored_and_rebuilt()
    for column, value in (("event", 999), ("sportvu_game", TRAIN_GAMES[2]), ("player_id", 7)):
        other = rebuilt.copy()
        other.loc[4, column] = value
        result = M.check_predictions(stored, other, p)
        assert not result["passed"] and result["n"] == 0 and result["max_abs_diff"] is None
        assert result["alignment"]["keys_equal"] is False and result["alignment"]["mismatches"][column] == 1
    lost = rebuilt.copy()
    lost.loc[1, "player_id"] = pd.NA                 # a null against a value is a mismatch; two nulls agree (row 0)
    assert M.check_predictions(stored, lost, p)["alignment"]["mismatches"] == {"sportvu_game": 0, "event": 0, "player_id": 1}
    for a, b, q in ((stored, rebuilt.iloc[:-1], p[:-1]), (stored.iloc[:-1], rebuilt, p), (stored, rebuilt, p[:-1]),
                    (stored.iloc[0:0], rebuilt.iloc[0:0], p[:0])):
        result = M.check_predictions(a, b, q)
        assert not result["passed"] and result["n"] == 0 and result["max_abs_diff"] is None
        assert result["n_expected"] == len(a)                  # what a full comparison would have covered


def test_a_prediction_check_record_passes_only_through_write_check(tmp_path):
    stored, rebuilt, p = stored_and_rebuilt()
    assert M.XFG_TOLERANCE == C.TOLERANCE_CAP["xfg"]
    cases = (("near", stored, rebuilt, p + 5e-7, True), ("far", stored, rebuilt, p + 5e-6, False),
             ("short", stored, rebuilt.iloc[:-1], p[:-1], False))            # not every stored prediction compared
    for case, a, b, q, expected in cases:
        folder = tmp_path / case
        folder.mkdir()
        (folder / M.MODEL_NAME).write_bytes(b"parts")
        C.write_check(folder, dict(M.check_predictions(a, b, q), name="xfg", compared="synthetic"), files=[M.MODEL_NAME])
        assert C.read_check(folder)["passed"] is expected, case


# ----------------------------------------------------------------------------- report parsers on the committed reports

def test_load_parser_reads_the_committed_report_as_the_frozen_results_parser_does():
    text = (REPO / "reports" / "load.md").read_text(encoding="utf-8")
    mine, frozen = M.parse_load_report(text), R.parse_report("load", text, "reports/load.md")
    log_loss = mine["logistic"]["log_loss"]
    assert frozen.oof_extra[0].name == "absent_next log loss"
    assert (frozen.oof_extra[0].value, frozen.oof_extra[0].interval) == (log_loss["point"], f"[{log_loss['lo']}, {log_loss['hi']}]")
    assert mine["selected"] in frozen.selected and mine["selected"].startswith("logistic")
    assert mine["protocol"] == f"out-of-fold under {M.FOLDS} grouped folds by sportvu_game (seed {M.SEED})"
    assert mine["rows"] == sum(mine["calibration_n"]) > mine["games"] > 0
    for model in ("base", "logistic"):
        for metric in ("log_loss", "brier", "ece"):
            cell = mine[model][metric]
            assert float(cell["lo"]) <= float(cell["point"]) <= float(cell["hi"])
            assert all(len(cell[part].partition(".")[2]) == 4 for part in ("point", "lo", "hi"))
    assert float(mine["logistic"]["log_loss"]["point"]) < float(mine["base"]["log_loss"]["point"])
    assert float(mine["paired"]["lo"]) <= float(mine["paired"]["diff"]) <= float(mine["paired"]["hi"]) < 0 and mine["excludes_zero"]
    assert 0 < float(mine["base_rate"]) < 1 and 0 < float(mine["calibration_slope"]) < 2
    # the held-out sections are never what the parser reads: without them the result is the same
    cut = text[:text.index("## Held-out games")]
    assert M.parse_load_report(cut[:cut.index("## Stage 8")] + cut[cut.index("## Inputs"):]) == mine


def test_embeddings_parser_reads_the_committed_report_as_the_frozen_results_parser_does():
    text = (REPO / "reports" / "embeddings.md").read_text(encoding="utf-8")
    mine, frozen = M.parse_embeddings_report(text), R.parse_report("embeddings", text, "reports/embeddings.md")
    accuracy, majority = mine["accuracy"], mine["majority"]
    assert (frozen.oof.value, frozen.oof.interval) == (accuracy["point"], f"[{accuracy['lo']}, {accuracy['hi']}]")
    assert (frozen.oof_extra[0].value, frozen.oof_extra[0].interval) == (majority["point"], f"[{majority['lo']}, {majority['hi']}]")
    assert frozen.selected == mine["chosen"]
    assert frozen.notes[0] == (f"Player-seasons: {mine['player_seasons']}, of which {mine['sufficient']} sufficient (at least "
                               f"{mine['possession_gate']} possessions) and {mine['insufficient']} insufficient.")
    assert int(mine["sufficient"]) + int(mine["insufficient"]) == int(mine["player_seasons"])
    rows = {name: int(value) for name, value in mine["table_rows"].items()}
    assert rows["player_season_features"] == int(mine["player_seasons"]) and rows["player_embeddings"] == int(mine["sufficient"])
    assert rows["embedding_neighbours"] == 5 * int(mine["sufficient"]) and int(mine["labelled"]) <= int(mine["sufficient"])
    with pytest.raises(R.ParseError):
        M.parse_embeddings_report(text[:text.index("## Candidates")])        # the shape of a run that fitted nothing
    games = M.trained_games(text)
    assert len(set(games)) == len(games) > 0 and C.training_games_only(games, "test") == len(games)


# ----------------------------------------------------------------------------- absence on the load lane's test generator

@pytest.fixture(scope="module")
def load_lane(tmp_path_factory):
    """The frozen load lane on its test generator: the stored training table, the report the frozen writer prints,
    and the out-of-fold absence rows the lane keeps in memory and never writes."""
    from analytics.load import features as F
    from analytics.load import train as L
    from tests import _fx_load
    base = tmp_path_factory.mktemp("load_lane")
    inputs = _fx_load.local_season_inputs(seed=0, n_games=10, heldout_every=5, n_possessions=20)
    tables = F.build_tables(inputs, n_boot=20, seed=M.SEED)
    ctx = L.run(inputs, n_folds=M.FOLDS, seed=M.SEED, n_boot=M.BOOTSTRAP, fixture=True, tables=tables)
    ctx["command"] = "tests/test_frozen_models.py"
    L.write_report(base / "load.md", ctx)
    F.write_tables(tables, base / "data_root", M.SEASON)
    return {"root": base / "data_root", "report": base / "load.md", "pred": ctx["abs_pred"]}


def test_absence_rebuild_reproduces_the_rows_the_lane_never_wrote(load_lane, tmp_path, monkeypatch):
    from analytics.eval import is_heldout_game
    monkeypatch.setenv("DATA_ROOT", str(load_lane["root"]))
    before = {p: p.stat().st_mtime_ns for p in load_lane["root"].rglob("*") if p.is_file()}
    artifacts = tmp_path / "artifacts"
    record = M.rebuild_absence(artifacts=artifacts, report=load_lane["report"])
    assert record["passed"] is True and record["status"] == "finished" and record["name"] == "absence"
    assert record == C.read_check(artifacts / "absence") and list(record["files"]) == [M.ABSENCE_FILE]
    required = [row for row in record["values"] if row["required"]]
    assert [row["what"] for row in required] == ["logistic log_loss point", "logistic brier point"]
    assert all(row["agrees"] for row in record["values"]) and all(row["agrees"] for row in record["exact"])
    assert record["n"] == len(record["values"]) == 23 and record["max_abs_diff"] <= record["tolerance"] == 5e-05
    assert record["fold_labels"]["max_abs_diff"] == 0.0 and record["code"]["tag"] == M.FROZEN_TAG
    assert (record["seed"], record["folds"], record["bootstrap"]) == (M.SEED, M.FOLDS, M.BOOTSTRAP)
    assert S.find_private((artifacts / "absence" / C.CHECK_NAME).read_text(encoding="utf-8")) == []
    assert as_site_check(record).passed and record["reasons"] == []

    rows, pred = M.load_absence(artifacts), load_lane["pred"]
    assert list(rows.columns) == M.ABSENCE_COLUMNS and len(rows) == len(pred) == record["rows"] > 0
    assert np.array_equal(rows["p_logistic"].to_numpy(), pred["p_logistic"].to_numpy(dtype=float))      # bit for bit
    assert rows["sportvu_game"].tolist() == pred["sportvu_game"].astype(str).tolist()
    assert rows["player_id"].tolist() == pred["player_id"].tolist() and rows["game_date"].tolist() == pred["game_date"].tolist()
    assert rows["absent_next"].tolist() == pred["absent_next"].astype(bool).tolist() and rows["absent_next"].any()
    assert rows.groupby("sportvu_game")["fold"].nunique().eq(1).all() and set(rows["fold"]) == set(range(record["folds"]))
    assert not any(is_heldout_game(game) for game in rows["sportvu_game"].unique())

    listed = {row["name"]: row for row in M.status(artifacts)}
    assert listed["absence"]["passed"] and listed["absence"]["exists"] and not listed["xfg"]["exists"]
    assert {p: p.stat().st_mtime_ns for p in load_lane["root"].rglob("*") if p.is_file()} == before     # DATA_ROOT untouched


def test_absence_check_fails_when_the_report_says_otherwise(load_lane, tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(load_lane["root"]))
    text = load_lane["report"].read_text(encoding="utf-8")
    point = M.parse_load_report(text)["logistic"]["log_loss"]["point"]
    cell = "| logistic on FEATURES | " + point
    assert text.count(cell) == 1
    other = tmp_path / "load.md"
    other.write_text(text.replace(cell, "| logistic on FEATURES | %.4f" % (float(point) + 0.0002)), encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    record = M.rebuild_absence(artifacts=artifacts, report=other)
    assert record["passed"] is False and record["status"] == "finished"
    assert [row["what"] for row in record["values"] if not row["agrees"]] == ["logistic log_loss point"]
    assert record["max_abs_diff"] == pytest.approx(0.0002, abs=5e-05) and record["max_abs_diff"] > record["tolerance"]
    with pytest.raises(RuntimeError, match="did not reproduce"):
        M.load_absence(artifacts)
    assert not {row["name"]: row for row in M.status(artifacts)}["absence"]["passed"]
    # a later good run makes the same folder usable again
    assert M.rebuild_absence(artifacts=artifacts, report=load_lane["report"])["passed"] is True
    assert len(M.load_absence(artifacts)) > 0


def test_absence_check_catches_wrong_fold_labels_and_changed_rows(load_lane):
    stored = pd.read_parquet(load_lane["root"] / "analytics" / "load" / M.SEASON / "player_games.parquet")
    reported = M.parse_load_report(load_lane["report"].read_text(encoding="utf-8"))
    pred, note = M.absence_oof(stored)
    labels = M.fold_labels(pred["sportvu_game"], note["folds"])
    table = M.absence_table(pred, labels)
    assert M.check_absence(table, pred, reported, note)["passed"] is True

    # swap a game of a fold that holds two games with a game of another fold (two one-game folds would only be renumbered)
    games, folds = sorted(labels), list(labels.values())
    a = next(g for g in games if folds.count(labels[g]) >= 2)
    b = next(g for g in games if labels[g] != labels[a])
    swapped = M.absence_table(pred, dict(labels, **{a: labels[b], b: labels[a]}))
    wrong = M.check_absence(swapped, pred, reported, note)
    assert wrong["passed"] is False and wrong["fold_labels"]["agrees"] is False and wrong["fold_labels"]["max_abs_diff"] > 1e-6

    moved = table.copy()
    moved.loc[0, "p_logistic"] += 1e-3
    assert M.check_absence(moved, pred, reported, note)["fold_labels"]["agrees"] is False
    assert M.check_absence(table.iloc[:-1], pred, reported, note)["passed"] is False
    with pytest.raises(RuntimeError):
        M.absence_table(pred.assign(p_logistic=np.nan), labels)


# ----------------------------------------------------------------------------- embeddings: the checks and the child

def as_the_trainer_would_write_it(committed: str) -> str:
    """The committed embeddings report as the frozen trainer writes one today with an output root: no stage 8
    preface, its command line and data root spelled out ({ARGS}, {DATA}), and the output-location line."""
    lines = committed.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("**Real data, stage 8"))
    lines = lines[:start] + lines[lines.index("## Inputs"):]
    text = "\n".join(lines) + "\n"
    text, n = re.subn(r"^Written \S+ by `[^`]+`\.$", "Written 2026-01-01T00:00:00Z by `python -m analytics.embeddings.train {ARGS}`.",
                      text, flags=re.M)
    tables = next(line for line in lines if line.startswith("Tables (docs/contract/embeddings.md)"))
    assert n == 1 and text.count("Source: $DATA_ROOT.") == 1 and text.count(tables) == 1
    return text.replace("Source: $DATA_ROOT.", "Source: {DATA}.").replace(
        tables, tables + "\nWritten under OUT_ROOT/analytics/embeddings/<season>/ (OUT_ROOT is the --out value): "
                         "embedding_neighbours, player_embeddings, player_season_features.")


def test_embeddings_check_compares_the_required_values_and_every_generated_section():
    committed = (REPO / "reports" / "embeddings.md").read_text(encoding="utf-8")
    rebuilt = as_the_trainer_would_write_it(committed).replace("{DATA}", "$DATA_ROOT").replace("{ARGS}", "--data $DATA_ROOT")
    rows = {name: int(value) for name, value in M.parse_embeddings_report(committed)["table_rows"].items()}
    record = M.check_embeddings(rebuilt, committed, rows)
    assert record["passed"] and record["n"] == 14 and record["max_abs_diff"] == 0.0 == record["tolerance"]
    assert [row["what"] for row in record["values"] if row["required"]] == [
        "player-seasons", "sufficient player-seasons", "insufficient player-seasons", "5-NN position accuracy point",
        "5-NN position accuracy lo", "5-NN position accuracy hi"]
    assert record["sections"]["only_in_committed"] == ["Stage 8: held-out next to out-of-fold (freeze-v1)"]
    assert record["sections"]["unexpected"] == [] and record["sections"]["differing_lines"] == 0
    assert {"Inputs", "Candidates", "Decision", "Outputs"} <= set(record["sections"]["compared"])

    old = M.parse_embeddings_report(committed)
    cell = "| %s | %s [%s, %s] |" % (old["chosen"], old["accuracy"]["point"], old["accuracy"]["lo"], old["accuracy"]["hi"])
    counts = "Player-seasons: %s, of which %s sufficient" % (old["player_seasons"], old["sufficient"])
    assert rebuilt.count(cell) == 1 and rebuilt.count(counts) == 1
    changes = {
        "5-NN position accuracy lo": rebuilt.replace(cell, "| %s | %s [%.3f, %s] |" % (
            old["chosen"], old["accuracy"]["point"], float(old["accuracy"]["lo"]) - 0.001, old["accuracy"]["hi"])),
        "sufficient player-seasons": rebuilt.replace(counts, "Player-seasons: %s, of which %d sufficient" % (
            old["player_seasons"], int(old["sufficient"]) - 1)),
    }
    for what, text in changes.items():
        changed = M.check_embeddings(text, committed, rows)
        assert not changed["passed"] and [row["what"] for row in changed["values"] if not row["agrees"]] == [what]
        assert changed["max_abs_diff"] > changed["tolerance"] and changed["sections"]["differing_lines"] == 1
    # a generated line outside the parsed values, a missing section, a new section, and tables of another size
    other = M.check_embeddings(rebuilt.replace("explained variance 0.", "explained variance 1."), committed, rows)
    assert not other["passed"] and all(row["agrees"] for row in other["values"]) and other["sections"]["differing_lines"] == 1
    assert other["sections"]["first_differences"][0]["section"] == "Candidates"
    assert not M.check_embeddings(rebuilt + "\n## Extra\n\ntext\n", committed, rows)["passed"]
    assert not M.check_embeddings(rebuilt[:rebuilt.index("## Assumptions")], committed, rows)["passed"]
    assert not M.check_embeddings(rebuilt, committed, dict(rows, player_embeddings=rows["player_embeddings"] - 1))["passed"]
    nothing = M.check_embeddings(rebuilt[:rebuilt.index("## Candidates")], committed, rows)
    assert not nothing["passed"] and nothing["n"] == 0 and "could not be parsed" in nothing["problem"]


def test_public_text_names_the_local_folders(tmp_path):
    root, artifacts = tmp_path / "Some Drive" / "harvest", tmp_path / "artifacts"
    text, left = M.public_text(f"--data {root} --out {artifacts}/embeddings --report {artifacts}/embeddings/report.md", artifacts, root)
    assert text == "--data $DATA_ROOT --out $V3_ARTIFACTS/embeddings --report $V3_ARTIFACTS/embeddings/report.md" and left == 0
    assert S.find_private(text) == []
    text, left = M.public_text(f"cache {Path.home()}/x", artifacts, root)        # a home path stays private, and is counted
    assert left == 1 and str(Path.home()) not in text


def test_run_child_returns_the_output_and_kills_a_child_over_the_memory_guard(tmp_path):
    code, output, peak = M.run_child([sys.executable, "-c", "print('hello'); raise SystemExit(3)"], C.MemoryGuard(), tmp_path,
                                     poll_s=0.2)
    assert (code, output.strip()) == (3, "hello") and peak >= 0
    marker = tmp_path / "finished"
    slow = [sys.executable, "-c", "import pathlib, sys, time\ntime.sleep(60)\npathlib.Path(sys.argv[1]).write_text('x')", str(marker)]
    started = time.time()
    with pytest.raises(MemoryError, match="above the 0.0 GB guard"):
        M.run_child(slow, C.MemoryGuard(limit_gb=0.0), tmp_path, poll_s=0.1)
    assert time.time() - started < 30 and not marker.exists()           # killed, not left to finish


STAND_IN = """
import argparse, json, sys
from pathlib import Path
import pandas as pd
parser = argparse.ArgumentParser()
for name in ("--data", "--seed", "--out", "--report"):
    parser.add_argument(name, required=True)
args = parser.parse_args()
here = Path(__file__).parent
text = (here / "body.md").read_text().replace("{DATA}", args.data).replace("{ARGS}", " ".join(sys.argv[1:]))
Path(args.report).parent.mkdir(parents=True, exist_ok=True)
Path(args.report).write_text(text)
folder = Path(args.out) / "analytics" / "embeddings" / "2015-16"
folder.mkdir(parents=True, exist_ok=True)
written = {}
for name, n in json.loads((here / "rows.json").read_text()).items():
    pd.DataFrame({"player_id": range(n), "season": "2015-16"}).to_parquet(folder / (name + ".parquet"), index=False)
    written[name] = [str(folder / (name + ".parquet"))]
print(text)
print(json.dumps({"written": written}))
"""


@pytest.fixture()
def stand_in_trainer(tmp_path, monkeypatch):
    """A stand-in for the frozen embeddings trainer (which needs the real frames): it takes the same four options,
    writes the committed report as the trainer would write it today, and writes three tables of the reported sizes
    in the trainer's layout. It tests the handling around the trainer, never the trainer."""
    committed = (REPO / "reports" / "embeddings.md").read_text(encoding="utf-8")
    home = tmp_path / "stand_in"
    home.mkdir()
    (home / "trainer.py").write_text(STAND_IN, encoding="utf-8")
    (home / "body.md").write_text(as_the_trainer_would_write_it(committed), encoding="utf-8")
    rows = {name: int(value) for name, value in M.parse_embeddings_report(committed)["table_rows"].items()}
    (home / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(M, "embeddings_command", lambda folder, root: [
        sys.executable, str(home / "trainer.py"), "--data", str(root), "--seed", str(M.SEED), "--out", str(folder),
        "--report", str(folder / M.REPORT_NAME)])
    return home


def test_embeddings_rebuild_redacts_copies_and_vouches_for_the_tables(stand_in_trainer, own_root, tmp_path):
    artifacts = tmp_path / "artifacts"
    record = M.rebuild_embeddings(artifacts=artifacts)
    folder = artifacts / "embeddings"
    assert record["passed"] is True and record["status"] == "finished" and record["n"] == 14
    assert as_site_check(record).passed and record["reasons"] == [] and record["training_games"] > 0
    assert all(row["agrees"] for row in record["values"]) and all(row["agrees"] for row in record["exact"])
    assert set(record["files"]) == {M.REPORT_NAME, M.LOG_NAME} | {f"{name}.parquet" for name in M.EMBEDDING_TABLES}
    report = (folder / M.REPORT_NAME).read_text(encoding="utf-8")
    assert "Source: $DATA_ROOT." in report and "--out $V3_ARTIFACTS/embeddings --report $V3_ARTIFACTS/embeddings/report.md" in report
    for name in (M.REPORT_NAME, M.LOG_NAME, C.CHECK_NAME):
        text = (folder / name).read_text(encoding="utf-8")
        assert S.find_private(text) == [] and str(own_root) not in text and str(tmp_path) not in text, name
    assert "$V3_ARTIFACTS/embeddings/analytics/embeddings/2015-16/player_embeddings.parquet" in (folder / M.LOG_NAME).read_text()
    tables = M.load_embeddings(artifacts)
    assert {name: len(table) for name, table in tables.items()} == {name: record["tables"][name]["rows"] for name in M.EMBEDDING_TABLES}
    assert list(own_root.iterdir()) == []                                  # nothing appeared under DATA_ROOT
    # a table changed after the check is refused, and the listing says why
    pd.DataFrame({"player_id": [1]}).to_parquet(folder / "player_embeddings.parquet", index=False)
    with pytest.raises(RuntimeError, match="changed after its check"):
        M.load_embeddings(artifacts)
    listed = {row["name"]: row for row in M.status(artifacts)}["embeddings"]
    assert listed["exists"] and not listed["passed"] and "changed after its check" in listed["reason"]


def test_embeddings_rebuild_fails_on_another_accuracy_or_a_failed_trainer(stand_in_trainer, tmp_path, monkeypatch):
    body = stand_in_trainer / "body.md"
    text = body.read_text(encoding="utf-8")
    old = M.parse_embeddings_report((REPO / "reports" / "embeddings.md").read_text(encoding="utf-8"))["accuracy"]
    cell = "%s [%s, %s]" % (old["point"], old["lo"], old["hi"])
    assert text.count(cell) == 1
    body.write_text(text.replace(cell, "%.3f [%s, %s]" % (float(old["point"]) + 0.002, old["lo"], old["hi"])), encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    record = M.rebuild_embeddings(artifacts=artifacts)
    assert record["passed"] is False and record["max_abs_diff"] == pytest.approx(0.002) and record["tolerance"] == 0.0
    assert [row["what"] for row in record["values"] if not row["agrees"]] == ["5-NN position accuracy point"]
    with pytest.raises(RuntimeError, match="did not reproduce"):
        M.load_embeddings(artifacts)

    listed = re.search(r"^Training games \((\d+)\): ", text, flags=re.M)
    body.write_text(text.replace(listed.group(0), "Training games (%d): %s, " % (int(listed.group(1)) + 1, HELD_ARENA)), encoding="utf-8")
    with pytest.raises(SystemExit, match="held-out"):
        M.rebuild_embeddings(artifacts=artifacts)                # a report that names a held-out game among the fitted ones
    assert C.read_check(artifacts / "embeddings")["status"] == "stopped"

    monkeypatch.setattr(M, "embeddings_command", lambda folder, root: [sys.executable, "-c", "print('boom'); raise SystemExit(2)"])
    with pytest.raises(RuntimeError, match="exit code 2"):
        M.rebuild_embeddings(artifacts=artifacts)
    check = C.read_check(artifacts / "embeddings")
    assert check["passed"] is False and check["status"] == "stopped" and "exit code 2" in check["error"]
    assert "boom" in (artifacts / "embeddings" / M.LOG_NAME).read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- xfg on the fixture season

@pytest.fixture(scope="module")
def xfg_tables():
    """The cleaned tables of a small synthetic season: 3 training games and the held-out game id."""
    from analytics.xfg import validate as V
    from tests import _fx_xfg
    tables, _summaries = V.clean(V.tables_from_season(_fx_xfg.synth_season(0, n_games=4, heldout_every=4, n_possessions=12)))
    return tables


@pytest.fixture(scope="module")
def xfg_fit(xfg_tables):
    from analytics.xfg import train as T
    train, heldout = T.assemble(dict(xfg_tables, shots=M.blank_heldout_labels(xfg_tables["shots"])))
    predict, parts = M.fit_xfg(train)
    return train, heldout.drop(columns=["made"]), predict, parts


def test_held_out_labels_are_blanked_and_training_rows_are_untouched(xfg_tables):
    from analytics.xfg import train as T
    shots = xfg_tables["shots"]
    held = shots["heldout"].to_numpy(dtype=bool)
    assert held.any() and shots.loc[held, "made"].any() and set(shots.loc[held, "sportvu_game"]) == {HELD}
    blank = M.blank_heldout_labels(shots)
    assert not blank.loc[held, "made"].any() and blank.loc[~held].equals(shots.loc[~held])
    assert blank.drop(columns=["made"]).equals(shots.drop(columns=["made"])) and shots.loc[held, "made"].any()   # a copy
    assert not M.blank_heldout_labels(shots.assign(heldout=False)).loc[held, "made"].any()       # by game id too
    a_train, a_held = T.assemble(xfg_tables)
    b_train, b_held = T.assemble(dict(xfg_tables, shots=blank))
    assert a_train.equals(b_train) and not b_held["made"].any()
    assert a_held.drop(columns=["made"]).equals(b_held.drop(columns=["made"]))


def test_xfg_fit_refuses_a_held_out_game_id(xfg_fit, monkeypatch):
    from analytics.xfg import train as T
    train = xfg_fit[0].copy()
    train.loc[train.index[0], "sportvu_game"] = HELD_ARENA
    monkeypatch.setattr(T, "fit_candidate", lambda *args, **kwargs: pytest.fail("a fit ran on a held-out game"))
    with pytest.raises(SystemExit, match="held-out"):
        M.fit_xfg(train)


def test_saved_parts_predict_exactly_as_the_frozen_predictor(xfg_fit, tmp_path):
    from analytics.xfg import features as F
    train, heldout, predict, parts = xfg_fit
    folder = tmp_path / "artifacts" / "xfg"
    folder.mkdir(parents=True)
    assert M._save_parts(folder, parts).name == M.MODEL_NAME and [p.name for p in folder.iterdir()] == [M.MODEL_NAME]
    model = M.FrozenXfg(M._read_parts(folder))
    assert (model.candidate, model.seed) == (M.XFG_CANDIDATE, M.SEED)
    assert model.needs() == F.FEATURES + ["player_id", "zone"]
    for frame in (train, heldout):
        before = frame.copy()
        assert np.array_equal(model.predict(frame), np.asarray(predict(frame), dtype=float))        # bit for bit
        assert frame.equals(before)
    assert np.array_equal(model.predict(heldout[model.needs()]), model.predict(heldout))           # extra columns are ignored
    with pytest.raises(KeyError, match="zone"):
        model.predict(heldout.drop(columns=["zone"]))
    with pytest.raises(RuntimeError):
        M.xfg_parts(lambda frame, r=0.5: None)                 # the constant predictor of a one-class fit is not the model


def test_load_xfg_needs_a_passed_check_that_vouches_for_this_very_file(xfg_fit, tmp_path):
    _train, heldout, predict, parts = xfg_fit
    artifacts = tmp_path / "artifacts"
    folder = artifacts / "xfg"
    with pytest.raises(RuntimeError, match="no rebuilt artifact"):
        M.load_xfg(artifacts)
    C.begin_rebuild(folder, "xfg")
    M._save_parts(folder, parts)
    with pytest.raises(RuntimeError, match="did not reproduce"):
        M.load_xfg(artifacts)                                   # a rebuild that started and wrote no check
    record = dict(name="xfg", compared="synthetic", n=10, n_expected=10, max_abs_diff=2e-6, tolerance=M.XFG_TOLERANCE,
                  versions=M.versions())
    C.write_check(folder, record, files=[M.MODEL_NAME])
    with pytest.raises(RuntimeError, match="did not reproduce"):
        M.load_xfg(artifacts)                                   # a check over the tolerance
    good = dict(record, max_abs_diff=1e-8)
    C.write_check(folder, good, files=[M.MODEL_NAME])
    model = M.load_xfg(artifacts)
    assert np.array_equal(model.predict(heldout), np.asarray(predict(heldout), dtype=float)) and model.check["passed"] is True
    C.write_check(folder, dict(good, versions=dict(M.versions(), scikit_learn="0.0")), files=[M.MODEL_NAME])
    with pytest.raises(RuntimeError, match="scikit-learn"):
        M.load_xfg(artifacts)
    (folder / "other.bin").write_bytes(b"something else")
    C.write_check(folder, good, files=["other.bin"])
    with pytest.raises(RuntimeError, match="does not vouch"):
        M.load_xfg(artifacts)                                   # a passed check that never hashed the model
    C.write_check(folder, good, files=[M.MODEL_NAME])
    (folder / M.MODEL_NAME).write_bytes(b"another model")
    with pytest.raises(RuntimeError, match="changed after its check"):
        M.load_xfg(artifacts)


@pytest.mark.slow
def test_xfg_rebuild_on_a_fixture_root(tmp_path, monkeypatch, capsys):
    """End to end on a synthetic DATA_ROOT: the frozen trainer writes its predictions there, the rebuild reproduces
    the held-out ones from saved parts, and a stored table that holds other predictions makes the check fail."""
    from analytics.xfg import train as T
    from tests import _fx_xfg
    root = _fx_xfg.write_fixture_root(tmp_path / "root", seed=3, n_games=6, heldout_every=3, n_possessions=12)
    # On a season this small the trainer selects whichever candidate the noise favours. The real run selected
    # XFG_CANDIDATE (reports/xfg.md), so the fixture run is told to, and its stored table is that candidate's.
    monkeypatch.setattr(T, "select", lambda scores, candidates=T.CANDIDATES: M.XFG_CANDIDATE)
    ctx = T.main(["--data", str(root), "--out", str(root), "--report", str(tmp_path / "xfg.md"), "--bootstrap", "10"])
    assert ctx["selected"] == M.XFG_CANDIDATE
    capsys.readouterr()
    monkeypatch.setenv("DATA_ROOT", str(root))
    stored_path = root / "analytics" / "xfg" / M.SEASON / "xfg_predictions.parquet"
    stored = pd.read_parquet(stored_path).drop(columns=["made"])
    stored.to_parquet(stored_path, index=False)             # the rebuild never reads the stored outcome: it is not even there
    held = stored["heldout"].to_numpy(dtype=bool)
    before = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    artifacts = tmp_path / "artifacts"

    record = M.rebuild_xfg(artifacts=artifacts)
    assert record["passed"] is True and record["status"] == "finished" and list(record["files"]) == [M.MODEL_NAME]
    assert record["n"] == record["n_expected"] == int(held.sum()) > 0 and record["reasons"] == []
    assert record["max_abs_diff"] <= record["tolerance"] == M.XFG_TOLERANCE
    assert record["rows_over_tolerance"] == 0 and record["alignment"]["keys_equal"] is True
    assert record["training"]["rows"] == int((~held).sum()) and not any(record["training"]["key_mismatches_against_stored"].values())
    assert record["saved_against_frozen_predictor"]["max_abs_diff"] == 0.0
    assert as_site_check(record).passed and record["unit"] == "probability"
    assert "no outcome column is read" in record["compared"] and "made" not in M.STORED_COLUMNS
    assert S.find_private((artifacts / "xfg" / C.CHECK_NAME).read_text(encoding="utf-8")) == []
    assert {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()} == before                # DATA_ROOT untouched

    model = M.load_xfg(artifacts)                           # a later use: predict without fitting
    _train, heldout = T.assemble(M.xfg_tables(root, C.MemoryGuard()))
    assert not heldout["made"].any()
    p = model.predict(heldout.drop(columns=["made"]))
    assert np.abs(p - stored.loc[held, "p_selected"].to_numpy(dtype=float)).max() <= M.XFG_TOLERANCE
    assert {row["name"]: row for row in M.status(artifacts)}["xfg"]["passed"]

    moved = stored.copy()
    moved.loc[held, "p_selected"] = (moved.loc[held, "p_selected"] + 1e-3).astype("float32")
    moved.to_parquet(stored_path, index=False)
    record = M.rebuild_xfg(artifacts=artifacts)
    assert record["passed"] is False and record["rows_over_tolerance"] == record["n"] == int(held.sum())
    assert record["max_abs_diff"] == pytest.approx(1e-3, rel=1e-3) and len(record["worst_rows"]) == min(10, record["n"])
    with pytest.raises(RuntimeError, match="did not reproduce"):
        M.load_xfg(artifacts)


# ----------------------------------------------------------------------------- listing, command line, hygiene

def test_status_lists_the_three_artifacts_and_any_other_checked_folder(tmp_path, capsys):
    artifacts = tmp_path / "artifacts"
    assert [(row["name"], row["exists"], row["passed"]) for row in M.status(artifacts)] == [(name, False, False) for name in M.NAMES]
    C.begin_rebuild(artifacts / "absence", "absence")
    (artifacts / "xfg").mkdir()
    (artifacts / "xfg" / "model.bin").write_bytes(b"weights")
    C.write_check(artifacts / "xfg", dict(name="xfg", compared="synthetic", n=3, n_expected=3, max_abs_diff=0.0, tolerance=1e-6),
                  files=["model.bin"])
    C.write_check(artifacts / "epv", dict(name="epv", compared="synthetic", n=3, n_expected=3, max_abs_diff=1.0, tolerance=1e-6))
    (artifacts / "notes").mkdir()                              # a folder without a check is not an artifact
    rows = {row["name"]: row for row in M.status(artifacts)}
    assert list(rows) == ["absence", "embeddings", "xfg", "epv"]
    assert rows["absence"]["exists"] and not rows["absence"]["passed"] and "rebuild started" in rows["absence"]["status"]
    assert not rows["embeddings"]["exists"] and rows["embeddings"]["reason"] == "no rebuilt artifact"
    assert rows["xfg"]["passed"] and rows["xfg"]["reason"] == "" and rows["xfg"]["n"] == 3
    assert rows["epv"]["exists"] and not rows["epv"]["passed"]
    capsys.readouterr()
    assert M.main(["status", "--artifacts", str(artifacts)]) == 0
    assert [row["name"] for row in json.loads(capsys.readouterr().out)] == list(rows)
    (artifacts / "xfg" / "model.bin").write_bytes(b"other weights")
    assert not {row["name"]: row for row in M.status(artifacts)}["xfg"]["passed"]


def test_command_line_exit_code_says_whether_the_check_passed(monkeypatch, capsys):
    monkeypatch.setitem(M.REBUILDS, "absence", lambda artifacts: {"name": "absence", "passed": True, "seen": artifacts})
    monkeypatch.setitem(M.REBUILDS, "xfg", lambda artifacts: {"name": "xfg", "passed": False})
    assert M.main(["absence", "--artifacts", "somewhere"]) == 0
    assert json.loads(capsys.readouterr().out) == {"name": "absence", "passed": True, "seen": "somewhere"}
    assert M.main(["xfg"]) == 1
    with pytest.raises(SystemExit):
        M.main(["unknown"])


def test_these_files_hold_no_dash_and_no_private_string():
    for path in (REPO / "tools" / "frozen_models.py", Path(__file__)):
        text = path.read_text(encoding="utf-8")
        assert chr(0x2013) not in text and chr(0x2014) not in text, path.name
        assert S.find_private(text) == [], path.name
