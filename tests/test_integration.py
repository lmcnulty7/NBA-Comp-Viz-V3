"""tests/test_integration.py: cross-lane integration checks (lane I6).

Fast part (not marked slow; one subprocess per check, timeouts everywhere, nothing imported in this process):
    every expected lane (analytics.interfaces.EXPECTED_LANES) exists, and for each
      - validate, features and train import in a fresh interpreter and expose main(argv) whose source accepts
        --fixtures
      - the train module source carries the guard (refuse_heldout or assert_time_forward)
      - a fixture generator exists (tests/_fx_<ext>.py or analytics/fixtures.py)
      - `python -m analytics.<ext>.validate --fixtures` exits 0 within 120 s

Slow part (@pytest.mark.slow, the marker registered in pytest.ini):
    1. features and train with --fixtures for every lane into one OUT_ROOT under tmp_path, in the dependency
       order of analytics.registry; every interface file the producer writes there passes check_frame.
    2. one DATA_ROOT from `python -m analytics.fixtures`, then validate, features and train with --data DATA_ROOT
       --out OUT_ROOT per lane in dependency order; every interface file passes check_frame, and the real
       held-out id the fixture root puts in its first held-out slot (12.16.2015.PHX.at.GSW) never reaches a
       lane's training outputs (it sits under heldout/, or carries heldout True and fold -1 in a single-file
       table, and no other file of a training folder names it).
    3. `python run_all.py --fixtures --all --continue-on-error --out OUT_ROOT` once; it exits 0 and every
       interface file exists under that OUT_ROOT and passes check_frame.

    python3 -m pytest -q tests/test_integration.py -m "not slow"    # fast part
    python3 -m pytest -q tests/test_integration.py -m slow          # slow part (10 to 20 minutes)
    python3 -m pytest -q tests/test_integration.py                  # both
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

slow = pytest.mark.slow                     # registered in pytest.ini

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analytics import interfaces  # noqa: E402

VALIDATE_TIMEOUT = 120
STEP_TIMEOUT = 600
IMPORT_TIMEOUT = 60
WORKERS = min(4, os.cpu_count() or 1)      # subprocesses of the fast part run concurrently (4 cores here)
PRESENT = interfaces.discover_lanes()
MISSING = interfaces.missing_lanes()
LANES = list(interfaces.EXPECTED_LANES)


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MPLBACKEND", "Agg")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # no BLAS oversubscription
        env.setdefault(k, "1")                                                  # across concurrent subprocesses
    return env


def _run(args: list[str], timeout: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run([sys.executable, *args], cwd=str(ROOT), env=_env(), capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        def _s(b):
            return b.decode("utf-8", "replace") if isinstance(b, bytes) else (b or "")
        return subprocess.CompletedProcess(exc.cmd, returncode=-999, stdout=_s(exc.stdout),
                                           stderr=("timed out after %d s\n" % timeout) + _s(exc.stderr))


def _run_many(jobs: dict, timeout: int, workers: int = WORKERS) -> dict:
    """{key: args} -> {key: CompletedProcess}, the subprocesses running `workers` at a time."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {k: pool.submit(_run, args, timeout) for k, args in jobs.items()}
        return {k: f.result() for k, f in futs.items()}


def _tail(text: str, n: int = 15) -> str:
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def _source(ext: str, step: str) -> str:
    return (interfaces.lane_dir(ext) / (step + ".py")).read_text(encoding="utf-8", errors="replace")


# ── registry sanity ──────────────────────────────────────────────────────────────────────────────
def test_registry_is_consistent():
    for name, e in interfaces.INTERFACES.items():
        assert e["producer"] in interfaces.EXPECTED_LANES, name
        assert all(c in interfaces.EXPECTED_LANES for c in e["consumers"]), name
        assert all(k in e["columns"] or k in e["optional"] for k in e["keys"]), name
        assert "<season>" in e["pattern"], name
        assert ("<game>" in e["pattern"]) == e["per_game"], name
    p = interfaces.interface_path("/o", "epv_states", "2015-16", "g1", heldout=True)
    assert str(p).endswith("analytics/epv/heldout/2015-16/g1.parquet")
    p = interfaces.interface_path("/o", "player_rapm", "2015-16")
    assert str(p).endswith("analytics/lineups/2015-16/player_rapm.parquet")
    with pytest.raises(ValueError):
        interfaces.interface_path("/o", "epv_states", "2015-16")


def test_check_frame_reports_missing_and_mistyped():
    import pandas as pd
    good = pd.DataFrame({"sportvu_game": ["g"], "possession": [1], "frame": [10], "epv_points": [1.0]})
    assert interfaces.check_frame("epv_states", good) == ["optional columns absent: heldout, fold"]
    bad = good.drop(columns=["epv_points"]).assign(frame=["x"])
    with pytest.raises(AssertionError) as exc:
        interfaces.check_frame("epv_states", bad)
    msg = str(exc.value)
    assert "missing columns: epv_points" in msg and "mistyped columns: frame" in msg


def test_check_frame_fold_convention_notes(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"sportvu_game": ["g", "g", "g"], "event": [1, 2, 3], "player_id": [7, 7, 7],
                       "p_selected": [0.4, 0.5, 0.6], "fold": [0, -1, -2]})
    train_path = tmp_path / "analytics" / "xfg" / "2015-16" / "xfg_predictions.parquet"
    notes = interfaces.check_frame("xfg_predictions", df, path=train_path)
    assert any("1 in-sample rows (fold == -2) in a training folder" in n for n in notes), notes
    held_path = tmp_path / "analytics" / "xfg" / "heldout" / "2015-16" / "xfg_predictions.parquet"
    assert not any("in-sample" in n for n in interfaces.check_frame("xfg_predictions", df, path=held_path))
    assert any("fold below -2" in n for n in interfaces.check_frame("xfg_predictions", df.assign(fold=[0, -1, -3])))
    assert (interfaces.FOLD_OOF_MIN, interfaces.FOLD_HELDOUT, interfaces.FOLD_IN_SAMPLE) == (0, -1, -2)


def test_consumers_match_the_review():
    assert interfaces.get("xfg_predictions")["consumers"] == ["decisions", "matchups"]
    assert interfaces.get("player_rapm")["consumers"] == ["betting"]


def test_discovery_lists_present_lanes():
    assert set(PRESENT) <= set(LANES), sorted(set(PRESENT) - set(LANES))
    assert not MISSING, MISSING                              # all fifteen lanes exist
    assert sorted(PRESENT) == sorted(LANES)


# ── fast part, per lane ──────────────────────────────────────────────────────────────────────────
_IMPORT_SNIPPET = r"""
import importlib, inspect, json, sys
ext = sys.argv[1]
out = {}
for step in ("validate", "features", "train"):
    name = "analytics.%s.%s" % (ext, step)
    try:
        m = importlib.import_module(name)
    except BaseException as exc:
        out[step] = {"import": "%s: %s" % (type(exc).__name__, exc)}
        continue
    main = getattr(m, "main", None)
    if not callable(main):
        out[step] = {"import": "ok", "main": "missing"}
        continue
    try:
        params = list(inspect.signature(main).parameters)
    except (TypeError, ValueError):
        params = ["?"]
    out[step] = {"import": "ok", "main": "ok", "params": params}
print(json.dumps(out))
"""


@pytest.fixture(scope="session")
def import_probes() -> dict:
    """One fresh interpreter per lane importing its three step modules, WORKERS at a time."""
    return _run_many({ext: ["-c", _IMPORT_SNIPPET, ext] for ext in LANES}, IMPORT_TIMEOUT)


@pytest.fixture(scope="session")
def validate_runs() -> dict:
    """`python -m analytics.<ext>.validate --fixtures` for every lane, WORKERS at a time."""
    return _run_many({ext: ["-m", "analytics.%s.validate" % ext, "--fixtures"] for ext in LANES}, VALIDATE_TIMEOUT)


@pytest.mark.parametrize("ext", LANES)
def test_imports_and_main(ext, import_probes):
    r = import_probes[ext]
    assert r.returncode == 0, "import probe failed for %s:\n%s" % (ext, _tail(r.stderr))
    info = json.loads(r.stdout.strip().splitlines()[-1])
    problems = []
    for step in interfaces.STANDARD_STEPS:
        d = info[step]
        if d["import"] != "ok":
            problems.append("analytics/%s/%s.py does not import: %s" % (ext, step, d["import"]))
            continue
        if d.get("main") != "ok":
            problems.append("analytics/%s/%s.py has no callable main(argv)" % (ext, step))
            continue
        if not d["params"]:
            problems.append("analytics/%s/%s.py main() takes no argv" % (ext, step))
        if "--fixtures" not in _source(ext, step):
            problems.append("analytics/%s/%s.py never mentions --fixtures" % (ext, step))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("ext", LANES)
def test_train_has_heldout_guard(ext):
    src = _source(ext, "train")
    assert re.search(r"refuse_heldout|assert_time_forward", src), \
        "analytics/%s/train.py:1 has neither refuse_heldout nor assert_time_forward" % ext


@pytest.mark.parametrize("ext", LANES)
def test_fixture_generator_exists(ext):
    assert interfaces.fixture_module(ext) is not None, \
        "neither tests/_fx_%s.py nor analytics/fixtures.py exists" % ext


@pytest.mark.parametrize("ext", LANES)
def test_validate_fixtures_exits_zero(ext, validate_runs):
    r = validate_runs[ext]
    assert r.returncode == 0, "python -m analytics.%s.validate --fixtures exited %d:\n%s" % (
        ext, r.returncode, _tail(r.stderr) or _tail(r.stdout))


# ── slow part ────────────────────────────────────────────────────────────────────────────────────
def _ordered(lanes: list[str]) -> list[str]:
    try:
        from analytics import registry
        known = [x for x in lanes if x in registry.EXTENSION_NAMES]
        order = [x for x in registry.order_for(known) if x in lanes]
        return order + [x for x in lanes if x not in order]
    except Exception:                                                     # noqa: BLE001
        return sorted(lanes)


@pytest.fixture(scope="session")
def slow_runs(tmp_path_factory) -> dict:
    """Runs features then train with --fixtures for every lane into one OUT_ROOT (dependency order).
    -> {"out_root": Path, "lanes": {ext: {"features": CompletedProcess, "train": CompletedProcess}}}"""
    out_root = tmp_path_factory.mktemp("out_root")
    reports = out_root / "reports"
    reports.mkdir()
    runs = {}
    for ext in _ordered(LANES):
        res = {}
        res["features"] = _run(["-m", "analytics.%s.features" % ext, "--fixtures", "--out", str(out_root)], STEP_TIMEOUT)
        res["train"] = _run(["-m", "analytics.%s.train" % ext, "--fixtures", "--out", str(out_root),
                             "--report", str(reports / ("%s.md" % ext)), "--folds", "3", "--seed", "0",
                             "--bootstrap", "20"], STEP_TIMEOUT)
        runs[ext] = res
    (out_root / "integration_runs.json").write_text(json.dumps(
        {e: {s: {"returncode": r.returncode, "stderr_tail": _tail(r.stderr)} for s, r in d.items()} for e, d in runs.items()},
        indent=1))
    return {"out_root": out_root, "lanes": runs}


@slow
@pytest.mark.parametrize("ext", LANES)
def test_slow_features_and_train_fixtures(ext, slow_runs):
    res = slow_runs["lanes"][ext]
    for step in ("features", "train"):
        r = res[step]
        assert r.returncode == 0, "python -m analytics.%s.%s --fixtures --out OUT_ROOT exited %d:\n%s" % (
            ext, step, r.returncode, _tail(r.stderr) or _tail(r.stdout))


@slow
@pytest.mark.parametrize("name", interfaces.INTERFACE_NAMES)
def test_slow_interface_outputs(name, slow_runs):
    e = interfaces.get(name)
    files = interfaces.find_outputs(slow_runs["out_root"], name)
    assert files, "python -m analytics.%s.train --fixtures --out OUT_ROOT wrote no %s under OUT_ROOT (pattern %s)" % (
        e["producer"], name, e["pattern"])
    notes = {}
    for f in files:
        notes[str(f.relative_to(slow_runs["out_root"]))] = interfaces.check_file(name, f)
    print("interface %s: %d file(s) checked; notes: %s" % (name, len(files), notes))


SHARED_TIMEOUT = 300


@pytest.fixture(scope="session")
def shared_root_runs(tmp_path_factory) -> dict:
    """One DATA_ROOT written by `python -m analytics.fixtures` (lane I1), then validate, features and train with
    --data DATA_ROOT --out OUT_ROOT for every lane in dependency order, so each consumer sees the real
    interface files of its producers (three steps per lane, 300 s each)."""
    base = tmp_path_factory.mktemp("shared")
    data_root, out_root = base / "data_root", base / "out_root"
    reports = out_root / "reports"
    reports.mkdir(parents=True)
    w = _run(["-m", "analytics.fixtures", "--out", str(data_root), "--seed", "0", "--games", "6", "--heldout-every", "3"], STEP_TIMEOUT)
    runs = {}
    for ext in _ordered(LANES):
        res = {}
        if w.returncode != 0:
            break
        res["validate"] = _run(["-m", "analytics.%s.validate" % ext, "--data", str(data_root), "--out", str(out_root)], SHARED_TIMEOUT)
        res["features"] = _run(["-m", "analytics.%s.features" % ext, "--data", str(data_root), "--out", str(out_root)], SHARED_TIMEOUT)
        res["train"] = _run(["-m", "analytics.%s.train" % ext, "--data", str(data_root), "--out", str(out_root),
                             "--report", str(reports / ("%s.md" % ext)), "--folds", "3", "--seed", "0", "--bootstrap", "20"], SHARED_TIMEOUT)
        runs[ext] = res
    (base / "shared_runs.json").write_text(json.dumps(
        {e: {s: {"returncode": r.returncode, "stderr_tail": _tail(r.stderr)} for s, r in d.items()} for e, d in runs.items()},
        indent=1))
    return {"data_root": data_root, "out_root": out_root, "writer": w, "lanes": runs}


@slow
def test_shared_fixture_root_is_written(shared_root_runs):
    """The premise of the held-out check: the root holds the real held-out id's frames under heldout/ and no
    training-folder copy of them (read from the layout, not from manifest.json)."""
    from analytics import fixtures
    w = shared_root_runs["writer"]
    assert w.returncode == 0, "python -m analytics.fixtures --out DATA_ROOT exited %d:\n%s" % (w.returncode, _tail(w.stderr))
    sportvu = Path(fixtures.fixture_paths(shared_root_runs["data_root"])["sportvu"])
    assert (sportvu / "heldout" / (fixtures.HELDOUT_GAME + ".parquet")).is_file(), \
        "analytics.fixtures wrote no heldout/%s.parquet under DATA_ROOT" % fixtures.HELDOUT_GAME
    assert not (sportvu / (fixtures.HELDOUT_GAME + ".parquet")).exists()


@slow
@pytest.mark.parametrize("ext", LANES)
def test_shared_root_validate_features_train(ext, shared_root_runs):
    res = shared_root_runs["lanes"].get(ext)
    assert res is not None, "shared DATA_ROOT was not written:\n%s" % _tail(shared_root_runs["writer"].stderr)
    for step in ("validate", "features", "train"):
        r = res[step]
        assert r.returncode == 0, "python -m analytics.%s.%s --data DATA_ROOT --out OUT_ROOT exited %d:\n%s" % (
            ext, step, r.returncode, _tail(r.stderr) or _tail(r.stdout))


@slow
@pytest.mark.parametrize("name", interfaces.INTERFACE_NAMES)
def test_shared_root_interface_outputs(name, shared_root_runs):
    e = interfaces.get(name)
    files = interfaces.find_outputs(shared_root_runs["out_root"], name) + interfaces.find_outputs(shared_root_runs["data_root"], name)
    assert files, "python -m analytics.%s.train --data DATA_ROOT --out OUT_ROOT wrote no %s under OUT_ROOT (pattern %s)" % (
        e["producer"], name, e["pattern"])
    for f in files:
        interfaces.check_file(name, f)


@pytest.fixture(scope="session")
def run_all_run(tmp_path_factory) -> dict:
    """`python run_all.py --fixtures --all --continue-on-error --out OUT_ROOT` once, 30 min cap."""
    out_root = tmp_path_factory.mktemp("run_all_out")
    r = _run(["run_all.py", "--fixtures", "--all", "--continue-on-error", "--out", str(out_root), "--bootstrap", "20"], 1800)
    return {"out_root": out_root, "run": r}


@slow
def test_run_all_fixtures_exits_zero(run_all_run):
    r = run_all_run["run"]
    assert r.returncode == 0, "run_all.py --fixtures --all --continue-on-error exited %d:\n%s" % (r.returncode, _tail(r.stderr) or _tail(r.stdout))


@slow
@pytest.mark.parametrize("name", interfaces.INTERFACE_NAMES)
def test_run_all_interface_file_exists(name, run_all_run):
    e = interfaces.get(name)
    files = interfaces.find_outputs(run_all_run["out_root"], name)
    assert files, "run_all.py --fixtures --all wrote no %s under OUT_ROOT (pattern %s, producer %s)" % (name, e["pattern"], e["producer"])
    for f in files:
        interfaces.check_file(name, f)



# ── held-out isolation on the shared DATA_ROOT ────────────────────────────────────────────────────
GAME_COLUMNS = ("sportvu_game", "game", "game_id")
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".md", ".yaml", ".yml"}


def heldout_game() -> str:
    """The real held-out id analytics.fixtures.write_fixture_root puts in its first held-out slot."""
    from analytics import fixtures
    return fixtures.HELDOUT_GAME


def _lane_folders(root: Path, ext: str) -> list[Path]:
    """analytics/<ext>/ and analytics/<ext>_*/ under root (epv also writes epv_features, for example)."""
    base = root / "analytics"
    if not base.is_dir():
        return []
    return [d for d in sorted(base.iterdir()) if d.is_dir() and (d.name == ext or d.name.startswith(ext + "_"))]


def _rows_of(df, game: str):
    """Boolean mask of the rows whose game column names `game` (exact id, or a pair id that contains it)."""
    import pandas as pd
    mask = pd.Series(False, index=df.index)
    for c in GAME_COLUMNS:
        if c in df.columns:
            s = df[c].astype("string")
            hit = {v for v in s.dropna().unique() if game in v}
            if hit:
                mask |= s.isin(hit).fillna(False).astype(bool)
    return mask


def _held_key(k) -> bool:
    k = str(k).lower()
    return "heldout" in k or "held_out" in k or "held-out" in k


def _json_mentions(x, game: str, path: str = "", held: bool = False) -> list[str]:
    """JSON paths that name `game` outside a held-out context. A record (object) is held-out context when it
    carries heldout true; so is anything under a key that names the held-out set (heldout_games, n_heldout ...)
    and a game-keyed entry whose value carries heldout true."""
    out = []
    if isinstance(x, dict):
        ctx = held or x.get("heldout") is True or x.get("held_out") is True
        for k, v in x.items():
            kctx = ctx or _held_key(k)
            if game in str(k) and not (kctx or (isinstance(v, dict) and v.get("heldout") is True)):
                out.append("%s/%s" % (path, k))
                continue
            out += _json_mentions(v, game, "%s/%s" % (path, k), kctx or (game in str(k)))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out += _json_mentions(v, game, "%s/%d" % (path, i), held)
    elif game in str(x) and not held:
        out.append(path or "/")
    return out


def _table_rows_problem(df, game: str):
    """None when the table has no rows of `game`; "" when its rows carry heldout True (and fold -1 when there is a
    fold column); else the problem."""
    import pandas as pd
    mask = _rows_of(df, game)
    if not mask.any():
        return None
    flagged = "heldout" in df.columns and bool(df.loc[mask, "heldout"].fillna(False).astype(bool).all())
    fold_ok = "fold" not in df.columns or bool((pd.to_numeric(df.loc[mask, "fold"], errors="coerce") == -1).all())
    if flagged and fold_ok:
        return ""
    return "%d rows of the held-out game in a training-folder table%s%s" % (
        int(mask.sum()), "" if flagged else " without heldout True", "" if fold_ok else " with fold != -1")


def heldout_placement(root: Path, ext: str, game: str) -> dict:
    """Where the held-out `game` sits in the lane's output folders under root.
    -> {"leaks": [problem, ...], "heldout_files": [rel path under heldout/ naming or holding it],
        "flagged_tables": [rel path of a training-folder table whose rows of it carry heldout True, fold -1]}

    A leak is: a file outside heldout/ named after the game; rows of the game in a parquet or csv table outside
    heldout/ that do not carry heldout True (and fold -1 when the table has a fold column); a json or jsonl file
    outside heldout/ that names the game outside a held-out record (_json_mentions); or a line of another text
    file outside heldout/ that names the game without saying held-out. Nothing here reads a manifest."""
    import pandas as pd
    out = {"leaks": [], "heldout_files": [], "flagged_tables": []}
    for folder in _lane_folders(root, ext):
        for f in sorted(p for p in folder.rglob("*") if p.is_file()):
            rel = f.relative_to(root)
            under_heldout = "heldout" in rel.parts
            if game in f.name:
                if under_heldout:
                    out["heldout_files"].append(str(rel))
                else:
                    out["leaks"].append("%s: file of the held-out game outside heldout/" % rel)
                continue
            suffix = f.suffix.lower()
            if suffix in (".parquet", ".csv"):
                try:
                    df = pd.read_parquet(f) if suffix == ".parquet" else pd.read_csv(f, low_memory=False)
                except Exception as exc:                                 # noqa: BLE001
                    out["leaks"].append("%s: unreadable (%s)" % (rel, exc))
                    continue
                problem = _table_rows_problem(df, game)
                if problem is None:
                    continue
                if under_heldout:
                    out["heldout_files"].append(str(rel))
                elif problem == "":
                    out["flagged_tables"].append(str(rel))
                else:
                    out["leaks"].append("%s: %s" % (rel, problem))
            elif not under_heldout and suffix in TEXT_SUFFIXES:
                text = f.read_text(encoding="utf-8", errors="replace")
                if game not in text:
                    continue
                if suffix in (".json", ".jsonl"):
                    try:
                        docs = [json.loads(text)] if suffix == ".json" else [json.loads(ln) for ln in text.splitlines() if ln.strip()]
                    except ValueError:
                        docs = None
                    if docs is not None:
                        paths = [m for d in docs for m in _json_mentions(d, game)]
                        if paths:
                            out["leaks"].append("%s: names the held-out game outside a held-out record at %s" % (
                                rel, ", ".join(paths[:3])))
                        else:
                            out["flagged_tables"].append(str(rel))
                        continue
                bad = [ln for ln in text.splitlines() if game in ln and not _held_key(ln.replace(" ", "-"))]
                if bad:
                    out["leaks"].append("%s: a training-folder file names the held-out game: %s" % (rel, bad[0].strip()[:120]))
    return out


def report_training_lines(reports_dir: Path, ext: str, game: str) -> list[str]:
    """Lines of reports/<ext>*.md that name training games and carry the held-out id in the training clause."""
    out = []
    for f in sorted(reports_dir.glob("*.md")) if reports_dir.is_dir() else []:
        if f.stem != ext and not f.stem.startswith(ext + "_"):
            continue
        for i, ln in enumerate(f.read_text(errors="replace").splitlines(), 1):
            low = ln.lower()
            if "training games" not in low:
                continue
            # only the training clause: a line may go on to list the held-out games set aside
            cut = min([k for k in (low.find("held-out"), low.find("heldout"), low.find("held out")) if k >= 0] or [len(ln)])
            if game in ln[:cut]:
                out.append("%s:%d: %s" % (f.name, i, ln.strip()[:160]))
    return out


@slow
@pytest.mark.parametrize("ext", LANES)
def test_shared_root_heldout_isolation(ext, shared_root_runs):
    """The fixture root's real held-out id never reaches a lane's training outputs: its rows sit under heldout/
    (or carry heldout True and fold -1 in a single-file table), no other training-folder file names it, and no
    report lists it as a training game."""
    res = shared_root_runs["lanes"].get(ext)
    assert res is not None, "shared DATA_ROOT was not written"
    game = heldout_game()
    problems = []
    seen = {}
    for label, root in (("OUT_ROOT", shared_root_runs["out_root"]), ("DATA_ROOT", shared_root_runs["data_root"])):
        r = heldout_placement(root, ext, game)
        problems += ["%s/%s" % (label, p) for p in r["leaks"]]
        seen[label] = {"heldout": len(r["heldout_files"]), "flagged": len(r["flagged_tables"])}
    problems += report_training_lines(shared_root_runs["out_root"] / "reports", ext, game)
    print("held-out %s in %s outputs: %s" % (game, ext, seen))
    assert not problems, "held-out game %s in %s training outputs:\n%s" % (game, ext, "\n".join(problems))
