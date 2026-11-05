"""tests/test_integration.py: cross-lane integration checks (lane I6).

Fast part (always runs; one subprocess per check, timeouts everywhere, nothing imported in this process):
    for every expected lane (analytics.interfaces.EXPECTED_LANES) that is present on disk
      - validate, features and train import in a fresh interpreter and expose main(argv) whose source accepts
        --fixtures
      - the train module source carries the guard (refuse_heldout or assert_time_forward)
      - a fixture generator exists (tests/_fx_<ext>.py or analytics/fixtures.py)
      - `python -m analytics.<ext>.validate --fixtures` exits 0 within 120 s
    a lane that is not present yet is a skip with the reason, not a failure.

Slow part (@pytest.mark.slow, and only when INTEGRATION_SLOW=1 is set so the default run stays fast):
    runs features and train with --fixtures for every present lane into one OUT_ROOT under tmp_path, in the
    dependency order of analytics.registry when it is importable, then runs analytics.interfaces.check_frame
    on every interface output found and checks that no held-out token of sportvu/splits.json appears in any
    training-set manifest the trainers wrote.

    python3 -m pytest -q tests/test_integration.py                       # fast part
    INTEGRATION_SLOW=1 python3 -m pytest -q tests/test_integration.py -m slow   # slow part only
    INTEGRATION_SLOW=1 python3 -m pytest -q tests/test_integration.py    # both

The `slow` marker: pytest only honours pytest_configure from conftest.py and plugins, and this lane owns no
conftest, so the mark is created once below with the unknown-mark warning silenced; `-m slow` and
`-m "not slow"` select on it as usual.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore", pytest.PytestUnknownMarkWarning)
    slow = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analytics import interfaces  # noqa: E402

SLOW = os.environ.get("INTEGRATION_SLOW", "") not in ("", "0", "false", "no")
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


def _need(ext: str) -> None:
    if ext not in PRESENT:
        pytest.skip("lane %s not present yet: %s" % (ext, MISSING.get(ext, "not discovered")))


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


def test_discovery_lists_present_lanes():
    assert set(PRESENT) <= set(LANES) | set(PRESENT), PRESENT
    assert set(MISSING).isdisjoint(PRESENT)
    assert set(MISSING) | set(PRESENT) >= set(LANES)


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
    """One fresh interpreter per present lane importing its three step modules, WORKERS at a time."""
    return _run_many({ext: ["-c", _IMPORT_SNIPPET, ext] for ext in PRESENT}, IMPORT_TIMEOUT)


@pytest.fixture(scope="session")
def validate_runs() -> dict:
    """`python -m analytics.<ext>.validate --fixtures` for every present lane, WORKERS at a time."""
    return _run_many({ext: ["-m", "analytics.%s.validate" % ext, "--fixtures"] for ext in PRESENT}, VALIDATE_TIMEOUT)


@pytest.mark.parametrize("ext", LANES)
def test_imports_and_main(ext, import_probes):
    _need(ext)
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
    _need(ext)
    src = _source(ext, "train")
    assert re.search(r"refuse_heldout|assert_time_forward", src), \
        "analytics/%s/train.py:1 has neither refuse_heldout nor assert_time_forward" % ext


@pytest.mark.parametrize("ext", LANES)
def test_fixture_generator_exists(ext):
    _need(ext)
    assert interfaces.fixture_module(ext) is not None, \
        "neither tests/_fx_%s.py nor analytics/fixtures.py exists" % ext


@pytest.mark.parametrize("ext", LANES)
def test_validate_fixtures_exits_zero(ext, validate_runs):
    _need(ext)
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
    """Runs features then train with --fixtures for every present lane into one OUT_ROOT (dependency order).
    -> {"out_root": Path, "lanes": {ext: {"features": CompletedProcess, "train": CompletedProcess}}}"""
    if not SLOW:
        pytest.skip("slow part: set INTEGRATION_SLOW=1")
    out_root = tmp_path_factory.mktemp("out_root")
    reports = out_root / "reports"
    reports.mkdir()
    runs = {}
    for ext in _ordered(PRESENT):
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
    _need(ext)
    res = slow_runs["lanes"][ext]
    for step in ("features", "train"):
        r = res[step]
        assert r.returncode == 0, "python -m analytics.%s.%s --fixtures --out OUT_ROOT exited %d:\n%s" % (
            ext, step, r.returncode, _tail(r.stderr) or _tail(r.stdout))


@slow
@pytest.mark.parametrize("name", interfaces.INTERFACE_NAMES)
def test_slow_interface_outputs(name, slow_runs):
    e = interfaces.get(name)
    if e["producer"] not in PRESENT:
        pytest.skip("producer lane %s not present yet" % e["producer"])
    files = interfaces.find_outputs(slow_runs["out_root"], name)
    if not files:
        pytest.skip("%s wrote no %s under OUT_ROOT/%s with --fixtures (pattern %s)" % (
            e["producer"], name, e["glob"].split("/")[1], e["pattern"]))
    notes = {}
    for f in files:
        notes[str(f.relative_to(slow_runs["out_root"]))] = interfaces.check_file(name, f)
    print("interface %s: %d file(s) checked; notes: %s" % (name, len(files), notes))


SHARED = os.environ.get("INTEGRATION_SHARED", "") not in ("", "0", "false", "no")
SHARED_TIMEOUT = 300


@pytest.fixture(scope="session")
def shared_root_runs(tmp_path_factory) -> dict:
    """One DATA_ROOT written by `python -m analytics.fixtures` (lane I1), then validate, features and train with
    --data DATA_ROOT --out OUT_ROOT for every present lane in dependency order, so each consumer sees the real
    interface files of its producers. Gated by INTEGRATION_SHARED=1 (three steps per lane, 300 s each)."""
    if not SHARED:
        pytest.skip("shared-root part: set INTEGRATION_SHARED=1")
    if not interfaces.module_available("analytics.fixtures"):
        pytest.skip("analytics/fixtures.py (lane I1) is not importable yet")
    base = tmp_path_factory.mktemp("shared")
    data_root, out_root = base / "data_root", base / "out_root"
    reports = out_root / "reports"
    reports.mkdir(parents=True)
    w = _run(["-m", "analytics.fixtures", "--out", str(data_root), "--seed", "0", "--games", "6", "--heldout-every", "3"], STEP_TIMEOUT)
    runs = {}
    for ext in _ordered(PRESENT):
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
    w = shared_root_runs["writer"]
    assert w.returncode == 0, "python -m analytics.fixtures --out DATA_ROOT exited %d:\n%s" % (w.returncode, _tail(w.stderr))
    assert (shared_root_runs["data_root"] / "manifest.json").exists() or any(shared_root_runs["data_root"].rglob("*.parquet")), \
        "analytics.fixtures wrote nothing under DATA_ROOT"


@slow
@pytest.mark.parametrize("ext", LANES)
def test_shared_root_validate_features_train(ext, shared_root_runs):
    _need(ext)
    res = shared_root_runs["lanes"].get(ext)
    if res is None:
        pytest.skip("shared DATA_ROOT was not written")
    for step in ("validate", "features", "train"):
        r = res[step]
        assert r.returncode == 0, "python -m analytics.%s.%s --data DATA_ROOT --out OUT_ROOT exited %d:\n%s" % (
            ext, step, r.returncode, _tail(r.stderr) or _tail(r.stdout))


@slow
@pytest.mark.parametrize("name", interfaces.INTERFACE_NAMES)
def test_shared_root_interface_outputs(name, shared_root_runs):
    e = interfaces.get(name)
    if e["producer"] not in PRESENT:
        pytest.skip("producer lane %s not present yet" % e["producer"])
    files = interfaces.find_outputs(shared_root_runs["out_root"], name) + interfaces.find_outputs(shared_root_runs["data_root"], name)
    if not files:
        pytest.skip("%s wrote no %s under OUT_ROOT or DATA_ROOT with --data (pattern %s)" % (e["producer"], name, e["pattern"]))
    for f in files:
        interfaces.check_file(name, f)


def _run_all_passes_out_to_train() -> bool:
    """True when run_all.py's train argv carries --out (lane I5 fix); read from the source, not imported."""
    src = (ROOT / "run_all.py").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"if step == \"train\":(.*?)(?:\n    else:|\n    return)", src, re.S)
    return bool(m and "--out" in m.group(1))


@pytest.fixture(scope="session")
def run_all_run(tmp_path_factory) -> dict:
    """`python run_all.py --fixtures --all --continue-on-error --out OUT_ROOT` once, 30 min cap."""
    if not SLOW:
        pytest.skip("slow part: set INTEGRATION_SLOW=1")
    if not (ROOT / "run_all.py").is_file():
        pytest.skip("run_all.py (lane I5) is not written yet")
    if not _run_all_passes_out_to_train():
        pytest.skip("run_all.py does not pass --out to the train step yet (lane I5); interface files would not land under OUT_ROOT")
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
    if e["producer"] not in PRESENT:
        pytest.skip("producer lane %s not present yet" % e["producer"])
    files = interfaces.find_outputs(run_all_run["out_root"], name)
    assert files, "run_all.py --fixtures --all wrote no %s under OUT_ROOT (pattern %s, producer %s)" % (name, e["pattern"], e["producer"])
    for f in files:
        interfaces.check_file(name, f)


def _training_items(obj, path="") -> list:
    """Every string reachable from a key whose name mentions train (and not held-out) in a JSON manifest."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).lower()
            if "train" in key and "held" not in key:
                out += [s for s in _flatten(v)]
            else:
                out += _training_items(v, path + "/" + str(k))
    elif isinstance(obj, list):
        for v in obj:
            out += _training_items(v, path)
    return out


def _flatten(v) -> list:
    if isinstance(v, dict):
        return [str(k) for k in v] + [x for vv in v.values() for x in _flatten(vv)]
    if isinstance(v, list):
        return [x for vv in v for x in _flatten(vv)]
    return [str(v)]


@slow
def test_slow_no_heldout_in_training_manifests(slow_runs):
    from sportvu import splits
    toks = splits.heldout_tokens()
    out_root = slow_runs["out_root"]
    manifests = [p for p in out_root.rglob("*") if p.is_file() and "manifest" in p.name.lower()]
    if not manifests:
        pytest.skip("no trainer wrote a manifest under OUT_ROOT with --fixtures")
    hits = {}
    for m in manifests:
        items = []
        if m.suffix == ".json":
            try:
                items = _training_items(json.loads(m.read_text()))
            except Exception as exc:                                     # noqa: BLE001
                hits[str(m)] = ["unreadable: %s" % exc]
                continue
        elif "train" in m.name.lower():
            items = m.read_text(errors="replace").split()
        bad = sorted({s for s in items if any(t in s for t in toks)})
        if bad:
            hits[str(m.relative_to(out_root))] = bad[:5]
    assert not hits, "held-out tokens in training manifests: %s" % hits
