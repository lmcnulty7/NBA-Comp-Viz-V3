"""Tests for run_all.py, analytics/registry.py, the Makefile target list and pytest.ini.

The extension packages themselves may be absent, so the runner is exercised against stub packages that the
tests write into a temporary directory placed on sys.path and name through a --registry JSON override."""
from __future__ import annotations

import importlib
import json
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import run_all  # noqa: E402
from analytics import registry  # noqa: E402

EXPECTED = ["xfg", "rebounds", "matchups", "gravity", "ghosting", "embeddings", "epv", "actions",
            "coverage", "decisions", "playcalling", "lineups", "winprob", "load", "betting"]
EXPECTED_DEPS = {"matchups": ["xfg"], "decisions": ["xfg", "epv"], "playcalling": ["epv", "actions", "coverage"],
                 "coverage": ["actions"], "betting": ["lineups"]}

STUB_STEP = textwrap.dedent('''
    import argparse
    from pathlib import Path

    STEP = {step!r}
    MARKS = Path({marks!r})
    EXPECTED_OUT = {expected_out!r}

    def main(argv=None):
        ap = argparse.ArgumentParser()
        ap.add_argument("--fixtures", action="store_true")
        ap.add_argument("--data")
        ap.add_argument("--out")
        ap.add_argument("--report")
        ap.add_argument("--seed", type=int, default=None)
        ap.add_argument("--bootstrap", type=int, default=None)
        a = ap.parse_args(argv)
        assert a.fixtures or a.data, "need --fixtures or --data"
        assert (STEP == "train") == (a.report is not None), "train takes --report"
        assert a.seed is not None, "every step gets --seed"
        if EXPECTED_OUT is not None:
            assert a.out is not None and Path(a.out) == Path(EXPECTED_OUT), "every step gets --out: %r" % a.out
        MARKS.mkdir(parents=True, exist_ok=True)
        with open(MARKS / "order.txt", "a") as fh:
            fh.write(STEP + "\\n")
        (MARKS / (STEP + ".done")).write_text("seed=%d bootstrap=%r" % (a.seed, a.bootstrap))
        if a.report:
            Path(a.report).parent.mkdir(parents=True, exist_ok=True)
            Path(a.report).write_text("# stub report\\n")
        print("stub " + STEP + " ran")
        import logging
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("stub").info("logged from " + STEP)
        {tail}
        return 0
''')


def make_stub(root: Path, package: str, boom_step: str | None = None, expected_out: Path | None = None) -> Path:
    """A stub package `package` with validate/features/train modules; `boom_step` raises instead of finishing.

    Each step appends its name to <root>/<package>_marks/order.txt and writes <step>.done there. With
    `expected_out`, every step asserts it received exactly that --out."""
    pkg = root / package
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    for m in [m for m in sys.modules if m == package or m.startswith(package + ".")]:
        del sys.modules[m]                      # the same stub name may have been imported from another tmp dir
    importlib.invalidate_caches()
    marks = str(root / (package + "_marks"))
    for step in ("validate", "features", "train"):
        tail = 'raise RuntimeError("stub %s exploded")' % step if step == boom_step else "pass"
        (pkg / (step + ".py")).write_text(STUB_STEP.format(
            step=step, tail=tail, marks=marks, expected_out=str(expected_out) if expected_out else None))
    reg = root / (package + "_registry.json")
    reg.write_text(json.dumps({"extensions": [
        {"name": package, "package": package, "steps": ["validate", "features", "train"],
         "depends_on": [], "phase": 1, "real_data_inputs": [], "notes": "test stub"}]}))
    return reg


@pytest.fixture
def no_fixture_root(monkeypatch):
    """Skip analytics.fixtures.write_fixture_root (about 3 s a call) where the test does not need it."""
    monkeypatch.setattr(run_all, "make_fixture_root", lambda root, seed: (None, "test: fixtures bypassed; passing --fixtures"))


def report_rows(out: Path, ext: str) -> list[str]:
    text = (out / "reports" / "run_all.md").read_text()
    return [l for l in text.splitlines() if l.startswith("| %s |" % ext)]


# ----------------------------------------------------------------------------- registry

def test_registry_has_fifteen_extensions_with_valid_dependencies():
    names = [e["name"] for e in registry.EXTENSIONS]
    assert names == EXPECTED
    assert len(set(names)) == 15
    for e in registry.EXTENSIONS:
        assert e["package"] == "analytics." + e["name"]
        assert e["steps"][0] == "validate" and e["steps"][-1] == "train" and "features" in e["steps"]
        assert 0 <= int(e["phase"]) <= 5
        assert e["real_data_inputs"] and e["notes"]
        for d in e["depends_on"]:
            assert d in names and d != e["name"]
        assert e["depends_on"] == EXPECTED_DEPS.get(e["name"], [])
        for path in e["real_data_inputs"]:
            assert not path.startswith("/")
            assert path.startswith(("sportvu_contract/", "collect/", "analytics/")), path
    assert "states" in registry.get("epv")["steps"]
    assert "stints" in registry.get("lineups")["steps"]
    assert [s["name"] for s in registry.PHASE0_STEPS] == ["gaps", "run_phase0"]
    assert "{contract}/manifest.json" in registry.PHASE0_STEPS[0]["argv"]
    assert "--steps" in registry.PHASE0_STEPS[1]["argv"]
    bref = [c for c in registry.COLLECTORS if c["name"].startswith("bref_")]
    assert bref and all(c["reachable"] for c in bref)
    nba = [c for c in registry.COLLECTORS if c["name"] in ("shots", "tracking", "context", "stints", "probe")]
    assert len(nba) == 5 and all(not c["reachable"] and c["reason"] for c in nba)
    drive = [c for c in registry.COLLECTORS if c["name"] == "drive"][0]
    assert "--sync-from" in drive["command"] and "--contract-dir" not in drive["command"]


def test_order_for_is_topological_and_stable():
    order = registry.order_for(EXPECTED)
    assert sorted(order) == sorted(EXPECTED)
    pos = {n: i for i, n in enumerate(order)}
    for e in registry.EXTENSIONS:
        for d in e["depends_on"]:
            assert pos[d] < pos[e["name"]], "%s must run before %s" % (d, e["name"])
    assert registry.order_for(list(reversed(EXPECTED))) == order        # input order does not matter
    assert registry.order_for(["decisions", "xfg"]) == ["xfg", "decisions"]
    assert registry.order_for(["betting"]) == ["betting"]              # unselected deps are not added
    assert registry.with_dependencies(["playcalling"]) == ["epv", "actions", "coverage", "playcalling"]
    assert registry.missing_dependencies(["matchups"]) == {"matchups": ["xfg"]}
    with pytest.raises(ValueError):
        registry.order_for(["nonexistent"])
    cyc = [{"name": "a", "package": "x.a", "steps": ["validate"], "depends_on": ["b"], "phase": 1},
           {"name": "b", "package": "x.b", "steps": ["validate"], "depends_on": ["a"], "phase": 1}]
    with pytest.raises(ValueError):
        registry.order_for(["a", "b"], cyc)


def test_available_uses_find_spec_without_importing():
    assert registry.available("definitely_not_a_package") is False
    assert registry.module_available("analytics.gaps") is True
    assert registry.module_available("analytics.no_such_module_here") is False
    assert "analytics.xfg" not in sys.modules or registry.available("xfg")


def test_contract_dir_and_classify_return(tmp_path):
    root = tmp_path / "root"
    (root / "sportvu_contract").mkdir(parents=True)
    assert run_all.contract_dir(root) == root                      # no manifest under the subfolder yet
    (root / "sportvu_contract" / "manifest.json").write_text("{}")
    assert run_all.contract_dir(root) == root / "sportvu_contract"
    assert run_all.contract_dir(root / "sportvu_contract") == root / "sportvu_contract"
    ok, failed = run_all.STATUS_OK, run_all.STATUS_FAILED
    assert run_all.classify_return(None)[0] == ok
    assert run_all.classify_return(0)[0] == ok
    assert run_all.classify_return(True)[0] == ok
    assert run_all.classify_return(3)[0] == failed
    assert run_all.classify_return(False)[0] == failed
    assert run_all.classify_return({"rows": 5})[0] == ok
    assert run_all.classify_return({"ok": True})[0] == ok
    assert run_all.classify_return({"ok": False, "problems": ["x"]})[0] == failed
    assert run_all.classify_return({"ok": 0, "problems": []})[0] == failed
    assert run_all.classify_return({"ok": 0})[0] == ok              # a count, not a verdict
    assert run_all.classify_return([tmp_path])[0] == ok


# ----------------------------------------------------------------------------- run_all CLI

def test_run_all_list(capsys):
    assert run_all.main(["--list"]) == 0
    out = capsys.readouterr().out
    for n in EXPECTED:
        assert re.search(r"^%s\s" % n, out, re.M), n
    assert "available" in out and "run order" in out
    assert run_all.main(["--collectors"]) == 0
    out = capsys.readouterr().out
    assert "17 requests a minute" in out and "bref_shots" in out and "--sync-from" in out


def test_run_all_usage_errors_exit_two(capsys, tmp_path):
    rc = run_all.main(["--fixtures", "--ext", "nonexistent_ext", "--out", str(tmp_path)])
    assert rc == 2
    out = capsys.readouterr().out
    assert "unknown extension 'nonexistent_ext'" in out
    assert "xfg" in out                       # the message names the known extensions
    assert out.strip().endswith("RUN FAILED")
    assert run_all.main(["--ext", "xfg"]) == 2  # neither --fixtures nor --data
    assert "RUN FAILED" in capsys.readouterr().out
    rc = run_all.main(["--data", str(tmp_path / "no_such_root"), "--ext", "xfg", "--out", str(tmp_path / "o")])
    assert rc == 2
    out = capsys.readouterr().out
    assert "does not exist" in out and out.strip().endswith("RUN FAILED")
    assert not (tmp_path / "o").exists()


def test_run_all_runs_stub_steps_in_order(tmp_path, monkeypatch, capsys):
    stubs = tmp_path / "stubs"
    out = tmp_path / "out"
    reg = make_stub(stubs, "xfg_stub", expected_out=out)
    monkeypatch.syspath_prepend(str(stubs))
    rc = run_all.main(["--fixtures", "--ext", "xfg_stub", "--registry", str(reg), "--out", str(out),
                       "--seed", "7", "--bootstrap", "3"])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert printed.strip().endswith("RUN OK")
    # the steps ran in order, each saw --out and --seed, only train saw --bootstrap
    marks = stubs / "xfg_stub_marks"
    assert (marks / "order.txt").read_text().split() == ["validate", "features", "train"]
    assert (marks / "validate.done").read_text() == "seed=7 bootstrap=None"
    assert (marks / "features.done").read_text() == "seed=7 bootstrap=None"
    assert (marks / "train.done").read_text() == "seed=7 bootstrap=3"
    # one log per step, holding the captured stdout and the lane's logging lines
    for step in ("validate", "features", "train"):
        log = out / "logs" / "run_all" / ("xfg_stub_%s.log" % step)
        assert log.exists(), log
        text = log.read_text()
        assert "stub %s ran" % step in text
        assert "logged from %s" % step in text, "logging output must reach the log of " + step
        assert "[ok in" in text
        assert "--out %s" % out in text.splitlines()[0] and "--seed 7" in text.splitlines()[0]
    # the report table
    report = out / "reports" / "run_all.md"
    text = report.read_text()
    assert "| extension | step | status | seconds | report |" in text
    rows = report_rows(out, "xfg_stub")
    assert len(rows) == 3
    assert all("| ok |" in r for r in rows)
    assert "xfg_stub.md" in rows[2]
    assert (out / "reports" / "xfg_stub.md").exists()
    assert text.rstrip().endswith("RUN OK")
    assert "3 ok, 0 failed" in text
    # fixtures either came from analytics.fixtures or were passed through; both are stated in the report
    assert "fixture" in text.lower()


def test_run_all_failed_train_exits_nonzero(tmp_path, monkeypatch, capsys, no_fixture_root):
    stubs = tmp_path / "stubs"
    out = tmp_path / "out"
    reg = make_stub(stubs, "xfg_boom", boom_step="train", expected_out=out)
    monkeypatch.syspath_prepend(str(stubs))
    rc = run_all.main(["--fixtures", "--ext", "xfg_boom", "--registry", str(reg), "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 1
    assert printed.strip().endswith("RUN FAILED")
    assert "RuntimeError: stub train exploded" in printed
    text = (out / "reports" / "run_all.md").read_text()
    rows = report_rows(out, "xfg_boom")
    assert len(rows) == 3
    assert "| validate | ok |" in rows[0] and "| features | ok |" in rows[1]
    assert "| train | failed |" in rows[2] and "stub train exploded" in rows[2]
    assert text.rstrip().endswith("RUN FAILED")
    log = (out / "logs" / "run_all" / "xfg_boom_train.log").read_text()
    assert "Traceback" in log and "[failed in" in log


def test_run_all_system_exit_message_reaches_log(tmp_path, monkeypatch, capsys, no_fixture_root):
    stubs = tmp_path / "stubs"
    pkg = stubs / "exiter"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "validate.py").write_text("import sys\ndef main(argv=None):\n    sys.exit('validate: no shots table under X')\n")
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"extensions": [{"name": "exiter", "package": "exiter", "steps": ["validate"]}]}))
    monkeypatch.syspath_prepend(str(stubs))
    out = tmp_path / "out"
    rc = run_all.main(["--fixtures", "--ext", "exiter", "--registry", str(reg), "--out", str(out)])
    capsys.readouterr()
    assert rc == 1
    log = (out / "logs" / "run_all" / "exiter_validate.log").read_text()
    assert "SystemExit: validate: no shots table under X" in log
    assert "| exiter | validate | failed |" in (out / "reports" / "run_all.md").read_text()


def test_run_all_skips_absent_package_and_continues(tmp_path, monkeypatch, capsys, no_fixture_root):
    stubs = tmp_path / "stubs"
    out = tmp_path / "out"
    make_stub(stubs, "xfg_present", expected_out=out)
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"extensions": [
        {"name": "ghost_pkg", "package": "ghost_pkg_not_there", "steps": ["validate", "train"], "depends_on": [], "phase": 1},
        {"name": "xfg_present", "package": "xfg_present", "depends_on": ["ghost_pkg"], "phase": 1}]}))
    monkeypatch.syspath_prepend(str(stubs))
    rc = run_all.main(["--fixtures", "--all", "--registry", str(reg), "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed                           # skipped is not failed
    assert "ghost_pkg" in printed and "not importable" in printed
    text = (out / "reports" / "run_all.md").read_text()
    assert "| ghost_pkg | validate | skipped |" in text
    assert "| xfg_present | train | ok |" in text
    assert "2 skipped" in text


@pytest.mark.parametrize("continue_on_error", [False, True])
def test_run_all_dependency_failure_skips_dependents(tmp_path, monkeypatch, capsys, continue_on_error, no_fixture_root):
    stubs = tmp_path / "stubs"
    out = tmp_path / "out"
    make_stub(stubs, "dep_a", boom_step="features", expected_out=out)
    make_stub(stubs, "dep_b", expected_out=out)
    make_stub(stubs, "dep_c", expected_out=out)
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"extensions": [
        {"name": "dep_a", "package": "dep_a", "phase": 1},
        {"name": "dep_b", "package": "dep_b", "phase": 1, "depends_on": ["dep_a"]},
        {"name": "dep_c", "package": "dep_c", "phase": 1}]}))
    monkeypatch.syspath_prepend(str(stubs))
    argv = ["--fixtures", "--all", "--registry", str(reg), "--out", str(out)]
    if continue_on_error:
        argv.append("--continue-on-error")
    rc = run_all.main(argv)
    printed = capsys.readouterr().out
    assert rc == 1
    assert printed.strip().endswith("RUN FAILED")
    a = report_rows(out, "dep_a")
    assert "| validate | ok |" in a[0] and "| features | failed |" in a[1] and "| train | not run |" in a[2]
    b = report_rows(out, "dep_b")
    assert len(b) == 3
    c = report_rows(out, "dep_c")
    if continue_on_error:
        assert all("| skipped |" in r and "dependency failed: dep_a" in r for r in b)
        assert "dep_b" in printed and "dependency failed (dep_a)" in printed
        assert all("| ok |" in r for r in c)             # independent extension still ran
        assert not (stubs / "dep_b_marks").exists()
    else:
        assert all("| not run |" in r for r in b) and all("| not run |" in r for r in c)
        assert not (stubs / "dep_c_marks").exists()


def test_run_all_phase_selection_and_with_deps(tmp_path, monkeypatch, capsys, no_fixture_root):
    stubs = tmp_path / "stubs"
    out = tmp_path / "out"
    make_stub(stubs, "ph_a", expected_out=out)
    make_stub(stubs, "ph_b", expected_out=out)
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"extensions": [
        {"name": "ph_a", "package": "ph_a", "phase": 2},
        {"name": "ph_b", "package": "ph_b", "phase": 3, "depends_on": ["ph_a"]}]}))
    monkeypatch.syspath_prepend(str(stubs))
    rc = run_all.main(["--fixtures", "--phase", "3", "--with-deps", "--registry", str(reg), "--out", str(out)])
    assert rc == 0, capsys.readouterr().out
    text = (out / "reports" / "run_all.md").read_text()
    assert text.index("| ph_a | validate | ok |") < text.index("| ph_b | validate | ok |")
    rc = run_all.main(["--fixtures", "--phase", "9", "--registry", str(reg), "--out", str(out)])
    assert rc == 2 and "no extension has phase 9" in capsys.readouterr().out


def test_run_all_phase0_uses_contract_folder(tmp_path, monkeypatch, capsys):
    """--phase0 --data DATA_ROOT: gaps reads DATA_ROOT/sportvu_contract/manifest.json and run_phase0 gets
    --data <contract> --out OUT_ROOT --steps <localisers>; nothing else runs."""
    root = tmp_path / "nba_harvest"
    contract = root / "sportvu_contract"
    contract.mkdir(parents=True)
    manifest = {"games": {"12.16.2015.PHX.at.GSW": {"gaps_over_2s": {"count": 1, "seconds": 12.0,
                                                                     "by_quarter": {"1": [[700.0, 688.0]]}}}}}
    (contract / "manifest.json").write_text(json.dumps(manifest))
    out = tmp_path / "out"
    # a stub run_phase0 so the test does not depend on lane I3's module being complete
    stubs = tmp_path / "stubs"
    (stubs / "analytics_stub").mkdir(parents=True)
    (stubs / "p0_stub.py").write_text(textwrap.dedent('''
        import argparse
        from pathlib import Path
        def main(argv=None):
            ap = argparse.ArgumentParser()
            ap.add_argument("--data"); ap.add_argument("--out"); ap.add_argument("--steps")
            ap.add_argument("--fixtures", action="store_true")
            a = ap.parse_args(argv)
            Path(a.out).mkdir(parents=True, exist_ok=True)
            (Path(a.out) / "p0_args.txt").write_text("%s|%s|%s" % (a.data, a.out, a.steps))
            return {"games": 1}
    '''))
    monkeypatch.syspath_prepend(str(stubs))
    monkeypatch.setattr(registry, "PHASE0_STEPS", [
        registry.PHASE0_STEPS[0],
        dict(registry.PHASE0_STEPS[1], module="p0_stub")])
    rc = run_all.main(["--phase0", "--data", str(root), "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert printed.strip().endswith("RUN OK")
    assert "phase0 gaps" in printed and "phase0 run_phase0" in printed
    assert (out / "reports" / "sportvu_gaps.md").exists()
    assert "12.16.2015.PHX.at.GSW" in (out / "reports" / "sportvu_gaps.md").read_text()
    assert (out / "p0_args.txt").read_text() == "%s|%s|%s" % (contract, out, run_all.PHASE0_LOCALISERS)
    gaps_log = (out / "logs" / "run_all" / "phase0_gaps.log").read_text().splitlines()[0]
    assert "--manifest %s" % (contract / "manifest.json") in gaps_log
    rows = report_rows(out, "phase0")
    assert len(rows) == 2 and all("| ok |" in r for r in rows)
    text = (out / "reports" / "run_all.md").read_text()
    assert "| xfg |" not in text                      # --phase0 alone runs no extension
    assert "(contract: %s)" % contract in text


def test_fixture_run_without_out_reports_under_reports_fixtures(tmp_path, monkeypatch, capsys, no_fixture_root):
    """With --fixtures and no --out the summary goes to reports/fixtures/, never over reports/<ext>.md."""
    stubs = tmp_path / "stubs"
    reg = make_stub(stubs, "fx_stub")
    monkeypatch.syspath_prepend(str(stubs))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    rc = run_all.main(["--fixtures", "--ext", "fx_stub", "--registry", str(reg)])
    capsys.readouterr()
    assert rc == 0
    assert (work / "reports" / "fixtures" / "run_all.md").exists()
    assert (work / "reports" / "fixtures" / "fx_stub.md").exists()
    assert not (work / "reports" / "run_all.md").exists()
    assert (work / "logs" / "run_all" / "fx_stub_train.log").exists()


# ----------------------------------------------------------------------------- Makefile and pytest.ini

def test_makefile_targets_match_registry():
    text = (REPO / "Makefile").read_text()
    targets = re.findall(r"^([a-zA-Z0-9_-]+):.*?## (.+)$", text, re.M)
    names = [t for t, _ in targets]
    assert names.count("help") == 1
    for required in ("test", "test-fast", "fixtures", "phase0", "all", "all-fixtures", "collect-bref", "site", "clean-logs"):
        assert required in names, required
    ext_targets = [t[4:] for t in names if t.startswith("ext-")]
    assert ext_targets == EXPECTED
    assert all(desc.strip() for _, desc in targets)         # one-line comment per target
    assert "DATA_ROOT ?= .fixture_root" in text
    assert "portfolio/hiring_manager_page.html" in text
    assert '$(or $(DATA),$(DATA_ROOT))' in text
    m = re.search(r"^EXTENSIONS = (.+)$", text, re.M)
    assert m and m.group(1).split() == EXPECTED
    # every expansion of a path variable is quoted (Drive paths have spaces)
    scan = text.replace("$(or $(DATA),$(DATA_ROOT))", "$(DATA)")
    for var in ("DATA", "OUT", "DATA_ROOT", "SITE_DIR"):
        for mm in re.finditer(r"\$\(%s\)" % var, scan):
            line = scan[scan.rfind("\n", 0, mm.start()) + 1:scan.find("\n", mm.end())]
            if line.startswith(("DATA", "OUT", "SITE_DIR", "#", "\t@echo")) or "$(if $(%s)," % var in line or "test -n" in line:
                continue
            assert scan[mm.start() - 1] == '"' and scan[mm.end():mm.end() + 1] in ('"', '/'), line
    if shutil.which("make"):
        res = subprocess.run(["make", "-s", "help"], cwd=str(REPO), capture_output=True, text=True, timeout=20)
        assert res.returncode == 0, res.stderr
        for t in names:
            assert re.search(r"^\s*%s\s" % re.escape(t), res.stdout, re.M), t
        res = subprocess.run(["make", "-n", "ext-xfg", "DATA=/tmp/my drive/nba_harvest", "OUT=/tmp/out dir"],
                             cwd=str(REPO), capture_output=True, text=True, timeout=20)
        assert 'run_all.py --data "/tmp/my drive/nba_harvest" --out "/tmp/out dir" --ext xfg' in res.stdout
        res = subprocess.run(["make", "-n", "ext-xfg"], cwd=str(REPO), capture_output=True, text=True, timeout=20)
        assert "run_all.py --fixtures --ext xfg" in res.stdout
        res = subprocess.run(["make", "-n", "phase0", "DATA=/tmp/a b"], cwd=str(REPO), capture_output=True, text=True, timeout=20)
        assert 'run_all.py --phase0 --data "/tmp/a b"' in res.stdout
        res = subprocess.run(["make", "-n", "phase0"], cwd=str(REPO), capture_output=True, text=True, timeout=20)
        assert 'run_all.py --phase0 --data ".fixture_root"' in res.stdout


def test_pytest_ini_registers_slow_marker_and_gitignore_covers_outputs():
    ini = (REPO / "pytest.ini").read_text()
    assert "[pytest]" in ini and re.search(r"^\s*slow:", ini, re.M)
    ignore = (REPO / ".gitignore").read_text().splitlines()
    for line in ("logs/", "reports/run_all.md", "reports/fixtures/", ".fixture_root/", "site/"):
        assert line in ignore, line
