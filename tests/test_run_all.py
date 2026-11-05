"""Tests for run_all.py, analytics/registry.py and the Makefile target list.

The extension packages themselves may be absent, so the runner is exercised against stub packages that the
tests write into a temporary directory placed on sys.path and name through a --registry JSON override."""
from __future__ import annotations

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
EXPECTED_DEPS = {"matchups": ["xfg"], "decisions": ["xfg", "epv"], "playcalling": ["epv", "actions"],
                 "coverage": ["actions"], "betting": ["lineups"]}

STUB_STEP = textwrap.dedent('''
    import argparse
    from pathlib import Path

    STEP = {step!r}
    MARKS = Path({marks!r})

    def main(argv=None):
        ap = argparse.ArgumentParser()
        ap.add_argument("--fixtures", action="store_true")
        ap.add_argument("--data")
        ap.add_argument("--out")
        ap.add_argument("--report")
        ap.add_argument("--seed", type=int, default=0)
        ap.add_argument("--bootstrap", type=int, default=None)
        a = ap.parse_args(argv)
        assert a.fixtures or a.data, "need --fixtures or --data"
        assert (STEP == "train") == (a.report is not None), "train takes --report, the others --out"
        MARKS.mkdir(parents=True, exist_ok=True)
        with open(MARKS / "order.txt", "a") as fh:
            fh.write(STEP + "\\n")
        (MARKS / (STEP + ".done")).write_text("seed=%d" % a.seed)
        if a.report:
            Path(a.report).parent.mkdir(parents=True, exist_ok=True)
            Path(a.report).write_text("# stub report\\n")
        print("stub " + STEP + " ran")
        {tail}
        return 0
''')


def make_stub(root: Path, package: str, boom_step: str | None = None) -> Path:
    """A stub package `package` with validate/features/train modules; `boom_step` raises instead of finishing.

    Each step appends its name to <root>/<package>_marks/order.txt and writes <step>.done there."""
    pkg = root / package
    marks = str(root / (package + "_marks"))
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    for step in ("validate", "features", "train"):
        tail = 'raise RuntimeError("stub %s exploded")' % step if step == boom_step else "pass"
        (pkg / (step + ".py")).write_text(STUB_STEP.format(step=step, tail=tail, marks=marks))
    reg = root / (package + "_registry.json")
    reg.write_text(json.dumps({"extensions": [
        {"name": package, "package": package, "steps": ["validate", "features", "train"],
         "depends_on": [], "phase": 1, "real_data_inputs": [], "notes": "test stub"}]}))
    return reg


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
    assert "states" in registry.get("epv")["steps"]
    assert "stints" in registry.get("lineups")["steps"]
    assert [s["name"] for s in registry.PHASE0_STEPS] == ["gaps", "run_phase0"]
    bref = [c for c in registry.COLLECTORS if c["name"].startswith("bref_")]
    assert bref and all(c["reachable"] for c in bref)
    nba = [c for c in registry.COLLECTORS if c["name"] in ("shots", "tracking", "context", "stints", "probe")]
    assert len(nba) == 5 and all(not c["reachable"] and c["reason"] for c in nba)


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
    assert registry.with_dependencies(["playcalling"]) == ["epv", "actions", "playcalling"]
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


# ----------------------------------------------------------------------------- run_all CLI

def test_run_all_list(capsys):
    assert run_all.main(["--list"]) == 0
    out = capsys.readouterr().out
    for n in EXPECTED:
        assert re.search(r"^%s\s" % n, out, re.M), n
    assert "available" in out and "run order" in out
    assert run_all.main(["--collectors"]) == 0
    out = capsys.readouterr().out
    assert "17 requests a minute" in out and "bref_shots" in out


def test_run_all_unknown_extension_exits_nonzero(capsys, tmp_path):
    rc = run_all.main(["--fixtures", "--ext", "nonexistent_ext", "--out", str(tmp_path)])
    assert rc != 0
    out = capsys.readouterr().out
    assert "unknown extension 'nonexistent_ext'" in out
    assert "xfg" in out                       # the message names the known extensions
    assert out.strip().endswith("RUN FAILED")
    assert run_all.main(["--ext", "xfg"]) == 2  # neither --fixtures nor --data
    assert "RUN FAILED" in capsys.readouterr().out


def test_run_all_runs_stub_steps_in_order(tmp_path, monkeypatch, capsys):
    stubs = tmp_path / "stubs"
    reg = make_stub(stubs, "xfg_stub")
    monkeypatch.syspath_prepend(str(stubs))
    out = tmp_path / "out"
    rc = run_all.main(["--fixtures", "--ext", "xfg_stub", "--registry", str(reg), "--out", str(out), "--seed", "7"])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert printed.strip().endswith("RUN OK")
    # the steps ran in order and saw the seed
    marks = stubs / "xfg_stub_marks"
    assert (marks / "order.txt").read_text().split() == ["validate", "features", "train"]
    assert (marks / "train.done").read_text() == "seed=7"
    # one log per step, holding the captured stdout
    for step in ("validate", "features", "train"):
        log = out / "logs" / "run_all" / ("xfg_stub_%s.log" % step)
        assert log.exists(), log
        assert "stub %s ran" % step in log.read_text()
        assert "[ok in" in log.read_text()
    # the report table
    report = out / "reports" / "run_all.md"
    text = report.read_text()
    assert "| extension | step | status | seconds | report |" in text
    rows = [l for l in text.splitlines() if l.startswith("| xfg_stub |")]
    assert len(rows) == 3
    assert all("| ok |" in r for r in rows)
    assert "xfg_stub.md" in rows[2]
    assert (out / "reports" / "xfg_stub.md").exists()
    assert text.rstrip().endswith("RUN OK")
    assert "3 ok, 0 failed" in text
    # fixtures either came from analytics.fixtures or were passed through; both are stated in the report
    assert "fixture" in text.lower()


def test_run_all_failed_train_exits_nonzero(tmp_path, monkeypatch, capsys):
    stubs = tmp_path / "stubs"
    reg = make_stub(stubs, "xfg_boom", boom_step="train")
    monkeypatch.syspath_prepend(str(stubs))
    out = tmp_path / "out"
    rc = run_all.main(["--fixtures", "--ext", "xfg_boom", "--registry", str(reg), "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 1
    assert printed.strip().endswith("RUN FAILED")
    assert "RuntimeError: stub train exploded" in printed
    text = (out / "reports" / "run_all.md").read_text()
    rows = [l for l in text.splitlines() if l.startswith("| xfg_boom |")]
    assert len(rows) == 3
    assert "| validate | ok |" in rows[0] and "| features | ok |" in rows[1]
    assert "| train | failed |" in rows[2] and "stub train exploded" in rows[2]
    assert text.rstrip().endswith("RUN FAILED")
    log = (out / "logs" / "run_all" / "xfg_boom_train.log").read_text()
    assert "Traceback" in log and "[failed in" in log


def test_run_all_skips_absent_package_and_continues(tmp_path, monkeypatch, capsys):
    stubs = tmp_path / "stubs"
    make_stub(stubs, "xfg_present")
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"extensions": [
        {"name": "ghost_pkg", "package": "ghost_pkg_not_there", "steps": ["validate", "train"], "depends_on": [], "phase": 1},
        {"name": "xfg_present", "package": "xfg_present", "depends_on": ["ghost_pkg"], "phase": 1}]}))
    monkeypatch.syspath_prepend(str(stubs))
    out = tmp_path / "out"
    rc = run_all.main(["--fixtures", "--all", "--registry", str(reg), "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed                           # skipped is not failed
    assert "ghost_pkg" in printed and "not importable" in printed
    text = (out / "reports" / "run_all.md").read_text()
    assert "| ghost_pkg | validate | skipped |" in text
    assert "| xfg_present | train | ok |" in text
    assert "2 skipped" in text


def test_run_all_phase_selection_and_with_deps(tmp_path, monkeypatch, capsys):
    stubs = tmp_path / "stubs"
    make_stub(stubs, "dep_a")
    make_stub(stubs, "dep_b")
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"extensions": [
        {"name": "dep_a", "package": "dep_a", "phase": 2},
        {"name": "dep_b", "package": "dep_b", "phase": 3, "depends_on": ["dep_a"]}]}))
    monkeypatch.syspath_prepend(str(stubs))
    out = tmp_path / "out"
    rc = run_all.main(["--fixtures", "--phase", "3", "--with-deps", "--registry", str(reg), "--out", str(out)])
    assert rc == 0, capsys.readouterr().out
    text = (out / "reports" / "run_all.md").read_text()
    a = text.index("| dep_a | validate | ok |")
    b = text.index("| dep_b | validate | ok |")
    assert a < b
    rc = run_all.main(["--fixtures", "--phase", "9", "--registry", str(reg), "--out", str(out)])
    assert rc == 2 and "no extension has phase 9" in capsys.readouterr().out


# ----------------------------------------------------------------------------- Makefile

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
    m = re.search(r"^EXTENSIONS = (.+)$", text, re.M)
    assert m and m.group(1).split() == EXPECTED
    # the Makefile parses and `make help` names every target, when make is installed
    if shutil.which("make"):
        res = subprocess.run(["make", "-s", "help"], cwd=str(REPO), capture_output=True, text=True, timeout=20)
        assert res.returncode == 0, res.stderr
        for t in names:
            assert re.search(r"^\s*%s\s" % re.escape(t), res.stdout, re.M), t
        res = subprocess.run(["make", "-n", "ext-xfg", "DATA=/tmp/root"], cwd=str(REPO),
                             capture_output=True, text=True, timeout=20)
        assert "run_all.py --data /tmp/root --ext xfg" in res.stdout
        res = subprocess.run(["make", "-n", "ext-xfg"], cwd=str(REPO), capture_output=True, text=True, timeout=20)
        assert "run_all.py --fixtures --ext xfg" in res.stdout
