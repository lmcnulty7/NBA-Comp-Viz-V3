#!/usr/bin/env python3
"""run_all.py: run the extension packages (analytics/<ext>/) in dependency order, in-process.

    python run_all.py --list                                   # the registry with availability
    python run_all.py --collectors                             # the collector run order and commands
    python run_all.py --fixtures [--ext NAME ...] [--all] [--phase N] [--out OUT_ROOT] [--seed 0]
                      [--continue-on-error] [--report-dir reports] [--with-deps] [--bootstrap N]
                      [--skip-heldout]
    python run_all.py --data DATA_ROOT [--out OUT_ROOT] [--ext NAME ...] ...
    python run_all.py --phase0 --data DATA_ROOT [--out OUT_ROOT]   # analytics.gaps then analytics.run_phase0
                      [--all | --ext NAME ...]                      # and then those extensions

DATA_ROOT is the parent of `sportvu_contract/` (the Step 1 contract: manifest.json, <game>.parquet,
heldout/) and of `collect/` (the collector tables); a DATA_ROOT that itself holds manifest.json is
accepted and treated as the contract folder.

Every step is `analytics.<ext>.<step>.main(argv)` imported at run time (never at import time here), its
stdout, stderr and logging captured to logs/run_all/<ext>_<step>.log and timed. The run stops at the first
failed step unless --continue-on-error. A package that is not importable is skipped with a printed reason,
and so are the dependents of an extension that failed. The summary table lands in reports/run_all.md
(under OUT_ROOT when --out is given). Exit code 0 when no step failed, 1 when one did, 2 on a usage error.
The last line is always RUN OK or RUN FAILED.

Step argv, per the package convention: every step gets `--data DATA_ROOT` (or `--fixtures`), `--seed N`
and `--out OUT_ROOT` when --out is given; train also gets `--report REPORT_DIR/<ext>.md`, `--bootstrap N`
when given and `--skip-heldout` when given (held-out games set aside as always but never scored; the validate,
features, states and stints steps score nothing, so only train takes it). With --fixtures and no --out, the
fixture root is written to a temporary directory through analytics.fixtures.write_fixture_root and every step
receives `--data FIXTURE_ROOT`; when that module is
absent (or fails) each step receives `--fixtures` instead and builds its own fixtures. Reports of such a
run go to reports/fixtures/ so the tracked reports/<ext>.md are never overwritten by fixture output.
Every lane report of a --fixtures run (fixture root or the per-step --fixtures fallback) carries the fixture
banner after its train step: " (fixture output, not a result)" on the `# ` title line and FIXTURE_BANNER right
under it, unless the lane already wrote the phrase (add_fixture_banner). Reports of a --data run are never touched.

--registry PATH points at a JSON file ({"extensions": [...]} in the analytics.registry shape); the tests
use it to run stub packages. Nothing here imports an extension package until it is run.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import io
import json
import logging
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analytics import registry  # noqa: E402  (plain data; imports no extension package)
from analytics.report_paths import redact_file  # noqa: E402  (standard library only)

# Extra train arguments on real data (never on --fixtures), from stage 6: where a lane's deep-model gate opened and the
# contract's torch model was kept over the sklearn stand-in, every later real run (stage 8 included) uses it.
DATA_TRAIN_ARGS = {
    "epv": ["--deep", "torch"],     # GRU kept 2026-11-12: MSE 1.2155 vs stand-in 1.2218, paired 0.0062 [0.0054, 0.0071]
}

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_NOT_RUN = "not run"

FIXTURE_N_GAMES = 4
FIXTURE_HELDOUT_EVERY = 4
FIXTURE_REPORT_DIR = "reports/fixtures"
CONTRACT_DIR = "sportvu_contract"
PHASE0_LOCALISERS = "rosters,shot_frames,rebound_frames,possessions"   # gaps runs on its own, before
FIXTURE_PHRASE = "fixture output, not a result"
FIXTURE_BANNER = ("**Fixture output, not a result.** Every number below comes from synthetic games (analytics.fixtures, "
                  "written by run_all.py --fixtures); it shows the pipeline runs end to end and nothing about NBA play.")


# ----------------------------------------------------------------------------- registry access

def load_extensions(path: str | None) -> list[dict]:
    """The registry's EXTENSIONS, or the entries of a JSON override file."""
    if not path:
        return registry.EXTENSIONS
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    entries = data["extensions"] if isinstance(data, dict) else data
    out = []
    for e in entries:
        e = dict(e)
        e.setdefault("package", "analytics." + e["name"])
        e.setdefault("steps", list(registry.STANDARD_STEPS))
        e.setdefault("depends_on", [])
        e.setdefault("phase", 0)
        e.setdefault("real_data_inputs", [])
        e.setdefault("notes", "")
        out.append(e)
    return out


def select(args, extensions: list[dict]) -> list[str]:
    """The extension names the flags select, in dependency order. ValueError on an unknown name."""
    names = list(args.ext or [])
    if args.phase is not None:
        names += [e["name"] for e in extensions if int(e["phase"]) == int(args.phase)]
        if not names:
            raise ValueError("no extension has phase %d" % args.phase)
    if args.all or (not names and args.phase is None and not args.phase0):
        names = [e["name"] for e in extensions]      # --phase0 alone runs only Phase 0
    if args.with_deps:
        return registry.with_dependencies(names, extensions)
    return registry.order_for(names, extensions)


def contract_dir(data_root: Path) -> Path:
    """The Step 1 contract folder under DATA_ROOT: DATA_ROOT/sportvu_contract when it holds a manifest,
    else DATA_ROOT itself (a DATA_ROOT given as the contract folder)."""
    sub = data_root / CONTRACT_DIR
    if (sub / "manifest.json").exists():
        return sub
    return data_root


# ----------------------------------------------------------------------------- listing

def print_list(extensions: list[dict]) -> None:
    print("%-12s %-5s %-9s %-28s %-14s %s" % ("extension", "phase", "available", "steps", "depends_on", "notes"))
    for e in extensions:
        avail = "yes" if registry.available(e["name"], extensions) else "no"
        print("%-12s %-5s %-9s %-28s %-14s %s" % (e["name"], e["phase"], avail, ",".join(e["steps"]),
                                                  ",".join(e["depends_on"]) or "-", e["notes"]))
    print()
    print("phase 0 steps: " + ", ".join("%s (%s, %s)" % (s["name"], s["module"],
                                                      "available" if registry.module_available(s["module"]) else "absent")
                                       for s in registry.PHASE0_STEPS))
    print("run order (all): " + " > ".join(registry.order_for([e["name"] for e in extensions], extensions)))


def print_collectors() -> None:
    print("Collector run order (DATA_ROOT is the parent of sportvu_contract/ and collect/, normally the Drive nba_harvest folder):")
    print(registry.BREF_RPM_NOTE)
    print()
    for i, c in enumerate(registry.COLLECTORS, 1):
        state = "reachable" if c["reachable"] else "NOT reachable"
        print("%2d. %-14s %-13s %s" % (i, c["name"], state, c["command"]))
        print("    %s. %s" % (c["reason"], c["notes"]))


# ----------------------------------------------------------------------------- running a step

def _short(text: str, n: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 3] + "..."


def classify_return(rc) -> tuple[str, str]:
    """(status, detail) for a step's return value.

    None, 0 and True are ok. A nonzero int (not bool) or False is failed. A dict is the step's summary:
    failed when its "ok" key is the boolean False, or when "ok" is a falsy number (0) and the dict also
    carries a "problems" key; absent "ok" means ok. Any other object (a list of written paths, a
    DataFrame) is ok and is noted in the log."""
    if rc is None or rc is True:
        return STATUS_OK, ""
    if rc is False:
        return STATUS_FAILED, "main returned False"
    if isinstance(rc, int):
        return (STATUS_OK, "") if rc == 0 else (STATUS_FAILED, "main returned %d" % rc)
    if isinstance(rc, dict):
        ok = rc.get("ok", True)
        bad = ok is False or (isinstance(ok, (int, float)) and not ok and "problems" in rc)
        if bad:
            return STATUS_FAILED, "main returned ok=%r: %s" % (ok, _short(rc.get("problems", ""), 120))
        return STATUS_OK, "summary"
    return STATUS_OK, "value"


def run_step(module_name: str, argv: list[str], log_path: Path) -> tuple[str, float, str]:
    """Import `module_name`, call main(argv) with stdout, stderr and root logging captured to log_path.

    Returns (status, seconds, detail). A return of None or 0, or SystemExit(0), is ok; anything else,
    including an exception, is failed with the traceback in the log. See classify_return for dicts."""
    buf = io.StringIO()
    t0 = time.perf_counter()
    status, detail = STATUS_OK, ""
    root = logging.getLogger()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    old_level = root.level
    if old_level == logging.NOTSET or old_level > logging.INFO:
        root.setLevel(logging.INFO)           # a lane's basicConfig(level=INFO) is a no-op once a handler exists
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            mod = importlib.import_module(module_name)
            main = getattr(mod, "main", None)
            if main is None:
                raise AttributeError("%s has no main(argv)" % module_name)
            rc = main(list(argv))
        status, detail = classify_return(rc)
        if detail and status == STATUS_OK:
            buf.write("\n[main returned %s]\n" % _short(repr(rc), 400))
            detail = ""
    except SystemExit as exc:                      # argparse errors and explicit exits
        code = exc.code
        if code not in (None, 0):
            buf.write("\nSystemExit: %s\n" % (code,))
            status, detail = STATUS_FAILED, _short("exit %r" % (code,))
    except KeyboardInterrupt:                      # Ctrl-C stops the whole run, never just one step
        raise
    except Exception as exc:                       # noqa: BLE001  (a failing step must never kill the runner)
        buf.write(traceback.format_exc())
        status, detail = STATUS_FAILED, _short("%s: %s" % (type(exc).__name__, exc))
    finally:
        # drop our handler and any handler a lane's logging.basicConfig bound to the captured buffer
        for h in list(root.handlers):
            if h is handler or getattr(h, "stream", None) is buf:
                root.removeHandler(h)
        root.setLevel(old_level)
    seconds = time.perf_counter() - t0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("$ %s %s\n" % (module_name, " ".join(argv)))
        fh.write(buf.getvalue())
        fh.write("\n[%s in %.2fs%s]\n" % (status, seconds, (" " + detail) if detail else ""))
    return status, seconds, detail


def step_argv(step: str, ext: str, data_root: Path | None, out_root: Path | None, report_path: Path, args) -> list[str]:
    """The argv for one step under the package convention (see the module docstring)."""
    argv = ["--data", str(data_root)] if data_root is not None else ["--fixtures"]
    if out_root is not None:
        argv += ["--out", str(out_root)]
    argv += ["--seed", str(args.seed)]
    if step == "train":
        argv += ["--report", str(report_path)]
        if args.bootstrap is not None:
            argv += ["--bootstrap", str(args.bootstrap)]
        if getattr(args, "skip_heldout", False):
            argv += ["--skip-heldout"]
        if not getattr(args, "fixtures", False):
            argv += DATA_TRAIN_ARGS.get(ext, [])
    return argv


# ----------------------------------------------------------------------------- fixtures

def make_fixture_root(root: Path, seed: int) -> tuple[Path | None, str]:
    """Write a fixture dataset with analytics.fixtures.write_fixture_root; (root, "") or (None, reason)."""
    try:
        spec = importlib.util.find_spec("analytics.fixtures")
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return None, "analytics.fixtures is absent; passing --fixtures to each step"
    try:
        mod = importlib.import_module("analytics.fixtures")
        fn = getattr(mod, "write_fixture_root")
        root.mkdir(parents=True, exist_ok=True)
        fn(root, seed=seed, n_games=FIXTURE_N_GAMES, heldout_every=FIXTURE_HELDOUT_EVERY)
        return root, ""
    except Exception as exc:                        # noqa: BLE001  (half-written lane; fall back)
        return None, "analytics.fixtures.write_fixture_root failed (%s: %s); passing --fixtures to each step" % (
            type(exc).__name__, exc)


# ----------------------------------------------------------------------------- report

def add_fixture_banner(path: Path) -> bool:
    """Label a lane report of a fixture run (docs/mac_runbook.md rule 4): append " (fixture output, not a result)" to
    the first `# ` title line when that line lacks the phrase, and insert FIXTURE_BANNER right after the title unless
    the report already carried the phrase anywhere (case-insensitive). No title line: the banner goes on top.
    True when the file changed; False when it is absent or already labelled."""
    path = Path(path)
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    had_phrase = FIXTURE_PHRASE in text.lower()
    lines = text.split("\n")
    title = next((i for i, l in enumerate(lines) if l.startswith("# ")), None)
    changed = False
    if title is not None and FIXTURE_PHRASE not in lines[title].lower():
        lines[title] = lines[title].rstrip() + " (" + FIXTURE_PHRASE + ")"
        changed = True
    if not had_phrase:
        at = 0 if title is None else title + 1
        lines[at:at] = ["", FIXTURE_BANNER] if title is not None else [FIXTURE_BANNER, ""]
        changed = True
    if changed:
        path.write_text("\n".join(lines), encoding="utf-8")
    return changed


def write_report(path: Path, rows: list[dict], header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# run_all", ""] + header + ["",
             "| extension | step | status | seconds | report | detail |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append("| %s | %s | %s | %.2f | %s | %s |" % (
            r["extension"], r["step"], r["status"], r["seconds"], r.get("report", ""),
            _short(r.get("detail", "")).replace("|", "/")))
    n_fail = sum(r["status"] == STATUS_FAILED for r in rows)
    n_ok = sum(r["status"] == STATUS_OK for r in rows)
    n_skip = sum(r["status"] == STATUS_SKIPPED for r in rows)
    lines += ["", "%d ok, %d failed, %d skipped, %d not run." % (
        n_ok, n_fail, n_skip, sum(r["status"] == STATUS_NOT_RUN for r in rows)),
        "", "RUN FAILED" if n_fail else "RUN OK", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------- phase 0

def run_phase0(args, data_root: Path | None, out_root: Path | None, report_dir: Path, log_dir: Path) -> list[dict]:
    """analytics.gaps on the contract manifest, then analytics.run_phase0 (the localisers only, since gaps
    already ran). With no data root (fixtures without analytics.fixtures) run_phase0 gets --fixtures and
    gaps is skipped for want of a manifest."""
    rows = []
    contract = contract_dir(data_root) if data_root is not None else None
    fmt = {"contract": str(contract), "data_root": str(data_root),
           "out_root": str(out_root if out_root else data_root), "report_dir": str(report_dir),
           "steps": PHASE0_LOCALISERS}
    for s in registry.PHASE0_STEPS:
        name, module = s["name"], s["module"]
        if not registry.module_available(module):
            print("phase0 %-12s skipped: %s is not importable" % (name, module))
            rows.append({"extension": "phase0", "step": name, "status": STATUS_SKIPPED, "seconds": 0.0,
                         "detail": module + " absent", "report": ""})
            continue
        if data_root is None:
            if name == "gaps":
                print("phase0 %-12s skipped: no manifest (fixtures without analytics.fixtures)" % name)
                rows.append({"extension": "phase0", "step": name, "status": STATUS_SKIPPED, "seconds": 0.0,
                             "detail": "no manifest: fixtures without analytics.fixtures", "report": ""})
                continue
            argv = ["--fixtures"] + (["--out", str(out_root)] if out_root else [])
        else:
            argv = [a.format(**fmt) for a in s["argv"]]
        log = log_dir / ("phase0_%s.log" % name)
        status, seconds, detail = run_step(module, argv, log)
        print("phase0 %-12s %-7s %7.2fs  %s%s" % (name, status, seconds, log, ("  " + detail) if detail else ""))
        rows.append({"extension": "phase0", "step": name, "status": status, "seconds": seconds,
                     "detail": detail, "report": ""})
        if status == STATUS_FAILED and not args.continue_on_error:
            break
    return rows


# ----------------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python run_all.py", description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--fixtures", action="store_true", help="run on synthetic fixtures")
    src.add_argument("--data", metavar="DATA_ROOT", help="run on the real dataset: the parent of sportvu_contract/ and collect/")
    ap.add_argument("--ext", nargs="+", metavar="NAME", help="extensions to run (dependency order is applied)")
    ap.add_argument("--all", action="store_true", help="every registered extension (the default when nothing is selected)")
    ap.add_argument("--phase", type=int, help="every extension of this phase")
    ap.add_argument("--with-deps", action="store_true", help="also run the dependencies of the selected extensions")
    ap.add_argument("--out", metavar="OUT_ROOT", help="where steps write; also holds logs/ and reports/")
    ap.add_argument("--seed", type=int, default=0, help="passed to every step")
    ap.add_argument("--bootstrap", type=int, default=None, help="passed to each train step when given")
    ap.add_argument("--skip-heldout", action="store_true",
                    help="passed to each train step: held-out games set aside as always but never scored "
                         "(docs/mac_runbook.md: every real run before stage 8)")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--report-dir", default=None,
                    help="report directory (relative to OUT_ROOT when --out is given); default reports, "
                         "or reports/fixtures for a --fixtures run without --out")
    ap.add_argument("--registry", metavar="PATH", help="JSON registry override (tests)")
    ap.add_argument("--list", action="store_true", help="print the registry with availability and exit")
    ap.add_argument("--collectors", action="store_true", help="print the collector run order and exit")
    ap.add_argument("--phase0", action="store_true", help="run analytics.gaps then analytics.run_phase0 first")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        extensions = load_extensions(args.registry)
    except (OSError, ValueError, KeyError) as exc:
        print("run_all: cannot read registry %s: %s" % (args.registry, exc))
        print("RUN FAILED")
        return 2

    if args.list:
        print_list(extensions)
        return 0
    if args.collectors:
        print_collectors()
        return 0

    if not args.fixtures and not args.data:
        print("run_all: one of --fixtures or --data DATA_ROOT is required (or --list, --collectors)")
        print("RUN FAILED")
        return 2

    try:
        names = select(args, extensions)
    except ValueError as exc:
        print("run_all: %s" % exc)
        print("RUN FAILED")
        return 2

    out_root = Path(args.out).resolve() if args.out else None
    base = out_root if out_root else Path.cwd()
    if args.report_dir:
        report_dir = Path(args.report_dir)
    elif args.fixtures and out_root is None:
        report_dir = Path(FIXTURE_REPORT_DIR)
    else:
        report_dir = Path("reports")
    if not report_dir.is_absolute():
        report_dir = base / report_dir
    log_dir = base / "logs" / "run_all"
    header = ["command: python run_all.py " + " ".join(argv if argv is not None else sys.argv[1:])]

    # Data root: real, or fixtures written by analytics.fixtures, or --fixtures passed through.
    tmp_holder = None
    data_root: Path | None = None
    if args.data:
        data_root = Path(args.data).resolve()
        if not data_root.exists():
            print("run_all: DATA_ROOT %s does not exist" % data_root)
            print("RUN FAILED")
            return 2
        header.append("data root: %s (contract: %s)" % (data_root, contract_dir(data_root)))
    else:
        if out_root is not None:
            fixture_root = out_root / "fixture_root"
        else:
            tmp_holder = tempfile.TemporaryDirectory(prefix="run_all_fixtures_")
            fixture_root = Path(tmp_holder.name) / "fixture_root"
        data_root, reason = make_fixture_root(fixture_root, args.seed)
        if data_root is None:
            print("fixtures: " + reason)
            header.append("fixtures: " + reason)
        else:
            print("fixtures: written to %s" % data_root)
            header.append("fixture root: %s" % data_root)

    rows: list[dict] = []
    failed = False

    if args.phase0:
        rows += run_phase0(args, data_root, out_root, report_dir, log_dir)
        failed = any(r["status"] == STATUS_FAILED for r in rows)
        if failed and not args.continue_on_error:
            names = []

    missing = registry.missing_dependencies(names, extensions)
    for n, deps in sorted(missing.items()):
        print("note: %s depends on %s, not selected; their outputs must already exist" % (n, ", ".join(deps)))

    by_name = {e["name"]: e for e in extensions}
    ext_status: dict[str, str] = {}
    stop = False
    for name in names:
        e = by_name[name]
        package = e["package"]
        report_path = report_dir / (name + ".md")
        if stop:
            for step in e["steps"]:
                rows.append({"extension": name, "step": step, "status": STATUS_NOT_RUN, "seconds": 0.0,
                             "report": "", "detail": "stopped after an earlier failure"})
            continue
        if not registry.module_available(package):
            print("%-12s skipped: package %s is not importable" % (name, package))
            for step in e["steps"]:
                rows.append({"extension": name, "step": step, "status": STATUS_SKIPPED, "seconds": 0.0,
                             "report": "", "detail": "package %s absent" % package})
            ext_status[name] = STATUS_SKIPPED
            continue
        bad_deps = [d for d in e["depends_on"] if ext_status.get(d) == STATUS_FAILED]
        if bad_deps:
            print("%-12s skipped: dependency failed (%s)" % (name, ", ".join(bad_deps)))
            for step in e["steps"]:
                rows.append({"extension": name, "step": step, "status": STATUS_SKIPPED, "seconds": 0.0,
                             "report": "", "detail": "dependency failed: " + ", ".join(bad_deps)})
            ext_status[name] = STATUS_SKIPPED
            continue
        ext_status[name] = STATUS_OK
        for i, step in enumerate(e["steps"]):
            module = package + "." + step
            log = log_dir / ("%s_%s.log" % (name, step))
            if not registry.module_available(module):
                print("%-12s %-9s skipped: %s is not importable" % (name, step, module))
                rows.append({"extension": name, "step": step, "status": STATUS_SKIPPED, "seconds": 0.0,
                             "report": "", "detail": "module %s absent" % module})
                continue
            argv_step = step_argv(step, name, data_root, out_root, report_path, args)
            status, seconds, detail = run_step(module, argv_step, log)
            if step == "train" and args.fixtures:      # never a --data run: those reports stay as the lane wrote them
                add_fixture_banner(report_path)
            # the repo is public: no DATA_ROOT, repo or home path (the Drive path names the account) in a report
            redact_file(report_path, data_root=data_root)
            rows.append({"extension": name, "step": step, "status": status, "seconds": seconds,
                         "report": str(report_path) if step == "train" and status == STATUS_OK else "",
                         "detail": detail})
            print("%-12s %-9s %-7s %7.2fs  %s%s" % (name, step, status, seconds, log, ("  " + detail) if detail else ""))
            if status == STATUS_FAILED:
                failed = True
                ext_status[name] = STATUS_FAILED
                for later in e["steps"][i + 1:]:
                    rows.append({"extension": name, "step": later, "status": STATUS_NOT_RUN, "seconds": 0.0,
                                 "report": "", "detail": "earlier step failed"})
                if not args.continue_on_error:
                    stop = True
                break

    report = report_dir / "run_all.md"
    write_report(report, rows, header)
    print("report: %s" % report)
    if tmp_holder is not None:
        tmp_holder.cleanup()
    print("RUN FAILED" if failed else "RUN OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
