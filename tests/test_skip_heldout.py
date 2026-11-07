"""Tests for --skip-heldout (docs/mac_runbook.md stage 4a): one test per trainer, the run_all passthrough and the
SKIP_HELDOUT make variable.

With the flag a trainer sets its held-out rows aside exactly as without it (same split, same guards, same
training rows) but never scores them and writes no held-out prediction; its report carries SKIP_HELDOUT_NOTE in
the held-out section. Each lane test runs the trainer on its smallest fixture, checks the fixture really has
held-out rows (so the test is not vacuous), that the held-out scoring path never ran and that the training set
matches a run without the flag."""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analytics.eval import SKIP_HELDOUT_NOTE, add_skip_heldout_arg, skip_heldout_lines  # noqa: E402


# ----------------------------------------------------------------------------- shared helper, run_all and Makefile

def test_skip_heldout_helper_parses_and_carries_the_exact_phrase():
    import argparse

    assert SKIP_HELDOUT_NOTE == "held-out games not scored (--skip-heldout)"
    ap = add_skip_heldout_arg(argparse.ArgumentParser())
    assert ap.parse_args([]).skip_heldout is False
    assert ap.parse_args(["--skip-heldout"]).skip_heldout is True
    assert SKIP_HELDOUT_NOTE in "\n".join(skip_heldout_lines(3))
    assert SKIP_HELDOUT_NOTE in "\n".join(skip_heldout_lines())


_RUN_ALL_STUB = '''
import json
from pathlib import Path

def main(argv=None):
    with open(Path({marks!r}), "a") as fh:
        fh.write(json.dumps([{step!r}, list(argv)]) + "\\n")
    return 0
'''


def _run_all_stub(root: Path, package: str) -> tuple:
    pkg = root / package
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    for m in [m for m in sys.modules if m == package or m.startswith(package + ".")]:
        del sys.modules[m]
    importlib.invalidate_caches()
    marks = root / "argv.jsonl"
    for step in ("validate", "features", "train"):
        (pkg / (step + ".py")).write_text(_RUN_ALL_STUB.format(marks=str(marks), step=step))
    reg = root / "registry.json"
    reg.write_text(json.dumps({"extensions": [
        {"name": package, "package": package, "steps": ["validate", "features", "train"], "depends_on": [],
         "phase": 1, "real_data_inputs": [], "notes": "test stub"}]}))
    return reg, marks


@pytest.mark.parametrize("flag", [True, False])
def test_run_all_passes_skip_heldout_to_train_steps_only(tmp_path, monkeypatch, capsys, flag):
    import run_all

    monkeypatch.setattr(run_all, "make_fixture_root", lambda root, seed: (None, "test: fixtures bypassed"))
    stubs = tmp_path / "stubs"
    reg, marks = _run_all_stub(stubs, "skipho_stub")
    monkeypatch.syspath_prepend(str(stubs))
    argv = ["--fixtures", "--ext", "skipho_stub", "--registry", str(reg), "--out", str(tmp_path / "out")]
    rc = run_all.main(argv + (["--skip-heldout"] if flag else []))
    assert rc == 0, capsys.readouterr().out
    seen = dict(json.loads(l) for l in marks.read_text().splitlines())
    assert sorted(seen) == ["features", "train", "validate"]
    assert ("--skip-heldout" in seen["train"]) is flag
    assert "--skip-heldout" not in seen["validate"] and "--skip-heldout" not in seen["features"]
    assert "--report" in seen["train"]


def test_run_all_step_argv_and_parser():
    import run_all

    args = run_all.build_parser().parse_args(["--fixtures", "--skip-heldout"])
    assert args.skip_heldout is True
    assert "--skip-heldout" in run_all.step_argv("train", "xfg", None, None, Path("r.md"), args)
    assert "--skip-heldout" not in run_all.step_argv("features", "xfg", None, None, Path("r.md"), args)
    args = run_all.build_parser().parse_args(["--fixtures"])
    assert "--skip-heldout" not in run_all.step_argv("train", "xfg", None, None, Path("r.md"), args)


@pytest.mark.skipif(not shutil.which("make"), reason="make not installed")
def test_makefile_skip_heldout_variable():
    def dry(*args):
        res = subprocess.run(["make", "-n", *args], cwd=str(REPO), capture_output=True, text=True, timeout=20)
        assert res.returncode == 0, res.stderr
        return res.stdout

    assert "run_all.py --fixtures --skip-heldout --ext xfg" in dry("ext-xfg", "SKIP_HELDOUT=1")
    assert 'run_all.py --data "/tmp/a b" --skip-heldout --ext betting' in dry("ext-betting", "DATA=/tmp/a b", "SKIP_HELDOUT=1")
    assert 'run_all.py --data "/tmp/a b" --all --continue-on-error --skip-heldout' in dry("all", "DATA=/tmp/a b", "SKIP_HELDOUT=1")
    assert "run_all.py --fixtures --all --continue-on-error --skip-heldout" in dry("all-fixtures", "SKIP_HELDOUT=1")
    for args in (("ext-xfg",), ("ext-xfg", "SKIP_HELDOUT=0"), ("all-fixtures",), ("all", "DATA=/tmp/x")):
        assert "--skip-heldout" not in dry(*args), args


# ----------------------------------------------------------------------------- lanes: xfg, train_xfg (Phase 1 baseline), rebounds, matchups

def _g1_section(text: str, heading: str) -> str:
    """The lines of the report section whose heading starts with ``heading``, up to the next ## heading."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(heading))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start:end])


def test_skip_heldout_xfg(tmp_path, monkeypatch, capsys):
    from analytics.xfg import train as T
    held = "12.16.2015.PHX.at.GSW"
    base = ["--fixtures", "--folds", "3", "--bootstrap", "5", "--fixture-games", "4", "--fixture-possessions", "12"]
    ref = T.main(base + ["--report", str(tmp_path / "ref.md"), "--out", str(tmp_path / "ref")])
    capsys.readouterr()

    def boom(*a, **k):
        raise AssertionError("held-out rows were scored under --skip-heldout")
    monkeypatch.setattr(T, "point_metrics", boom)
    real_fit = T.fit_candidate
    fits = []
    monkeypatch.setattr(T, "fit_candidate", lambda name, train, seed=0: (fits.append(len(train)), real_fit(name, train, seed))[1])
    ctx = T.main(base + ["--report", str(tmp_path / "skip.md"), "--out", str(tmp_path / "skip"), "--skip-heldout"])
    printed = capsys.readouterr().out
    text = (tmp_path / "skip.md").read_text()
    assert ctx["heldout_rows"] > 0 and ctx["heldout_game_ids"] == [held]           # the fixture has a held-out game
    assert SKIP_HELDOUT_NOTE in text and SKIP_HELDOUT_NOTE in printed
    sec = _g1_section(text, "## Held-out score")
    assert "| candidate |" not in sec and "log loss" not in sec
    assert ctx["heldout_p"] == {} and ctx["heldout_scores"] == {} and ctx["heldout_block"] == ""
    assert all(n < ctx["train_rows"] for n in fits)                                   # no refit on all training rows
    out = pd.read_parquet(tmp_path / "skip" / "analytics" / "xfg" / "2015-16" / "xfg_predictions.parquet")
    assert not out["heldout"].any() and (out["fold"] >= 0).all() and held not in set(out["sportvu_game"])
    ref_out = pd.read_parquet(tmp_path / "ref" / "analytics" / "xfg" / "2015-16" / "xfg_predictions.parquet")
    pd.testing.assert_frame_equal(out, ref_out[~ref_out["heldout"]].reset_index(drop=True))
    assert ctx["train_rows"] == ref["train_rows"] and ctx["heldout_rows"] == ref["heldout_rows"]


def _train_xfg_shots(path, n_games=4, per_game=60, held="12.16.2015.PHX.at.GSW"):
    """A small 10a-shaped shots table: n_games training games plus the held-out SportVU game."""
    rng = np.random.default_rng(0)
    rows = []
    for g in ["01.%02d.2016.AAA.at.BBB" % (d + 1) for d in range(n_games)] + [held]:
        for _ in range(per_game):
            x, y = rng.uniform(-240, 240), rng.uniform(-5, 280)
            dist = float(np.hypot(x, y) / 10)
            rows.append({"game": g, "sportvu_game": g, "period": int(rng.integers(1, 5)), "clock_s": float(rng.uniform(0, 720)),
                         "made": bool(rng.uniform() < 1 / (1 + np.exp(-(1.0 - 0.06 * dist)))), "loc_x": round(x, 1),
                         "loc_y": round(y, 1), "shot_type": "3PT" if dist >= 22 else "2PT", "shot_distance_ft": float(round(dist)),
                         "court_x": 5.25 + y / 10, "court_y": 25 + x / 10, "split": "heldout_game" if g == held else None,
                         "heldout": g == held, "source": "bref", "release_frame": None, "closest_defender_dist_ft": None})
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_skip_heldout_train_xfg(tmp_path, monkeypatch, capsys):
    from analytics import train_xfg as tx
    src = _train_xfg_shots(tmp_path / "shots.parquet")
    base = ["--shots", str(src), "--folds", "3", "--bootstrap", "5"]
    ref = tx.main(base + ["--report", str(tmp_path / "ref.md")])
    capsys.readouterr()
    real_fit = tx.fit_models
    seen = []
    monkeypatch.setattr(tx, "fit_models", lambda train: (seen.append(sorted(train["sportvu_game"].unique())), real_fit(train))[1])
    ctx = tx.main(base + ["--report", str(tmp_path / "skip.md"), "--skip-heldout"])
    printed = capsys.readouterr().out
    text = (tmp_path / "skip.md").read_text()
    assert ctx["heldout_rows"] == 60 and ctx["heldout_games"] == 1                  # the input has a held-out game
    assert SKIP_HELDOUT_NOTE in text and SKIP_HELDOUT_NOTE in printed
    sec = _g1_section(text, "## Held-out rows")
    assert "| model |" not in sec and "point estimates" not in sec
    assert ctx["heldout_scores"] == {}
    assert len(seen) == 3 and all(len(g) < 4 for g in seen)                         # three folds, no refit on all 4 games
    assert ctx["train_game_ids"] == ref["train_game_ids"] and repr(ctx["scores"]) == repr(ref["scores"])
    pd.testing.assert_frame_equal(ctx["calibration"], ref["calibration"])


def test_skip_heldout_rebounds(tmp_path, monkeypatch, capsys):
    from analytics.rebounds import train as T
    base = ["--fixtures", "--fixture-games", "4", "--fixture-possessions", "8", "--bootstrap", "5", "--folds", "3"]
    ref = T.main(base + ["--report", str(tmp_path / "ref.md"), "--out", str(tmp_path / "ref")])
    capsys.readouterr()
    real_fit = T.fit_candidates

    def fit(train, seed=0, only=None):
        if only is not None:
            raise AssertionError("the held-out refit ran under --skip-heldout")
        return real_fit(train, seed, only)
    monkeypatch.setattr(T, "fit_candidates", fit)
    ctx = T.main(base + ["--report", str(tmp_path / "skip.md"), "--out", str(tmp_path / "skip"), "--skip-heldout"])
    printed = capsys.readouterr().out
    text = (tmp_path / "skip.md").read_text()
    assert ctx["heldout_rows"] > 0 and ctx["heldout_games"] == 1
    assert SKIP_HELDOUT_NOTE in text and SKIP_HELDOUT_NOTE in printed
    sec = _g1_section(text, "## Held-out games")
    assert "| model |" not in sec and "point estimates" not in sec
    assert ctx["heldout_scores"] == {}
    pr = ctx["predictions"]
    assert not pr["heldout"].any() and (pr["fold"] >= 0).all() and not set(pr["sportvu_game"]) & set(ctx["heldout_game_ids"])
    assert not (tmp_path / "skip" / "analytics" / "rebounds" / "heldout").exists()
    assert (tmp_path / "ref" / "analytics" / "rebounds" / "heldout").exists()       # the reference run did write them
    rp = ref["predictions"]
    pd.testing.assert_frame_equal(pr, rp[~rp["heldout"]].reset_index(drop=True))
    pd.testing.assert_frame_equal(ctx["train"], ref["train"])


def test_skip_heldout_matchups(tmp_path, monkeypatch, capsys):
    from analytics.matchups import train as T
    base = ["--fixtures", "--fixture-games", "3", "--fixture-heldout-every", "3", "--fixture-possessions", "16",
            "--folds", "3", "--bootstrap", "5", "--n-min", "5"]
    ref = T.main(base + ["--report", str(tmp_path / "ref.md"), "--out", str(tmp_path / "ref")])
    capsys.readouterr()

    def boom(*a, **k):
        raise AssertionError("the held-out game was scored under --skip-heldout")
    monkeypatch.setattr(T, "score_heldout", boom)
    ctx = T.main(base + ["--report", str(tmp_path / "skip.md"), "--out", str(tmp_path / "skip"), "--skip-heldout"])
    printed = capsys.readouterr().out
    text = (tmp_path / "skip.md").read_text()
    assert len(ctx["held_ids"]) == 1 and ref["heldout"]                              # a held-out game, scored without the flag
    assert SKIP_HELDOUT_NOTE in text and SKIP_HELDOUT_NOTE in printed
    sec = _g1_section(text, "## Held-out game(s)")
    assert "MSE" not in sec and "stability" not in sec
    assert ctx["heldout"] == {}
    x = ctx["xfg"]                                                                   # no held-out xFG probability was predicted
    assert (x["fold"] != -1).all() and not set(x["sportvu_game"]) & set(ctx["held_ids"])
    rx = ref["xfg"]
    pd.testing.assert_frame_equal(x, rx[rx["fold"] != -1].reset_index(drop=True))
    assert ctx["inputs"] == ref["inputs"] and ctx["train_ids"] == ref["train_ids"]
    pd.testing.assert_frame_equal(ctx["cv"]["pred"], ref["cv"]["pred"])
    pd.testing.assert_frame_equal(ctx["tables"]["train"]["matchup_possessions"], ref["tables"]["train"]["matchup_possessions"])


# ----------------------------------------------------------------------------- lanes: gravity, ghosting, embeddings, epv

def test_skip_heldout_gravity(tmp_path, monkeypatch):
    from analytics.gravity import train as T

    def run(tag, *extra):
        report = tmp_path / ("gravity_%s.md" % tag)
        ctx = T.main(["--fixtures", "--no-pull-section", "--fixture-games", "4", "--fixture-possessions", "12",
                      "--fixture-heldout-every", "4", "--folds", "2", "--bootstrap", "5", "--min-possessions", "5",
                      "--report", str(report), "--out", str(tmp_path / tag)] + list(extra))
        return ctx, report.read_text()

    base, _ = run("base")
    assert base["heldout_games"] == 1 and base["heldout_scores"], "the fixture must hold a scored held-out game"
    real_build = T.F.build

    def build_training_only(games):
        assert not any(T.F.is_heldout(d) for d in games), "a held-out game was built for scoring"
        return real_build(games)

    monkeypatch.setattr(T.F, "build", build_training_only)
    ctx, text = run("skip", "--skip-heldout")
    assert SKIP_HELDOUT_NOTE in text
    assert ctx["heldout_games"] == 1 and ctx["heldout_scores"] == {} and ctx["heldout_rows"] == 0
    held = set(ctx["heldout_game_ids"])
    for name in ("gravity_frames", "spacing_frames"):
        df = pd.read_parquet(tmp_path / "skip" / "analytics" / "gravity" / ("%s.parquet" % name))
        assert not df["heldout"].any() and not set(df["sportvu_game"]) & held
    assert not (ctx["gravity_frames"]["fold"] == T.HELDOUT_FOLD).any()
    assert ctx["train_game_ids"] == base["train_game_ids"] and ctx["train_rows"] == base["train_rows"]
    pd.testing.assert_frame_equal(ctx["pred"], base["pred"])


def test_skip_heldout_ghosting(tmp_path, monkeypatch):
    from analytics.ghosting import train as T
    monkeypatch.setattr(T, "MAX_ITER", 30)

    def run(tag, *extra):
        report = tmp_path / ("ghosting_%s.md" % tag)
        ctx = T.main(["--fixtures", "--fixture-games", "3", "--fixture-possessions", "4", "--fixture-heldout-every", "3",
                      "--folds", "2", "--bootstrap", "5", "--report", str(report), "--out", str(tmp_path / tag)] + list(extra))
        return ctx, report.read_text()

    base, _ = run("base")
    assert base["heldout_games"] and base["heldout_scores"], "the fixture must hold a scored held-out game"
    real_roll = T.roll_forward

    def roll_training_only(models, rows, panel, fold):
        assert fold != T.HELDOUT_FOLD and not rows["heldout"].astype(bool).any(), "held-out rows rolled forward"
        return real_roll(models, rows, panel, fold)

    monkeypatch.setattr(T, "roll_forward", roll_training_only)
    ctx, text = run("skip", "--skip-heldout")
    assert SKIP_HELDOUT_NOTE in text and "## Held-out games" in text
    assert ctx["heldout_games"] == base["heldout_games"] and ctx["heldout_scores"] == {} and ctx["heldout_validation"] is None
    for df in (ctx["ghost_frames"], ctx["ghost_possessions"]):
        assert not set(df["sportvu_game"]) & set(ctx["heldout_games"])
    assert not (ctx["ghost_frames"]["fold"] == T.HELDOUT_FOLD).any()
    assert not list((tmp_path / "skip").rglob("heldout")) and list((tmp_path / "base").rglob("heldout"))
    assert ctx["train_games"] == base["train_games"] and ctx["train_rows"] == base["train_rows"]
    pd.testing.assert_frame_equal(ctx["errors"], base["errors"])


def test_skip_heldout_embeddings(tmp_path):
    from analytics.embeddings import train as T

    def run(tag, *extra):
        out = tmp_path / tag
        report = out / "embeddings.md"
        ctx = T.main(["--fixtures", "--report", str(report), "--out", str(out), "--bootstrap", "5",
                      "--fixture-games", "4", "--fixture-heldout-every", "2", "--seed", "0", "--folds", "2"] + list(extra))
        return ctx, report.read_text()

    base, base_text = run("base")
    ctx, text = run("skip", "--skip-heldout")
    assert SKIP_HELDOUT_NOTE in text and SKIP_HELDOUT_NOTE not in base_text
    held = set(ctx["dropped"]["heldout_games_by_id"])
    assert ctx["dropped"]["heldout_frame_files"] >= 1, "the fixture must hold a held-out game"
    assert not set(ctx["games"]) & held
    assert not list((tmp_path / "skip").rglob("heldout"))
    assert ctx["games"] == base["games"]
    pd.testing.assert_frame_equal(ctx["features"], base["features"])


def test_skip_heldout_epv(tmp_path, monkeypatch):
    from analytics.epv import train as T

    def run(tag, *extra):
        report = tmp_path / ("epv_%s.md" % tag)
        ctx = T.main(["--fixtures", "--fixture-games", "3", "--fixture-possessions", "16", "--fixture-heldout-every", "3",
                      "--folds", "2", "--bootstrap", "0", "--hgb-iter", "10", "--report", str(report),
                      "--out", str(tmp_path / tag)] + list(extra))
        return ctx, report.read_text()

    base, _ = run("base")
    assert base["heldout"] is not None and base["heldout_games"], "the fixture must hold a scored held-out game"
    assert (tmp_path / "base" / "analytics" / "epv" / "heldout").exists()
    monkeypatch.setattr(T, "heldout_once", lambda *a, **k: pytest.fail("held-out games scored under --skip-heldout"))
    ctx, text = run("skip", "--skip-heldout")
    assert SKIP_HELDOUT_NOTE in text and "## Held-out games" in text
    assert ctx["heldout"] is None and ctx["heldout_games"] == base["heldout_games"]
    assert not (tmp_path / "skip" / "analytics" / "epv" / "heldout").exists()
    for p in (tmp_path / "skip" / "analytics" / "epv").rglob("*.parquet"):
        df = pd.read_parquet(p)
        assert not df["sportvu_game"].isin(ctx["heldout_games"]).any(), p
        assert "fold" not in df.columns or not (df["fold"] == -1).any(), p
    assert ctx["train_games"] == base["train_games"] and ctx["train_rows"] == base["train_rows"]
    pd.testing.assert_frame_equal(ctx["pred"], base["pred"])


# ----------------------------------------------------------------------------- lanes: actions, coverage, decisions, playcalling

def _g3_section(text: str, heading: str) -> str:
    """The body of the first report section whose heading starts with ``heading``, up to the next ## heading."""
    part = text.split(heading, 1)[1]
    return part.split("\n## ", 1)[0]


def test_skip_heldout_actions(tmp_path, monkeypatch, capsys):
    from analytics.actions import train as T
    args = ["--fixtures", "--fixture-games", "3", "--fixture-possessions", "10", "--fixture-heldout-every", "3",
            "--folds", "2", "--bootstrap", "5"]
    base = T.main(args + ["--report", str(tmp_path / "base.md")])
    assert base["heldout_games"] >= 1 and base["heldout_scores"]           # the fixture has held-out games, scored without the flag

    real_score, real_table = T.score, T.candidate_table

    def score_guard(frame, *a, **k):
        assert not frame["heldout"].astype(bool).any(), "a held-out candidate was scored"
        return real_score(frame, *a, **k)

    def table_guard(frame, *a, **k):
        assert not frame["heldout"].astype(bool).any(), "a held-out candidate was predicted"
        return real_table(frame, *a, **k)

    monkeypatch.setattr(T, "score", score_guard)
    monkeypatch.setattr(T, "candidate_table", table_guard)
    capsys.readouterr()
    out = tmp_path / "out"
    ctx = T.main(args + ["--report", str(tmp_path / "skip.md"), "--out", str(out), "--skip-heldout"])
    text = (tmp_path / "skip.md").read_text()
    assert SKIP_HELDOUT_NOTE in text and SKIP_HELDOUT_NOTE in capsys.readouterr().out
    held = _g3_section(text, "## Held-out games")
    assert "| " not in held and "F1" not in held                          # no held-out metric row
    assert ctx["heldout_scores"] == {} and ctx["heldout_games"] == base["heldout_games"]
    assert not ctx["table"]["heldout"].any() and not (out / "analytics" / "actions" / "heldout").exists()
    assert not set(ctx["table"]["sportvu_game"]) & set(base["heldout_game_ids"])
    assert ctx["train_game_ids"] == base["train_game_ids"]
    pd.testing.assert_frame_equal(ctx["table"].reset_index(drop=True),
                                  base["table"][~base["table"]["heldout"]].reset_index(drop=True))


def test_skip_heldout_coverage(tmp_path, monkeypatch, capsys):
    from analytics.coverage import train as T
    args = ["--fixtures", "--fixture-games", "4", "--fixture-heldout-every", "3", "--fixture-possessions", "12",
            "--folds", "2", "--bootstrap", "5"]
    base = T.main(args + ["--report", str(tmp_path / "base.md"), "--out", str(tmp_path / "base")])
    assert base["heldout_rows"] > 0 and base["heldout_scores"]

    real = T.prediction_rows

    def guard(frame, *a, **k):
        assert not frame["heldout"].astype(bool).any(), "a held-out row was predicted"
        return real(frame, *a, **k)

    monkeypatch.setattr(T, "prediction_rows", guard)
    capsys.readouterr()
    out = tmp_path / "out"
    ctx = T.main(args + ["--report", str(tmp_path / "skip.md"), "--out", str(out), "--skip-heldout"])
    text = (tmp_path / "skip.md").read_text()
    assert SKIP_HELDOUT_NOTE in text and SKIP_HELDOUT_NOTE in capsys.readouterr().out
    held = _g3_section(text, "## Held-out rows")
    assert "| " not in held and "macro F1" not in held
    assert ctx["heldout_scores"] == {} and ctx["heldout_rows"] == base["heldout_rows"]
    assert not ctx["predictions"]["heldout"].any() and not (out / "analytics" / "coverage" / "heldout").exists()
    bp = base["predictions"]
    pd.testing.assert_frame_equal(ctx["predictions"].reset_index(drop=True), bp[~bp["heldout"]].reset_index(drop=True))


def test_skip_heldout_decisions(tmp_path, monkeypatch, capsys):
    from analytics.decisions import train as T
    args = ["--fixtures", "--fixture-games", "3", "--fixture-heldout-every", "3", "--fixture-possessions", "10",
            "--folds", "2", "--bootstrap", "5"]
    base = T.main(args + ["--report", str(tmp_path / "base.md"), "--out", str(tmp_path / "base")])
    assert base["heldout_ids"] and len(base["events_heldout"]) > 0

    real = T.value_game

    def guard(contexts, bundle, *a, **k):
        assert not bundle["heldout"], "a held-out game was valued"
        return real(contexts, bundle, *a, **k)

    monkeypatch.setattr(T, "value_game", guard)
    capsys.readouterr()
    out = tmp_path / "out"
    res = T.main(args + ["--report", str(tmp_path / "skip.md"), "--out", str(out), "--skip-heldout"])
    text = (tmp_path / "skip.md").read_text()
    assert SKIP_HELDOUT_NOTE in text and SKIP_HELDOUT_NOTE in capsys.readouterr().out
    held = _g3_section(text, "## Held-out games")
    assert "| " not in held and "decisions from" not in held
    assert len(res["events_heldout"]) == 0 and res["heldout_ids"] == base["heldout_ids"]
    assert not (out / "analytics" / "decisions" / "heldout").exists()
    assert res["train_ids"] == base["train_ids"]
    pd.testing.assert_frame_equal(res["events"], base["events"])


def test_skip_heldout_playcalling(tmp_path, monkeypatch, capsys):
    from analytics.playcalling import train as T
    args = ["--fixtures", "--fixture-games", "4", "--fixture-possessions", "8", "--fixture-heldout-every", "4",
            "--folds", "2", "--bootstrap", "5"]
    base = T.main(args + ["--report", str(tmp_path / "base.md")])
    assert base["heldout_rows"] > 0 and set(base["heldout_scores"]) == {"ridge", "mean_floor"}

    real = T.predict_ridge

    def guard(model, frame, *a, **k):
        assert not frame["heldout"].astype(bool).any(), "a held-out row was scored"
        return real(model, frame, *a, **k)

    monkeypatch.setattr(T, "predict_ridge", guard)
    capsys.readouterr()
    ctx = T.main(args + ["--report", str(tmp_path / "skip.md"), "--skip-heldout"])
    text = (tmp_path / "skip.md").read_text()
    assert SKIP_HELDOUT_NOTE in text and SKIP_HELDOUT_NOTE in capsys.readouterr().out
    held = _g3_section(text, "## Held-out rows")
    assert "| " not in held and "MSE" not in held
    assert ctx["heldout_scores"] == {} and ctx["heldout_rows"] == base["heldout_rows"]
    pd.testing.assert_frame_equal(ctx["train"], base["train"])
    pd.testing.assert_frame_equal(ctx["pred"], base["pred"])


# ----------------------------------------------------------------------------- lanes: lineups, winprob, load, betting

def test_skip_heldout_lineups(tmp_path, monkeypatch):
    from analytics.lineups import train as T

    held = "2022-23"                                   # the last fixture season, the default held-out season of --fixtures
    seen = []
    real_eval = T.evaluate_pair

    def spy(fits, train, test, *a, **k):
        seen.append(sorted(test["season"].astype(str).unique()))
        return real_eval(fits, train, test, *a, **k)

    monkeypatch.setattr(T, "evaluate_pair", spy)
    args = ["--fixtures", "--fixture-generator", "local", "--bootstrap", "5", "--folds", "2"]
    ctx = {}
    for flag in (False, True):
        out = tmp_path / ("skip" if flag else "full")
        seen.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            ctx[flag] = T.main(args + ["--out", str(out), "--report", str(out / "r.md")] + (["--skip-heldout"] if flag else []))
        if not flag:
            assert [held] in seen, "the fixture must carry a held-out season for the test to mean anything"
    assert all(held not in s for s in seen), seen                      # never scored
    c = ctx[True]
    assert c["heldout_season"] == held and c.get("heldout") is None
    assert held not in set(c["cross_season"]["season_to"].astype(str))
    text = (tmp_path / "skip" / "r.md").read_text()
    assert SKIP_HELDOUT_NOTE in text and "## Held-out season, scored once" in text
    assert not (tmp_path / "skip" / "analytics" / "lineups" / held).exists()
    assert (tmp_path / "full" / "analytics" / "lineups" / held).exists()
    # the training side is unchanged: same validation pairs, same final fit, same training-season tables
    assert [p["metrics"] for p in ctx[False]["pairs"]] == [p["metrics"] for p in c["pairs"]]
    assert ctx[False]["final"] == c["final"] and ctx[False]["selected"] == c["selected"]
    for f in ("2020-21/lineup_ratings.parquet", "2021-22/player_rapm_prior.parquet"):
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "full" / "analytics" / "lineups" / f),
                                      pd.read_parquet(tmp_path / "skip" / "analytics" / "lineups" / f))


def test_skip_heldout_winprob(tmp_path, monkeypatch):
    from analytics.winprob import train as T

    held = "2024-25"
    real_score = T.bucket_scores

    def guarded(pred, *a, **k):
        if (pred["season"].astype(str) == held).any():
            raise AssertionError("winprob scored the held-out season under --skip-heldout")
        return real_score(pred, *a, **k)

    args = ["--fixtures", "--fixture-games", "8", "--bootstrap", "0", "--folds", "2"]
    with contextlib.redirect_stdout(io.StringIO()):
        full = T.main(args + ["--out", str(tmp_path / "full"), "--report", str(tmp_path / "full.md")])
        assert full["heldout_rows"] > 0 and full["heldout_pred"] is not None
        monkeypatch.setattr(T, "bucket_scores", guarded)
        skip = T.main(args + ["--out", str(tmp_path / "skip"), "--report", str(tmp_path / "skip.md"), "--skip-heldout"])
    text = (tmp_path / "skip.md").read_text()
    assert SKIP_HELDOUT_NOTE in text and "## Held-out season %s, scored once" % held in text
    assert skip["heldout_pred"] is None and skip["heldout_scores"] == {}
    assert skip["heldout_rows"] == full["heldout_rows"]                  # set aside exactly as before
    assert not (tmp_path / "skip" / "analytics" / "winprob" / held).exists()
    assert (tmp_path / "full" / "analytics" / "winprob" / held).exists()
    for f in (tmp_path / "skip").rglob("winprob_states.parquet"):
        assert not pd.read_parquet(f)["heldout"].any(), f
    pd.testing.assert_frame_equal(full["pred"], skip["pred"])          # same training rows, same out-of-fold predictions
    assert full["selected"] == skip["selected"]


def test_skip_heldout_load(tmp_path, monkeypatch):
    from analytics.load import train as T

    real_rows = T.regression_rows

    def guarded(df):
        if df["heldout"].astype(bool).any():
            raise AssertionError("load scored held-out player-games under --skip-heldout")
        return real_rows(df)

    args = ["--fixtures", "--fixture-games", "4", "--fixture-possessions", "20", "--fixture-heldout-every", "2",
            "--bootstrap", "5", "--folds", "2"]
    with contextlib.redirect_stdout(io.StringIO()):
        full = T.main(args + ["--report", str(tmp_path / "full.md")])
        assert full["heldout_games"] > 0 and full["heldout"] is not None
        monkeypatch.setattr(T, "regression_rows", guarded)
        skip = T.main(args + ["--report", str(tmp_path / "skip.md"), "--skip-heldout"])
    text = (tmp_path / "skip.md").read_text()
    assert SKIP_HELDOUT_NOTE in text and "## Held-out games, scored once" in text
    assert skip["heldout"] is None and skip["heldout_games"] == full["heldout_games"]
    assert "MSE, selected model" not in text and "MSE, selected model" in (tmp_path / "full.md").read_text()
    pd.testing.assert_frame_equal(full["reg_pred"], skip["reg_pred"])  # same training rows, same out-of-fold predictions
    assert full["train_games"] == skip["train_games"]


def test_skip_heldout_betting(tmp_path, monkeypatch):
    from analytics.betting import train as T

    real_predict = T.predict_frame

    def guarded(models, frame):
        if frame["heldout"].astype(bool).any():
            raise AssertionError("betting predicted the held-out season under --skip-heldout")
        return real_predict(models, frame)

    args = ["--fixtures", "--fixture-games", "12", "--bootstrap", "5", "--folds", "2"]
    with contextlib.redirect_stdout(io.StringIO()):
        full = T.main(args + ["--out", str(tmp_path / "full"), "--report", str(tmp_path / "full.md")])
        held = full["heldout_season"]
        assert held and full["predictions"]["heldout"].any()
        monkeypatch.setattr(T, "predict_frame", guarded)
        skip = T.main(args + ["--out", str(tmp_path / "skip"), "--report", str(tmp_path / "skip.md"), "--skip-heldout"])
    text = (tmp_path / "skip.md").read_text()
    assert SKIP_HELDOUT_NOTE in text
    assert not skip["predictions"]["heldout"].any() and (skip["predictions"]["fold"] != -1).all()
    assert all(r["mode"] != "heldout" for r in skip["summary"]) and "| heldout |" not in text
    assert not (tmp_path / "skip" / "analytics" / "betting" / held / "betting_predictions.parquet").exists()
    assert (tmp_path / "full" / "analytics" / "betting" / held / "betting_predictions.parquet").exists()
    assert skip["heldout_games"] == full["heldout_games"] and skip["train_games"] == full["train_games"]
    train_full = full["predictions"][~full["predictions"]["heldout"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(train_full, skip["predictions"].reset_index(drop=True))
    assert full["chosen"] == skip["chosen"]
