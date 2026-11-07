"""sportvu.splits (ROADMAP R1.4): the committed splits and the guard every training script calls."""
import re
from pathlib import Path

import pytest

from sportvu import splits

ROOT = Path(__file__).resolve().parents[1]
GUARDED = ["gate_ship_v2.py", "gate_retrain_experiment.py", "colab_train_player.py"]
TRIPWIRE = ("refuse_heldout", "assert_time_forward")


def guarded_scripts(root: Path) -> list:
    """Every training script that must call the R1.4 guard: root/train_*.py, root/analytics/train_*.py
    (globbed at test time, so a new analytics trainer is checked the moment it exists) and GUARDED."""
    scripts = sorted(p.name for p in root.glob("train_*.py"))
    scripts += sorted("analytics/" + p.name for p in (root / "analytics").glob("train_*.py"))
    scripts += sorted("analytics/%s/%s" % (p.parent.name, p.name) for p in (root / "analytics").glob("*/train.py"))
    return scripts + [g for g in GUARDED if (root / g).exists() or root == ROOT]


def guard_missing(root: Path) -> list:
    return [s for s in guarded_scripts(root) if "refuse_heldout" not in (root / s).read_text()]


def fitting_files(root: Path) -> list:
    """Files under root/analytics that import sklearn or call .fit( : the crude leakage tripwire."""
    out = []
    for p in sorted((root / "analytics").rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        text = p.read_text()
        if re.search(r"^\s*(import|from)\s+sklearn\b", text, re.M) or ".fit(" in text:
            out.append(p)
    return out


def test_committed_splits_hold_the_decisions():
    sp = splits.load()
    assert set(sp["heldout_game"]["games"]) == {"gsw_phx_2016"}
    assert set(sp["heldout_era"]["games"]) == {"gsw_nyk_curry54"}
    assert {"cle_nyk_2015", "cle_gsw_2016"} <= set(sp["heldout_arena"]["games"])
    assert all(g["arena"] == "CLE" for g in sp["heldout_arena"]["games"].values())
    train = set(sp["train"]["games"])
    held = {t for s in splits.HELDOUT_SPLITS for t in sp[s]["games"]}
    assert train and not (train & held)
    assert "CLE" not in sp["train"]["arenas"]
    assert not (train & set(sp["excluded"]))


def test_guard_refuses_held_out_ids_and_passes_train_items():
    sp = splits.load()
    assert splits.refuse_heldout(["data/sportvu/labels/gsw_bkn_2015/img/x.jpg"], "t", sp) == 1
    for bad in ("label_corpus/gsw_phx_2016/f0000001.jpg", "12.23.2015.NYK.at.CLE_moments.json",
                "frames/gsw_nyk_curry54_s03.mp4"):
        with pytest.raises(SystemExit):
            splits.refuse_heldout(["ok.jpg", bad], "t", sp)


def test_missing_splits_file_stops_training(monkeypatch, tmp_path):
    monkeypatch.setattr(splits, "PATH", tmp_path / "splits.json")
    with pytest.raises(SystemExit):
        splits.load()


def test_every_training_script_calls_the_guard():
    missing = guard_missing(ROOT)
    assert not missing, "training scripts without the R1.4 guard: %s" % missing


def test_guard_check_catches_an_analytics_trainer_without_the_guard(tmp_path):
    """The check itself: passes with no analytics/train_*.py, fails once one lacks refuse_heldout."""
    (tmp_path / "analytics").mkdir()
    assert guard_missing(tmp_path) == []
    (tmp_path / "analytics" / "train_ok.py").write_text("from analytics.eval import refuse_heldout\n")
    assert guard_missing(tmp_path) == []
    (tmp_path / "analytics" / "train_bad.py").write_text("import sklearn\nmodel.fit(X, y)\n")
    assert guard_missing(tmp_path) == ["analytics/train_bad.py"]


def test_analytics_files_that_fit_carry_a_leakage_tripwire():
    """Every file under analytics/ that imports sklearn or calls .fit( must mention refuse_heldout or
    assert_time_forward: crude, but a fit without either guard cannot land unnoticed."""
    files = fitting_files(ROOT) if (ROOT / "analytics").exists() else []
    if not files:
        pytest.skip("no file under analytics/ imports sklearn or calls .fit(")
    bare = [str(p.relative_to(ROOT)) for p in files if not any(t in p.read_text() for t in TRIPWIRE)]
    assert not bare, "analytics files that fit without refuse_heldout or assert_time_forward: %s" % bare
