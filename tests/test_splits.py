"""sportvu.splits (ROADMAP R1.4): the committed splits and the guard every training script calls."""
from pathlib import Path

import pytest

from sportvu import splits

ROOT = Path(__file__).resolve().parents[1]
GUARDED = ["gate_ship_v2.py", "gate_retrain_experiment.py", "colab_train_player.py"]


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
    scripts = sorted(p.name for p in ROOT.glob("train_*.py")) + GUARDED
    missing = [s for s in scripts if "refuse_heldout" not in (ROOT / s).read_text()]
    assert not missing, "training scripts without the R1.4 guard: %s" % missing
