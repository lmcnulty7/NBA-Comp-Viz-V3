"""sportvu.scaling: arms are deterministic subsets with a fixed train-list length."""
import json
from pathlib import Path

from sportvu import scaling


def fake_ds(tmp: Path) -> Path:
    ds = tmp / "ds"
    for split in ("train", "val"):
        (ds / "images" / split).mkdir(parents=True)
        (ds / "labels" / split).mkdir(parents=True)
    def add(split, name):
        (ds / "images" / split / name).write_bytes(b"x")
        (ds / "labels" / split / (name[:-4] + ".txt")).write_text("0\n")
    for g in scaling.GAMES:
        for w in range(50):
            for f in range(6):
                add("train", "r13_%s_s01_w%02d_%06d.jpg" % (g, w, 12 * f))
    for k in range(3):
        for i in range(5):
            add("train", "old_nbacourt_%04d.rf.x_r%d.jpg" % (i, k))
    add("val", "r13_gsw_bkn_2015_s09_w00_000000.jpg")
    add("val", "old_nbacourt_9999.rf.x.jpg")
    (ds / "data.yaml").write_text("path: x\ntrain: images/train\nval: images/val\nkpt_shape: [91, 3]\n")
    return ds


def test_pools_follow_the_spec(tmp_path):
    ds = fake_ds(tmp_path)
    r13, old, wins = scaling.pools("B_f1", ds)
    assert len(wins) == 4 * 50 and len(r13) == len(wins) and len(old) == 5
    r13, _, wins = scaling.pools("C_all4", ds)
    assert len(wins) == 48 and {w.rsplit("_s", 1)[0] for w in wins} == set(scaling.GAMES)
    assert scaling.pools("C_all4", ds) == scaling.pools("C_all4", ds)          # deterministic
    r13, _, wins = scaling.pools("C_bkn", ds)
    assert all(w.startswith("gsw_bkn_2015") for w in wins)


def test_build_has_fixed_length_and_r13_only_val(tmp_path):
    ds = fake_ds(tmp_path)
    out = scaling.build("B_f4", tmp_path / "arms", ds)
    meta = json.loads((out / "arm.json").read_text())
    assert meta["train_lines"] == scaling.OLD_LINES + scaling.R13_LINES
    assert len(list((out / "images" / "train").iterdir())) == meta["train_lines"]
    assert len(list((out / "labels" / "train").iterdir())) == meta["train_lines"]
    assert [p.name for p in (out / "images" / "val").iterdir()] == ["r13_gsw_bkn_2015_s09_w00_000000_k0.jpg"]
