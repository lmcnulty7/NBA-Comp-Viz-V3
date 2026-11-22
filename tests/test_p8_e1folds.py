"""E1 leave one game out (D156): the fold config, the corpus packer, the build proxy and the scoring report, with a
stub model and synthetic windows. No training, no Colab, no Drive."""

import json
import tarfile

import numpy as np
import pytest

from tools import p8_e1_colab as C
from tools import p8_e1_pack as K
from tools import p8_e1_score as S


# ── folds ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def _corpus(tmp_path):
    root = tmp_path / "corpus"
    for g in C.GAMES:
        for w in range(10):
            for f in (1, 2):
                for sub, ext in (("images", "jpg"), ("labels", "txt")):
                    p = root / sub / g / f"{g}_s00_w{w:02d}_f{f:06d}.{ext}"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text("x")
    meta = {"games": list(C.GAMES), "code_commit": "abc",
            "datasets": {g: {"sportvu": f"sv_{g}", "video_id": f"vid_{g}"} for g in C.GAMES}}
    (root / "meta.json").write_text(json.dumps(meta))
    (root / "manifest.json").write_text(json.dumps({"counts": {g: {"frames": 20} for g in C.GAMES}, "frames": {}}))
    (root / "plan.json").write_text("{}")
    return root


def test_each_fold_trains_on_the_other_two_games_only(tmp_path):
    corpus = _corpus(tmp_path)
    for fold in C.GAMES:
        rec = C.prepare(corpus, fold, 0, tmp_path / fold)
        assert rec["held_out_game"] == fold and fold not in rec["training_games"] and len(rec["training_games"]) == 2
        listed = (tmp_path / fold / "train.txt").read_text() + (tmp_path / fold / "val.txt").read_text()
        assert f"/{fold}/" not in listed
        assert rec["split"] == {"train": 36, "val": 4, "val_share": 0.1}     # the last window of each training game
        assert set(rec["corpus_counts"]) == set(rec["training_games"])
    with pytest.raises(SystemExit):
        C.fold_games("hou_cle_2016")


# ── packer ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_pack_writes_one_tar_per_game_and_verify_counts_members(tmp_path):
    corpus = _corpus(tmp_path)
    dest = tmp_path / "drive"
    man = K.pack(corpus, dest)
    assert set(man["parts"]) == {"common", *C.GAMES}
    assert man["parts"]["gsw_bkn_2015"]["members"] == 40 and man["parts"]["common"]["members"] == 3
    assert K.verify(dest)["gsw_cha_2016"] == (40, 40)
    # a tar that lost a member on the way fails the count
    with tarfile.open(dest / "gsw_sac_2015.tar", "w") as tf:
        tf.add(corpus / "meta.json", arcname="meta.json")
    with pytest.raises(SystemExit):
        K.verify(dest)
    out = tmp_path / "unpacked"
    K.unpack_checked(dest, out, ["common", "gsw_bkn_2015"])
    assert (out / "images" / "gsw_bkn_2015").exists() and (out / "meta.json").exists()


# ── build proxy ─────────────────────────────────────────────────────────────────────────────────────────────────────
def test_the_proxy_routes_only_the_build_script_and_only_with_a_model_set():
    cmd = ["py", "build_trajectories.py", "--source", "x.mp4"]
    assert S.rewrite(cmd, env={}) == cmd
    assert S.rewrite(cmd, env={S.ENV: "w.pt"}) == ["py", "-m", "tools.p8_e1_score", "child", *cmd[1:]]
    assert S.rewrite(["py", "other.py"], env={S.ENV: "w.pt"}) == ["py", "other.py"]
    assert S.scored_games("gsw_cha_2016") == ["gsw_cha_2016", "hou_cle_2016"]
    with pytest.raises(SystemExit):
        S.build_name("gsw_ind_2016")


# ── report with a stub model ────────────────────────────────────────────────────────────────────────────────────────
def _win(rng, shift=0.0, missed=20, ghosts=10, team=80, same=False):
    def side(sh, mi, gh, tm):
        m = {f"{f}:{j}": [float(abs(rng.normal(1.5, 0.5)) + sh), float(rng.normal(0.3, 0.2)), ["near", "far", "middle"][j % 3]]
             for f in range(10) for j in range(10)}
        return {"missed": mi, "in_frame": 100, "nr_ghosts": gh, "nr_boxes": 100,
                "team": {"n": 100, "majority": tm, "fixed": tm}, "cov_acc": 9.0, "cov_live": 10.0, "matched": m}
    base = side(0.0, 25, 12, 78)
    new = json.loads(json.dumps(base)) if same else side(shift, missed, ghosts, team)
    return {"r23": base, "new": new}


def test_a_stub_that_reproduces_r23_changes_no_row_and_is_not_adopted():
    rng = np.random.default_rng(0)
    res = {(g, g): [_win(rng, same=True) for _ in range(6)] for g in S.FOLDS}
    rep = S.report(res)
    assert all(v == 0 for v in rep["pool"]["diff"].values())
    assert not rep["pool"]["verdict"]["adopt"]                       # nothing improves
    assert all(not r.get("worse", False) for r in rep["pool"]["verdict"]["rows"].values())


def test_a_better_model_is_adopted_and_a_worse_row_blocks_it():
    rng = np.random.default_rng(1)
    better = {(g, g): [_win(rng, shift=-0.2, missed=15, ghosts=6, team=85) for _ in range(8)] for g in S.FOLDS}
    better[("gsw_bkn_2015", "hou_cle_2016")] = [_win(rng, shift=-0.2) for _ in range(4)]
    rep = S.report(better)
    v = rep["pool"]["verdict"]
    assert rep["pool"]["diff"]["missed_rate"] == pytest.approx(-0.10) and rep["pool"]["game_interval"]["missed_rate"]
    assert "gsw_bkn_2015__hou_cle_2016" in rep["apart"] and "gsw_bkn_2015__gsw_bkn_2015" in rep["per_game"]
    assert v["rows"]["missed_rate"]["improves"] and v["rows"]["team_accuracy"]["improves"]
    worse_team = {(g, g): [_win(rng, shift=-0.2, missed=15, ghosts=6, team=60) for _ in range(8)] for g in S.FOLDS}
    vw = S.report(worse_team)["pool"]["verdict"]
    assert vw["rows"]["team_accuracy"]["worse"] and not vw["adopt"]


def test_d5_coverage_is_a_floor():
    point = {k: (-0.1 if not h else 0.1) for k, h in S.ROWS.items()}
    iv = {k: ([-0.2, -0.05] if not h else [0.05, 0.2]) for k, h in S.ROWS.items()}
    assert S.d5_verdict(point, iv, (0.9, 0.6))["adopt"]
    assert not S.d5_verdict(point, iv, (0.9, 0.45))["adopt"]
    assert S.d5_verdict(point, iv, (0.4, 0.4))["adopt"] and not S.d5_verdict(point, iv, (0.4, 0.39))["adopt"]


def test_every_new_tool_guards():
    for m in (C, K, S):
        assert "refuse_cv_test(" in open(m.__file__).read()
