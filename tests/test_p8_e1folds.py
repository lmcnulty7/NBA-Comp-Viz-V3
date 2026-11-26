"""E1 leave one game out (D156, D158): the fold config, the corpus packer, the build proxy, weights provenance and the
scoring report, with a stub model and synthetic windows. No training, no Colab, no Drive."""

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


def _spy(monkeypatch, module):
    calls = []
    monkeypatch.setattr(module, "guard", lambda items, who, homes=(): calls.append(list(items)) or items)
    return calls


def test_each_fold_trains_on_the_other_two_games_only(tmp_path, monkeypatch):
    corpus = _corpus(tmp_path)
    calls = _spy(monkeypatch, C)
    for fold in C.GAMES:
        rec = C.prepare(corpus, fold, 0, tmp_path / fold)
        assert rec["held_out_game"] == fold and fold not in rec["training_games"] and len(rec["training_games"]) == 2
        listed = (tmp_path / fold / "train.txt").read_text() + (tmp_path / fold / "val.txt").read_text()
        assert f"/{fold}/" not in listed
        assert rec["split"] == {"train": 36, "val": 4, "val_share": 0.1}     # the last window of each training game
        assert set(rec["corpus_counts"]) == set(rec["training_games"])
        assert rec["image_counts"] == {g: 20 for g in rec["training_games"]}
    # the guard ran on every game, SportVU id and video id, once per prepare
    assert len(calls) == 3 and all({"gsw_bkn_2015", "sv_gsw_cha_2016", "vid_gsw_sac_2015"} <= set(c) for c in calls)
    with pytest.raises(SystemExit):
        C.fold_games("hou_cle_2016")


def test_prepare_refuses_when_images_labels_and_manifest_disagree(tmp_path, monkeypatch):
    corpus = _corpus(tmp_path)
    _spy(monkeypatch, C)
    (corpus / "labels" / "gsw_cha_2016" / "gsw_cha_2016_s00_w00_f000001.txt").unlink()
    with pytest.raises(SystemExit):
        C.prepare(corpus, "gsw_bkn_2015", 0, tmp_path / "out")
    assert C.check_counts(corpus, ["gsw_sac_2015"]) == {"gsw_sac_2015": 20}


def test_label_check_counts_found_missing_empty_and_corrupt(tmp_path):
    pytest.importorskip("ultralytics")
    import cv2
    root = tmp_path / "c"
    (root / "images" / "g").mkdir(parents=True)
    (root / "labels" / "g").mkdir(parents=True)
    img = np.full((64, 64, 3), 128, np.uint8)
    for i in range(3):
        cv2.imwrite(str(root / "images" / "g" / f"a{i}.jpg"), img)
    (root / "labels" / "g" / "a0.txt").write_text("0 0.5 0.5 0.2 0.4\n")
    (root / "labels" / "g" / "a1.txt").write_text("")
    (root / "images" / "g" / "bad.jpg").write_text("not an image")
    files = [str(p) for p in sorted((root / "images" / "g").glob("*.jpg"))]
    c = C.label_check(files)
    assert c == {"images": 4, "found": 2, "missing": 1, "empty": 1, "corrupt": 1}


def test_colab_guards_before_it_reads_drive_and_sets_an_old_result_aside(tmp_path, monkeypatch):
    order = []
    monkeypatch.setattr(C, "guard", lambda items, who, homes=(): order.append("guard"))
    monkeypatch.setattr(C, "DRIVE_RUNS", tmp_path / "runs")
    monkeypatch.setattr(C, "DRIVE_CORPUS", tmp_path / "corpus")
    monkeypatch.setattr(C, "LOCAL", tmp_path / "local")
    (tmp_path / "runs").mkdir()
    old = tmp_path / "runs" / "e1_gsw_bkn_2015_seed0.tar"
    old.write_text("earlier run")

    def stop(*a, **k):
        order.append("unpack")
        raise RuntimeError("stop here")
    import tools.p8_e1_pack as pack
    monkeypatch.setattr(pack, "unpack_checked", stop)
    with pytest.raises(RuntimeError):
        C.colab("gsw_bkn_2015", 0)
    assert order == ["guard", "unpack"]
    assert not old.exists() and len(list((tmp_path / "runs").glob("e1_gsw_bkn_2015_seed0.tar.prev-*"))) == 1


# ── packer ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_pack_writes_one_tar_per_game_and_verify_checks_bytes_and_members(tmp_path, monkeypatch):
    corpus = _corpus(tmp_path)
    calls = _spy(monkeypatch, K)
    dest = tmp_path / "drive"
    man = K.pack(corpus, dest)
    assert calls and {"gsw_bkn_2015", "sv_gsw_bkn_2015", "vid_gsw_bkn_2015"} <= set(calls[0])
    assert set(man["parts"]) == {"common", *C.GAMES}
    assert man["parts"]["gsw_bkn_2015"]["members"] == 40 and man["parts"]["common"]["members"] == 3
    assert K.verify(dest)["gsw_cha_2016"] == (40, 40)
    # a tar cut short on the way fails on bytes
    data = (dest / "gsw_cha_2016.tar").read_bytes()
    (dest / "gsw_cha_2016.tar").write_bytes(data[:-512])
    with pytest.raises(SystemExit):
        K.verify(dest)
    out = tmp_path / "unpacked"
    K.unpack_checked(dest, out, ["common", "gsw_bkn_2015"])
    assert (out / "images" / "gsw_bkn_2015").exists() and (out / "meta.json").exists()


# ── build proxy and weights provenance ──────────────────────────────────────────────────────────────────────────────
def test_the_proxy_routes_only_the_build_script_and_only_with_a_model_set():
    cmd = ["py", "build_trajectories.py", "--source", "x.mp4"]
    assert S.rewrite(cmd, env={}) == cmd
    assert S.rewrite(cmd, env={S.ENV: "w.pt"}) == ["py", "-m", "tools.p8_e1_score", "child", *cmd[1:]]
    assert S.rewrite(["py", "other.py"], env={S.ENV: "w.pt"}) == ["py", "other.py"]
    assert S.scored_games("gsw_cha_2016") == ["gsw_cha_2016", "hou_cle_2016"]
    assert S.build_name("gsw_cha_2016", 1) == "e1_gsw_cha_2016_seed1"
    assert S.score_name("stub", 0, "hou_cle_2016") == "stub__seed0__hou_cle_2016.json"
    with pytest.raises(SystemExit):
        S.build_name("gsw_ind_2016", 0)


def _run(tmp_path, fold="gsw_bkn_2015", seed=0, training=("gsw_cha_2016", "gsw_sac_2015"), good_hash=True):
    run = tmp_path / f"e1_{fold}_seed{seed}"
    w = run / "train" / "weights" / "best.pt"
    w.parent.mkdir(parents=True)
    w.write_bytes(b"weights")
    (run / "run_record.json").write_text(json.dumps({"fold": fold, "held_out_game": fold, "seed": seed,
                                                     "training_games": list(training)}))
    sha = S._sha(w) if good_hash else "0" * 64
    (run / "complete.json").write_text(json.dumps({"weights": {"best.pt": sha}}))
    return w


def test_provenance_accepts_only_the_fold_seed_and_hash_it_was_trained_for(tmp_path):
    w = _run(tmp_path)
    assert S.check_provenance("gsw_bkn_2015", "gsw_bkn_2015", 0, w)["sha256"] == S._sha(w)
    assert S.check_provenance("gsw_bkn_2015", "hou_cle_2016", 0, w)["run"] == "e1_gsw_bkn_2015_seed0"
    for model, game, seed in (("gsw_cha_2016", "gsw_cha_2016", 0), ("gsw_bkn_2015", "gsw_bkn_2015", 1)):
        with pytest.raises(SystemExit):
            S.check_provenance(model, game, seed, w)
    leak = _run(tmp_path / "leak", training=("gsw_bkn_2015", "gsw_cha_2016"))
    with pytest.raises(SystemExit):
        S.check_provenance("gsw_bkn_2015", "gsw_bkn_2015", 0, leak)
    unfinished = _run(tmp_path / "bad", good_hash=False)
    with pytest.raises(SystemExit):
        S.check_provenance("gsw_bkn_2015", "gsw_bkn_2015", 0, unfinished)
    stub = tmp_path / "stub.pt"
    stub.write_bytes(b"production")
    assert S.check_provenance("stub", "gsw_cha_2016", 0, stub, prod_sha=S._sha(stub))["run"] is None
    with pytest.raises(SystemExit):
        S.check_provenance("stub", "gsw_cha_2016", 0, stub, prod_sha="1" * 64)


def test_detector_records_must_agree_and_frame_sets_are_compared(tmp_path):
    for i, (sha, seed) in enumerate((("a", 0), ("a", 0))):
        (tmp_path / f"w{i}_detector.json").write_text(json.dumps({"sha256": sha, "seed": seed}))
    assert S.detector_records(tmp_path, ["w0", "w1"]) == {"weights_sha256": "a", "seed": 0}
    (tmp_path / "w2_detector.json").write_text(json.dumps({"sha256": "b", "seed": 0}))
    with pytest.raises(SystemExit):
        S.detector_records(tmp_path, ["w0", "w2"])
    with pytest.raises(SystemExit):
        S.detector_records(tmp_path, ["w0", "w9"])
    fs = S.frame_sets({"frames": [{"frame": 0}, {"frame": 6}]}, {"frames": [{"frame": 6}, {"frame": 12}]})
    assert fs == {"shared": [6], "only_r23": 1, "only_new": 1, "equal": False}


def test_filtered_sidecars_keep_only_the_shared_frames(tmp_path):
    src = tmp_path / "b"
    src.mkdir()
    (src / "w_frames.json").write_text(json.dumps({"fps": 60, "frames": [{"frame": 0}, {"frame": 6}]}))
    (src / "w_identity.json").write_text("{}")
    d = S._filtered(src, "w", [6], tmp_path / "f")
    assert [r["frame"] for r in json.loads((d / "w_frames.json").read_text())["frames"]] == [6]
    assert (d / "w_identity.json").is_symlink()


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


def _file(model, seed, wins, sha="h", drop=0):
    names = [f"w{i}" for i in range(len(wins))]
    return {"model": model, "seed": seed, "weights_sha256": sha, "expected_windows": names,
            "scored_windows": names[drop:], "gate_empty_windows": [], "windows": dict(zip(names[drop:], wins[drop:]))}


def _files(rng, seeds=(0,), **kw):
    return {(g, s, g): _file(g, s, [_win(rng, **kw) for _ in range(6)], sha=f"{g}{s}") for g in S.FOLDS for s in seeds}


def test_a_stub_that_reproduces_r23_changes_no_row_and_is_not_adopted():
    rep = S.report(_files(np.random.default_rng(0), same=True))
    pool = rep["pools"]["0"]
    assert all(v == 0 for v in pool["diff"].values())
    assert not pool["verdict"]["adopt"] and not rep["verdict"]["adopt"]
    assert all(not r.get("worse", False) for r in pool["verdict"]["rows"].values())


def test_a_better_model_needs_all_three_seeds():
    rng = np.random.default_rng(1)
    kw = dict(shift=-0.2, missed=15, ghosts=6, team=85)
    one = _files(rng, seeds=(0,), **kw)
    one[("gsw_bkn_2015", 0, "hou_cle_2016")] = _file("gsw_bkn_2015", 0, [_win(rng, **kw) for _ in range(4)], sha="gsw_bkn_20150")
    rep = S.report(one)
    assert rep["pools"]["0"]["verdict"]["adopt"]
    assert rep["verdict"] == {"seeds_scored": "1 of 3 seeds", "adopt": False}
    assert "gsw_bkn_2015__seed0__hou_cle_2016" in rep["apart"]
    three = S.report(_files(rng, seeds=(0, 1, 2), **kw))
    assert three["verdict"] == {"seeds_scored": "3 of 3 seeds", "adopt": True}
    worse_team = S.report(_files(rng, seeds=(0, 1, 2), shift=-0.2, missed=15, ghosts=6, team=60))
    assert worse_team["pools"]["0"]["verdict"]["rows"]["team_accuracy"]["worse"] and not worse_team["verdict"]["adopt"]


def test_a_partial_pool_never_adopts_and_mixed_hashes_are_refused():
    rng = np.random.default_rng(2)
    kw = dict(shift=-0.2, missed=15, ghosts=6, team=85)
    files = _files(rng, seeds=(0, 1, 2), **kw)
    f = files[("gsw_cha_2016", 1, "gsw_cha_2016")]
    files[("gsw_cha_2016", 1, "gsw_cha_2016")] = _file("gsw_cha_2016", 1, list(f["windows"].values()), sha=f["weights_sha256"], drop=2)
    rep = S.report(files)
    assert rep["pools"]["1"]["verdict"].get("partial") and not rep["pools"]["1"]["verdict"]["adopt"]
    assert rep["verdict"] == {"seeds_scored": "2 of 3 seeds", "adopt": False}
    files[("gsw_cha_2016", 0, "hou_cle_2016")] = _file("gsw_cha_2016", 0, [_win(rng, **kw)], sha="other")
    with pytest.raises(SystemExit):
        S.report(files)


def test_d5_coverage_is_a_floor():
    point = {k: (-0.1 if not h else 0.1) for k, h in S.ROWS.items()}
    iv = {k: ([-0.2, -0.05] if not h else [0.05, 0.2]) for k, h in S.ROWS.items()}
    assert S.d5_verdict(point, iv, (0.9, 0.6))["adopt"]
    assert not S.d5_verdict(point, iv, (0.9, 0.45))["adopt"]
    assert S.d5_verdict(point, iv, (0.4, 0.4))["adopt"] and not S.d5_verdict(point, iv, (0.4, 0.39))["adopt"]


def test_d5_game_interval_also_blocks(monkeypatch):
    """D158: a row whose game interval lies wholly on the bad side is worse, even when its window interval is not."""
    point = {k: (-0.1 if not h else 0.1) for k, h in S.ROWS.items()}
    iv = {k: ([-0.2, 0.05] if not h else [-0.05, 0.2]) for k, h in S.ROWS.items()}
    giv = dict(iv, team_accuracy=[-0.3, -0.01])
    assert S.d5_verdict(point, iv, (0.9, 0.9))["adopt"]
    v = S.d5_verdict(point, iv, (0.9, 0.9), giv)
    assert v["rows"]["team_accuracy"]["worse"] and not v["adopt"]
    assert v["rows"]["missed_rate"]["game_interval"] == [-0.2, 0.05]


def test_e1_score_uses_the_full_guard():
    src = open(S.__file__).read()
    assert src.count("guard([args.game], WHO)") == 2 and "guard([src], WHO)" in src


def test_every_new_tool_guards():
    for m in (C, K, S):
        assert "refuse_cv_test(" in open(m.__file__).read()


def test_e1data_writes_no_root_data_yaml():
    from tools import p8_e1data as D
    assert 'root / "data.yaml"' not in open(D.__file__).read()


def test_colab_takes_api_folders_instead_of_drive(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(C, "guard", lambda items, who, homes=(): None)
    monkeypatch.setattr(C, "LOCAL", tmp_path / "local")

    def stop(src, dest, parts):
        seen["src"] = src
        raise RuntimeError("stop here")
    import tools.p8_e1_pack as pack
    monkeypatch.setattr(pack, "unpack_checked", stop)
    with pytest.raises(RuntimeError):
        C.main(["colab", "--fold", "gsw_bkn_2015", "--corpus-dir", str(tmp_path / "in"), "--runs-dir", str(tmp_path / "out")])
    assert seen["src"] == tmp_path / "in" and (tmp_path / "out").is_dir()
