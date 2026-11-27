"""tools.p8_posefoot_pose (docs/research/P8_posefoot.md section 15): colour order, the ankle point, the lift and its
sign, the hybrid, the mechanical check, partners, the hit sample, the stage A requirement, the refusals and the writes.
No video, no model, no workspace."""

import json

import numpy as np
import pytest

from tools import p8_posefoot_pose as M

W, H = 1280, 720
CAM = ([47.23, 137.06, -30.72], [-1.3447, -0.1647, -0.2669], 4300.0)


def test_bgr_to_rgb():
    img = np.zeros((2, 2, 3), np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = 10, 20, 30            # B, G, R
    out = M.bgr_to_rgb(img)
    assert out[0, 0].tolist() == [30, 20, 10] and out.flags["C_CONTIGUOUS"]


def test_to_coco_and_ankle_point():
    assert M.to_coco([10, 20, 50, 120]) == [10.0, 20.0, 40.0, 100.0]
    kps = np.zeros((17, 2)); kps[15] = [100, 200]; kps[16] = [110, 204]
    sc = np.zeros(17); sc[15] = 0.9; sc[16] = 0.5
    assert M.ankle_point(kps, sc) == ([105.0, 202.0], 2)
    sc[16] = 0.2
    assert M.ankle_point(kps, sc) == ([100.0, 200.0], 1)
    sc[15] = 0.1
    assert M.ankle_point(kps, sc) == (None, 0)


def test_lift_at_zero_is_the_ground_mapping_and_the_sign():
    from sportvu.camera import h_px2ft
    Hc = np.asarray(h_px2ft(np.array(CAM[0]), np.array(CAM[1]), CAM[2], W, H)).reshape(3, 3)
    px = [640.0, 500.0]
    g0 = M.lift_ground(px, *CAM, W, H, 0.0)
    assert np.allclose(g0, M.PF.map_px(Hc, [px])[0], atol=1e-3)
    g25 = M.lift_ground(px, *CAM, W, H, 0.25)
    cam_ground = np.array(CAM[0][:2])
    # the lifted ankle's floor point lies nearer the camera than the unlifted ray's floor point
    assert np.hypot(*(g25 - cam_ground)) < np.hypot(*(g0 - cam_ground))
    shift = np.hypot(*(g0 - g25))
    assert 0.2 < shift < 2.0                                    # about h / tan(elevation) for a broadcast camera
    assert np.hypot(*(M.lift_ground(px, *CAM, W, H, 0.35) - g0)) > shift


def test_hybrid_pixel_and_mechanical_check():
    assert M.hybrid_pixel([100.0, 300.0], [102.0, 305.0], [101.0, 298.0]) == [99.0, 293.0]
    box = [100, 100, 140, 300]
    assert M.inside_or_below([120, 290], box) and M.inside_or_below([120, 339], box)
    assert not M.inside_or_below([120, 341], box) and not M.inside_or_below([150, 290], box)


def test_partner_order_and_hit_sample():
    assert M.partner_order([180, 150, 175, 200, 179], 0, [True, True, True, True, False]) == [2, 3, 1]
    assert M.partner_order([180], 0, [True]) == []
    a = M.sample_hits([0, 1, 3, 5], np.random.default_rng(0))
    b = M.sample_hits([0, 1, 3, 5], np.random.default_rng(0))
    assert a == b and a[0] == [] and a[1] == [0] and len(a[2]) == 2 and len(a[3]) == 2
    assert all(len(set(x)) == len(x) for x in a)


def test_require_aligned(tmp_path):
    p = tmp_path / "align.json"
    with pytest.raises(SystemExit):
        M.require_aligned(p)
    rec = {"stop": False, "smoke": False, "per_game": {
        g: {"not_aligned": [f"{g}_w1"], "M_and_L0": {f"{g}_w0": {}, f"{g}_w1": {}}} for g in M.PF.POOL}}
    p.write_text(json.dumps(rec))
    assert M.require_aligned(p) == {g: [f"{g}_w0"] for g in M.PF.POOL}
    p.write_text(json.dumps({**rec, "stop": True}))
    with pytest.raises(SystemExit):
        M.require_aligned(p)
    p.write_text(json.dumps({**rec, "smoke": True}))
    with pytest.raises(SystemExit):
        M.require_aligned(p)


def test_hou_is_refused_before_any_read(monkeypatch):
    original = M.collect_pose
    monkeypatch.setattr(M, "require_aligned", lambda *a: (_ for _ in ()).throw(AssertionError("a file was read")))
    with pytest.raises(SystemExit):
        original("hou_cle_2016")
    called = []
    monkeypatch.setattr(M, "collect_pose", lambda g, s: called.append(g))
    with pytest.raises(SystemExit):
        M.main(["pose", "hou_cle_2016"])
    assert called == []


def test_offline_is_required(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(SystemExit):
        M.load_pose()


def test_the_only_writes_are_the_records():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(M))
    names = [getattr(c.func, "id", getattr(c.func, "attr", None)) for c in ast.walk(tree) if isinstance(c, ast.Call)]
    assert names.count("atomic_write") == 2
    assert not any(n in ("write_text", "write_bytes", "unlink", "rmtree", "remove", "rename", "open") for n in names)


def test_constants_match_the_design():
    assert (M.ANKLE_SCORE, M.ANKLE_FT, M.ANKLE_FT_ALT, M.HITS_PER_FRAME, M.PARTNERS) == (0.3, 0.25, 0.35, 2, 2)
    assert (M.MECH_SHARE, M.MECH_BELOW) == (0.95, 0.2) and M.ANKLES == (15, 16)
