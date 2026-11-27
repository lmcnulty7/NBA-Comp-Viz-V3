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
    assert (M.ANKLE_SCORE, M.ANKLE_FT, M.ANKLE_FT_ALT, M.HITS_PER_FRAME) == (0.3, 0.25, 0.35, 2)
    assert (M.POSE_CAP_S, M.READ_FAIL_MAX) == (3600, 3)
    assert (M.MECH_SHARE, M.MECH_BELOW) == (0.95, 0.2) and M.ANKLES == (15, 16)


# a real stored B: builds/r23 gsw_cha_2016_s00_w00, frame 30, and its first box
HB_REAL = [0.036258996322889585, 0.06051632856056825, -40.51878316199468, -0.00912497739965471, 0.169207630803979,
           -51.694168081089614, 3.458406276211347e-05, 0.0012031769592921, 0.5769749223111552]
RAW_REAL, STAB_REAL = [369.99407958984375, 553.6014404296875], [352.46978759765625, 552.8682250976562]


def test_lift_pixel_at_zero_is_the_pixel_on_a_real_b_and_the_hybrid_is_foot_stab():
    from sportvu.camera import pose_from_h
    pose = pose_from_h(np.asarray(HB_REAL).reshape(3, 3), W, H)
    assert pose is not None
    camB = (pose[2], pose[0], pose[1])
    pl = M.lift_pixel(RAW_REAL, camB, W, H, 0.0)
    assert np.allclose(pl, RAW_REAL, atol=1e-6)
    assert np.allclose(M.hybrid_pixel(STAB_REAL, RAW_REAL, pl), STAB_REAL, atol=1e-6)
    # B's own mapping is used for distances, so at lift 0 the hybrid maps exactly where foot_stab maps under B
    Hb = np.asarray(HB_REAL).reshape(3, 3)
    assert np.allclose(M.PF.map_px(Hb, [M.hybrid_pixel(STAB_REAL, RAW_REAL, pl)]), M.PF.map_px(Hb, [STAB_REAL]))


def test_lift_pixel_moves_down_the_image_toward_the_camera():
    pl0 = M.lift_pixel([640.0, 500.0], CAM, W, H, 0.0)
    pl = M.lift_pixel([640.0, 500.0], CAM, W, H, 0.25)
    assert np.allclose(pl0, [640.0, 500.0], atol=1e-6)
    assert pl[1] > 500.0                                      # the floor point beneath the ankle sits lower in the frame
    assert M.lift_pixel([640.0, 500.0], CAM, W, H, 0.35)[1] > pl[1]


def test_fill_no_change():
    row = {"dT": {"stab": 3.0, "hyb": None}, "gT": {"stab": True, "hyb": None}}
    out = M.fill_no_change(row)
    assert out["dT"]["hyb"] == 3.0 and out["gT"]["hyb"] is True and row["dT"]["hyb"] is None
    row2 = {"dT": {"stab": 3.0, "hyb": 1.0}, "gT": {"stab": True, "hyb": False}}
    assert M.fill_no_change(row2)["dT"]["hyb"] == 1.0 and M.fill_no_change(row2)["gT"]["hyb"] is False


def test_project_stage2():
    per = {g: {"rates": [{"crops": 100, "pose_seconds": 20.0}], "counters": {"crops_all_aligned": 6000},
               "selection_seconds": 60.0} for g in M.PF.POOL}
    align = {g: {"windows": {"a": {"verdict": "aligned", "seconds": 300.0}, "b": {"verdict": "undetermined"}}}
             for g in M.PF.POOL}
    out = M.project_stage2(per, align)
    assert out["pose_seconds_per_crop"] == pytest.approx(0.2)
    assert out["per_game"][M.PF.CHA]["total_s"] == pytest.approx(300 + 60 + 1200)
    assert out["total_min"] == pytest.approx(2 * 1560 / 60) and out["over_120_min"] is False
