"""tools.p8_posefoot (docs/research/P8_posefoot.md version 1.1), stage 0a: the refusal, the null, the rules, the
readings and the report rows, on synthetic data. No video, no model, no workspace."""

import numpy as np
import pytest

from tools import p8_posefoot as M


def test_hou_is_refused_before_any_file_is_read(monkeypatch):
    original = M.collect0a
    called = []
    monkeypatch.setattr(M, "collect0a", lambda g: called.append(g))
    with pytest.raises(SystemExit):
        M.main(["stage0a", "hou_cle_2016"])
    assert called == []
    with pytest.raises(SystemExit):
        M.refuse_game("gsw_sac_2015")
    import tools.p8_rebaseline as RB
    monkeypatch.setattr(RB, "read_json", lambda p: (_ for _ in ()).throw(AssertionError("a file was read")))
    with pytest.raises(SystemExit):
        original("hou_cle_2016")


def test_scorer_mapping_matches_footbias():
    from tools.p8_footbias import scorer_court
    H = np.array([[0.1, 0.01, -3.0], [0.0, 0.12, -5.0], [0.0, 0.0001, 1.0]])
    px = [[100.3, 200.7], [640.1, 400.9]]
    assert np.allclose(M.map_px(H, px), scorer_court(H, px))
    assert M.map_px(H, []).shape == (0, 2)


def test_null_partner_and_foot():
    hts = [180.0, 150.0, 175.0, 200.0]
    assert M.null_partner(hts, 0, [True] * 4) == 2
    assert M.null_partner(hts, 0, [True, True, False, True]) == 3        # 200 is nearer 180 than 150 (tie broken low)
    assert M.null_partner([180.0], 0, [True]) is None
    assert M.null_partner([100.0, 90.0, 110.0], 0, [True] * 3) == 1      # equal distance: lower index
    assert M.null_foot([10.0, 20.0], [2.0, -4.0], 180.0, 90.0) == [14.0, 12.0]


def test_speed_bins():
    assert [M.speed_bin(v) for v in (None, 0.0, 4.99, 5.0, 9.99, 10.0, 30.0)] == \
        [None, "under 5", "under 5", "5 to 10", "5 to 10", "10 and over", "10 and over"]


def own(R=0.30, Rn=0.15, mt=(-0.8, [-1.0, -0.6]), mb=-0.1):
    return {"R": R, "R_null": Rn, "med_T": {"value": mt[0], "interval": mt[1]}, "med_B": {"value": mb}}


def hits(mt=0.05, lost=0.03):
    return {"med_T": {"value": mt}, "lost": lost}


def test_rules():
    assert M.rule_set(own(), hits(), {"value": -0.02}, "x")["all"]
    assert not M.rule_set(own(R=0.24, Rn=0.15), hits(), {"value": -0.02}, "x")["rule1_null_margin"]
    assert M.rule_set(own(R=0.25, Rn=0.15), hits(), {"value": -0.02}, "x")["rule1_null_margin"]
    assert not M.rule_set(own(mt=(-0.6, [-0.9, 0.01])), hits(), {"value": -0.02}, "x")["rule2_median_T"]
    assert not M.rule_set(own(mt=(-0.4, [-0.9, -0.1])), hits(), {"value": -0.02}, "x")["rule2_median_T"]
    assert not M.rule_set(own(mb=0.01), hits(), {"value": -0.02}, "x")["rule3_median_B"]
    assert not M.rule_set(own(), hits(lost=0.06), {"value": -0.02}, "x")["rule4_hits"]
    assert not M.rule_set(own(), hits(mt=0.11), {"value": -0.02}, "x")["rule4_hits"]
    assert not M.rule_set(own(), hits(), {"value": -0.009}, "x")["rule5_est"]


def test_readings_0a():
    ok = M.rule_set(own(), hits(), {"value": -0.02}, "x")
    bad4 = M.rule_set(own(), hits(lost=0.2), {"value": -0.02}, "x")
    r = M.readings_0a({M.CHA: ok, M.BKN: bad4})
    assert r == {"c_stabiliser_finding": True, "d_pose_reads_hybrid_only": True}
    r = M.readings_0a({M.CHA: ok, M.BKN: M.rule_set(own(R=0.1), hits(), {"value": -0.02}, "x")})
    assert r == {"c_stabiliser_finding": False, "d_pose_reads_hybrid_only": False}


def row(wi, kind, cls, stab, raw, null, g_stab=False, g_raw=False, speed=7.0):
    return {"wi": wi, "kind": kind, "class": cls, "h": 180.0, "speed": speed,
            "dT": {"stab": stab, "raw": raw, "null": null}, "dB": {"stab": stab, "raw": raw, "null": null},
            "gT": {"stab": g_stab, "raw": g_raw, "null": False}}


def test_unit_rows_counts_and_est():
    rows = []
    for wi in range(4):
        rows += [row(wi, "P-ii", "own", 3.0, 2.0, 3.5, g_stab=True), row(wi, "P-ii", "own", 3.2, 3.0, 2.0),
                 row(wi, "P-ii", "switch", 3.0, 1.0, 3.0),
                 row(wi, "hit", "own", 1.0, 1.2, 1.0), row(wi, "hit", "own", 2.0, 2.6, 2.0, g_raw=True)]
    per = {M.CHA: {"windows": 4, "rows": rows, "in_frame_by_window": [100] * 4, "nr_boxes_by_window": [90] * 4}}
    out = M.unit_rows(per, [M.CHA], "raw", {M.CHA: 16})
    assert out["P-ii own"]["n"] == 8 and out["P-ii own"]["R"] == pytest.approx(0.5)
    assert out["P-ii own"]["R_null"] == pytest.approx(0.5)
    assert out["P-ii own"]["med_T"]["value"] == pytest.approx(np.median([-1.0, -0.2]))
    assert out["hits"]["lost"] == pytest.approx(0.5) and out["s_hit"] == pytest.approx(2.0)
    # est: per window -(2 rescued P-ii) + 2 x (1 lost hit) = 0, over 100 in-frame players
    assert out["est"]["value"] == pytest.approx(0.0)
    # ghosts: P-ii own row 1 stops being a ghost (-1); one hit becomes a ghost (+1, scaled by 2)
    assert out["ghost_change_per_box"]["value"] == pytest.approx((-1 + 2) / 90)
    assert out["R_by_speed"]["5 to 10"]["value"] == pytest.approx(0.5)
    assert out["R_by_speed"]["under 5"]["value"] is None


def test_constants_match_the_design():
    assert (M.NULL_MARGIN, M.MED_BAR_FT, M.HIT_MED_FT, M.HIT_LOST, M.EST_BAR) == (0.10, -0.5, 0.10, 0.05, -0.010)
    assert M.OWN == ("own", "own, with crossings", "own, misplaced, with crossings", "own, misplaced")
    assert M.POOL == (M.CHA, M.BKN) and (M.BOOT_N, M.SEED) == (2000, 0)


def test_cross_frame_partner_frame():
    fr = [0, 6, 60, 66, 120]
    assert M.cross_frame(fr, 6, 59.94) == 66
    assert M.cross_frame(fr, 120, 59.94) == 60
    assert M.cross_frame([0, 6, 12], 6, 59.94) is None


def test_rule1_reads_r_and_r_null_on_the_same_boxes():
    rows = []
    for wi in range(2):
        rows += [row(wi, "P-ii", "own", 3.0, 2.0, 3.5), row(wi, "P-ii", "own", 3.0, 2.0, None),
                 row(wi, "hit", "own", 1.0, 1.0, 1.0)]
    per = {M.CHA: {"windows": 2, "rows": rows, "in_frame_by_window": [50, 50], "nr_boxes_by_window": [40, 40]}}
    out = M.unit_rows(per, [M.CHA], "raw", {M.CHA: 2})
    assert out["P-ii own"]["R"] == pytest.approx(1.0) and out["P-ii own"]["R_null"] == pytest.approx(0.0)
    assert out["P-ii own"]["R_all_boxes"]["value"] == pytest.approx(1.0)
    assert out["P-ii own"]["cross_frame_null"]["R"]["value"] is None


def test_the_only_writes_are_the_records():
    """Writer check: the module writes through atomic_write only, at its two record paths."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(M))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    names = [getattr(c.func, "id", getattr(c.func, "attr", None)) for c in calls]
    assert names.count("atomic_write") == 5                  # the 0a, alignment and model records
    assert not any(n in ("write_text", "write_bytes", "unlink", "rmtree", "remove", "rename") for n in names)
    assert "open" not in names



# ── stage A (section 14) ───────────────────────────────────────────────────────────────────────────────────────────
def box(x, y, tid, w=40, h=100):
    return {"bbox": [x, y, x + w, y + h], "tid": tid, "foot_raw": [x + w / 2, y + h], "foot_stab": [x + w / 2, y + h]}


def side_of(n_frames, n_boxes=5, stride=6, speed=None):
    speed = speed or (lambda f: 2.0)
    frames = []
    for i in range(n_frames):
        f = i * stride
        frames.append({"frame": f, "boxes": [box(100 * t + speed(f) * i, 300, t) for t in range(n_boxes)]})
    return {"stride": stride, "frames": frames}


def test_motion_and_pick_frames():
    a, b = {"boxes": [box(0, 0, 1), box(50, 0, 2)]}, {"boxes": [box(3, 4, 1), box(50, 0, 3)]}
    assert M.motion(a, b) == pytest.approx(5.0)
    assert M.motion(a, {"boxes": [box(0, 0, 9)]}) is None
    picks, n = M.pick_frames(side_of(11))                      # 9 eligible (the two ends lack a neighbour)
    assert picks == [] and n == 9
    picks, n = M.pick_frames(side_of(32))
    assert n == 30 and len(picks) == 15 and picks == sorted(picks)
    s = side_of(32)
    s["frames"][7]["boxes"] = s["frames"][7]["boxes"][:3]      # fewer than 4 boxes: not eligible
    assert M.pick_frames(s)[1] == 29
    s["frames"][7]["boxes"] = []                                # v1.3: frames 6 and 8 lose a neighbour with boxes
    assert M.pick_frames(s)[1] == 27
    fast = side_of(32, speed=lambda f: 9.0 if f == 6 * 5 else 1.0)
    assert 6 * 4 in M.pick_frames(fast)[0] or 6 * 6 in M.pick_frames(fast)[0]   # neighbours of the fast frame move most


def test_frame_ious():
    side = [[0, 0, 10, 10], [100, 100, 110, 110], [500, 500, 510, 510]]
    pred = [[0, 0, 10, 10], [101, 100, 111, 110], [505, 505, 515, 515]]
    v = M.frame_ious(side, pred)
    assert v[0] == pytest.approx(1.0) and v[1] == pytest.approx(90 / 110)
    assert v[2] == pytest.approx(25 / 175)                     # above the 0.1 floor
    assert M.frame_ious(side, [[900, 900, 910, 910]]) == [0.0, 0.0, 0.0]
    assert M.frame_ious(side, []) == [0.0, 0.0, 0.0] and M.frame_ious([], pred) == []
    assert M.frame_ious([[0, 0, 10, 10]], [[8, 8, 18, 18]]) == [0.0]   # IoU 4/196 is below the floor


def test_align_rule():
    st = lambda m1, m0, p1, l0: {"M": {"-1": m1, "0": m0, "+1": p1}, "L0": l0}
    assert M.align_verdict(st(0.80, 0.85, 0.80, 0.90)) == "aligned"
    assert M.align_verdict(st(0.90, 0.85, 0.80, 0.90)) == "misaligned"     # a neighbour beats f by 0.05
    assert M.align_verdict(st(0.899, 0.85, 0.80, 0.90)) == "aligned"
    assert M.align_verdict(st(0.85, 0.85, 0.85, 0.90)) == "aligned"         # ties read aligned
    assert M.align_verdict(st(0.70, 0.85, 0.70, 0.84)) == "misaligned"     # v1.3: level under 0.85
    assert M.align_verdict(st(0.70, 0.85, 0.70, 0.85)) == "aligned"
    assert M.align_verdict(st(0.70, 0.81, 0.70, 0.90)) == "misaligned"     # v1.3: M_0 under 0.82
    assert M.align_verdict(st(0.70, 0.82, 0.70, 0.90)) == "aligned"
    assert M.align_verdict(st(None, None, None, None)) == "undetermined"
    s = M.align_stats([{"-1": [0.5, 0.0], "0": [0.9, 0.2], "+1": [0.6, 0.6]}, {"-1": [0.5], "0": [0.8], "+1": [0.4]}])
    assert s["M"]["0"] == pytest.approx((0.9 + 0.2 + 0.8) / 3) and s["L0"] == pytest.approx(0.85)


def test_game_stop_bar_and_cross_table():
    v = {f"w{i}": "aligned" for i in range(15)} | {"w15": "misaligned", "w16": "undetermined", "w17": "misaligned",
                                                     "w18": "misaligned", "w19": "misaligned"}
    g = M.game_alignment(v)
    assert g["windows"] == 20 and g["share_not_aligned"] == pytest.approx(0.25) and not g["stop"]
    v["w0"] = "misaligned"
    assert M.game_alignment(v)["stop"]
    assert M.cross_table({"a": "aligned", "b": "misaligned", "c": "aligned"}, {"a": False, "b": True, "z": True}) == \
        {"aligned|a3align_aligned": 1, "misaligned|a3align_misaligned": 1}


class FakeCap:
    """A capture whose frames carry their own index; set() is a seek and is logged."""
    def __init__(self, path=None, api=None, n=1200, fps=59.94, opened=True):
        self.n, self.fps, self.pos, self.sets, self.opened = n, fps, 0, [], opened
    def isOpened(self):
        return self.opened
    def get(self, prop):
        import cv2
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_COUNT: self.n}.get(prop, 0)
    def set(self, prop, v):
        self.sets.append(int(v)); self.pos = int(v)
    def grab(self):
        if self.pos >= self.n:
            return False
        self.pos += 1
        return True
    def read(self):
        if self.pos >= self.n:
            return False, None
        f = np.full((2, 2, 3), self.pos % 256, np.uint8); f[0, 1, 0] = self.pos // 256
        self.pos += 1
        return True, f
    def getBackendName(self):
        return "FAKE"
    def release(self):
        pass


def idx_of(img):
    return int(img[0, 0, 0]) + 256 * int(img[0, 1, 0])


def test_pregate_intervals_are_the_builds():
    from videoseq import SeqReader
    cap = FakeCap(n=600)
    live = {60, 90, 300}
    iv = M.pregate_intervals(SeqReader(cap), 600, 30, 90, lambda img: idx_of(img) in live)
    assert iv == [[-30, 180], [210, 390]]                     # 60 and 90 merge; 300 apart
    assert cap.sets == []                                     # 30-frame steps never seek


def test_main_reads_jump_and_seek_like_the_build():
    from videoseq import SeqReader
    cap = FakeCap(n=1200)
    reader = M.SeekCount(SeqReader(cap))
    M.pregate_intervals(reader, 1200, 30, 90, lambda img: False)          # the decoder ends at the clip's end
    seeks_pre = len(reader.seeks)
    accepted = set(range(0, 97, 6)) | set(range(702, 900, 6))
    got = {}
    preds, reads = M.main_reads(reader, 1200, [[0, 100], [700, 900]], 6, accepted, 10_000, {48, 706},
                                lambda img: got.setdefault(idx_of(img), True))
    assert reads[0] == 0 and 102 not in reads and reads[reads.index(96) + 1] == 700     # jump to the interval start
    assert reader.seeks[seeks_pre:] == [0, 700]               # back to 0 after the pre-gate pass, then the long jump
    assert cap.sets[-2:] == [0, 700]
    assert 48 in preds and 706 in preds and got == {48: True, 706: True}   # the decoded frame is the frame asked for
    preds, reads = M.main_reads(M.SeekCount(SeqReader(FakeCap(n=1200))), 1200, [[0, 100], [700, 900]], 6, accepted,
                                5, set(), lambda img: None)
    assert reads == [0, 6, 12, 18, 24]                        # stops at max_frames accepted frames


def test_open_like_build_tries_ffmpeg_first(monkeypatch):
    import cv2
    calls = []

    def fake(path, *api):
        calls.append(api)
        return FakeCap(opened=bool(not api))                  # the FFMPEG open fails, the default one works
    monkeypatch.setattr(cv2, "VideoCapture", fake)
    cap = M.open_like_build("clip.mp4")
    assert calls == [(cv2.CAP_FFMPEG,), ()] and cap.isOpened()


def test_build_read_validates_the_sidecar_frames(monkeypatch):
    import cv2
    import config
    monkeypatch.setattr(cv2, "VideoCapture", lambda path, *api: FakeCap(n=1200))
    pg, pad = config.pregate_params(59.94)
    side = {"stride": 6, "frames": [{"frame": f, "boxes": []} for f in range(0, 120, 6)]}
    out = M.build_read("clip.mp4", side, 10_000, lambda img: idx_of(img) in (0, 30, 60), {12},
                       lambda img: idx_of(img))
    assert out["validation"]["passes"] and out["preds"] == {12: 12} and out["backend"] == "FAKE"
    side["frames"].append({"frame": 600, "boxes": []})       # a frame the replay never reads
    assert not M.build_read("clip.mp4", side, 10_000, lambda img: idx_of(img) in (0, 30, 60), set(),
                            lambda img: None)["validation"]["passes"]


def test_align_refuses_hou_before_any_read(monkeypatch):
    original = M.collect_align
    called = []
    monkeypatch.setattr(M, "collect_align", lambda g: called.append(g))
    with pytest.raises(SystemExit):
        M.main(["align", "hou_cle_2016"])
    assert called == []
    import tools.p8_rebaseline as RB
    monkeypatch.setattr(RB, "read_json", lambda p: (_ for _ in ()).throw(AssertionError("a file was read")))
    with pytest.raises(SystemExit):
        original("hou_cle_2016")


def test_stage_a_constants_match_section_14():
    assert (M.ALIGN_SEGMENTS, M.ALIGN_MIN_ELIGIBLE, M.ALIGN_MIN_BOXES) == (15, 10, 4)
    assert (M.ALIGN_MARGIN, M.ALIGN_LEVEL, M.ALIGN_M0, M.PAIR_FLOOR, M.LEVEL_PAIR, M.STOP_SHARE) == \
        (0.05, 0.85, 0.82, 0.1, 0.3, 0.25)
    assert len(M.SMOKE_WINDOWS) == 6


def test_offset_sizes(tmp_path):
    import json as js
    p = tmp_path / "w_frames.json"
    p.write_text(js.dumps({"frames": [{"boxes": [{"foot_raw": [3.0, 4.0], "foot_stab": [0.0, 0.0]},
                                                  {"foot_raw": [1.0, 1.0], "foot_stab": [1.0, 1.0]}]}]}))
    o = M.offset_sizes([p])
    assert o["boxes"] == 2 and o["median_px"] == pytest.approx(2.5) and o["sidecars"] == 1


def test_by_cause_and_env_guard(monkeypatch):
    w = {"a": {"verdict": "aligned"}, "b": {"verdict": "misaligned"},
         "c": {"verdict": "undetermined", "why": "fewer than 10 eligible frames"},
         "d": {"verdict": "undetermined", "why": "fewer than 10 eligible frames"}}
    assert M.by_cause(w) == {"misaligned": {"n": 1, "windows": ["b"]},
                             "undetermined: fewer than 10 eligible frames": {"n": 2, "windows": ["c", "d"]}}
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(SystemExit):
        M.collect_align("gsw_cha_2016")
