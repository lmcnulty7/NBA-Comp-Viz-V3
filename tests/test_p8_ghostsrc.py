"""tools.p8_ghostsrc: the pure classification and summary."""

from tools import p8_ghostsrc as G


def test_iou_and_off_court():
    assert G.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0 and G.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert G.off_court([47.0, 25.0]) is None and G.off_court([-1.9, 25.0]) is None
    assert G.off_court([-3.0, 25.0]) == "near" and G.off_court([47.0, 57.9]) == "near"
    assert G.off_court([47.0, 60.0]) == "far" and G.off_court([110.0, 25.0]) == "far"


def test_cut_before():
    assert G.cut_before(100, [50], 59.94) and not G.cut_before(200, [50], 59.94) and not G.cut_before(40, [50], 59.94)


def test_classify_order():
    base = {"A": False, "B": None, "C": False, "dist": 4.0}
    assert G.classify(base) == "D_near" and G.classify({**base, "dist": 9.0}) == "D_far"
    assert G.classify({**base, "C": True}) == "C"
    assert G.classify({**base, "C": True, "B": "far"}) == "B_far"
    assert G.classify({**base, "A": True, "B": "near", "C": True}) == "A"


def _row(cls, after=True, ghost=True, src="ours", **f):
    return {"src": src, "after": after, "ghost": ghost, "cls": cls, "a1": f.get("a1", False), "a2": False,
            "a3": False, "b": f.get("b"), "c": f.get("c", False)}


def test_summarize_and_leads():
    rows = [_row("B_near", b="near"), _row("B_far", b="far"), _row("D_far"), _row("A", a1=True),
            _row("C", c=True, after=False), _row("D_near", src="sidecar"),
            _row("D_near", ghost=False, b="near")]
    s = G.summarize(rows)
    assert s["ours_after"]["n"] == 4 and s["ours_after"]["classes"] == {"A": 1, "B": 2, "C": 0, "D": 1}
    assert s["ours_after"]["shares"]["B"] == 0.5 and s["ours_before"]["n"] == 1 and s["sidecar_after"]["n"] == 1
    assert s["ours_after_not_ghost"] == {"n": 1, "with_flag": {"A": 0, "B": 1, "C": 0}}
    named = G.leads({"gsw_cha_2016": s}, s)
    assert [x["class"] for x in named] == ["B"]



def test_frame_pairs():
    import numpy as np
    sv = np.array([[0.0, 0.0], [20.0, 0.0], [40.0, 0.0]] + [[100.0, 100.0]] * 7)
    inside = np.array([True, True, True] + [False] * 7)
    feet = np.array([[0.5, 0.0], [24.0, 0.0], [60.0, 0.0]])   # finds p0; p1's box 4 ft off; one far ghost
    c = G.frame_pairs(feet, sv, inside)
    assert c == {"in_frame": 3, "missed": 2, "ghosts": 2, "pairs": 1, "pairs_tight": 1}
    c = G.frame_pairs(np.array([[0.5, 0.0], [25.0, 0.0]]), sv, inside)
    assert c["pairs"] == 1 and c["pairs_tight"] == 0                   # 5 ft: within 6.0, not 4.5
    assert G.frame_pairs(np.zeros((0, 2)), sv, inside)["missed"] == 3


def test_stab_feet_and_explained():
    import numpy as np
    raw = np.array([[1.0, 1.0], [5.0, 5.0]])
    out = G.stab_feet([[0, 0, 10, 10], [50, 50, 60, 60]], [[0, 0, 10, 11]], raw, np.array([[2.0, 2.0]]))
    assert out.tolist() == [[2.0, 2.0], [5.0, 5.0]]
    e = G.explained({"in_frame": 100, "missed": 20, "ghosts": 15, "pairs": 6},
                    {"in_frame": 100, "missed": 25, "ghosts": 5, "pairs": 2})
    assert abs(e["dE"] - 0.05) < 1e-12 and abs(e["dE_from_pairs"] - 0.08) < 1e-12 and abs(e["explained_share"] - 1.6) < 1e-12
    assert G.explained({"in_frame": 100, "missed": 20, "ghosts": 5, "pairs": 0},
                       {"in_frame": 100, "missed": 25, "ghosts": 5, "pairs": 0})["explained_share"] is None
