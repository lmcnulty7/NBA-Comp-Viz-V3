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
