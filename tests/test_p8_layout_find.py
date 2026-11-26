"""tools.p8_layout_find: token parsing, grouping, the proposal and the validation bar, on synthetic OCR reads. No OCR,
no video."""

import pytest

from tools import p8_layout_find as L


def test_clock_and_period_tokens():
    assert L.clock_token("2:17") and L.clock_token("11.46") and L.clock_token("43.3") and L.clock_token("12:00")
    assert not L.clock_token(":10") and not L.clock_token("28") and not L.clock_token("13:00") and not L.clock_token("9:75")
    assert L.period_token("1ST")[0] == "ordinal" and L.period_token("IST")[0] == "ordinal"
    kind, share = L.period_token("1ST QTR")
    assert kind == "ordinal" and share == pytest.approx(3 / 7)
    assert L.period_token("OT")[0] == "ordinal"
    assert L.period_token("SECOND QTR") == ("word", 1.0)
    assert L.period_token("GOLDEN STATE") is None and L.period_token("28") is None


def test_iou_and_overlap_take_the_best_position():
    assert L.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0 and L.iou([0, 0, 10, 10], [20, 0, 30, 10]) == 0.0
    layout = {"period_box": (0, 0, 10, 10), "clock_box": (20, 0, 30, 10),
              "alternates": [((100, 100, 110, 110), (120, 100, 130, 110))]}
    ov = L.overlap_with({"period_box": [100, 100, 110, 110], "clock_box": [120, 100, 130, 110]}, layout)
    assert ov == {"clock_iou": 1.0, "period_iou": 1.0}


def _reads(n=40, clock_at=(1000, 630, 1060, 652), period_at=(1090, 630, 1160, 652), period_text="1ST QTR",
           every=1):
    reads = []
    for f in range(0, n, every):
        reads.append((f, list(clock_at), f"{10 - f % 10}:{(f * 7) % 60:02d}"))
        reads.append((f, list(period_at), period_text))
        reads.append((f, [300, 630, 330, 652], "28"))                  # a score, ignored
        reads.append((f, [1200, 630, 1230, 652], ":10"))               # the shot clock, ignored
    return reads


def test_proposal_finds_the_clock_and_cuts_the_period_to_its_ordinal():
    p = L.propose(_reads(), 40, 1280, 720)
    assert p["status"] == "ok" and not p["period_words"]
    pad = round(L.PAD_FRAC * 22)
    assert p["clock_box"] == [1000 - pad, 630 - pad, 1060 + pad, 652 + pad]
    assert p["period_box"][0] == 1090 - pad and p["period_box"][2] == round(1090 + 70 * 3 / 7 + L.PAD_FRAC * 22)


def test_no_proposal_when_the_clock_is_rare_or_no_period_on_its_row():
    assert L.propose(_reads(every=8), 40, 1280, 720)["status"] == "no_clock"
    far = L.propose(_reads(period_at=(100, 100, 170, 122)), 40, 1280, 720)
    assert far["status"] == "no_period"


def test_validation_bar():
    ok = {"chosen": L.CHECK_NAME}
    assert L.validation_passes({"clock_iou": 0.5, "period_iou": 0.6}, ok)
    assert not L.validation_passes({"clock_iou": 0.49, "period_iou": 0.9}, ok)
    assert not L.validation_passes({"clock_iou": 0.9, "period_iou": 0.9}, {"chosen": None})


def test_the_tool_guards_every_video_and_validates_on_existing_layouts_only():
    src = open(L.__file__).read()
    assert "refuse_cv_test(" in src and "game_of(tag)" in src
    assert "nop_gsw_2015" not in L.VALIDATION and len(L.VALIDATION) == 8


# ── D180: register one validated proposal, unedited ─────────────────────────────────────────────────────────────────
VAL = {"videos": {
    "okc_sac_2015": {"proposal": {"status": "ok", "clock_box": [942, 586, 1008, 624], "period_box": [828, 586, 879, 624],
                                  "period_words": False, "clock_frames": 26, "period_frames": 27, "frames": 40},
                     "check": {"chosen": "proposal", "plausible_share": 0.7, "consistent_share": 0.65, "pairs": 40}},
    "nyk_bos_2016": {"proposal": {"status": "no_period", "clock_box": [1110, 621, 1184, 655]}, "check": None},
    "por_okc_2016": {"proposal": {"status": "no_clock"}}}}


def test_only_a_proposal_that_passed_the_frozen_check_is_taken():
    from tools import p8_layout_find as L
    prop, check = L.validated_proposal(VAL, "okc_sac_2015")
    assert prop["clock_box"] == [942, 586, 1008, 624] and check["consistent_share"] == 0.65
    for tag in ("nyk_bos_2016", "por_okc_2016", "hou_cle_2016"):
        with pytest.raises(SystemExit):
            L.validated_proposal(VAL, tag)


def test_register_validated_is_the_d180_exception_only(tmp_path, monkeypatch):
    import json
    from argparse import Namespace
    from tools import p8_layout_find as L
    for tag, dec in (("nyk_bos_2016", "D180"), ("okc_sac_2015", "D179"), ("por_okc_2016", "D180")):
        with pytest.raises(SystemExit):
            L.cmd_register_validated(Namespace(tag=tag, decision=dec))
    repo = tmp_path / "repo"
    (repo / "reports" / "research").mkdir(parents=True)
    reg = json.loads((L.REPO / "reports" / "research" / "p8_registry.json").read_text())
    (repo / "reports" / "research" / "p8_registry.json").write_text(json.dumps(reg))
    root = tmp_path / "p8"
    (root / "layout_find").mkdir(parents=True)
    (root / "layout_find" / "validation.json").write_text(json.dumps(VAL))
    monkeypatch.setattr(L, "REPO", repo)
    monkeypatch.setattr(L, "p8_root", lambda: root)
    L.cmd_register_validated(Namespace(tag="okc_sac_2015", decision="D180"))
    out = json.loads((repo / "reports" / "research" / "p8_registry.json").read_text())
    lay = out["layouts"]["auto_okc_sac_2015"]
    assert lay["clock_box"] == [942, 586, 1008, 624] and lay["period_box"] == [828, 586, 879, 624]   # unedited
    assert "period_words" not in lay and "D180" in lay["measured_on"] and "3 of 8" in lay["measured_on"]
    assert out["games"]["okc_sac_2015"]["layout"] == "auto_okc_sac_2015"
    assert out["layouts"]["msg15_720"] == reg["layouts"]["msg15_720"]           # nothing else changes
    assert {k: v for k, v in out["games"].items() if k != "okc_sac_2015"} == \
        {k: v for k, v in reg["games"].items() if k != "okc_sac_2015"}
