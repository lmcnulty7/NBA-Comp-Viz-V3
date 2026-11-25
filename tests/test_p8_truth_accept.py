"""tools.p8_truth_accept and tools.p8_truth_v2 measure-dev: the acceptance checks of docs/research/P8_truth_arenas.md
section 5 on synthetic inputs, and the report-only measure path. No video, no model, no SportVU file."""

import json
from argparse import Namespace

import pytest

from tools import p8_truth_accept as A
from tools import p8_truth_v2 as V


def test_the_bars_are_the_registered_ones():
    assert A.BARS == {"reproj_px_max": 10.0, "accepted_min": 1000, "mapped_s_min": 1500.0, "offset_median_s_max": 0.54,
                      "offset_share_max": 0.7945, "offset_n_share_min": 0.8}


def test_check_a_registration():
    assert A.check_a({"partial": False, "camera": {"status": "ok", "reproj_px_median_after": 10.0}})["pass"]
    assert not A.check_a({"partial": False, "camera": {"status": "ok", "reproj_px_median_after": 10.01}})["pass"]
    assert not A.check_a({"partial": False, "status": "camera_fit_failed",
                          "camera": {"status": "failed", "reproj_px_median_after": 5.0}})["pass"]
    assert not A.check_a({"partial": False, "camera": {"status": "ok"}})["pass"]


def test_a_and_b_need_a_complete_fit():
    fit = {"camera": {"status": "ok", "reproj_px_median_after": 6.0}, "accepted": 3000}
    for partial in (True, None):                                  # a smoke fit, or a result that does not say
        r = {**fit, **({"partial": partial} if partial is not None else {})}
        assert not A.check_a(r)["pass"] and not A.check_b(r, 2500.0)["pass"]
    assert A.check_a({**fit, "partial": False})["pass"] and A.check_b({**fit, "partial": False}, 2500.0)["pass"]


def test_check_b_coverage():
    assert A.check_b({"partial": False, "accepted": 1000}, 1500.0)["pass"]
    assert not A.check_b({"partial": False, "accepted": 999}, 3000.0)["pass"]
    assert not A.check_b({"partial": False, "accepted": 4000}, 1499.9)["pass"]
    assert not A.check_b({"partial": False}, 3000.0)["pass"]


def test_check_c_offset_stability_with_the_in_use_bars():
    assert A.check_c({"n": 70, "median_abs_s": 0.54, "share_over_0_2_s": 0.7945}, 80)["pass"]
    assert not A.check_c({"n": 70, "median_abs_s": 0.5401, "share_over_0_2_s": 0.5}, 80)["pass"]   # unrounded
    assert not A.check_c({"n": 70, "median_abs_s": 0.3, "share_over_0_2_s": 0.7946}, 80)["pass"]
    assert not A.check_c({"n": 89, "median_abs_s": 0.6, "share_over_0_2_s": 0.7978}, 92)["pass"]  # gsw_ind_2016
    assert not A.check_c({"n": 0, "median_abs_s": None, "share_over_0_2_s": None}, 80)["pass"]


def test_check_c_needs_most_windows_measured():
    st = {"n": 64, "median_abs_s": 0.4, "share_over_0_2_s": 0.6}
    assert A.check_c(st, 80)["pass"]                              # 0.8 of 80
    assert not A.check_c(st, 81)["pass"]
    assert not A.check_c(st, 0)["pass"]


def test_verdicts_on_games_already_built():
    # the fit D98 found broken (gsw_ind_2016 on MPS): 18.35 px, 970 accepted -> unused
    broken = {"camera": {"status": "ok", "reproj_px_median_after": 18.35}, "accepted": 970}
    broken = {**broken, "partial": False}
    a, b = A.check_a(broken), A.check_b(broken, 2400.0)
    assert A.verdict(a, b, A.check_c({"n": 89, "median_abs_s": 0.6, "share_over_0_2_s": 0.7978}, 92)) == "unused"
    # hou_cle_2016's fit and gsw_sac_2015's frozen stability (73 of 82 windows) -> use
    good = {"partial": False, "camera": {"status": "ok", "reproj_px_median_after": 6.4}, "accepted": 4427}
    c = A.check_c({"n": 73, "median_abs_s": 0.52, "share_over_0_2_s": 0.7945}, 82)
    assert A.verdict(A.check_a(good), A.check_b(good, 2635.0), c) == "use"
    # every check passes but one window was not built with the CPU detector -> unused
    assert A.verdict(A.check_a(good), A.check_b(good, 2635.0), c, platform_ok=False) == "unused"
    # coverage fine, registration off -> apart
    off = {"partial": False, "camera": {"status": "ok", "reproj_px_median_after": 12.0}, "accepted": 3000}
    assert A.verdict(A.check_a(off), A.check_b(off, 2000.0), c) == "apart"


def test_platform_ok_is_enforced_per_window():
    cpu = {"resolved": {"detector": "cpu", "court": "mps", "gate": "mps"}}
    mps = {"resolved": {"detector": "mps", "court": "mps", "gate": "mps"}}
    assert A.platform_ok([cpu, cpu])
    assert not A.platform_ok([cpu, mps]) and not A.platform_ok([cpu, None]) and not A.platform_ok([])


def test_platform_tally():
    cpu = {"resolved": {"detector": "cpu", "court": "mps", "gate": "mps"}}
    t = A.platform_tally([cpu, cpu, None])
    assert t == {json.dumps(cpu["resolved"], sort_keys=True): 2, "no record": 1}


def _rows():
    def row(fv3, fr23, vv3, vr23, stored=None):
        return {"frozen": {"v3": fv3, "r23": fr23},
                "v2": {"v3": {"offset_s": vv3, "at_edge": False}, "r23": {"offset_s": vr23, "at_edge": False}},
                "stored_frozen_v3": fv3 if stored is None else stored, "candidates_v3": {"frozen": 40, "v2": 44}}
    return {"okc_sac_2015_s01_w00": row(0.1, 0.9, 0.2, 0.3), "okc_sac_2015_s01_w01": row(0.4, 0.5, 0.4, 0.45),
            "okc_sac_2015_s02_w00": row(-0.2, 0.4, 0.0, 0.1), "okc_sac_2015_s02_w01": row(0.3, 0.3, 0.2, 0.2, stored=1.0)}


def test_measure_dev_record_is_report_only_and_summarised():
    rec = V.dev_record("okc_sac_2015", _rows(), 12.34)
    assert rec["report_only"] is True and "not part of the truth v2 registration" in rec["rule"]
    s = rec["summary"]
    assert s["frozen_recompute_mismatch"] == ["okc_sac_2015_s02_w01"]        # stored offset differs: left out
    assert s["frozen"]["n"] == 3 and s["frozen"]["median_abs_s"] == pytest.approx(0.6)
    assert "_pairs" not in s


def test_measure_dev_refuses_registered_games_and_report_never_reads_dev(tmp_path, monkeypatch):
    for t in V.POOLED + V.APART:
        with pytest.raises(SystemExit):
            V.cmd_measure_dev(Namespace(tag=t))
    monkeypatch.setenv("V3_P8_ROOT", str(tmp_path / "root"))
    assert V.dev_path("okc_sac_2015") == tmp_path / "root" / "truth_v2" / "dev" / "okc_sac_2015.json"
    p = V.dev_path("okc_sac_2015")
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(V.dev_record("okc_sac_2015", _rows(), 1.0)))
    res = V.cmd_report(Namespace())
    assert res["per_game"] == {} and res["pooled"] is None


def test_accept_end_to_end(tmp_path, monkeypatch):
    root = tmp_path / "ref"
    tag = "okc_sac_2015"
    (root / "labels").mkdir(parents=True)
    (root / "labels" / f"{tag}_result.json").write_text(json.dumps(
        {"game": tag, "partial": False, "camera": {"status": "ok", "reproj_px_median_after": 6.9}, "accepted": 2200}))
    (root / "sync").mkdir()
    for k, m in enumerate((900.0, 800.0)):
        (root / "sync" / f"{tag}_s{k:02d}_local_timemap.json").write_text(json.dumps({"mapped_video_s": m}))
    (root / "label_build").mkdir()
    (root / "builds" / "r23").mkdir(parents=True)
    for folder in (root / "label_build", root / "builds" / "r23"):
        for w in ("s01_w00", "s01_w01"):
            (folder / f"{tag}_{w}_frames.json").write_text("{}")
            (folder / f"{tag}_{w}_device.json").write_text(json.dumps(
                {"requested": {"detector": "cpu"}, "resolved": {"detector": "cpu", "court": "mps", "gate": "mps"}}))
    with pytest.raises(SystemExit):                                            # no measure-dev record yet
        A.accept(tag, root)
    (root / "truth_v2" / "dev").mkdir(parents=True)
    (root / "truth_v2" / "dev" / f"{tag}.json").write_text(json.dumps(V.dev_record(tag, _rows(), 1.0)))
    rows = _rows()
    del rows["okc_sac_2015_s02_w01"]                                      # 3 of 3 measured windows usable
    (root / "truth_v2" / "dev" / f"{tag}.json").write_text(json.dumps(V.dev_record(tag, {
        k: {**v, "frozen": {"v3": v["frozen"]["v3"], "r23": v["frozen"]["v3"] + 0.1}} for k, v in rows.items()}, 1.0)))
    out = A.accept(tag, root)
    assert out["use"] == "use" and out["platform_ok"] and out["checks"]["B"]["mapped_s"] == 1700.0
    assert out["label_windows"] == 2 and out["rebuild_windows"] == 2
    assert list(out["truth_platform"]["label"].values()) == [2]
    assert json.loads((root / "truth_accept" / f"{tag}.json").read_text()) == out
    from export.schemas import find_private
    assert find_private(json.dumps(out)) == []
    (root / "builds" / "r23" / f"{tag}_s01_w01_device.json").unlink()      # one rebuild window without a record
    assert A.accept(tag, root)["use"] == "unused"
