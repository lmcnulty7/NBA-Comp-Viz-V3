"""tools.p8_rebaseline: the pure rows, intervals, the truth change, the reproduction gate and the manifests."""

import json

import numpy as np
import pytest

from tools import p8_rebaseline as R


def _w(missed=30, far=10, ghosts=5, boxes=90, inf=100, m1=40, matched=None):
    return {"missed": missed, "far": far, "nr_ghosts": ghosts, "nr_boxes": boxes, "in_frame": inf,
            "unmatched_1to1": m1, "cov_acc": 6.0, "cov_live": 8.0, "team": {"n": 10, "majority": 8},
            "matched": matched or {"0:0": [1.0, 0.2, "near"], "0:1": [2.0, -0.1, "far"]}}


def test_ratio_values_and_placement():
    v = R.ratio_values([_w(), _w(missed=20, far=4)])
    assert v["missed"]["value"] == 0.25 and v["detection"]["value"] == 0.07
    assert abs(v["placement"]["value"] - 0.18) < 1e-12                   # missed minus detection
    assert abs(v["ghosts_per_box"]["value"] - 10 / 180) < 1e-12 and v["ghosts_per_player"]["value"] == 0.05
    assert v["net_error"]["value"] == 0.3 and v["net_error_1to1"]["value"] == 0.4


def test_unit_rows_intervals_contain_the_value():
    rng = np.random.default_rng(0)
    wins = [_w(missed=int(rng.integers(10, 40)), matched={f"{i}:0": [float(rng.random()), 0.1, "near"]})
            for i in range(12)]
    u = R.unit_rows(wins, n_boot=200)
    for k in ("missed", "net_error", "position_p50_ft", "team"):
        lo, hi = u[k]["interval"]
        assert lo <= u[k]["value"] <= hi, k
    assert u["counts"] == {"windows": 12, "in_frame": 1200, "nr_boxes": 1080}


def test_change_rows_match_the_registered_diffs():
    from tools.p8_e1_score import diffs
    prev = [_w(missed=30), _w(missed=20), _w(missed=25)]
    curr = [_w(missed=27), _w(missed=18), _w(missed=26)]
    c = R.change_rows(prev, curr, n_boot=100)
    assert abs(c["missed"]["change"] - (71 - 75) / 300) < 1e-12
    d = diffs([{"r23": a, "new": b} for a, b in zip(prev, curr)])
    assert c["position_p50_ft"]["change"] == d["position_p50_ft"] and c["team"]["players"] == "matched under both truths"


def test_gate_problems():
    mine = {"w": {f: 1 for f in R.GATE_FIELDS}}
    arm = {"w": {"new": {f: 1 for f in R.GATE_FIELDS}}}
    assert R.gate_problems(mine, arm) == []
    arm["w"]["new"]["far"] = 2
    assert R.gate_problems(mine, arm) == ["w: far differs"]
    assert R.gate_problems({}, arm)


def test_manifest_and_pin(tmp_path):
    (tmp_path / "labels" / "g").mkdir(parents=True)
    (tmp_path / "labels" / "g" / "g_s00_w00.jsonl").write_text("{}\n")
    (tmp_path / "labels" / "g_result.json").write_text("{}")
    (tmp_path / "other").mkdir()
    (tmp_path / "lb").symlink_to(tmp_path / "other")
    m = R.manifest(tmp_path, ("labels", "lb", "cache"))
    assert "labels/g/g_s00_w00.jsonl" in m and m["lb/"] == "link:other" and m["cache/"] == "absent"
    pin = R.labels_pin(tmp_path, "g")
    assert set(pin) == {"g_s00_w00.jsonl", "g_result.json"}


def test_timemap_content_ignores_run_fields():
    a = {"frames": [[0, 1, 700.0]], "spans": [1], "ocr_seconds": 3.0}
    assert R.timemap_content(a) == R.timemap_content({**a, "ocr_seconds": 9.0})
    assert R.timemap_content(a) != R.timemap_content({**a, "frames": [[0, 1, 700.1]]})


def test_order_puts_the_gate_first():
    assert R.ORDER[0] == R.BKN and list(R.truths(R.roots())[R.BKN])[0] == "previous"
    assert set(R.ORDER) == set(R.GAMES) and R.POOL == (R.CHA, R.BKN)


def test_report_refuses_score_files_from_another_run(tmp_path, monkeypatch):
    d = tmp_path / "work_ref" / "rebaseline"
    d.mkdir(parents=True)
    pairs = {f"{g}__{n}": {} for g, ns in (("gsw_cha_2016", ("current", "previous")),
                                           ("gsw_bkn_2015", ("previous", "current")), ("hou_cle_2016", ("current",)))
             for n in ns}
    (d / "score_run.json").write_text(json.dumps({"run_id": "A", "head": "h", "seconds": 1.0, "pairs": pairs}))
    for k in pairs:
        (d / f"score__{k}.json").write_text(json.dumps({"run_id": "A", "windows": {}, "cache": {}}))
    for g in R.POOL:
        for n in ("previous", "current"):
            (d / f"score__{g}__{n}__common.json").write_text(json.dumps({"run_id": "B", "windows": {}}))
    monkeypatch.setattr(R, "roots", lambda: {"ws": tmp_path / "work", "ref": tmp_path / "work_ref",
                                            "ms": tmp_path / "work_ms"})
    with pytest.raises(SystemExit):
        R.cmd_report(None)


def test_gate_fields_cover_team_coverage_and_duplicates():
    assert {"team", "cov_acc", "cov_live", "duplicates"} <= set(R.GATE_FIELDS)
