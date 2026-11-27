"""tools.p8_teamcause: the pure rules, and stage A on synthetic records writing only its own record."""

import builtins
import json

import numpy as np
import pytest

from tools import p8_teamcause as T


def test_majority_tie_rule_and_fixed_mapping():
    rows = [(1, 0, 5), (2, 0, 7), (3, 1, 7), (4, 1, 7), (5, 1, 5)]
    assert T.majority_teams(rows) == {0: 5, 1: 7}                        # label 0 ties 5 and 7: the smaller id
    from tools.p8_e3 import window_team
    m = T.fixed_mapping(rows)
    assert sum(1 for r in rows if m.get(r[1]) == r[2]) == window_team(rows)["fixed"]


def test_worst_windows():
    err = {"a": 5, "b": 1, "c": 0, "d": 3, "e": 9}
    rows = {"a": 10, "b": 10, "c": 0, "d": 10, "e": 10}
    assert T.worst_windows(err, rows) == ["e"]                           # ceil(0.1 * 4) = 1; c has no rows
    err2 = {f"w{i:02d}": i for i in range(12)}
    assert T.worst_windows(err2, {k: 5 for k in err2}) == ["w11", "w10"]


def test_bins_and_switches():
    assert T.bin_of(2.0, (3.0, 6.0), ("<3", "3-6", ">6")) == "<3" and T.bin_of(7, (3.0, 6.0), ("<3", "3-6", ">6")) == ">6"
    assert T.bin_of(None, (1.0,), ("a", "b")) == "none"
    assert T.switch_tracks([(1, 10), (1, 10), (1, 11), (2, 12)]) == {1}
    a, b = T.tercile_edges(list(range(9)))
    assert a < b


def test_colour_distances():
    bgr, lab = T.colour_distances({"0": [0, 0, 0], "1": [255, 255, 255]})
    assert bgr > 400 and lab > 90
    assert T.colour_distances({"0": [1, 2, 3]}) == (None, None)


def _rows(rng):
    out = []
    for w in range(20):
        for k in range(40):
            crowd = "<3" if k < 10 else ">6"
            err = bool(rng.random() < (0.6 if crowd == "<3" else 0.05))
            out.append({"window": f"w{w:02d}", "abstained": False, "error": err, "fixed_error": err,
                        "crowd_box": crowd, "third": "near" if k % 2 else "far"})
    out.append({"window": "w00", "abstained": True, "error": False, "fixed_error": False, "crowd_box": "<3",
                "third": "near"})
    return out


def test_carry_reading_names_the_concentrated_bin():
    rows = _rows(np.random.default_rng(1))
    r = T.carry_reading(rows, "crowd_box", n_boot=300)
    assert r["<3"]["carries"] and not r[">6"]["carries"] and r["<3"]["ratio_lower"] > 1
    assert T.bin_table(rows, "crowd_box")["<3"]["abstentions"] == 1
    rk = T.rank_carriers({"crowd_box": r, "third": T.carry_reading(rows, "third", n_boot=100)}, [])
    assert rk["top"] == {"feature": "crowd_box", "bin": "<3"}
    rk2 = T.rank_carriers({"crowd_box": r}, [{"pair": ["crowd_box", "x"], "rho": 0.9}])
    assert rk2["not_separated"] == []


def test_overlap_pairs():
    lv = ("<3", "3-6", ">6"); ev = ("<50", "50-150", ">150"); tv = ("far", "middle", "near")
    rows = [{"abstained": False, "crowd_box": lv[i % 3], "edge": ev[i % 3], "third": tv[(i * 7) % 3 if i % 2 else 0]}
            for i in range(60)]
    ov = T.overlap_pairs(rows, ["crowd_box", "edge", "third"])
    assert [o["pair"] for o in ov] == [["crowd_box", "edge"]]
    assert T.cross_tab([{**r, "error": False} for r in rows], "crowd_box", "edge")["<3|<50"]["rows"] == 20


def test_stored_writes_only_its_record(tmp_path, monkeypatch):
    from tools import p8_common
    ref = tmp_path / "work_ref"
    (ref / "rebaseline").mkdir(parents=True)
    (ref / "builds" / "r23").mkdir(parents=True)
    for g in (T.CHA, T.BKN, T.HOU):
        wins = {f"{g}_s00_w{i:02d}": {"team": {"n": 10 if i else 0, "majority": 8 if i else 0, "fixed": 8 if i else 0}}
                for i in range(5)}
        (ref / "rebaseline" / f"score__{g}__current.json").write_text(json.dumps({"run_id": T.RUN_ID, "windows": wins}))
        for w in wins:
            (ref / "builds" / "r23" / f"{w}_identity.json").write_text(json.dumps(
                {"team_colors_bgr": {"0": [10, 20, 30], "1": [200, 190, 180]},
                 "identity": {"teams": {"silhouette": 0.5, "tracks_abstained": 2}, "team_vetoed": 1}}))
    monkeypatch.setattr(T, "D202_SHA", {g: T.sha256(ref / "rebaseline" / f"score__{g}__current.json")
                                        for g in T.POOL})
    out = tmp_path / "stored.json"
    monkeypatch.setattr(T, "out_stored", lambda: out)
    written = {}
    monkeypatch.setattr(p8_common, "atomic_write", lambda p, text: written.__setitem__(str(p), text))
    real_open = builtins.open

    def guarded_open(file, mode="r", *a, **k):
        if any(m in mode for m in ("w", "a", "x", "+")):
            raise AssertionError(f"write attempted: {file} ({mode})")
        return real_open(file, mode, *a, **k)
    monkeypatch.setattr(builtins, "open", guarded_open)
    res = T.cmd_stored(None, root=ref)
    assert list(written) == [str(out)]
    g = res["games"][T.CHA]
    assert g["windows_without_rows"] == 1 and g["windows_below_0_5"] == 0 and g["accuracy"] == 0.8
    assert res["hou_cle_2016_aggregate_only"]["team_accuracy"] == 0.8 and T.HOU not in res["per_window"]


def test_stored_refuses_an_unpinned_record(tmp_path, monkeypatch):
    ref = tmp_path / "work_ref"
    (ref / "rebaseline").mkdir(parents=True)
    for g in (T.CHA, T.BKN):
        (ref / "rebaseline" / f"score__{g}__current.json").write_text(json.dumps({"run_id": T.RUN_ID, "windows": {}}))
    with pytest.raises(SystemExit):
        T.cmd_stored(None, root=ref)



def test_rank_codes_follow_the_fixed_orders():
    assert T.rank_code("crowd_box", "<3") == 0 and T.rank_code("crowd_box", "3-6") == 1 and T.rank_code("crowd_box", ">6") == 2
    assert T.rank_code("since_cut", "censored") is None and T.rank_code("team_id", 5) is None
    assert T.rank_code("third", "far") == 0 and T.rank_code("switch_track", True) == 1
    # text sort would put ">6" before "3-6"; the fixed order does not
    rows = [{"abstained": False, "crowd_box": c, "edge": e} for c, e in
            [("<3", "<50"), ("3-6", "50-150"), (">6", ">150")] * 20]
    ov = T.overlap_pairs(rows, ["crowd_box", "edge", "team_id"])
    assert ov and ov[0]["pair"] == ["crowd_box", "edge"] and ov[0]["rho"] > 0.99
