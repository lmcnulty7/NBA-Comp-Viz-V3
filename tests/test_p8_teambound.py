"""tools.p8_teambound (docs/research/P8_teambound.md 1 and 1.1): scoring, oracles, O2's cuts and tags, the split,
the bar, the reading table, the refusal and the writes. Synthetic data only."""

import numpy as np
import pytest

from tools import p8_teambound as M

A, B = 1610612744, 1610612751


def row(frame, cid, tid, label, team, pid, id_pair=True, crop=False, crowd=">6"):
    return {"frame": frame, "cid": cid, "tid": tid, "label": label, "team": team, "pid": pid, "id_pair": id_pair,
            "crop": crop, "crowd_sv": crowd}


def test_majority_map_matches_window_team():
    from tools.p8_e3 import window_team
    rows = [(1, 0, A), (1, 0, A), (2, 0, B), (3, 1, B), (4, 1, B), (4, 1, A)]
    ok = M.correct(rows)
    assert sum(ok) == window_team(rows)["majority"] == 4
    assert M.majority_map([(1, 0, A), (2, 0, B)]) == {0: min(A, B)}    # tie: the smaller team id


def test_runs_cuts_and_a_b_a():
    fps = 10.0
    idr = [(t / fps, 7) for t in range(0, 20)] + [(t / fps, 9) for t in range(20, 25)] + \
          [(t / fps, 7) for t in range(25, 40)]
    assert M.cuts(idr, 1.0) == []                              # a short B (0.4 s) does not cut
    idr2 = [(t / fps, 7) for t in range(0, 20)] + [(t / fps, 9) for t in range(20, 35)] + \
           [(t / fps, 7) for t in range(35, 50)]
    c = M.cuts(idr2, 1.0)
    assert [(round(x[0], 3), x[1], x[2]) for x in c] == [(2.0, 7, 9), (3.5, 9, 7)]   # A-B-A: two cuts
    assert M.segment_of(1.0, [2.0, 3.5]) == 0 and M.segment_of(2.0, [2.0, 3.5]) == 1 and M.segment_of(4, [2, 3.5]) == 2


def test_swap_and_orphan_tags():
    pairs = [(2.1, 5, 7), (2.5, 6, 3)]
    assert M.tag_cut(2.0, 7, 4, pairs) == "swap"               # the old id 7 is paired with track 5 after the cut
    assert M.tag_cut(2.0, 3, 6, pairs) == "orphan"             # paired only with the same track
    assert M.tag_cut(2.0, 8, 4, pairs) == "orphan"


def window(fps=10.0):
    rows = []
    # track 1: pure team A, labelled 0 (correct); track 2: pure team B but labelled 0 (a colour error)
    rows += [row(f, 1, 11, 0, A, 101) for f in range(0, 30)]
    rows += [row(f, 2, 21, 0, B, 201) for f in range(0, 10)]
    # track 3: labelled 1, team B for 2 s (pid 301), then a sustained switch to an A player (pid 102) for 2 s
    rows += [row(f, 3, 31, 1, B, 301) for f in range(0, 20)]
    rows += [row(f, 3, 32, 1, A, 102) for f in range(20, 40)]
    # pid 301 is then paired with track 4 (a swap)
    rows += [row(f, 4, 41, 1, B, 301) for f in range(20, 30)]
    return rows, fps


def test_oracles_on_a_small_window():
    rows, fps = window()
    o = M.window_oracles(rows, fps)
    n = o["n"]
    assert n == 30 + 10 + 40 + 10
    cur, o1, o2 = sum(o["current"]), sum(o["O1"]), sum(o["O2_swap"])
    # current: label 0 -> A (30 of 40 rows A), so track 2's 10 rows are wrong; label 1 -> B (30 of 50), track 3's
    # 20 A rows are wrong
    assert cur == n - 10 - 20
    assert o1 == n - 20                                         # colour fixed; track 3's minority rows stay wrong
    assert o2 == n                                              # the swap cut splits track 3
    assert sum(o["O2_all"]) == n
    assert [t["tag"] for t in o["cut_tags"]] == ["swap"] and o["cut_tags"][0]["fragment_boundary"] is True
    assert sum(o["O1f"]) == n                                   # fragments 31 and 32 are pure here


def test_abstained_rows_are_left_out_and_o1c_falls_back():
    rows, fps = window()
    rows += [row(f, 9, 91, None, A, 109) for f in range(0, 10)]
    o = M.window_oracles(rows, fps)
    assert o["n"] == 90                                         # abstained rows do not enter the denominator
    assert sum(o["O1c"]) == sum(o["current"])                    # no crop rows: every track keeps its label


def test_bin_split_sums():
    rows, fps = window()
    rec = {"windows": {"w": {"rows": rows, "fps": fps}}}
    per_w = {"w": M.window_oracles(rows, fps)}
    b = M.bin_split(rec, per_w)[">6"]
    assert b["errors"] == b["tracking"] + b["colour"] == 30 and b["tracking"] == 20 and b["colour"] == 10


def test_pattern_table():
    bars = {"colour": 0.03, "tracking": 0.03, "combined": 0.03}
    g = lambda c, t, k: {"colour": c, "tracking": t, "combined": k}
    assert M.pattern({1: g(0.04, 0.0, 0.04), 2: g(0.05, 0.0, 0.05)}, bars) == "colour headroom"
    assert M.pattern({1: g(0.0, 0.04, 0.04), 2: g(0.0, 0.05, 0.05)}, bars) == "tracking headroom"
    assert M.pattern({1: g(0.04, 0.04, 0.08), 2: g(0.05, 0.05, 0.1)}, bars) == "both"
    assert M.pattern({1: g(0.02, 0.02, 0.04), 2: g(0.02, 0.02, 0.04)}, bars) == "combined"
    assert M.pattern({1: g(0.04, 0.0, 0.04), 2: g(0.0, 0.04, 0.04)}, bars) == "split by kind"
    assert M.pattern({1: g(0.04, 0.0, 0.04), 2: g(0.0, 0.0, 0.01)}, bars) == "one game only"
    assert M.pattern({1: g(0.01, 0.01, 0.02), 2: g(0.0, 0.0, 0.01)}, bars) == "neither"


def test_mde():
    gw = [(10, 100)] * 50
    assert M.mde(gw, n_draw=93, n=200) == pytest.approx(0.0)
    gw2 = [(10, 100), (0, 100)] * 50
    m = M.mde(gw2, n_draw=93, n=500)
    assert 0.0 < m < 0.05


def test_crop_frames_rule():
    side = {"frames": [{"frame": f, "boxes": [{"tid": 1, "bbox": [0, 0, 60, 120]}, {"tid": 2, "bbox": [0, 0, 20, 20]}]}
                       for f in range(0, 60, 6)]}
    cf = M.crop_frames(side, 1280, 720)
    assert {f for f, t in cf if t == 1} == {0, 18, 36, 54}     # every 3rd processed frame
    assert not any(t == 2 for _, t in cf)                       # the inner patch is under 14 px


def test_hou_is_refused_before_any_read(monkeypatch):
    import tools.p8_rebaseline as RB
    monkeypatch.setattr(RB, "read_json", lambda p: (_ for _ in ()).throw(AssertionError("a file was read")))
    with pytest.raises(SystemExit):
        M.collect("hou_cle_2016")
    called = []
    monkeypatch.setattr(M, "collect", lambda g: called.append(g))
    with pytest.raises(SystemExit):
        M.main(["collect", "hou_cle_2016"])
    assert called == []


def test_the_only_writes_are_the_records_and_no_cmd_stored():
    import ast
    import inspect
    src = inspect.getsource(M)
    tree = ast.parse(src)
    names = [getattr(c.func, "id", getattr(c.func, "attr", None)) for c in ast.walk(tree) if isinstance(c, ast.Call)]
    assert names.count("atomic_write") == 2
    assert not any(n in ("write_text", "write_bytes", "unlink", "rmtree", "remove", "rename", "open") for n in names)
    assert "cmd_stored" not in src
    assert names.count("find_private") == 2


def test_constants_match_the_design():
    assert (M.L_RULE, M.L_GRID, M.SWAP_SPAN_S) == (1.0, (0.5, 0.75, 1.0, 1.5, 2.0), 1.0)
    assert (M.CROP_EVERY, M.CROP_CAP, M.CROP_MIN_PX, M.MIN_CROP_FRAMES) == (3, 40, 14, 2)
    assert (M.F_RECOVERY, M.BAR_MIN, M.HOU_WINDOWS, M.N_BOOT, M.SEED) == (0.5, 0.03, 93, 2000, 0)


# ── 9.12 ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_the_bar_divides_the_mde_by_f_so_f_does_not_cancel():
    gw = [(10, 100), (0, 100), (0, 0)] * 40                    # empty windows add zero rows
    m = M.mde(gw, n_draw=93, n=500)
    assert m > 0
    assert M.bar_of(m, f=0.5) == pytest.approx(max(0.03, m / 0.5))
    assert M.bar_of(0.04, f=0.5) == pytest.approx(0.08) and M.bar_of(0.04, f=1.0) == pytest.approx(0.04)
    assert M.bar_of(0.001, f=0.5) == 0.03


def test_pattern_one_game_combined_and_unclassified():
    bars = {"colour": 0.03, "tracking": 0.03, "combined": 0.03}
    g = lambda c, t, k: {"colour": c, "tracking": t, "combined": k}
    assert M.pattern({1: g(0.02, 0.02, 0.04), 2: g(0.0, 0.0, 0.01)}, bars) == "one game only"
    assert M.pattern({1: g(0.01, 0.01, 0.02), 2: g(0.0, 0.0, 0.02)}, bars) == "neither"
    assert M.pattern({1: g(0.01, 0.01, None), 2: g(0.0, 0.0, 0.02)}, bars) == "unclassified"


def test_o3_relabels_pure_tracks_only_at_row_level():
    rows = [row(f, 1, 11, 0, A, 101) for f in range(10)]          # pure A, label 0 -> A (correct)
    rows += [row(f, 2, 21, 0, B, 201) for f in range(4)]          # pure B with label 0 (wrong): O3 fixes it
    rows += [row(f, 3, 31, 1, B, 301) for f in range(6)] + [row(f, 3, 32, 1, A, 102) for f in range(6, 9)]
    o = M.window_oracles(rows, 10.0)
    cur, o3 = o["current"], o["O3"]
    idx2 = [i for i, r in enumerate([x for x in rows if x["label"] is not None]) if r["cid"] == 2]
    idx3a = [i for i, r in enumerate([x for x in rows if x["label"] is not None]) if r["cid"] == 3 and r["team"] == A]
    assert not any(cur[i] for i in idx2) and all(o3[i] for i in idx2)
    assert [o3[i] for i in idx3a] == [cur[i] for i in idx3a]      # the mixed track keeps its label


def test_o1c_uses_crop_rows_only():
    # track 1's crop rows are team B, its other rows team A: O1c labels it B, O1 labels it A
    rows = [row(f, 1, 11, 0, A, 101) for f in range(10)] + [row(f, 1, 11, 0, B, 201, crop=True) for f in range(10, 13)]
    rows += [row(f, 2, 21, 1, B, 202) for f in range(5)]
    o = M.window_oracles(rows, 10.0)
    assert sum(o["O1"]) > sum(o["O1c"])


def test_tracks_with_two_ids_and_no_sustained_run():
    rows = [row(f, 1, 11, 0, A, 101 if f % 2 else 102) for f in range(6)] + [row(f, 2, 21, 1, B, 201) for f in range(30)]
    assert M.window_oracles(rows, 10.0)["tracks_multi_id_no_sustained_run"] == 1


def test_report_refuses_unchecked_records():
    ok = {"git": {"head": "h", "tools_and_sportvu_clean": True}, "manifest_unchanged": True, "d202": {"sha256": "x"}}
    pinned = {M.CHA: "x", M.BKN: "x"}
    M.check_records({M.CHA: ok, M.BKN: ok}, pinned)
    for bad in ({**ok, "manifest_unchanged": False}, {**ok, "git": {"head": "other", "tools_and_sportvu_clean": True}},
                {**ok, "d202": {"sha256": "y"}}, {**ok, "git": {"head": "h", "tools_and_sportvu_clean": False}}):
        with pytest.raises(SystemExit):
            M.check_records({M.CHA: ok, M.BKN: bad}, pinned)
