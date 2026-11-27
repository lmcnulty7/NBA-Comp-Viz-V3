"""tools.p8_sac_checkc: the pure rules and a stage 1 run on synthetic roots that may write only its one record."""

import builtins
import json

import pytest

from tools import p8_sac_checkc as C


def test_checkc_d_keeps_game_summary_windows():
    rec = {"windows": {"a": {"frozen": {"v3": 0.4, "r23": 0.0}, "stored_frozen_v3": 0.4},
                       "b": {"frozen": {"v3": 0.4, "r23": None}, "stored_frozen_v3": 0.4},
                       "c": {"frozen": {"v3": 0.2, "r23": 0.0}, "stored_frozen_v3": 0.6}}}
    assert C.checkc_d(rec) == {"a": pytest.approx(0.4)}


def test_window_class():
    assert C.window_class(1, 0, 0, 2) == "both" and C.window_class(0, 1, 0, 0) == "MPS only"
    assert C.window_class(0, 0, 0, 1) == "CPU only" and C.window_class(0, 0, 0, 0) == "none"


def test_crossings_and_hybrids():
    dm = {"a": 0.1, "b": 0.3, "c": 0.1, "d": 0.5}
    df = {"a": 0.3, "b": 0.1, "c": 0.4, "d": 0.6}
    cls = {"a": "none", "b": "none", "c": "both", "d": "both"}
    x = C.crossings(dm, df, cls)
    assert x["none"]["up"] == 1 and x["none"]["down"] == 1 and x["none"]["net"] == 0 and x["both"]["net"] == 1
    assert C.hybrid(dm, df, {"a", "b"}) == {"a": 0.3, "b": 0.1, "c": 0.1, "d": 0.5}


def test_moves_and_readings():
    assert C.moves(0.58, 0.52, 0.64) and not C.moves(0.57, 0.52, 0.64) and not C.moves(0.5, 0.52, 0.52)
    assert C.moves(0.48, 0.52, 0.44)                                     # a fall counts the same way
    assert C.decomposition_reading(True, True, False) == "Clock"
    assert C.decomposition_reading(False, True, False) == "Platform"
    assert C.decomposition_reading(False, False, True) == "Fix"
    assert C.decomposition_reading(False, True, True) == "Joint"
    assert C.decomposition_reading(False, False, False) == "Neither"
    assert C.side_reading(True, False) == "V3 side" and C.side_reading(False, True) == "r23 side"
    assert C.side_reading(True, True) == "both" and C.side_reading(False, False) == "not separated"
    assert C.combine_side("V3 side", "V3 side") == "V3 side" and C.combine_side("V3 side", "both") == "not separated"
    assert C.within_controls(0.03, [0.04, 0.02]) == "within" and C.within_controls(0.05, [0.04]) == "outside"


def test_share_reading_turns_on_single_windows():
    """Section 7.4: the share change is the net crossings, split between the hybrids."""
    dm = {f"w{i}": 0.1 for i in range(10)}
    df = dict(dm)
    df["w0"], df["w1"] = 0.5, 0.5                                        # +2 windows over 0.2 s
    for none, want in (({"w0", "w1"}, "Platform"), ({"w0"}, "Joint"), (set(), "Fix")):
        si, sii = C.stats(C.hybrid(dm, df, none)), C.stats(C.hybrid(dm, df, set(dm) - none))
        sm, sf = C.stats(dm), C.stats(df)
        k = "share_over_0_2_s"
        assert C.decomposition_reading(False, C.moves(si[k], sm[k], sf[k]), C.moves(sii[k], sm[k], sf[k])) == want


def test_first_logged_reset():
    assert C.first_logged_reset({"resets": [{"frame": 90}, {"frame": 30}]}) == 30
    assert C.first_logged_reset({"resets": []}) is None and C.first_logged_reset(None) is None


# ── a synthetic run ───────────────────────────────────────────────────────────────────────────────────────────────
def _w(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def _record(vals, dev):
    wins = {s: {"frozen": {"v3": a, "r23": b}, "stored_frozen_v3": a} for s, (a, b) in vals.items()}
    rec = {"windows": wins}
    if dev:
        from tools.p8_truth_v2 import stability
        rec["summary"] = {"frozen": stability([(a, b) for a, b in vals.values()])}
    return rec


def _roots(tmp):
    ws, ref = tmp / "work", tmp / "work_ref"
    sac = [f"gsw_sac_2015_s00_w{i:02d}" for i in range(4)]
    cha = [f"gsw_cha_2016_s00_w{i:02d}" for i in range(3)]
    side = {"frames": [{"frame": f, "H": [1, 0, 0, 0, 1, 0, 0, 0, 1], "boxes": [{"b": 1}] * 5} for f in range(0, 60, 6)]}
    tm = {"spans": [{"f0": 0, "f1": 100, "period": 1, "c0": 700.0, "c1": 698.0}]}
    for s in sac:
        for root in (ws, ref):
            _w(root / "label_build" / f"{s}_timemap.json", tm)
            _w(root / "label_build" / f"{s}_frames.json", side)
        for folder in (ws / "label_build", ws / "builds" / "r23", ref / "label_build", ref / "builds" / "r23_nofix",
                       ref / "builds" / "r23"):
            _w(folder / f"{s}_identity.json", {"identity": {"camera_cut_resets": 1 if s.endswith("w00") else 0}})
            _w(folder / f"{s}_frames.json", side)
    _w(ref / "builds" / "resetfix_r23_detcpu__gsw_sac_2015" / f"{sac[0]}_resetfix.json",
       {"resets": [{"frame": 30, "stripped": 2}]})
    vm = {s: (0.0, x) for s, x in zip(sac, (0.4, 0.1, 0.1, 0.3))}
    vf = {s: (0.0, x) for s, x in zip(sac, (0.4, 0.3, 0.1, 0.3))}
    _w(ws / "truth_v2" / "gsw_sac_2015.json", _record(vm, False))
    _w(ref / "truth_v2" / "dev" / "gsw_sac_2015.json", _record(vf, True))
    bk = {f"gsw_bkn_2015_s00_w{i:02d}": (0.0, 0.1 * i) for i in range(4)}
    _w(ws / "truth_v2" / "gsw_bkn_2015.json", _record(bk, False))
    _w(ref / "truth_v2" / "dev" / "gsw_bkn_2015.json", _record(bk, True))
    ch = {s: (0.0, 0.1) for s in cha}
    _w(ws / "truth_v2" / "gsw_cha_2016.json", _record(ch, False))
    _w(ref / "superseded_nofix" / "truth_v2" / "dev" / "gsw_cha_2016.json", _record(ch, True))
    _w(ref / "truth_v2" / "dev" / "gsw_cha_2016.json", _record(ch, True))
    for s in cha:
        for root in (ref / "label_build", ref / "superseded_nofix" / "label_build"):
            for f, obj in (("frames", side), ("trajectories", {}), ("identity", {"identity": {"camera_cut_resets": 0}})):
                _w(root / f"{s}_{f}.json", obj)
    _w(ref / "resetfix" / "plan__gsw_sac_2015.json", {"reset_windows": [sac[0]]})
    for g in C.GAMES:
        _w(ref / "resetfix" / f"equiv__{g}.json", {"passes": True, "equivalence": {"x": {"byte_identical": {"frames": True}}},
                                                  "reset_windows": {"y": {"prefix": {"equal": True}}}})
    for root in (ws, ref):
        _w(root / "labels" / "gsw_sac_2015_offsets.json", {s: 0.0 for s in sac})
    return ws, ref


def test_stage1_on_synthetic_roots_writes_only_its_record(tmp_path, monkeypatch):
    from tools import p8_common
    ws, ref = _roots(tmp_path)
    out = tmp_path / "out" / "p8_sac_checkc.json"
    monkeypatch.setattr(C, "out_path", lambda: out)
    written = {}
    monkeypatch.setattr(p8_common, "atomic_write", lambda p, text: written.__setitem__(str(p), text))
    real_open = builtins.open

    def guarded_open(file, mode="r", *a, **k):
        if any(m in mode for m in ("w", "a", "x", "+")):
            raise AssertionError(f"write attempted: {file} ({mode})")
        return real_open(file, mode, *a, **k)
    monkeypatch.setattr(builtins, "open", guarded_open)
    res = C.run_stage1(ws, ref, expect=None,
                       clock=lambda w, r, g: {"workspace": {"m": 1}, "reference": {"m": 1}})
    assert list(written) == [str(out)]
    assert res["classes"]["counts"] == {"both": 1, "MPS only": 0, "CPU only": 0, "none": 3}
    assert res["crossings"]["none"]["up"] == 1 and res["clock"]["differs"] is False
    assert res["decomposition"]["share_over_0_2_s"]["reading"] == "Platform"
    assert res["qualifying_frames"]["gsw_sac_2015_s00_w00"]["fixed_v3"] == {"frames": 10, "after_cpu_first_reset": 5}
    assert res["cha_no_reset_v3_vs_unfixed"] == {"identical": 3, "differing": [], "unfixed_missing": 0}
    assert res["stopped"] == []


def test_stage1_stops_when_a_quoted_fact_differs(tmp_path, monkeypatch):
    from tools import p8_common
    ws, ref = _roots(tmp_path)
    monkeypatch.setattr(C, "out_path", lambda: tmp_path / "o.json")
    monkeypatch.setattr(p8_common, "atomic_write", lambda p, text: None)
    with pytest.raises(SystemExit):
        C.run_stage1(ws, ref, expect=C.KNOWN, clock=lambda w, r, g: {"workspace": {}, "reference": {}})


def test_no_cmd_import_and_one_output_path():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(C))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "tools.p8_truth_v2":
            assert not any(a.name.startswith("cmd_") for a in node.names)
    src = inspect.getsource(C)
    assert src.count("atomic_write(out_path()") == 2 and src.count("atomic_write(out_path_v22()") == 1
    assert src.count("atomic_write(out_path_close()") == 1 and src.count("atomic_write(") == 4
    assert '"truth_accept"' not in src                                    # no path is built there
    assert C.out_path().name == "p8_sac_checkc.json" and C.out_path().parent.name == "research"



# ── version 2.2: the clock on time-map content, the side reading ──────────────────────────────────────────────────
def test_timemap_content_equal_ignores_run_fields():
    a = {"frames": [[0, 1, 700.0]], "spans": [{"f0": 0}], "ocr_seconds": 24.6, "reads": 10}
    assert C.timemap_content_equal(a, {**a, "ocr_seconds": 21.7, "reads": 11})
    assert not C.timemap_content_equal(a, {**a, "frames": [[0, 1, 700.04]]})
    assert not C.timemap_content_equal(a, {**a, "spans": []})


def test_stage1_side_writes_only_its_record(tmp_path, monkeypatch):
    from tools import p8_common
    ws, ref = _roots(tmp_path)
    tm2 = json.loads((ref / "label_build" / "gsw_sac_2015_s00_w00_timemap.json").read_text())
    tm2["ocr_seconds"] = 3.0                                             # metadata only: the clock holds
    (ref / "label_build" / "gsw_sac_2015_s00_w00_timemap.json").write_text(json.dumps(tm2))
    out = tmp_path / "v22.json"
    monkeypatch.setattr(C, "out_path_v22", lambda: out)
    written = {}
    monkeypatch.setattr(p8_common, "atomic_write", lambda p, text: written.__setitem__(str(p), text))
    real_open = builtins.open

    def guarded_open(file, mode="r", *a, **k):
        if any(m in mode for m in ("w", "a", "x", "+")):
            raise AssertionError(f"write attempted: {file} ({mode})")
        return real_open(file, mode, *a, **k)
    monkeypatch.setattr(builtins, "open", guarded_open)
    same = lambda w, r, g: {"workspace": {"m": 1}, "reference": {"m": 1}}
    res = C.run_stage1_side(ws, ref, clock=same)
    assert list(written) == [str(out)] and res["clock_content"]["holds"] and "per_statistic" in res["side"]
    monkeypatch.setattr(builtins, "open", real_open)
    tm2["frames"] = [[0, 1, 701.0]]                                       # a real clock difference
    (ref / "label_build" / "gsw_sac_2015_s00_w00_timemap.json").write_text(json.dumps(tm2))
    monkeypatch.setattr(builtins, "open", guarded_open)
    res = C.run_stage1_side(ws, ref, clock=same)
    assert res["side"]["reading"] == "invalid (clock differs)"



def test_close_figures():
    dm = {"a": 0.1, "b": 0.3, "c": 0.1, "d": 0.5, "e": 0.2}
    df = {"a": 0.3, "b": 0.3, "c": 0.5, "d": 0.1, "e": 0.25}
    cls = {"a": "none", "b": "none", "e": "none", "c": "both", "d": "MPS only"}
    f = C.close_figures(dm, df, cls)
    assert f["measured"] == 5 and f["class_split"] == {"none": 3, "both": 1, "MPS only": 1, "CPU only": 0}
    assert f["no_reset"]["net"] == 2 and f["reset"]["net"] == 0
    assert abs(f["no_reset"]["median_abs_d_change_s"] - 0.05) < 1e-12 and abs(f["reset"]["median_abs_d_change_s"] - 0.4) < 1e-12
    assert f["median_abs_d_change_by_class_s"]["CPU only"] is None and abs(f["median_abs_d_change_by_class_s"]["both"] - 0.4) < 1e-12
