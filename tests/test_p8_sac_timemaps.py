"""tools.p8_sac_timemaps: the content comparison."""

from tools import p8_sac_timemaps as T


def test_compare_timemaps():
    a = {"spans": [1], "frames": [[0, 1, 700.0], [6, 1, 699.9]], "ocr_seconds": 24.6}
    b = {"spans": [1], "frames": [[0, 1, 700.0], [6, 1, 699.9]], "ocr_seconds": 21.7}
    c = T.compare_timemaps(a, b)
    assert c["keys_differing"] == ["ocr_seconds"] and c["frames_equal_sets"] and c["spans_equal"]
    assert c["clock_abs_diff_max"] == 0.0 and c["period_differences"] == 0
    b2 = {"spans": [2], "frames": [[0, 2, 700.5], [12, 1, 699.0]], "ocr_seconds": 24.6}
    c = T.compare_timemaps(a, b2)
    assert c["keys_differing"] == ["frames", "spans"] and not c["frames_equal_sets"] and c["frames_common"] == 1
    assert abs(c["clock_abs_diff_max"] - 0.5) < 1e-12 and c["period_differences"] == 1



def test_run_writes_only_its_record(tmp_path, monkeypatch):
    """Write scope: on synthetic roots the tool writes its one record and nothing else."""
    import builtins
    import json
    from tools import p8_common
    ws, ref, co = tmp_path / "work", tmp_path / "work_ref", tmp_path / "checkout"
    (co / "data" / "sportvu" / "sync").mkdir(parents=True)
    for g in T.GAMES:
        tgt = co / "data" / "sportvu" / "sync" / f"{g}_s00_local_timemap.json"
        tgt.write_text("{}")
        for root, ocr in ((ws, 24.6), (ref, 21.7)):
            (root / "label_build").mkdir(parents=True, exist_ok=True)
            (root / "sync").mkdir(parents=True, exist_ok=True)
            (root / "label_build" / f"{g}_s00_w00_timemap.json").write_text(
                json.dumps({"frames": [[0, 1, 700.0]], "spans": [1], "ocr_seconds": ocr}))
            link = root / "sync" / tgt.name
            if not link.exists():
                link.symlink_to(tgt)
    out = tmp_path / "out.json"
    monkeypatch.setattr(T, "out_path", lambda: out)
    monkeypatch.setattr("tools.p8_common.main_checkout", lambda: co)
    written = {}
    monkeypatch.setattr(p8_common, "atomic_write", lambda p, text: written.__setitem__(str(p), text))
    real_open = builtins.open

    def guarded_open(file, mode="r", *a, **k):
        if any(m in mode for m in ("w", "a", "x", "+")):
            raise AssertionError(f"write attempted: {file} ({mode})")
        return real_open(file, mode, *a, **k)
    monkeypatch.setattr(builtins, "open", guarded_open)
    res = T.run(ws, ref)
    assert list(written) == [str(out)]
    g = res["games"]["gsw_sac_2015"]
    assert g["sha_differing"] == 1 and g["keys_differing_counts"] == {"ocr_seconds": 1}
    assert g["windows"]["gsw_sac_2015_s00_w00"]["ocr_seconds_workspace"] == 24.6
    assert g["sync_maps"]["gsw_sac_2015_s00_local_timemap.json"] == {
        "link_workspace": True, "link_reference": True, "same_file": True, "in_main_checkout_sync": True}
