"""tools.p8_cuda_check: manifest and hash checks, the result round trip over a fake runtime, the comparison arithmetic
and the reading rule (docs/research/P8_cuda_check.md section 4) on synthetic inputs. No video, no model, no SportVU."""

import hashlib
import io
import json
import tarfile
from argparse import Namespace

import numpy as np
import pytest

from tools import p8_colab_xfer as X
from tools import p8_cuda_check as C

W3, W4 = C.WINDOWS


# ── hashes, tars, manifest ──────────────────────────────────────────────────────────────────────────────────────────
def _files(tmp_path):
    (tmp_path / "src").mkdir()
    a, b = tmp_path / "src" / "a.mp4", tmp_path / "src" / "b.json"
    a.write_bytes(b"clip" * 1000)
    b.write_text('{"x": 1}')
    return {"clips/a.mp4": a, "sync/b.json": b}


def test_tar_round_trip_is_deterministic_and_checked(tmp_path):
    files = _files(tmp_path)
    rec = C.member_records(files)
    assert rec["sync/b.json"] == {"sha256": hashlib.sha256(b'{"x": 1}').hexdigest(), "bytes": 8}
    s1 = C.write_tar(tmp_path / "p1.tar", files)
    s2 = C.write_tar(tmp_path / "p2.tar", dict(reversed(list(files.items()))))
    assert s1 == s2 == C.sha256_file(tmp_path / "p1.tar")          # same bytes whatever the order or the clock
    assert set(C.check_tar(tmp_path / "p1.tar", rec)) == set(files)
    out = C.extract_checked(tmp_path / "p1.tar", rec, tmp_path / "x")
    assert (out / "clips" / "a.mp4").read_bytes() == b"clip" * 1000


def test_check_tar_refuses_tampered_extra_missing_and_unsafe_members(tmp_path):
    files = _files(tmp_path)
    rec = C.member_records(files)
    C.write_tar(tmp_path / "p.tar", files)
    bad = {**rec, "sync/b.json": {**rec["sync/b.json"], "sha256": "0" * 64}}
    with pytest.raises(SystemExit):
        C.check_tar(tmp_path / "p.tar", bad)
    with pytest.raises(SystemExit):                                # a record the tar lacks
        C.check_tar(tmp_path / "p.tar", {**rec, "models/h.joblib": {"sha256": "1" * 64, "bytes": 1}})
    with pytest.raises(SystemExit):                                # a member no record names
        C.check_tar(tmp_path / "p.tar", {"clips/a.mp4": rec["clips/a.mp4"]})
    evil = tmp_path / "evil.tar"
    with tarfile.open(evil, "w") as tf:
        ti = tarfile.TarInfo("../outside.txt")
        ti.size = 1
        tf.addfile(ti, io.BytesIO(b"x"))
    with pytest.raises(SystemExit):
        C.check_tar(evil, {"../outside.txt": {"sha256": hashlib.sha256(b"x").hexdigest(), "bytes": 1}})
    assert not C.safe_member("/abs") and not C.safe_member("a/../../b") and C.safe_member("a/b.json")


def _pack(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    files = _files(tmp_path)
    sha = C.write_tar(pack / f"{C.PACK}.tar", files)
    man = {"game": C.GAME, "windows": {w: {"window": w} for w in C.ALL_WINDOWS}, "tar": f"{C.PACK}.tar",
           "tar_sha256": sha, "members": C.member_records(files)}
    (pack / "pack_manifest.json").write_text(json.dumps(man))
    return pack, man


def test_check_pack_refuses_a_changed_tar_or_another_game(tmp_path):
    pack, man = _pack(tmp_path)
    assert C.check_pack(pack)["tar_sha256"] == man["tar_sha256"]
    (pack / man["tar"]).write_bytes(b"changed")
    with pytest.raises(SystemExit):
        C.check_pack(pack)
    (pack / "pack_manifest.json").write_text(json.dumps({**man, "game": "gsw_bkn_2015"}))
    with pytest.raises(SystemExit):
        C.check_pack(pack)


def test_check_weights(tmp_path):
    p = tmp_path / "w.pt"
    p.write_bytes(b"weights")
    want = {"models/w.pt": hashlib.sha256(b"weights").hexdigest()}
    assert C.check_weights(want, {"models/w.pt": p}) == want
    with pytest.raises(SystemExit):
        C.check_weights({"models/w.pt": "0" * 64}, {"models/w.pt": p})
    with pytest.raises(SystemExit):
        C.check_weights(want, {"models/w.pt": tmp_path / "missing.pt"})


def test_tensor_fingerprint_ignores_order_and_file_format_but_not_values():
    a = [("b", np.ones((2, 3), np.float32)), ("a", np.arange(4, dtype=np.float32))]
    fp = C.tensor_fingerprint(a)
    assert fp == C.tensor_fingerprint(list(reversed(a))) and fp["tensors"] == 2
    changed = [("b", np.ones((2, 3), np.float32)), ("a", np.arange(4, dtype=np.float32) + 1e-6)]
    assert C.tensor_fingerprint(changed)["sha256"] != fp["sha256"]
    assert C.tensor_fingerprint([("b", np.ones((3, 2), np.float32)), a[1]])["sha256"] != fp["sha256"]


# ── put and get over a fake runtime ─────────────────────────────────────────────────────────────────────────────────
class FakeRuntime:
    def __init__(self, files=None):
        self.files, self.puts = dict(files or {}), []

    def request(self, method, path, body=None, headers=None, timeout=600):
        if method == "PUT":
            self.puts.append((path, body))
            return b"{}"
        lo, hi = headers["Range"].split("=")[1].split("-")
        return self.files[path[len("files/"):]][int(lo):int(hi) + 1]


def test_put_sends_the_manifest_and_the_checked_tar(tmp_path):
    pack, man = _pack(tmp_path)
    rt = FakeRuntime()
    C.cmd_put(Namespace(pack=str(pack)), rt=rt)
    paths = [p for p, b in rt.puts if b.get("type") == "file"]
    assert paths == [f"api/contents/{C.REMOTE_IN}/pack_manifest.json", f"api/contents/{C.REMOTE_IN}/{man['tar']}"]
    (pack / man["tar"]).write_bytes(b"changed")
    with pytest.raises(SystemExit):                                # a damaged local tar never leaves the Mac
        C.cmd_put(Namespace(pack=str(pack)), rt=FakeRuntime())


def _result(tmp_path, arms=("cuda_a",), purpose="result"):
    work_out, out_dir = tmp_path / "work" / "cuda_check", tmp_path / "out"
    out_dir.mkdir(parents=True)
    builds = {}
    for arm in arms:
        (work_out / arm).mkdir(parents=True)
        builds[arm] = {}
        for w in C.WINDOWS:
            (work_out / arm / f"{w}_frames.json").write_text(json.dumps({"frames": []}))
            (work_out / arm / f"{w}_device.json").write_text(json.dumps({"resolved": {"detector": C.ARMS[arm]}}))
            (work_out / arm / f"{w}.mp4").write_bytes(b"not kept")
            builds[arm][w] = {"ok": True, "seconds": 1.0,
                              "device": {"resolved": {k: C.ARMS[arm] for k in ("detector", "court", "gate")}}}
    record = {"purpose": purpose, "arms_done": list(arms), "builds": builds, "pack_manifest_sha256": "m" * 64}
    complete = C.write_result(out_dir, work_out, record)
    return out_dir, complete


def test_write_result_keeps_only_the_outputs_and_get_checks_them(tmp_path, monkeypatch):
    out_dir, complete = _result(tmp_path)
    names = set(complete["members"])
    assert f"{C.RESULT}/run_record.json" in names and not any(n.endswith(".mp4") for n in names)
    assert len(names) == 1 + 2 * 2                                 # run record, frames and device per window
    monkeypatch.setattr(X, "CHUNK", 100)                          # several Range reads
    files = {f"{C.REMOTE_OUT}/{C.RESULT}.tar": (out_dir / f"{C.RESULT}.tar").read_bytes(),
             f"{C.REMOTE_OUT}/{C.RESULT}.complete.json": (out_dir / f"{C.RESULT}.complete.json").read_bytes()}
    dest = tmp_path / "got"
    info = C.cmd_get(Namespace(dest=str(dest)), rt=FakeRuntime(files))
    assert info["arms_done"] == ["cuda_a"] and (dest / C.RESULT / "cuda_a" / f"{W3}_frames.json").exists()
    bad = json.loads(files[f"{C.REMOTE_OUT}/{C.RESULT}.complete.json"])
    bad["tar_sha256"] = "0" * 64
    files[f"{C.REMOTE_OUT}/{C.RESULT}.complete.json"] = json.dumps(bad).encode()
    with pytest.raises(SystemExit):
        C.cmd_get(Namespace(dest=str(dest)), rt=FakeRuntime(files))
    assert any(p.name.startswith(f"{C.RESULT}.tar.prev-") for p in dest.iterdir())   # the first get was kept


def test_check_result_marks_failed_or_wrong_device_arms_unusable(tmp_path):
    out_dir, complete = _result(tmp_path, arms=("cuda_a", "cuda_b", "colab_cpu"))
    with tarfile.open(out_dir / f"{C.RESULT}.tar") as tf:
        tf.extractall(tmp_path / "res", filter="data")
    rr = tmp_path / "res" / C.RESULT / "run_record.json"
    rec = json.loads(rr.read_text())
    rec["builds"]["cuda_b"][W4]["device"]["resolved"]["court"] = "cpu"
    rec["builds"]["colab_cpu"][W3]["ok"] = False
    rr.write_text(json.dumps(rec))
    _, usable, unusable = C.check_result(tmp_path / "res", None, plumbing=False)
    assert usable == ["cuda_a"] and set(unusable) == {"cuda_b", "colab_cpu"}
    rec["purpose"] = "plumbing only, not a result"
    rr.write_text(json.dumps(rec))
    with pytest.raises(SystemExit):
        C.check_result(tmp_path / "res", None, plumbing=False)


# ── the comparison arithmetic ───────────────────────────────────────────────────────────────────────────────────────
H_ID = [1, 0, 0, 0, 1, 0, 0, 0, 1]


def _frame(f, n, H=H_ID, x0=0):
    return {"frame": f, "H": H, "boxes": [{"bbox": [x0 + 10 * i, 0, x0 + 10 * i + 5, 20]} for i in range(n)]}


def test_pair_metrics():
    r = {f: _frame(f, 8) for f in range(10)}
    x = {f: _frame(f, 8 if f < 8 else 7) for f in range(10)}
    x[3] = _frame(3, 8, x0=1)                                      # same count, other coordinates
    x[10] = _frame(10, 8)                                          # a frame only X built
    pm = C.pair_metrics(x, r)
    assert pm["common"] == 10 and pm["frames_jaccard"] == pytest.approx(10 / 11)
    assert pm["same_count_share"] == 0.8 and pm["identical_boxes_share"] == pytest.approx(0.7)
    assert pm["boxes_mean_diff"] == pytest.approx((8 * 9 + 7 * 2) / 11 - 8)          # unrounded
    assert pm["h_gap_px"]["median"] == 0.0 and pm["frames_with_h_x"] == 11
    x[4] = {**x[4], "H": None}
    assert C.pair_metrics(x, r)["frames_with_h_x"] == 10
    shifted = [1, 0, -2, 0, 1, 0, 0, 0, 1]                         # court H 2 px apart
    pm2 = C.pair_metrics({f: _frame(f, 8, H=shifted) for f in range(10)}, r)
    assert pm2["h_gap_px"]["median"] == pytest.approx(2.0)


def test_offset_set_and_candidate_tolerance():
    assert C.offset_set(0.62, [0.9, 1.42, None, 0.94]) == [0.3, 0.62, 0.9, 0.94, 1.2, 1.42]
    assert C.offset_set(-0.06, [-1.34, -0.78]) == [-1.34, -0.78, -0.06, 0.3, 0.62, 0.9, 1.2]
    assert C.cand_ok(85, 88) and not C.cand_ok(45, 88)            # within 8.8; MPS's 43 short is out
    assert C.cand_ok(3, 0) and not C.cand_ok(4, 0)                 # the floor of 3
    assert C.cand_ok(107, 98) and not C.cand_ok(108, 98)                # 9.8 allowed
    assert not C.cand_ok(None, 5)


def test_recorded_offset_mismatch():
    assert C.recorded_offset_mismatch(W3, {"mac_mps": 0.9, "mac_cpu": 1.42, "detector_cpu": 0.94}) == {}
    assert C.recorded_offset_mismatch(W4, {"mac_cpu": -0.74, "mac_mps": None}) == {
        "mac_cpu": {"recorded": -0.78, "now": -0.74}, "mac_mps": {"recorded": -1.34, "now": None}}


# ── the reading rule ────────────────────────────────────────────────────────────────────────────────────────────────
CPU_CANDS = {"0.30": 98, "0.62": 88, "0.90": 55, "1.20": 35}       # s02_w03, P8_PLATFORM.md
MPS_CANDS = {"0.30": 65, "0.62": 45, "0.90": 30, "1.20": 16}


def _pm(jac=1.0, diff=0.0, same=1.0, gap=0.0):
    return {"frames_jaccard": jac, "boxes_mean_diff": diff, "same_count_share": same,
            "h_gap_px": {"median": gap, "p90": gap, "frames": 100}}


def test_window_tests_on_the_mac_numbers():
    mps_vs_cpu = C.window_tests(_pm(diff=-1.04, same=0.20, gap=3.84), MPS_CANDS, CPU_CANDS)
    assert not any(mps_vs_cpu[k] for k in ("boxes_mean", "same_count", "court_h_gap", "candidates"))
    det_vs_cpu = C.window_tests(_pm(gap=3.84), {"0.30": 97, "0.62": 88, "0.90": 55, "1.20": 35}, CPU_CANDS)
    assert det_vs_cpu == {"frames_jaccard": True, "boxes_mean": True, "same_count": True, "court_h_gap": False,
                          "candidates": True}
    near = C.window_tests(_pm(jac=0.95, diff=0.25, same=0.80, gap=1.0), {"0.30": 89, "0.62": 80, "0.90": 52,
                                                                        "1.20": 32}, CPU_CANDS)
    assert all(near.values())                                       # every bound is inclusive
    assert not C.window_tests(_pm(jac=0.949), CPU_CANDS, CPU_CANDS)["frames_jaccard"]
    assert not C.window_tests(_pm(), {"0.30": 98}, CPU_CANDS)["candidates"]          # a missing offset fails
    assert not C.window_tests({**_pm(), "h_gap_px": None}, CPU_CANDS, CPU_CANDS)["court_h_gap"]


def _tests(**pairs):
    """pairs: name -> per-window pass (True for all tests, or a list of failing test names)."""
    names = ("frames_jaccard", "boxes_mean", "same_count", "court_h_gap", "candidates")
    out = {}
    for k, v in pairs.items():
        out[k.replace("__", "~")] = {w: {n: (v is True) or (n not in v) for n in names} for w in C.WINDOWS}
    return out


SEP = {"mac_cpu__mac_mps": ["boxes_mean", "same_count"]}


def test_verdicts():
    v = C.verdict(_tests(**SEP, cuda_b__cuda_a=True, cuda_a__mac_cpu=True, cuda_a__mac_mps=["boxes_mean"]))
    assert v["verdict"] == "CUDA matches CPU" and v["repeatability"] == "measured"
    v = C.verdict(_tests(**SEP, cuda_b__cuda_a=True, cuda_a__mac_cpu=["same_count"], cuda_a__mac_mps=True))
    assert v["verdict"] == "CUDA matches MPS"
    v = C.verdict(_tests(**SEP, cuda_b__cuda_a=True, cuda_a__mac_cpu=["court_h_gap"], cuda_a__mac_mps=["boxes_mean"],
                         colab_cpu__mac_cpu=True))
    assert v["verdict"] == "matches neither" and v["colab_cpu_matches_mac_cpu"] is True
    assert v["failed_tests"]["cuda_a~mac_cpu"] == {W3: ["court_h_gap"], W4: ["court_h_gap"]}
    v = C.verdict(_tests(**SEP, cuda_b__cuda_a=["same_count"], cuda_a__mac_cpu=True, cuda_a__mac_mps=["boxes_mean"]))
    assert v["verdict"] == "CUDA not repeatable"
    v = C.verdict(_tests(**SEP, cuda_a__mac_cpu=True, cuda_a__mac_mps=["boxes_mean"]))
    assert v["verdict"] == "CUDA matches CPU" and v["repeatability"] == "not measured"
    v = C.verdict(_tests(mac_cpu__mac_mps=True, cuda_b__cuda_a=True, cuda_a__mac_cpu=True, cuda_a__mac_mps=True))
    assert v["verdict"] == "rule cannot separate"
    assert C.verdict(_tests(**SEP))["verdict"] == "no cuda_a build"


def test_a_match_needs_both_windows():
    one = {W3: {"frames_jaccard": True, "boxes_mean": True, "same_count": True, "court_h_gap": True, "candidates": True}}
    assert not C.matches(one)
    assert C.matches({**one, W4: dict(one[W3])})


def test_guard_refuses_nothing_of_this_check_and_the_tolerances_are_the_registered_ones():
    C.guard_all()
    assert C.TOL == {"frames_jaccard_min": 0.95, "boxes_mean_abs_max": 0.25, "same_count_share_min": 0.80,
                     "h_gap_median_px_max": 1.0, "cand_rel": 0.10, "cand_floor": 3}
    assert C.FIXED_OFFSETS == (0.30, 0.62, 0.90, 1.20)
    assert C.ARMS == {"cuda_a": "cuda", "cuda_b": "cuda", "colab_cpu": "cpu"}


# ── amendments of 2026-11-25 ────────────────────────────────────────────────────────────────────────────────────────
def test_the_rule_reads_unrounded_values():
    r = {f: _frame(f, 8) for f in range(103)}
    x = {f: _frame(f, 9 if f < 26 else 8) for f in range(103)}     # 26/103 = 0.2524 more boxes a frame
    pm = C.pair_metrics(x, r)
    assert round(pm["boxes_mean_diff"], 2) == 0.25                 # would pass if the rule read the rounded value
    assert not C.window_tests(pm, CPU_CANDS, CPU_CANDS)["boxes_mean"]
    assert C.round_floats({"a": [0.123456, 2], "b": {"c": 1.0 / 3}}) == {"a": [0.1235, 2], "b": {"c": 0.3333}}


def test_report_only_window_is_cuda_only_and_never_in_the_verdict():
    w5 = list(C.EXTRA)[0]
    assert C.arm_windows("cuda_a", list(C.ALL_WINDOWS)) == list(C.ALL_WINDOWS)
    assert C.arm_windows("colab_cpu", list(C.ALL_WINDOWS)) == list(C.WINDOWS)
    assert w5 in C.arm_windows("colab_cpu", list(C.ALL_WINDOWS), plumbing=True)
    bad5 = {"frames_jaccard": False, "boxes_mean": False, "same_count": False, "court_h_gap": False, "candidates": False}
    ok = {k: True for k in bad5}
    assert C.matches({W3: ok, W4: dict(ok), w5: bad5})             # the extra window cannot fail a match
    C.guard_all()                                                  # gsw_bkn_2015 passes the guard too


def test_deadline_skips_arms_not_started():
    assert not C.past_deadline(100.0, None, now=10_000.0)
    assert not C.past_deadline(100.0, 600, now=699.0) and C.past_deadline(100.0, 600, now=701.0)


def test_build_window_sets_the_r23_court_for_that_build_only(monkeypatch):
    monkeypatch.delenv("COURT_GRID_WEIGHTS", raising=False)
    seen = []

    def fake(P, w, snip, out, extra_args=()):
        seen.append(__import__("os").environ.get("COURT_GRID_WEIGHTS"))
        return {"ok": True}
    C.build_window(fake, {}, {}, "s.mp4", "out", (), court="/m/court_grid_r23.pt")
    C.build_window(fake, {}, {}, "s.mp4", "out", (), court=None)
    assert seen == ["/m/court_grid_r23.pt", None]

    def boom(*a, **k):
        raise RuntimeError("build died")
    with pytest.raises(RuntimeError):
        C.build_window(boom, {}, {}, "s.mp4", "out", (), court="/m/court_grid_r23.pt")
    assert "COURT_GRID_WEIGHTS" not in __import__("os").environ


def test_run_flags():
    rec = {"commit": "abc123def", "versions": {"transformers": "4.49.0"}, "clip_revision_matches_manifest": True}
    man = {"mac_versions": {"transformers": "4.49.0"}}
    f = C.run_flags(rec, man, "abc123d")
    assert f["pin_mismatch"] is False and not f["clip_revision_differs"] and f["transformers_differs"] is False
    f = C.run_flags({**rec, "versions": {"transformers": "4.57.1"}, "clip_revision_matches_manifest": False}, man, "fff000")
    assert f["pin_mismatch"] is True and f["clip_revision_differs"] and f["transformers_differs"] is True
    assert C.run_flags({**rec, "commit": None}, man, "abc")["pin_mismatch"] is True
    assert C.run_flags(rec, None, None)["pin_mismatch"] is None


def test_attribution_when_colab_cpu_is_missing():
    v = C.verdict(_tests(**SEP, cuda_b__cuda_a=True, cuda_a__mac_cpu=["court_h_gap"], cuda_a__mac_mps=["boxes_mean"]))
    assert v["verdict"] == "matches neither" and "cannot be traced" in v["attribution"]
    v = C.verdict(_tests(**SEP, cuda_b__cuda_a=True, cuda_a__mac_cpu=["court_h_gap"], cuda_a__mac_mps=["boxes_mean"],
                         cuda_a__colab_cpu=["same_count"]))
    assert "cuda_a~colab_cpu" in v["failed_tests"] and v["verdict"] == "matches neither"   # reported, not deciding


def test_check_result_keeps_an_arm_whose_report_only_window_failed(tmp_path):
    out_dir, _ = _result(tmp_path, arms=("cuda_a",))
    with tarfile.open(out_dir / f"{C.RESULT}.tar") as tf:
        tf.extractall(tmp_path / "res", filter="data")
    rr = tmp_path / "res" / C.RESULT / "run_record.json"
    rec = json.loads(rr.read_text())
    w5 = list(C.EXTRA)[0]
    rec["builds"]["cuda_a"][w5] = {"ok": True, "seconds": 1.0, "court_weights": "court_grid_snapped.pt",
                                   "device": {"resolved": {"detector": "cuda", "court": "cuda", "gate": "cuda"}}}
    rr.write_text(json.dumps(rec))
    _, usable, unusable = C.check_result(tmp_path / "res", None, plumbing=False)
    assert usable == ["cuda_a"] and "cuda_a (report-only window)" in unusable     # wrong court model, flagged
