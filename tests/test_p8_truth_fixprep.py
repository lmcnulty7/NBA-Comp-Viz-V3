"""tools.p8_truth_fixprep: moving unfixed label builds and their truth aside, and the fixed-label check. Synthetic
folders only; no video, no model."""

import json

import pytest

from tools import p8_truth_fixprep as F

G = "gsw_cha_2016"
FIXED = {"requested": {"detector": "cpu", "court": "auto", "gate": "auto"},
         "resolved": {"detector": "cpu", "court": "mps", "gate": "mps"}, "reset_fix": True}
UNFIXED = {k: v for k, v in FIXED.items() if k != "reset_fix"}


def window(lb, stem, rec, built=True):
    lb.mkdir(parents=True, exist_ok=True)
    (lb / f"{stem}.mp4").write_bytes(b"clip")
    if rec is not None:
        (lb / f"{stem}_device.json").write_text(json.dumps(rec))
    if built:
        for k in ("frames", "trajectories", "identity", "timemap"):
            (lb / f"{stem}_{k}.json").write_text("{}")
        (lb / f"team_clusters_{stem}.png").write_bytes(b"png")
    else:
        (lb / f"{stem}_gate_empty.json").write_text("{}")


def truth(root):
    (root / "labels" / G).mkdir(parents=True)
    (root / "labels" / G / f"{G}_s01_w00.jsonl").write_text("{}\n")
    for k in ("result.json", "offsets.json", "paint.jsonl"):
        (root / "labels" / f"{G}_{k}").write_text("{}")
    for rel in (f"truth_v2/dev/{G}.json", f"truth_accept/{G}.json", f"truth_moved/{G}.json",
                f"meta/done/fit__{G}.json", f"meta/done/label__{G}__{G}_s01.json", f"meta/done/link__{G}.json",
                f"builds/r23/{G}_s01_w00_frames.json"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("{}")


def test_supersede_moves_unfixed_outputs_and_truth_and_keeps_clips(tmp_path):
    root = tmp_path / "work_ref"
    lb = root / "label_build"
    window(lb, f"{G}_s01_w00", UNFIXED)
    window(lb, f"{G}_s01_w01", None, built=False)                    # a gate-empty window with no record
    window(lb, f"{G}_s01_w02", FIXED)                                # already fixed: stays
    window(lb, "hou_cle_2016_s01_w00", UNFIXED)                      # another game: stays
    truth(root)
    out = F.supersede(G, root)
    dest = root / "superseded_nofix"
    assert (dest / "label_build" / f"{G}_s01_w00_frames.json").exists()
    assert (dest / "label_build" / f"team_clusters_{G}_s01_w00.png").exists()
    assert (dest / "label_build" / f"{G}_s01_w01_gate_empty.json").exists()
    assert (lb / f"{G}_s01_w00.mp4").exists() and (lb / f"{G}_s01_w00_timemap.json").exists()
    assert not (lb / f"{G}_s01_w00_frames.json").exists()
    assert (lb / f"{G}_s01_w02_frames.json").exists() and (lb / "hou_cle_2016_s01_w00_frames.json").exists()
    assert (dest / "labels" / G / f"{G}_s01_w00.jsonl").exists() and not (root / "labels" / G).exists()
    for rel in (f"truth_accept/{G}.json", f"truth_v2/dev/{G}.json", f"meta/done/fit__{G}.json",
                f"meta/done/label__{G}__{G}_s01.json"):
        assert (dest / rel).exists() and not (root / rel).exists()
    assert (root / f"meta/done/link__{G}.json").exists() and (dest / f"builds/r23/{G}_s01_w00_frames.json").exists()
    assert not (root / f"builds/r23/{G}_s01_w00_frames.json").exists()
    assert out["label_windows_moved"] == 1 and (dest / f"{G}.json").exists()


def test_supersede_runs_once_and_never_overwrites(tmp_path):
    root = tmp_path / "work_ref"
    window(root / "label_build", f"{G}_s01_w00", UNFIXED)
    F.supersede(G, root)
    window(root / "label_build", f"{G}_s01_w00", UNFIXED)            # a stray unfixed build appears later
    assert F.supersede(G, root)["game"] == G                          # no-op once recorded
    assert (root / "label_build" / f"{G}_s01_w00_frames.json").exists()
    (root / "superseded_nofix" / f"{G}.json").unlink()
    with pytest.raises(SystemExit, match="already in"):
        F.supersede(G, root)


def test_label_check_needs_the_fix_and_the_reference_devices(tmp_path):
    lb = tmp_path / "work_ref" / "label_build"
    window(lb, f"{G}_s01_w00", FIXED)
    window(lb, f"{G}_s01_w01", FIXED, built=False)
    assert F.label_check(G, tmp_path / "work_ref")["ok"]
    window(lb, f"{G}_s01_w02", UNFIXED)
    window(lb, f"{G}_s01_w03", {**FIXED, "resolved": {**FIXED["resolved"], "detector": "mps"}})
    window(lb, f"{G}_s01_w04", None)
    window(lb, f"{G}_s01_w05", {**FIXED, "clip_fix": True})
    bad = F.label_problems(lb, G)
    assert bad == {f"{G}_s01_w02": "no reset fix", f"{G}_s01_w03": bad[f"{G}_s01_w03"],
                   f"{G}_s01_w04": "no device record", f"{G}_s01_w05": "clip fix"}
    assert bad[f"{G}_s01_w03"].startswith("devices")
    assert not F.label_check(G, tmp_path / "work_ref")["ok"]
    assert not F.label_check(G, tmp_path / "empty")["ok"]


def test_guards(tmp_path):
    with pytest.raises(SystemExit, match="apart"):
        F.supersede("gsw_ind_2016", tmp_path)
    with pytest.raises(SystemExit, match="CV test"):
        F.label_check("tor_gsw_2016", tmp_path)
