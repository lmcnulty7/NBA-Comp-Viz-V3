"""export/public_data.py: the public data root carries no SportVU frame data (PLAN.md D60).

Two inputs, both fast and without DATA_ROOT: the synthetic sample (export.sample, with demo scenes) and a data root
that export.site_data writes from the small synthetic tables of tests/_fx_site_data.py (the real layout: real
source names, ghosts, located releases and shot chart fallbacks). On each: nothing per-frame survives, the values
that should survive are equal, every other file is the same bytes, the tree validates, and the command gives the
same bytes twice and changes nothing when run on its own output.
"""

import json
from pathlib import Path

import pytest

from export import public_data as P
from export import sample
from export import schemas as S
from export import site_data
from export import site_players
from tests import _fx_site_data as FX

REPO = Path(__file__).resolve().parents[1]
QUIET = lambda message: None  # noqa: E731
KEPT = ("schema_version", "game_id", "game_slug", "possession", "period", "hz", "frame0", "offense", "defense",
        "attack_x", "result", "points", "gap", "epv")
SHOT_KEPT = ("event", "t", "t_source", "shooter_id", "shooter", "made", "three", "xfg", "has_contest")


@pytest.fixture(scope="module")
def sample_pair(tmp_path_factory):
    base = tmp_path_factory.mktemp("public_sample")
    sample.write_sample(base / "full")
    P.build_public(base / "full", base / "public")
    return base / "full", base / "public"


@pytest.fixture(scope="module")
def fixture_pair(tmp_path_factory):
    """A real-layout data root written by export.site_data from the synthetic tables, and its public copy."""
    base = tmp_path_factory.mktemp("public_fixture")
    tables = FX.write_data_root(base / "data", site_players.read_baseline(REPO)[2])
    FX.write_artifacts(base / "artifacts", base / "data", tables)
    checks = FX.tracked_checks(base / "artifacts")
    problems, _ = site_data.run(base / "full", base / "data", base / "artifacts", log=QUIET, allow_fixture_root=True,
                                checks_copy=checks)
    assert problems == []
    P.build_public(base / "full", base / "public")
    return base / "full", base / "public"


PAIRS = ["sample_pair", "fixture_pair"]


def _files(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _json(root: Path, rel: str) -> dict:
    return json.loads((root / rel).read_text(encoding="utf-8"))


def _possessions(root: Path) -> list:
    return sorted(rel for rel in _files(root) if S.model_for(rel) is S.Possession)


def _tracks(value, key=None) -> list:
    """Every list of two or more numbers under a key x, y or z, and every non-empty truth list, anywhere."""
    found = []
    if isinstance(value, dict):
        for name, item in value.items():
            found += _tracks(item, name)
    elif isinstance(value, list):
        numbers = [v for v in value if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if key in ("x", "y", "z") and len(numbers) >= 2:
            found.append(key)
        if key == "truth" and value:
            found.append(key)
        for item in value:
            found += _tracks(item)
    return found


@pytest.mark.parametrize("pair", PAIRS)
def test_nothing_per_frame_survives(pair, request):
    full, public = request.getfixturevalue(pair)
    rels = _possessions(public)
    assert rels and rels == _possessions(full)
    for rel in rels:
        p = _json(public, rel)
        assert p["t"] == [] and p["game_clock_s"] == [] and p["shot_clock_s"] == [], rel
        assert p["players"] == [] and p["ball"] == {"x": [], "y": [], "z": []}, rel
        assert p["ghosts"] is None, rel
        for shot in p["shots"]:
            assert shot["location_source"] != "tracking_at_release", rel
            if shot["location_source"] == "none":
                assert shot["x"] is None and shot["y"] is None, rel
        assert P.FRAME_NOTE in p["notes"]
        assert not any(note.startswith(P.FRAME_NOTE_PREFIXES) for note in p["notes"]), rel
    for rel, data in _files(public).items():
        assert _tracks(json.loads(data)) == [], rel


@pytest.mark.parametrize("pair", PAIRS)
def test_the_values_that_should_survive_are_equal(pair, request):
    full, public = request.getfixturevalue(pair)
    moved = 0
    for rel in _possessions(full):
        a, b = _json(full, rel), _json(public, rel)
        for key in KEPT:
            assert a[key] == b[key], (rel, key)
        assert len(a["shots"]) == len(b["shots"])
        for old, new in zip(a["shots"], b["shots"]):
            for key in SHOT_KEPT:
                assert old[key] == new[key], (rel, key)
            if old["location_source"] == "tracking_at_release":
                moved += 1
                assert (new["x"], new["y"], new["location_source"]) == (None, None, "none")
            else:
                assert (new["x"], new["y"], new["location_source"]) == (old["x"], old["y"], old["location_source"])
        kept_notes = [note for note in a["notes"] if not note.startswith(P.FRAME_NOTE_PREFIXES)]
        assert b["notes"][:len(kept_notes)] == kept_notes
    assert moved > 0, "the input has no located release, so the test proves nothing"
    # Every other contract file is the same bytes; the manifest and the stripped files are the only ones written anew.
    a_files, b_files = _files(full), _files(public)
    assert set(a_files) == set(b_files)
    for rel, data in a_files.items():
        if S.model_for(rel) not in (S.Possession, S.DemoFrames, S.Manifest):
            assert b_files[rel] == data, rel


def test_demo_frames_keep_boxes_broadcast_and_error(sample_pair):
    full, public = sample_pair
    a, b = _json(full, "demo/frames.json"), _json(public, "demo/frames.json")
    assert any(frame["truth"] for scene in a["scenes"] for frame in scene["frames"])
    assert [s["id"] for s in a["scenes"]] == [s["id"] for s in b["scenes"]]
    for old_scene, new_scene in zip(a["scenes"], b["scenes"]):
        assert {k: v for k, v in old_scene.items() if k != "frames"} == {k: v for k, v in new_scene.items() if k != "frames"}
        for old, new in zip(old_scene["frames"], new_scene["frames"], strict=True):
            assert new["truth"] == []
            assert {k: v for k, v in old.items() if k != "truth"} == {k: v for k, v in new.items() if k != "truth"}
    assert P.DEMO_NOTE in b["notes"] and b["sources"] == a["sources"]


def test_sources_name_only_what_a_file_still_holds(fixture_pair):
    full, public = fixture_pair
    saw_tracking = saw_ghosts = False
    for rel in _possessions(full):
        before = _json(full, rel)["sources"]
        after = _json(public, rel)["sources"]
        saw_ghosts |= any("ghosting" in s["path"] for s in before)
        assert not any("/analytics/ghosting/" in s["path"] or s["path"] == "reports/ghosting.md" for s in after), rel
        for source in after:
            if source["path"].startswith(P.TRACKING_PREFIX) and not source["path"].endswith("_pbp.parquet"):
                saw_tracking = True
                assert source["what"] == P.TRACKING_WHAT
        assert {s["path"] for s in after} == {s["path"] for s in before
                                              if "/analytics/ghosting/" not in s["path"] and s["path"] != "reports/ghosting.md"}
    assert saw_tracking and saw_ghosts
    manifest = _json(public, "manifest.json")
    for entry in manifest["files"]:
        if entry["model"] == "Possession":
            assert not any("ghosting" in s["path"] for s in entry["sources"]), entry["path"]


@pytest.mark.parametrize("pair", PAIRS)
def test_the_public_tree_validates_and_its_manifest_is_rebuilt(pair, request):
    full, public = request.getfixturevalue(pair)
    assert S.validate_tree(public, allow_missing=S.STAGED_FILES) == []
    a, b = _json(full, "manifest.json"), _json(public, "manifest.json")
    assert b["build_id"] == a["build_id"] + P.PUBLIC_SUFFIX
    for key in ("generated_at", "repo_commit", "synthetic", "n_games", "n_possessions", "freeze_tag", "season"):
        assert a[key] == b[key], key
    assert [f["path"] for f in a["files"]] == [f["path"] for f in b["files"]]
    assert sum(f["bytes"] for f in b["files"]) < sum(f["bytes"] for f in a["files"])


@pytest.mark.parametrize("pair", PAIRS)
def test_same_bytes_twice_and_nothing_changes_on_its_own_output(pair, request, tmp_path):
    full, public = request.getfixturevalue(pair)
    P.build_public(full, tmp_path / "again")
    assert _files(tmp_path / "again") == _files(public)
    P.build_public(full, tmp_path / "again")  # over its own earlier output
    assert _files(tmp_path / "again") == _files(public)
    P.build_public(public, tmp_path / "twice")
    assert _files(tmp_path / "twice") == _files(public)


def test_the_output_folder_is_checked(sample_pair, tmp_path):
    full, _ = sample_pair
    stranger = tmp_path / "stranger"
    stranger.mkdir()
    (stranger / "notes.txt").write_text("not a public root", encoding="utf-8")
    with pytest.raises(P.PublicDataProblem, match="not empty"):
        P.build_public(full, stranger)
    assert (stranger / "notes.txt").exists()
    with pytest.raises(P.PublicDataProblem, match="outside the source"):
        P.build_public(full, full / "public")
    with pytest.raises(P.PublicDataProblem, match="no manifest"):
        P.build_public(tmp_path / "missing", tmp_path / "out")


def test_a_file_of_an_earlier_run_that_is_gone_from_the_source_is_removed(tmp_path):
    sample.write_sample(tmp_path / "full")
    P.build_public(tmp_path / "full", tmp_path / "public")
    # A data root without the demo files (they are staged, so the source still validates once its manifest drops them).
    for rel in S.STAGED_FILES:
        (tmp_path / "full" / rel).unlink()
    manifest = S.Manifest.model_validate_json((tmp_path / "full" / "manifest.json").read_bytes())
    manifest = manifest.model_copy(update={"files": [f for f in manifest.files if f.path not in S.STAGED_FILES]})
    (tmp_path / "full" / "manifest.json").write_text(P.dump(manifest), encoding="utf-8")
    P.build_public(tmp_path / "full", tmp_path / "public")
    assert not (tmp_path / "public" / "demo").exists()
    assert S.validate_tree(tmp_path / "public", allow_missing=S.STAGED_FILES) == []


def test_the_command_line(sample_pair, tmp_path, capsys):
    full, public = sample_pair
    assert P.main(["--src", str(full), "--out", str(tmp_path / "cli")]) == 0
    assert "possessions without frames" in capsys.readouterr().out
    assert _files(tmp_path / "cli") == _files(public)
    assert P.main(["--src", str(tmp_path / "nothing"), "--out", str(tmp_path / "x")]) == 1
