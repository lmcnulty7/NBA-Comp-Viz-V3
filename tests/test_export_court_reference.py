"""The court reference (export/court_reference.py): feet to canvas pixels, stated in Python.

The web court renderer states the same transform in web/components/court/geometry.ts, and a browser test
compares the two within half a pixel. These tests pin the Python side on its own: the transform on known
numbers, null handling, the command line, and that the two files still name the same transform version and
margin. No network, no real data unless SITE_DATA_DIR points at a data root; nothing here is a result.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from export import court_reference as CR
from export import sample
from export import schemas as S

REPO = Path(__file__).resolve().parents[1]
GEOMETRY_TS = REPO / "web" / "components" / "court" / "geometry.ts"
DASHES = (chr(0x2013), chr(0x2014))


@pytest.fixture(scope="module")
def sample_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("court_reference_root")
    sample.write_sample(root)
    return root


@pytest.fixture(scope="module")
def possession(sample_root):
    return CR.load_possession(sample_root / "games" / "fixture-game-a" / "1.json")


def test_full_view_on_a_canvas_of_the_same_shape():
    # 94 + 6 by 50 + 6 feet at 10 pixels per foot: the widened region fills the canvas exactly.
    t = CR.court_transform(1000, 560)
    assert (t.scale, t.left, t.top) == (10.0, 30.0, 30.0)
    assert t.to_px(0, 0) == (30.0, 30.0)
    assert t.to_px(94, 50) == (970.0, 530.0)
    assert t.to_px(S.RIM_X_FT[0], S.RIM_Y_FT) == (82.5, 280.0)
    assert t.to_px(S.RIM_X_FT[1], S.RIM_Y_FT) == (917.5, 280.0)
    assert t.to_px(47, 25) == (500.0, 280.0)


@pytest.mark.parametrize("width, height, scale, left, top", [
    (1000, 1000, 10.0, 30.0, 250.0),   # a taller canvas: bands above and below
    (2000, 560, 10.0, 530.0, 30.0),    # a wider canvas: bands left and right
    (358, 200.48, 3.58, 10.74, 10.74),  # a phone: 390 px less two 16 px gutters
])
def test_the_region_is_centred_and_never_stretched(width, height, scale, left, top):
    t = CR.court_transform(width, height)
    assert t.scale == pytest.approx(scale)
    assert t.left == pytest.approx(left)
    assert t.top == pytest.approx(top)
    # The centre of the court is the centre of the canvas, whatever the shape of the canvas.
    px, py = t.to_px(47, 25)
    assert (px, py) == pytest.approx((width / 2, height / 2))


def test_half_view_shows_the_half_the_offense_attacks():
    right = CR.court_transform(530, 560, "half", attack_x=S.RIM_X_FT[1])
    assert (right.x0, right.region_width_ft, right.scale) == (47.0, 47.0, 10.0)
    assert right.to_px(47, 0) == (30.0, 30.0)
    assert right.to_px(94, 50) == (500.0, 530.0)
    left = CR.court_transform(530, 560, "half", attack_x=S.RIM_X_FT[0])
    assert (left.x0, left.scale) == (0.0, 10.0)
    assert left.to_px(0, 0) == (30.0, 30.0)
    assert left.to_px(S.RIM_X_FT[0], S.RIM_Y_FT) == (82.5, 280.0)


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        CR.court_transform(0, 100)
    with pytest.raises(ValueError):
        CR.court_transform(100, -1)
    with pytest.raises(ValueError):
        CR.court_transform(100, 100, "half")          # no attack_x
    with pytest.raises(ValueError):
        CR.court_transform(100, 100, "quarter", 5.25)


@pytest.mark.parametrize("view", CR.VIEWS)
def test_a_larger_canvas_scales_every_position(view):
    small = CR.court_transform(358, 200, view, attack_x=88.75)
    large = CR.court_transform(358 * 3, 200 * 3, view, attack_x=88.75)
    for x, y in ((0, 0), (5.25, 25), (61.3, 12.9), (94, 50), (-2, 53)):
        sx, sy = small.to_px(x, y)
        lx, ly = large.to_px(x, y)
        assert (lx, ly) == pytest.approx((3 * sx, 3 * sy))


def test_reference_frame_places_every_player_the_ball_and_the_ghosts(possession):
    frame = 30
    ref = CR.reference_frame(possession, frame, 1000, 560)
    assert ref["transform_version"] == CR.TRANSFORM_VERSION
    assert (ref["game_slug"], ref["possession"], ref["frame"]) == ("fixture-game-a", 1, frame)
    assert ref["n_frames"] == len(possession.t)
    assert ref["t"] == possession.t[frame]
    assert len(ref["players"]) == 10
    assert sum(p["offense"] for p in ref["players"]) == 5
    for entry, track in zip(ref["players"], possession.players):
        assert entry["player_id"] == track.player_id
        assert entry["px"] == pytest.approx(30 + 10 * track.x[frame])
        assert entry["py"] == pytest.approx(30 + 10 * track.y[frame])
    assert ref["ball"]["px"] == pytest.approx(30 + 10 * possession.ball.x[frame])
    assert ref["ball"]["py"] == pytest.approx(30 + 10 * possession.ball.y[frame])
    assert [g["player_id"] for g in ref["ghosts"]] == [g.player_id for g in possession.ghosts.tracks]
    for entry, track in zip(ref["ghosts"], possession.ghosts.tracks):
        assert entry["px"] == pytest.approx(30 + 10 * track.x[frame])
    assert ref["landmarks"]["rim_left"]["px"] == pytest.approx(82.5)
    assert ref["landmarks"]["corner_x94_y50"] == {"x_ft": 94.0, "y_ft": 50.0, "px": 970.0, "py": 530.0}
    assert set(ref["landmarks"]) == {"rim_left", "rim_right", "centre", "corner_x0_y0", "corner_x94_y0",
                                     "corner_x94_y50", "corner_x0_y50"}


def test_a_missing_value_has_no_pixel_position(possession, sample_root):
    # The sample has no ghost output in its first frames, and its third possession has no ghosts at all.
    early = CR.reference_frame(possession, 0, 800, 448)
    assert early["ghosts"] and all(g["px"] is None and g["py"] is None for g in early["ghosts"])
    assert all(p["px"] is not None for p in early["players"])
    no_ghosts = CR.load_possession(sample_root / "games" / "fixture-game-a" / "3.json")
    assert CR.reference_frame(no_ghosts, 5, 800, 448)["ghosts"] == []
    # A player with no tracking row in a frame keeps his place in the list, with null pixels.
    doc = json.loads((sample_root / "games" / "fixture-game-a" / "1.json").read_text())
    doc["players"][3]["x"][12] = None
    doc["ball"]["y"][12] = None
    holed = S.Possession.model_validate(doc)
    ref = CR.reference_frame(holed, 12, 800, 448)
    assert ref["players"][3]["px"] is None and ref["players"][3]["py"] is None
    assert ref["players"][2]["px"] is not None
    assert ref["ball"]["px"] is None and ref["ball"]["py"] is None


def test_frame_must_be_inside_the_possession(possession):
    last = len(possession.t) - 1
    CR.reference_frame(possession, last, 400, 224)
    for bad in (-1, last + 1):
        with pytest.raises(ValueError):
            CR.reference_frame(possession, bad, 400, 224)


def test_half_view_follows_the_possession(sample_root):
    # Possession 1 of the sample attacks the left rim and possession 2 the right one.
    first = CR.load_possession(sample_root / "games" / "fixture-game-a" / "1.json")
    second = CR.load_possession(sample_root / "games" / "fixture-game-a" / "2.json")
    assert (first.attack_x, second.attack_x) == (S.RIM_X_FT[0], S.RIM_X_FT[1])
    assert CR.reference_frame(first, 10, 530, 560, "half")["transform"]["x0"] == 0.0
    assert CR.reference_frame(second, 10, 530, 560, "half")["transform"]["x0"] == 47.0


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "export.court_reference", *args], cwd=REPO, capture_output=True,
                          text=True)


def test_command_line_prints_the_reference(possession, sample_root, tmp_path):
    path = sample_root / "games" / "fixture-game-a" / "1.json"
    done = run_cli("--possession", str(path), "--frame", "30", "--width", "358", "--height", "200.48")
    assert done.returncode == 0, done.stderr
    printed = json.loads(done.stdout)
    assert printed == CR.reference_frame(possession, 30, 358, 200.48)
    out = tmp_path / "nested" / "reference.json"
    done = run_cli("--possession", str(path), "--frame", "30", "--width", "530", "--height", "560", "--view",
                   "half", "--out", str(out))
    assert done.returncode == 0 and done.stdout == ""
    assert json.loads(out.read_text()) == CR.reference_frame(possession, 30, 530, 560, "half")


def test_command_line_fails_clearly(sample_root, tmp_path):
    path = sample_root / "games" / "fixture-game-a" / "1.json"
    too_far = run_cli("--possession", str(path), "--frame", "9999", "--width", "358", "--height", "200")
    assert too_far.returncode == 1 and "frame 9999 is outside" in too_far.stderr
    missing = run_cli("--possession", str(tmp_path / "none.json"), "--frame", "0", "--width", "358", "--height",
                      "200")
    assert missing.returncode == 1 and "court_reference:" in missing.stderr
    index = run_cli("--possession", str(sample_root / "games" / "fixture-game-a" / "possessions.json"), "--frame",
                    "0", "--width", "358", "--height", "200")
    assert index.returncode == 1, "a file that is not a possession must be refused"


def test_the_web_geometry_names_the_same_transform():
    # The half-pixel comparison runs in the browser (npm run check). This keeps the two files from drifting
    # apart between runs: same transform version, same margin, same court.
    text = GEOMETRY_TS.read_text(encoding="utf-8")

    def constant(name):
        found = re.search(rf"^export const {name} = ([^;]+);", text, flags=re.M)
        assert found, f"{name} is not in geometry.ts"
        return found.group(1).strip()

    assert int(constant("TRANSFORM_VERSION")) == CR.TRANSFORM_VERSION
    assert float(constant("MARGIN_FT")) == CR.MARGIN_FT
    assert float(constant("COURT_LENGTH_FT")) == S.COURT_LENGTH_FT
    assert float(constant("COURT_WIDTH_FT")) == S.COURT_WIDTH_FT
    assert float(constant("RIM_Y_FT")) == S.RIM_Y_FT
    rims = re.search(r"\[([^\]]+)\]", constant("RIM_X_FT"))
    assert tuple(float(v) for v in rims.group(1).split(",")) == S.RIM_X_FT


def test_owned_files_hold_no_dash_and_no_private_string():
    for path in (REPO / "export" / "court_reference.py", Path(__file__)):
        text = path.read_text(encoding="utf-8")
        assert not any(dash in text for dash in DASHES), path.name
        assert S.find_private(text) == [], path.name


def test_real_data_root_when_one_is_named():
    """Runs only when SITE_DATA_DIR names a data root: every possession of its first game gets a reference."""
    named = os.environ.get("SITE_DATA_DIR", "").strip()
    if not named:
        pytest.skip("SITE_DATA_DIR is not set")
    root = Path(named)
    if not root.is_absolute():
        root = REPO / "web" / root
    if not (root / "manifest.json").exists():
        pytest.skip("SITE_DATA_DIR has no manifest.json")
    manifest = S.Manifest.model_validate_json((root / "manifest.json").read_text(encoding="utf-8"))
    game = manifest.games[0]
    index = S.PossessionIndex.model_validate_json(
        (root / "games" / game.slug / "possessions.json").read_text(encoding="utf-8"))
    assert index.possessions, "the first game has no possession"
    for row in index.possessions[:25]:
        loaded = CR.load_possession(root / "games" / game.slug / f"{row.possession}.json")
        frame = len(loaded.t) // 2
        for view in CR.VIEWS:
            ref = CR.reference_frame(loaded, frame, 358, 200.48 if view == "full" else 378.26, view)
            t = CR.Transform(**ref["transform"])
            for entry in ref["players"] + ref["ghosts"] + [ref["ball"]]:
                if entry["px"] is None:
                    continue
                assert entry["px"] == pytest.approx(t.left + (entry["x_ft"] - t.x0) * t.scale)
                assert entry["py"] == pytest.approx(t.top + (entry["y_ft"] - t.y0) * t.scale)
