"""export/court_reference.py: where the court renderer must draw one frame of a possession (PLAN.md M3).

The web court renderer (web/components/court/geometry.ts) turns court feet into canvas pixels. This module states
the same transform a second time, in Python, so a web test can draw a possession at one frame and compare the
positions the renderer used with the positions computed here. The two must agree within half a pixel. Nothing
here reads a model output or scores anything: it is geometry only.

The transform (transform version 1)
-----------------------------------
Court positions are in feet in the contract frame (export/schemas.py): x from 0 to 94 along the court, y from 0 to
50 across it, rims at (5.25, 25) and (88.75, 25). The canvas has a width and a height in CSS pixels, its origin at
the top left, x to the right and y down. Court x maps to canvas x and court y to canvas y: no flip, no rotation.

1. Region. The full view shows x in [0, 94]. The half view shows the half the offense attacks: x in [0, 47] when
   attack_x is below 47, otherwise x in [47, 94]. Both views show y in [0, 50].
2. Margin. The region is widened by MARGIN_FT (3 ft) on each of its four sides, so a player who steps out of
   bounds is still on the canvas.
3. Scale. s = min(width / (region_width + 2 * margin), height / (region_height + 2 * margin)), in pixels per foot.
   The widened region is as large as it can be while staying whole, with the same scale on both axes.
4. Centring. The widened region is centred in the canvas:
   left = (width - s * (region_width + 2 * margin)) / 2 + s * margin, and the same for top with the heights.
5. Position. px = left + (x - region_x0) * s and py = top + (y - region_y0) * s. Nothing is rounded.

Every step is linear in the canvas size, so a canvas k times larger gives positions k times larger. Given the
size of the backing store (the CSS size times the device pixel ratio) the same code returns backing-store pixels.

    python3 -m export.court_reference --possession web/data/games/<slug>/<possession>.json --frame 30 \
        --width 358 --height 200 --view full

prints one JSON document: the transform, and the expected pixel position of every player, of the ball, of every
ghost defender and of four court landmarks at that frame. A position is null where the track has no value.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from export import schemas as S

TRANSFORM_VERSION = 1
MARGIN_FT = 3.0
HALF_X_FT = S.COURT_LENGTH_FT / 2.0
VIEWS = ("full", "half")


@dataclass(frozen=True)
class Transform:
    """Feet to canvas pixels for one canvas size and one view. `left` and `top` are the canvas position of the
    region's corner (x0, y0); `scale` is in pixels per foot."""

    view: str
    width: float
    height: float
    x0: float
    y0: float
    region_width_ft: float
    region_height_ft: float
    margin_ft: float
    scale: float
    left: float
    top: float

    def to_px(self, x: float, y: float) -> tuple[float, float]:
        return (self.left + (x - self.x0) * self.scale, self.top + (y - self.y0) * self.scale)


def region(view: str, attack_x: Optional[float] = None) -> tuple[float, float, float, float]:
    """The court region a view shows, as (x0, y0, width, height) in feet."""
    if view == "full":
        return (0.0, 0.0, S.COURT_LENGTH_FT, S.COURT_WIDTH_FT)
    if view == "half":
        if attack_x is None:
            raise ValueError("the half view needs attack_x, the x of the rim the offense attacks")
        x0 = 0.0 if attack_x < HALF_X_FT else HALF_X_FT
        return (x0, 0.0, HALF_X_FT, S.COURT_WIDTH_FT)
    raise ValueError(f"unknown view {view!r}: use one of {VIEWS}")


def court_transform(width: float, height: float, view: str = "full",
                    attack_x: Optional[float] = None) -> Transform:
    """The transform for a canvas of `width` by `height` CSS pixels (steps 1 to 4 of the module docstring)."""
    if not (width > 0 and height > 0):
        raise ValueError("the canvas width and height must be positive")
    x0, y0, region_w, region_h = region(view, attack_x)
    wide_w = region_w + 2.0 * MARGIN_FT
    wide_h = region_h + 2.0 * MARGIN_FT
    scale = min(width / wide_w, height / wide_h)
    left = (width - scale * wide_w) / 2.0 + scale * MARGIN_FT
    top = (height - scale * wide_h) / 2.0 + scale * MARGIN_FT
    return Transform(view=view, width=float(width), height=float(height), x0=x0, y0=y0, region_width_ft=region_w,
                     region_height_ft=region_h, margin_ft=MARGIN_FT, scale=scale, left=left, top=top)


def _point(transform: Transform, x: Optional[float], y: Optional[float]) -> dict:
    """One position in feet and in pixels. Both pixel values are null when either coordinate is missing."""
    if x is None or y is None:
        return {"x_ft": x, "y_ft": y, "px": None, "py": None}
    px, py = transform.to_px(x, y)
    return {"x_ft": x, "y_ft": y, "px": px, "py": py}


def landmarks(transform: Transform) -> dict:
    """Court landmarks in pixels: the two rims, the centre spot and the four corners of the court."""
    named = {
        "rim_left": (S.RIM_X_FT[0], S.RIM_Y_FT),
        "rim_right": (S.RIM_X_FT[1], S.RIM_Y_FT),
        "centre": (HALF_X_FT, S.COURT_WIDTH_FT / 2.0),
        "corner_x0_y0": (0.0, 0.0),
        "corner_x94_y0": (S.COURT_LENGTH_FT, 0.0),
        "corner_x94_y50": (S.COURT_LENGTH_FT, S.COURT_WIDTH_FT),
        "corner_x0_y50": (0.0, S.COURT_WIDTH_FT),
    }
    return {name: _point(transform, x, y) for name, (x, y) in named.items()}


def reference_frame(possession: S.Possession, frame: int, width: float, height: float,
                    view: str = "full") -> dict:
    """Expected canvas positions for frame `frame` of `possession` on a canvas of `width` by `height` pixels."""
    n_frames = len(possession.t)
    if not 0 <= frame < n_frames:
        raise ValueError(f"frame {frame} is outside 0 to {n_frames - 1}")
    transform = court_transform(width, height, view, possession.attack_x)
    players = []
    for track in possession.players:
        entry = {"player_id": track.player_id, "offense": track.offense}
        entry.update(_point(transform, track.x[frame], track.y[frame]))
        players.append(entry)
    ghosts = []
    if possession.ghosts is not None:
        for track in possession.ghosts.tracks:
            entry = {"player_id": track.player_id}
            entry.update(_point(transform, track.x[frame], track.y[frame]))
            ghosts.append(entry)
    return {
        "transform_version": TRANSFORM_VERSION,
        "game_slug": possession.game_slug,
        "possession": possession.possession,
        "n_frames": n_frames,
        "frame": frame,
        "t": possession.t[frame],
        "attack_x": possession.attack_x,
        "transform": asdict(transform),
        "players": players,
        "ball": _point(transform, possession.ball.x[frame], possession.ball.y[frame]),
        "ghosts": ghosts,
        "landmarks": landmarks(transform),
    }


def load_possession(path) -> S.Possession:
    """Read one possession file of the data root and validate it against the contract."""
    return S.Possession.model_validate_json(Path(path).read_text(encoding="utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Expected canvas pixel positions for one frame of a possession file (the court reference).")
    parser.add_argument("--possession", required=True, help="a possession file, games/<slug>/<possession>.json")
    parser.add_argument("--frame", required=True, type=int, help="frame index, 0 for the first frame")
    parser.add_argument("--width", required=True, type=float, help="canvas width in CSS pixels")
    parser.add_argument("--height", required=True, type=float, help="canvas height in CSS pixels")
    parser.add_argument("--view", choices=VIEWS, default="full", help="full court, or the half the offense attacks")
    parser.add_argument("--out", help="write the JSON here instead of printing it")
    args = parser.parse_args(argv)
    try:
        possession = load_possession(args.possession)
        reference = reference_frame(possession, args.frame, args.width, args.height, args.view)
    except (OSError, ValueError) as exc:
        print(f"court_reference: {type(exc).__name__}: {str(exc)[:500]}", file=sys.stderr)
        return 1
    text = json.dumps(reference, indent=2) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
