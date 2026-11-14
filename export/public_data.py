"""export/public_data.py: the public data root, with no SportVU frame data (PLAN.md D60).

    python3 -m export.public_data --src <full data root> --out <public data root>

The SportVU source is marked not for redistribution, so the public site carries none of its frame data. This
command reads a full data root (written by export.site_data or export.sample) and writes a copy without it:

- games/<slug>/<possession>.json keeps the per-possession values (game, possession, period, offense and defense,
  attack_x, result, points, gap, the EPV curve with its own times, the shots, notes and sources) and empties the
  per-frame part: t, game_clock_s, shot_clock_s, players and the ball arrays are empty and ghosts is null (a ghost
  position is model output fed by SportVU frames, so it is frame data too). A shot located by the tracking at its
  release loses x and y and says location_source "none"; a shot chart location (basketball-reference) is kept.
  Notes about frames are dropped and one note says where the positions are. The sources name only what the file
  still holds: the ghosting sources go, and the tracking-frames source says what is still read from it.
- demo/frames.json keeps the boxes, the broadcast points and error_ft of every frame; truth (the sensor
  positions at the synced moment) is empty.
- Every other contract file is copied byte for byte. The manifest is written again with the new sizes and hashes.

The output validates with export.schemas.validate_tree, and the same input gives the same bytes. Running it on
its own output changes nothing. Nothing here reads a model or computes a value: it only removes.

The output folder must be empty, absent, or a public root this command wrote before (its manifest's build_id ends
in "-public"). Contract files of an earlier run that this run does not write are removed from it; nothing else is.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from export import schemas as S

PUBLIC_SUFFIX = "-public"
FRAME_NOTE = "Per-frame positions are on the private copy only (PLAN.md D60)."
SHOT_NOTE = ("Shot locations measured from the tracking at the release are on the private copy only (PLAN.md D60): "
             "{n}. Their time, outcome and xFG are kept.")
DEMO_NOTE = ("truth is empty in this copy: the sensor positions of the synced moments are on the private copy only "
             "(PLAN.md D60). boxes, broadcast and error_ft are kept.")
# Notes of export/site_games.py that describe the frames of a possession, which a public file no longer holds.
FRAME_NOTE_PREFIXES = (
    "Single frame:",
    "Frames missing from the tracking table:",
    "Frames with fewer than",
    "Players in this possession:",
    "Frames with no ball position:",
    "No ghost defenders are stored",
    "Ghost points more than",
)
# The tracking table of a game. A public possession file no longer holds its positions, ball or clocks; the
# attacked rim, the clock gap flag and the release times are still read from it.
TRACKING_PREFIX = "$DATA_ROOT/sportvu_contract/"
TRACKING_WHAT = ("tracking frames: read for the attacked rim, the clock gap flag and the release times; the "
                 "positions, the ball and the clocks themselves are on the private copy only (PLAN.md D60)")


class PublicDataProblem(Exception):
    """The input or the output folder is not what the command needs."""


def dump(model: S.Strict) -> str:
    """The file text of a contract model, in the form export.site_data and export.sample write."""
    return json.dumps(model.model_dump(mode="json"), separators=(",", ":"), allow_nan=False) + "\n"


def _is_ghost_source(source: S.Source) -> bool:
    return "/analytics/ghosting/" in source.path or source.path == "reports/ghosting.md"


def _is_tracking_source(source: S.Source) -> bool:
    return source.path.startswith(TRACKING_PREFIX) and not source.path.endswith("_pbp.parquet")


def public_sources(sources: list) -> list:
    """The sources of a stripped possession: no ghosting source, and the tracking source saying what it still
    feeds. A source no rule names is kept as it is."""
    out, seen = [], set()
    for source in sources:
        if _is_ghost_source(source):
            continue
        if _is_tracking_source(source):
            source = S.Source(path=source.path, what=TRACKING_WHAT, rows=source.rows)
        key = (source.path, source.what, source.rows)
        if key not in seen:
            seen.add(key)
            out.append(source)
    if not out:
        raise PublicDataProblem("a stripped possession would name no source")
    return out


def strip_possession(possession: S.Possession) -> S.Possession:
    """The possession without its frame data (module docstring)."""
    shots, moved = [], 0
    for shot in possession.shots:
        if shot.location_source == "tracking_at_release":
            shot = shot.model_copy(update={"x": None, "y": None, "location_source": "none"})
            moved += 1
        shots.append(shot)
    notes = [note for note in possession.notes if not note.startswith(FRAME_NOTE_PREFIXES)]
    if moved:
        notes.append(SHOT_NOTE.format(n=moved))
    if FRAME_NOTE not in notes:
        notes.append(FRAME_NOTE)
    return S.Possession(
        schema_version=possession.schema_version, game_id=possession.game_id, game_slug=possession.game_slug,
        possession=possession.possession, period=possession.period, hz=possession.hz, frame0=possession.frame0,
        t=[], game_clock_s=[], shot_clock_s=[], offense=possession.offense, defense=possession.defense,
        attack_x=possession.attack_x, result=possession.result, points=possession.points, gap=possession.gap,
        players=[], ball=S.BallTrack(x=[], y=[], z=[]), epv=possession.epv, ghosts=None, shots=shots, notes=notes,
        sources=public_sources(possession.sources))


def strip_frames(frames: S.DemoFrames) -> S.DemoFrames:
    """The demo scenes with no sensor position: every frame keeps its boxes, broadcast points and error."""
    scenes = [scene.model_copy(update={"frames": [frame.model_copy(update={"truth": []}) for frame in scene.frames]})
              for scene in frames.scenes]
    notes = list(frames.notes)
    if DEMO_NOTE not in notes:
        notes.append(DEMO_NOTE)
    return frames.model_copy(update={"scenes": scenes, "notes": notes})


def _sources_in(text: str) -> list:
    """Every source record inside a contract file, in the order they appear."""
    found = []

    def walk(value):
        if isinstance(value, dict):
            if set(value) == set(S.Source.model_fields):
                found.append(S.Source(**value))
            else:
                for item in value.values():
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(json.loads(text))
    return found


def _members(entry: S.ManifestFile, written: dict) -> list:
    """The files of one manifest entry among the files written, by the rule export.schemas checks."""
    if entry.count == 1 and "<" not in entry.path:
        return [entry.path] if entry.path in written else []
    folder = entry.path.rsplit("/", 1)[0]
    return sorted(rel for rel in written if rel.startswith(folder + "/") and rel.endswith(".json")
                  and S.model_for(rel) is not None and S.model_for(rel).__name__ == entry.model)


def _entry_sources(entry: S.ManifestFile, members: list, written: dict, changed: set) -> list:
    """The sources of an entry. An entry whose files were not changed keeps its own list. Otherwise the list keeps
    the entry's sources the files still name, in their order, then any source the files name that it lacks."""
    if not changed.intersection(members):
        return entry.sources
    held = []
    for rel in members:
        held += _sources_in(written[rel].decode("utf-8"))
    keys = {(s.path, s.what, s.rows) for s in held}
    kept = [s for s in entry.sources if (s.path, s.what, s.rows) in keys]
    have = {(s.path, s.what, s.rows) for s in kept}
    extra = {(s.path, s.what, s.rows): s for s in held if (s.path, s.what, s.rows) not in have}
    sources = kept + [extra[key] for key in sorted(extra)]
    if not sources:
        raise PublicDataProblem(f"manifest entry {entry.path} would name no source")
    return sources


def public_manifest(manifest: S.Manifest, written: dict, changed: set) -> S.Manifest:
    """The manifest of the public root: each entry and each game with the sizes and hashes of the files written.
    Every other field is the source manifest's, and the build id says it is the public copy."""
    files = []
    for entry in manifest.files:
        members = _members(entry, written)
        if len(members) != entry.count:
            raise PublicDataProblem(f"manifest entry {entry.path} lists {entry.count} file(s), {len(members)} written")
        hashes = [hashlib.sha256(written[rel]).hexdigest() for rel in members]
        single = entry.count == 1 and "<" not in entry.path
        digest = hashes[0] if single else hashlib.sha256("".join(sorted(hashes)).encode()).hexdigest()
        files.append(entry.model_copy(update={
            "bytes": sum(len(written[rel]) for rel in members), "sha256": digest,
            "sources": _entry_sources(entry, members, written, changed)}))
    games = []
    for game in manifest.games:
        prefix = f"games/{game.slug}/"
        size = sum(len(data) for rel, data in written.items() if rel.startswith(prefix))
        games.append(game.model_copy(update={"bytes": size}))
    build_id = manifest.build_id if manifest.build_id.endswith(PUBLIC_SUFFIX) else manifest.build_id + PUBLIC_SUFFIX
    return manifest.model_copy(update={"build_id": build_id, "files": files, "games": games})


def _contract_files(root: Path) -> tuple[list, list]:
    """The files under a data root the contract knows, and the others, as sorted posix paths."""
    known, unknown = [], []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name in S.IGNORED_NAMES:
            continue
        rel = path.relative_to(root).as_posix()
        (known if S.model_for(rel) is not None else unknown).append(rel)
    return known, unknown


def _check_folders(src: Path, out: Path) -> None:
    if not (src / "manifest.json").is_file():
        raise PublicDataProblem("the source folder has no manifest.json")
    a, b = src.resolve(), out.resolve()
    if a == b or a in b.parents or b in a.parents:
        raise PublicDataProblem("the output folder must be outside the source folder, and the source outside it")
    if out.exists():
        if not out.is_dir():
            raise PublicDataProblem("the output path is a file")
        if any(out.iterdir()):
            try:
                previous = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                previous = {}
            if not str(previous.get("build_id", "")).endswith(PUBLIC_SUFFIX):
                raise PublicDataProblem("the output folder is not empty and is not a public root this command wrote")


def build_public(src, out) -> dict:
    """Write the public root of `src` to `out`. Returns counts: files, stripped possessions, shots whose location
    was removed, demo frames emptied, bytes, and the files of `src` the contract does not know (not copied)."""
    src, out = Path(src), Path(out)
    _check_folders(src, out)
    problems = S.validate_tree(src, allow_missing=S.STAGED_FILES)
    if problems:
        raise PublicDataProblem(f"the source root does not validate ({len(problems)} problem(s)); first: {problems[0]}")
    known, unknown = _contract_files(src)
    written: dict = {}
    changed: set = set()
    info = {"possessions": 0, "shots_moved": 0, "demo_frames": 0, "unknown": unknown}
    for rel in known:
        if rel == "manifest.json":
            continue
        model = S.model_for(rel)
        data = (src / rel).read_bytes()
        if model is S.Possession:
            source = S.Possession.model_validate_json(data)
            info["shots_moved"] += sum(s.location_source == "tracking_at_release" for s in source.shots)
            data = dump(strip_possession(source)).encode("utf-8")
            info["possessions"] += 1
        elif model is S.DemoFrames:
            source = S.DemoFrames.model_validate_json(data)
            info["demo_frames"] += sum(len(scene.frames) for scene in source.scenes)
            data = dump(strip_frames(source)).encode("utf-8")
        else:
            written[rel] = data
            continue
        found = S.find_private(data.decode("utf-8"))
        if found:
            raise PublicDataProblem(f"{rel}: private string {found[0]!r} would be written")
        written[rel] = data
        changed.add(rel)
    manifest = S.Manifest.model_validate_json((src / "manifest.json").read_bytes())
    written["manifest.json"] = dump(public_manifest(manifest, written, changed)).encode("utf-8")

    out.mkdir(parents=True, exist_ok=True)
    for rel, data in written.items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_bytes() != data:
            path.write_bytes(data)
    # Contract files an earlier run wrote that this run did not: a game or a demo file that is gone from the source.
    stale, _ = _contract_files(out)
    for rel in stale:
        if rel not in written:
            (out / rel).unlink()
    for folder in sorted((p for p in out.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if not any(folder.iterdir()):
            folder.rmdir()

    problems = S.validate_tree(out, allow_missing=S.STAGED_FILES)
    if problems:
        raise PublicDataProblem(f"the public root does not validate ({len(problems)} problem(s)); first: {problems[0]}")
    info["files"] = len(written)
    info["bytes"] = sum(len(data) for data in written.values())
    return info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Write the public data root: the full one without SportVU frame "
                                                 "data (PLAN.md D60).")
    parser.add_argument("--src", required=True, help="the full data root (written by export.site_data)")
    parser.add_argument("--out", required=True, help="the folder of the public data root")
    args = parser.parse_args(argv)
    try:
        info = build_public(args.src, args.out)
    except PublicDataProblem as exc:
        print(f"public_data: {exc}", file=sys.stderr)
        return 1
    for rel in info["unknown"]:
        print(f"public_data: {rel} is not a contract file and was not copied")
    print(f"public_data: {info['files']} files, {info['bytes'] / 1e6:.2f} MB; {info['possessions']} possessions "
          f"without frames, {info['shots_moved']} shot locations from the tracking removed, "
          f"{info['demo_frames']} demo frames without sensor positions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
