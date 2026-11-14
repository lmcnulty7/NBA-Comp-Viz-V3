"""tools/check_no_frame_data.py finds position tracks and video, and passes summary files (PLAN.md D60, D62)."""

import json

from export import sample
from tools import check_no_frame_data as C


def _scan(tmp_path, files):
    for name, data in files.items():
        (tmp_path / name).write_bytes(data if isinstance(data, bytes) else data.encode())
    return C.scan(list(files), lambda name: (tmp_path / name).read_bytes())


def test_tracks_video_and_tables_are_found(tmp_path):
    root = tmp_path / "sample"
    sample.write_sample(root)
    possession = (root / "games" / "fixture-game-a" / "1.json").read_text()
    frames = (root / "demo" / "frames.json").read_text()
    found = _scan(tmp_path, {
        "possession.json": possession,
        "frames.json": frames,
        "moments.json": json.dumps({"moments": [[1, 2, 3.0, [[1, 2, 3.0, 4.0, 5.0]]]]}),
        "track.csv": "player_id,x,y\n" + "".join(f"1,{i}.0,{i}.5\n" for i in range(60)),
        "clip.mp4": b"not really a video",
        "frames.parquet": b"PAR1",
    })
    assert set(found) == {"possession.json", "frames.json", "moments.json", "track.csv", "clip.mp4", "frames.parquet"}


def test_summaries_pass(tmp_path):
    root = tmp_path / "sample"
    sample.write_sample(root)
    found = _scan(tmp_path, {
        "scorecard.json": (root / "scorecard.json").read_text(),
        "results.json": (root / "results.json").read_text(),
        "summary.json": json.dumps({"position_error_ft": {"p50": 1.92, "p90": 4.98}, "counts": list(range(10))}),
    })
    assert found == {}
