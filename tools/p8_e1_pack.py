"""tools/p8_e1_pack.py: the E1 corpus as one tar per game, for Drive (PLAN.md D156). Many small files written to a
Drive folder from Colab or the Mac mount have been lost before (the async upload drops some), so the corpus travels as
a few tars, each checked by member count where it lands.

    python3 -m tools.p8_e1_pack pack --dest <folder>      write common.tar and <game>.tar plus pack_manifest.json
    python3 -m tools.p8_e1_pack verify --dest <folder>    count the members of each tar in <folder> (for example the
                                                          Drive folder on the Mac mount) against pack_manifest.json

The agent never runs pack or verify against Drive itself (Lucien copies, D137); <folder> can be any local folder.
common.tar holds meta.json, manifest.json and plan.json; <game>.tar holds images/<game>/ and labels/<game>/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from tools.p8_common import guard
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_e1_pack"
COMMON = ("meta.json", "manifest.json", "plan.json")


def members_for(corpus: Path, part: str) -> list:
    """Corpus-relative paths in one part: "common" or a game."""
    if part == "common":
        return [n for n in COMMON if (corpus / n).exists()]
    out = []
    for sub in ("images", "labels"):
        out += sorted(str(p.relative_to(corpus)) for p in (corpus / sub / part).glob("*") if p.is_file())
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack(corpus: Path, dest: Path) -> dict:
    meta = json.loads((corpus / "meta.json").read_text())
    games = list(meta["games"])
    ids = games + [d["sportvu"] for d in meta["datasets"].values()] + [d["video_id"] for d in meta["datasets"].values()]
    refuse_cv_test(ids, WHO)
    guard(ids, WHO)
    dest.mkdir(parents=True, exist_ok=True)
    man = {"corpus_meta_commit": meta.get("code_commit"), "parts": {}}
    for part in ["common", *games]:
        names = members_for(corpus, part)
        tp = dest / f"{part}.tar"
        with tarfile.open(tp, "w") as tf:
            for n in names:
                tf.add(corpus / n, arcname=n)
        man["parts"][part] = {"tar": tp.name, "members": len(names),
                              "images": sum(n.startswith("images/") for n in names),
                              "labels": sum(n.startswith("labels/") for n in names),
                              "bytes": tp.stat().st_size, "sha256": sha256(tp)}
    (dest / "pack_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))
    return man


def count_members(tar_path: Path) -> int:
    with tarfile.open(tar_path) as tf:
        return sum(1 for m in tf.getmembers() if m.isfile())


def verify(dest: Path, parts=None) -> dict:
    """Bytes and member count of each tar in dest against pack_manifest.json -> {part: (members found, expected)};
    SystemExit when either differs or a tar is missing."""
    man = json.loads((dest / "pack_manifest.json").read_text())
    out, bad = {}, []
    for part, rec in man["parts"].items():
        if parts and part not in parts:
            continue
        tp = dest / rec["tar"]
        if not tp.exists() or tp.stat().st_size != rec["bytes"]:
            out[part] = (None, rec["members"])
            bad.append(part)
            continue
        found = count_members(tp)
        out[part] = (found, rec["members"])
        if found != rec["members"]:
            bad.append(part)
    if bad:
        raise SystemExit(f"{WHO}: tars short or missing in {dest}: {bad} {[out[b] for b in bad]}")
    return out


def unpack_checked(src: Path, dest: Path, parts: list) -> dict:
    """On Colab: verify the named parts' bytes and member counts in src (the Drive mount), then extract them from
    there straight into dest (no separate copy step)."""
    counts = verify(src, parts)
    for part in parts:
        with tarfile.open(src / f"{part}.tar") as tf:
            try:
                tf.extractall(dest, filter="data")
            except TypeError:                         # a Python without extraction filters
                tf.extractall(dest)
    return counts


def main(argv=None) -> None:
    from tools.p8_e1data import corpus_root
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_e1_pack")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("pack", "verify"):
        s = sub.add_parser(c)
        s.add_argument("--dest", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "pack":
        pack(corpus_root(), Path(a.dest))
    else:
        print(json.dumps(verify(Path(a.dest)), indent=1))


if __name__ == "__main__":
    main()
