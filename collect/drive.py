"""collect/drive.py: the owner's Mac and its synced Google Drive folder (nba_harvest).

The Drive desktop client mounts the account under ~/Library/CloudStorage/GoogleDrive-<account>/My Drive; the
collectors' outputs live under nba_harvest/collect there (collect.common.DRIVE_OUT is the same folder as Colab
sees it, /content/drive/MyDrive/nba_harvest/collect). Two helpers:

  drive_root()                 the synced nba_harvest folder ($NBA_DRIVE_ROOT first, then the CloudStorage glob)
  sync(src, dst, ...)          copy a tree with rsync-like semantics in Python: a file is copied when the
                               destination is missing or has a different size (same size: skipped, no hashing);
                               nothing is ever deleted at the destination; .part files are skipped; copies go
                               through a .part rename so a torn copy never looks complete

  python -m collect.drive --sync-to DATA_ROOT             # data/collect -> DATA_ROOT/collect (Drive by default)
  python -m collect.drive --sync-from DATA_ROOT           # DATA_ROOT/collect -> data/collect
  python -m collect.drive --sync-to DATA_ROOT --only raw bref_shots --dry-run
  python -m collect.drive --where                         # print the resolved Drive root and stop
DATA_ROOT is the folder that holds collect/ (nba_harvest on Drive); --local names the local collect/ root
(default data/collect, or $NBA_COLLECT_OUT). --only limits the copy to these top-level entries of collect/ (raw,
a family name, ...). Manifests and raw pages travel with their family, so a run resumed on either machine sees
the same units done.
"""
from __future__ import annotations
import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

from collect import common

DRIVE_GLOB = "~/Library/CloudStorage/GoogleDrive-*/My Drive/nba_harvest"
COLAB_ROOT = "/content/drive/MyDrive/nba_harvest"
SKIP_SUFFIXES = (".part", ".json.part")


def drive_root(env: dict | None = None) -> Path | None:
    """The synced nba_harvest folder: $NBA_DRIVE_ROOT, else the first CloudStorage match, else the Colab mount; None
    when none exists."""
    env = os.environ if env is None else env
    if env.get("NBA_DRIVE_ROOT"):
        p = Path(env["NBA_DRIVE_ROOT"]).expanduser()
        return p if p.is_dir() else None
    hits = sorted(glob.glob(os.path.expanduser(DRIVE_GLOB)))
    if hits:
        return Path(hits[0])
    return Path(COLAB_ROOT) if os.path.isdir(COLAB_ROOT) else None


def plan(src: Path, dst: Path, only: list[str] | None = None) -> list[tuple[Path, Path, str]]:
    """(source file, destination file, reason) for every file under src that sync would copy: 'missing' or
    'size <src> != <dst>'. Files whose destination has the same size are left out."""
    src, dst = Path(src), Path(dst)
    out = []
    if not src.is_dir():
        return out
    tops = sorted(p for p in src.iterdir()) if not only else [src / o for o in only]
    for top in tops:
        if not top.exists():
            continue
        files = [top] if top.is_file() else sorted(p for p in top.rglob("*") if p.is_file())
        for f in files:
            if f.name.endswith(SKIP_SUFFIXES):
                continue
            rel = f.relative_to(src)
            d = dst / rel
            if not d.exists():
                out.append((f, d, "missing"))
            elif d.stat().st_size != f.stat().st_size:
                out.append((f, d, "size %d != %d" % (f.stat().st_size, d.stat().st_size)))
    return out


def sync(src: Path, dst: Path, only: list[str] | None = None, dry_run: bool = False, log=print) -> dict:
    """Copy what plan() lists. Returns {"copied": n, "bytes": n, "skipped": n, "files": [relative paths]}."""
    src, dst = Path(src), Path(dst)
    todo = plan(src, dst, only)
    total = sum(1 for top in (src.iterdir() if src.is_dir() else []) for f in ([top] if top.is_file() else top.rglob("*")) if f.is_file() and not f.name.endswith(SKIP_SUFFIXES))
    copied, nbytes, files = 0, 0, []
    for f, d, why in todo:
        rel = str(f.relative_to(src))
        log("%s %s (%s)" % ("would copy" if dry_run else "copy", rel, why))
        files.append(rel)
        if dry_run:
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        tmp = d.with_name(d.name + ".part")
        shutil.copyfile(f, tmp)
        os.replace(tmp, d)
        copied += 1
        nbytes += f.stat().st_size
    log("%s: %d files to copy, %d copied (%.1f MB), %d skipped (same size)" % ("dry run" if dry_run else "sync", len(todo), copied, nbytes / 1e6, max(total - len(todo), 0)))
    return {"copied": copied, "bytes": nbytes, "skipped": max(total - len(todo), 0), "files": files}


def main(argv=None) -> dict | None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sync-to", metavar="DATA_ROOT", nargs="?", const="", help="copy the local collect/ root to DATA_ROOT/collect (default DATA_ROOT: the Drive nba_harvest folder)")
    ap.add_argument("--sync-from", metavar="DATA_ROOT", nargs="?", const="", help="copy DATA_ROOT/collect to the local collect/ root")
    ap.add_argument("--local", default=os.environ.get("NBA_COLLECT_OUT") or str(common.DEFAULT_OUT), help="the local collect/ root (default data/collect)")
    ap.add_argument("--only", nargs="*", help="only these top-level entries of collect/ (raw, bref_shots, ...)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--where", action="store_true", help="print the resolved Drive root and stop")
    a = ap.parse_args(argv)
    root = drive_root()
    if a.where:
        print(root or "no Drive root: set NBA_DRIVE_ROOT or sync nba_harvest in the Google Drive client")
        return None
    if a.sync_to is None and a.sync_from is None:
        ap.error("give --sync-to or --sync-from")
    data_root = a.sync_to if a.sync_to else a.sync_from
    data_root = Path(data_root).expanduser() if data_root else root
    if data_root is None:
        sys.exit("no DATA_ROOT given and no Drive root found (%s)" % DRIVE_GLOB)
    local = Path(a.local).expanduser()
    if a.sync_to is not None:
        return sync(local, data_root / "collect", a.only, a.dry_run)
    return sync(data_root / "collect", local, a.only, a.dry_run)


if __name__ == "__main__":
    main()
