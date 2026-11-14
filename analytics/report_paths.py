"""analytics/report_paths.py: take local paths out of a lane report before it is committed.

Reports record the command that wrote them, and on the owner's Mac that command names DATA_ROOT, a Google Drive
folder whose path carries the Drive account, and the repo's absolute path. The repo is public, so a committed report
names neither: DATA_ROOT is written as $DATA_ROOT, the repo root is dropped (paths become repo-relative), the home
folder is written as ~, and any remaining CloudStorage folder name loses its account part.

    python -m analytics.report_paths [--data DATA_ROOT] REPORT.md [REPORT.md ...]

run_all calls redact_file on each lane report after every step. --data defaults to the DATA_ROOT environment variable.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLOUD_ACCOUNT = re.compile(r"(CloudStorage/[A-Za-z]+)-[^/\s`\"']+")


def redact(text: str, data_root=None, repo_root=REPO_ROOT, home=None) -> str:
    """``text`` with DATA_ROOT as $DATA_ROOT, the repo root removed, the home folder as ~ and any CloudStorage
    account name removed. Longer prefixes go first, so DATA_ROOT under the home folder is still $DATA_ROOT."""
    home = str(Path.home()) if home is None else str(home)
    prefixes = []
    if data_root:
        prefixes.append((str(data_root).rstrip("/"), "$DATA_ROOT"))
    if repo_root:
        root = str(repo_root).rstrip("/")
        prefixes.append((root + "/", ""))
        prefixes.append((root, "."))
    if home:
        prefixes.append((home.rstrip("/"), "~"))
    for old, new in sorted(prefixes, key=lambda p: -len(p[0])):
        text = text.replace(old, new)
    return CLOUD_ACCOUNT.sub(r"\1", text)


def redact_file(path, data_root=None, repo_root=REPO_ROOT, home=None) -> bool:
    """Rewrite ``path`` redacted; True when it changed, False when it is absent or already clean."""
    path = Path(path)
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    new = redact(text, data_root=data_root, repo_root=repo_root, home=home)
    if new == text:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.environ.get("DATA_ROOT"), help="DATA_ROOT (default: $DATA_ROOT)")
    ap.add_argument("reports", nargs="+")
    a = ap.parse_args(argv)
    for r in a.reports:
        if redact_file(r, data_root=a.data):
            print("redacted", r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
