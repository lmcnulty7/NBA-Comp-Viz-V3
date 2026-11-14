"""export/heavy.py: one heavy job at a time on this machine (PLAN.md P9, D39).

The Mac has 16 GB of memory and has crashed under a large job before. Every job that reads DATA_ROOT at scale,
runs model code, runs the full test suite, builds the site or drives a browser takes this lock first, so parallel
teammates queue instead of overlapping.

    from export.heavy import heavy_job
    with heavy_job("site_data build"):
        ...

    python3 -m export.heavy --name "next build" -- npm run build

The lock is a file lock (fcntl), so it is released when the process ends, even if it is killed.
"""

import argparse
import contextlib
import fcntl
import os
import resource
import subprocess
import sys
import time
from pathlib import Path


def lock_path() -> Path:
    custom = os.environ.get("V3_HEAVY_LOCK")
    return Path(custom) if custom else Path.home() / ".cache" / "nba-v3" / "heavy.lock"


def peak_rss_gb() -> float:
    """Peak resident memory of this process in GB (macOS reports bytes, Linux kilobytes)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


@contextlib.contextmanager
def heavy_job(name: str, wait: bool = True, poll_s: float = 5.0, log=print):
    """Hold the machine-wide heavy-job lock for the duration of the block. Waits for the current holder unless
    `wait` is false, in which case it raises RuntimeError when the lock is taken."""
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        waited = 0.0
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not wait:
                    raise RuntimeError(f"another heavy job is running; {name!r} did not start") from None
                if waited == 0.0:
                    handle.seek(0)
                    log(f"[heavy] {name!r} waits for: {handle.read().strip() or 'another heavy job'}")
                time.sleep(poll_s)
                waited += poll_s
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid {os.getpid()} {name}\n")
        handle.flush()
        started = time.time()
        try:
            yield
        finally:
            log(f"[heavy] {name!r} done in {time.time() - started:.0f} s, peak memory {peak_rss_gb():.1f} GB")
            handle.seek(0)
            handle.truncate()
            fcntl.flock(handle, fcntl.LOCK_UN)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a command while holding the heavy-job lock.")
    parser.add_argument("--name", required=True, help="what the job is, shown to anything that waits for it")
    parser.add_argument("--no-wait", action="store_true", help="fail at once when the lock is taken")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="the command, after --")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the command to run after --")
    with heavy_job(args.name, wait=not args.no_wait):
        return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
