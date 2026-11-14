"""export/heavy.py: one heavy job at a time on this machine (PLAN.md P9, D39).

The Mac has 16 GB of memory and has crashed under a large job before. Every job that reads DATA_ROOT at scale,
runs model code, runs the full test suite, builds the site or drives a browser takes this lock first, so parallel
teammates queue instead of overlapping.

    from export.heavy import heavy_job
    with heavy_job("site_data build"):
        ...

    python3 -m export.heavy --name "next build" -- npm run build

The lock is a file lock (fcntl) held by the process that took it. It is reentrant down the process tree: a job
started by a lock holder (a command run through the wrapper, or a nested heavy_job) sees V3_HEAVY_HELD in its
environment and does not wait for itself. The wrapper passes SIGTERM and SIGINT on to its command and waits for it,
so stopping the wrapper stops the job before the lock is released. A wrapper killed with SIGKILL cannot do that:
its command keeps running without the lock, so never kill -9 a wrapper.
"""

import argparse
import contextlib
import fcntl
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

HELD_ENV = "V3_HEAVY_HELD"


def lock_path() -> Path:
    custom = os.environ.get("V3_HEAVY_LOCK")
    return Path(custom) if custom else Path.home() / ".cache" / "nba-v3" / "heavy.lock"


def _gb(raw: int) -> float:
    """ru_maxrss in GB (macOS reports bytes, Linux kilobytes)."""
    return raw / 1e9 if sys.platform == "darwin" else raw / 1e6


def peak_rss_gb(children: bool = False) -> float:
    """Peak resident memory in GB: of this process, or of its finished child processes when `children` is true."""
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    return _gb(resource.getrusage(who).ru_maxrss)


@contextlib.contextmanager
def heavy_job(name: str, wait: bool = True, poll_s: float = 5.0, log=print):
    """Hold the machine-wide heavy-job lock for the duration of the block. Waits for the current holder unless
    `wait` is false, in which case it raises RuntimeError when the lock is taken. Inside a process tree that
    already holds the lock the block runs at once (the holder is an ancestor, so nothing overlaps)."""
    if os.environ.get(HELD_ENV):
        yield
        return
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
        os.environ[HELD_ENV] = str(os.getpid())
        started = time.time()
        try:
            yield
        finally:
            os.environ.pop(HELD_ENV, None)
            peak = max(peak_rss_gb(), peak_rss_gb(children=True))
            log(f"[heavy] {name!r} done in {time.time() - started:.0f} s, peak memory {peak:.1f} GB")
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
        child = subprocess.Popen(command)

        def forward(signum, _frame):
            child.send_signal(signum)

        previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            return child.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
