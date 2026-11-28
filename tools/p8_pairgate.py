"""tools/p8_pairgate.py: the scorer's pairing with the gate applied before the assignment (docs/research/P8_pairgate.md,
version 1.1). Report only: the frozen scorer stays the reported rule and no frozen file is edited; the capped pairing
replaces the sportvu.truth._match attribute for one scoring pass only.

    <frozen CV python> -m tools.p8_pairgate run    inside the heavy-job lock ("p8 pairgate"; V3_P8_ROOT = the reference
                                                   root) -> reports/research/p8_pairgate.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import numpy as np

P8_NO_GAMES = False
WHO = "tools.p8_pairgate"
ALLOWED = ("gsw_cha_2016", "gsw_bkn_2015")
SAME_FIELDS = ("missed", "in_frame", "nr_ghosts", "nr_boxes", "far", "missed_1to1", "unmatched_1to1", "duplicates",
               "cov_acc", "cov_live")      # section 2: rows that do not use _match; they must not move
CAP_S = 15 * 60


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def _pairs(D: np.ndarray, C: np.ndarray, gate: float) -> list:
    from scipy.optimize import linear_sum_assignment
    ri, ci = linear_sum_assignment(C)
    return [(int(i), int(j), float(D[i, j])) for i, j in zip(ri, ci) if D[i, j] <= gate]


def capped_match(court: np.ndarray, sv_xy: np.ndarray, gate: float) -> list:
    """Section 3: Hungarian on min(D, gate), pairs kept at D at most the gate (the frozen _match keeps the same pairs
    from a Hungarian on D itself)."""
    D = np.sqrt(((court[:, None, :] - sv_xy[None, :, :]) ** 2).sum(-1))
    return _pairs(D, np.minimum(D, gate), gate)


def big_match(court: np.ndarray, sv_xy: np.ndarray, gate: float) -> list:
    """8.2, report only: D214's 1e6 rule (pairs beyond the gate cost 1e6)."""
    D = np.sqrt(((court[:, None, :] - sv_xy[None, :, :]) ** 2).sum(-1))
    return _pairs(D, np.where(D <= gate, D, 1e6), gate)


class Counter:
    """Calls and pair differences seen by the swap (8.6, 8.2)."""

    def __init__(self):
        self.frozen_calls = self.capped_calls = 0
        self.pairs = {"frozen": 0, "capped": 0, "gained": 0, "lost": 0, "frames_differ": 0,
                      "capped_vs_1e6_frames_differ": 0, "capped_vs_1e6_pairs_differ": 0}


@contextlib.contextmanager
def swapped(mode: str, counter: Counter):
    """Replace sportvu.truth._match for one pass: "frozen" counts calls to the unchanged function; "capped" returns the
    capped pairs and records how they differ from the frozen and the 1e6 ones. Restored in a finally block."""
    import sportvu.truth as TR
    orig = TR._match

    def frozen(court, sv_xy, gate):
        counter.frozen_calls += 1
        return orig(court, sv_xy, gate)

    def capped(court, sv_xy, gate):
        counter.capped_calls += 1
        a = {(i, j) for i, j, _ in orig(court, sv_xy, gate)}
        out = capped_match(court, sv_xy, gate)
        b = {(i, j) for i, j, _ in out}
        c = {(i, j) for i, j, _ in big_match(court, sv_xy, gate)}
        p = counter.pairs
        p["frozen"] += len(a); p["capped"] += len(b); p["gained"] += len(b - a); p["lost"] += len(a - b)
        p["frames_differ"] += a != b
        p["capped_vs_1e6_frames_differ"] += b != c
        p["capped_vs_1e6_pairs_differ"] += len(b ^ c)
        return out
    TR._match = frozen if mode == "frozen" else capped
    try:
        yield
    finally:
        TR._match = orig


def reading(change: dict, d202: dict) -> dict:
    """8.1: per row, the paired change against D202's half-width, (hi - lo) / 2 -> "scorer-sensitive" or
    "insensitive"."""
    out = {}
    for row, ch in change.items():
        iv = (d202.get(row) or {}).get("interval")
        if not iv or ch.get("change") is None:
            continue
        hw = (iv[1] - iv[0]) / 2.0
        out[row] = {"change": ch["change"], "interval": ch.get("interval"), "d202_half_width": round(hw, 5),
                    "label": "scorer-sensitive" if abs(ch["change"]) > hw else "insensitive"}
    return out


def allowed(names) -> list:
    from tools.p8_common import guard
    from tools.research_testset import refuse_cv_test
    names = [str(n) for n in names]
    bad = [n for n in names if not any(n == g or n.startswith(g + "_") for g in ALLOWED)]
    if bad:
        raise SystemExit(f"{WHO}: {bad[0]} is not in this item's allow-list {ALLOWED}")
    refuse_cv_test(names, WHO)
    guard(names, WHO)
    return names


# ── run (frozen CV python, inside the lock) ────────────────────────────────────────────────────────────────────────
def cmd_run(args) -> dict:
    import subprocess
    from sportvu import bench as B
    from sportvu.sync import MIRRORS
    from tools import p8_frozen as F
    from tools import p8_pin
    from tools import p8_rebaseline as RB
    from tools import p8_scorecard as S
    from tools.p8_common import REPO, atomic_write, guard, p8_root
    from tools.p8_placecause import cache_gaps
    from tools.p8_resetfix import score_pair
    from tools.p8_teamcause import D202_SHA, RUN_ID
    from export.schemas import find_private
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    r = RB.roots()
    if p8_root().resolve() != r["ref"].resolve():
        raise SystemExit(f"{WHO}: V3_P8_ROOT must be the reference root")
    st = subprocess.run(["git", "status", "--porcelain", "--", "tools", "sportvu"], cwd=REPO, capture_output=True,
                        text=True, check=True).stdout.strip()
    if st:
        raise SystemExit(f"{WHO}: tools/ or sportvu/ has uncommitted changes")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    pinned, now = RB.read_json(p8_pin.PIN_FILE), p8_pin.current()
    if [k for k in now if now[k] != pinned.get(k)]:
        raise SystemExit(f"{WHO}: the scorecard does not match its pin")
    allowed(list(ALLOWED))
    recs = {}
    for g in ALLOWED:
        p = r["ref"] / "rebaseline" / f"score__{g}__current.json"
        if RB.sha256(p) != D202_SHA[g]:
            raise SystemExit(f"{WHO}: {p.name} is not the pinned D202 record")
        recs[g] = RB.read_json(p)
        if recs[g].get("run_id") != RUN_ID:
            raise SystemExit(f"{WHO}: {p.name} is not run {RUN_ID}")
        allowed(sorted(recs[g]["windows"]))

    def watch():                                     # 8.3: the pool windows' files and the pool truth_accept records
        files = []
        for g in ALLOWED:
            for s_ in sorted(recs[g]["windows"]):
                files += sorted((r["ref"] / "builds" / "r23").glob(f"{s_}_*"))
                files += sorted((r["ref"] / "label_build").glob(f"{s_}*"))
                files += [r["ref"] / "cache" / "scorecard" / f"{s_}.json",
                          RB.truths(r)[g]["current"] / "labels" / g / f"{s_}.jsonl"]
            files += [r["ms"] / "truth_accept" / f"{g}.json", r["ref"] / "truth_accept" / f"{g}.json"]
        return {str(p.relative_to(p.anchor)): RB.sha256(p) for p in files}
    before = watch()
    t0 = time.time()
    P = F.install_paths()
    S.load_fixed_constants()
    refs = B.References()
    comp = P["builds"] / "r23"
    counter_u, counter_c = Counter(), Counter()
    wins_u, wins_c, g1_bad, same_bad = {g: {} for g in ALLOWED}, {g: {} for g in ALLOWED}, [], []
    for g in ALLOWED:
        gd = F.game_of(g)
        guard([gd["sportvu"], gd["video_id"]], WHO)
        mirror = F.mirror_of(P, g)
        mom = S.Moments(RB.read_json(P["sportvu"] / f"{gd['sportvu']}_moments.json")["moments"])
        ctx = {"mom": mom, "mirror": mirror, "refs": refs, "events": S.pbp_events(gd["sportvu"]),
               "fx": MIRRORS[mirror][0], "fy": MIRRORS[mirror][1]}
        P_t = {**P, "labels": RB.truths(r)[g]["current"] / "labels"}
        wins = sorted((w for w in F.label_windows(P, g) if (comp / f"{w['window']}_frames.json").exists()),
                      key=lambda w: w["window"])
        if [w["window"] for w in wins] != sorted(recs[g]["windows"]):
            raise SystemExit(f"{WHO}: {g}: the windows differ from D202's record")
        for w in wins:                               # score_set's loop (tools/p8_rebaseline.py), D202's own call
            if time.time() - t0 > CAP_S:
                raise SystemExit(f"{WHO}: past the {CAP_S // 60}-minute cap; nothing is recorded")
            s = w["window"]
            hashes = (recs[g].get("hashes") or {}).get(s, {})
            for key, pth in (("sidecar", comp / f"{s}_frames.json"),
                             ("timemap", P_t["label_build"] / f"{s}_timemap.json"),
                             ("labels", P_t["labels"] / g / f"{s}.jsonl"), ("identity", comp / f"{s}_identity.json")):
                want = hashes.get(key)
                if (want is None and key != "identity") or (want is not None and RB.sha256(pth) != want):
                    raise SystemExit(f"{WHO}: {s} {key} does not hash as D202 recorded it")
            gap = cache_gaps(P_t, g, w, comp, mom, mirror)      # 8.4
            if gap:
                raise SystemExit(f"{WHO}: {s}: the scorecard cache lacks {gap}")
            side = RB.read_json(comp / f"{s}_frames.json")
            with swapped("frozen", counter_u):
                u = score_pair(P_t, g, w, {"r23": comp, "new": comp}, {"r23": side, "new": side}, ctx)["r23"]
            if counter_u.capped_calls:
                raise SystemExit(f"{WHO}: the capped function ran in the uncapped pass")
            want = recs[g]["windows"][s]
            norm = lambda x: json.loads(json.dumps(x))
            if any(norm(u.get(f)) != norm(want.get(f)) for f in RB.GATE_FIELDS):
                g1_bad.append(s)
                continue
            with swapped("capped", counter_c):
                c = score_pair(P_t, g, w, {"r23": comp, "new": comp}, {"r23": side, "new": side}, ctx)["r23"]
            if any(norm(u.get(f)) != norm(c.get(f)) for f in SAME_FIELDS):
                same_bad.append(s)
            wins_u[g][s], wins_c[g][s] = norm(u), norm(c)
    if g1_bad:
        raise SystemExit(f"{WHO}: G1 fails on {len(g1_bad)} windows (first {g1_bad[:3]}); nothing is recorded")
    if same_bad:
        raise SystemExit(f"{WHO}: rows that do not use _match moved on {same_bad[:3]}; a code fault")
    if counter_c.capped_calls != counter_u.frozen_calls or counter_c.frozen_calls:
        raise SystemExit(f"{WHO}: the capped pass called the capped function {counter_c.capped_calls} times, against "
                         f"{counter_u.frozen_calls} frozen calls in the uncapped pass")
    if counter_c.pairs["frames_differ"] == 0:
        raise SystemExit(f"{WHO}: no pair differs between the two rules; the swap had no effect")
    if watch() != before:
        raise SystemExit(f"{WHO}: a watched file changed; nothing is recorded")
    d202 = RB.read_json(REPO / "reports" / "research" / "p8_rebaseline.json")
    scopes = {g: [g] for g in ALLOWED} | {"pooled": list(ALLOWED)}
    out_rows = {}
    for sc, gs in scopes.items():
        keys = [(g, s) for g in gs for s in sorted(wins_u[g])]
        prev, curr = [wins_u[g][s] for g, s in keys], [wins_c[g][s] for g, s in keys]
        paired = RB.change_rows(prev, curr)
        own_u, own_c = RB.unit_rows(prev), RB.unit_rows(curr)
        own = {k: {"uncapped": own_u[k]["value"], "capped": own_c[k]["value"],
                   "difference": (None if own_u[k]["value"] is None or own_c[k]["value"] is None
                                  else own_c[k]["value"] - own_u[k]["value"])}
               for k in own_u if isinstance(own_u[k], dict) and "value" in own_u[k]}
        ref = d202["pooled_cha_bkn"] if sc == "pooled" else d202["per_game"][f"{sc}__current"]
        out_rows[sc] = {"paired_common_players": paired, "own_rule_values": own, "reading": reading(paired, ref),
                        "counts": {"windows": len(keys), "team_n_uncapped": sum(w["team"]["n"] for w in prev),
                                   "team_n_capped": sum(w["team"]["n"] for w in curr)}}
    out = {"what": "P8 pair gate: the scorer's rows with the gate applied before the assignment "
                   "(docs/research/P8_pairgate.md v1.1); report only; the frozen rule stays reported", "head": head,
           "d202": {"run_id": RUN_ID, "sha256": {g: D202_SHA[g] for g in ALLOWED}}, "g1": "passed",
           "calls": {"uncapped_pass_frozen": counter_u.frozen_calls, "capped_pass_capped": counter_c.capped_calls},
           "pairs": counter_c.pairs, "rows": out_rows, "seconds": round(time.time() - t0, 1)}
    text = json.dumps(out, indent=1, default=float)
    if find_private(text):
        raise SystemExit(f"{WHO}: the record holds private data")
    atomic_write(REPO / "reports" / "research" / "p8_pairgate.json", text)
    print(json.dumps({sc: {k: v["label"] for k, v in d["reading"].items()} for sc, d in out_rows.items()}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=WHO)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    args = ap.parse_args(argv)
    {"run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
