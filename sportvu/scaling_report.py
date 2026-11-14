"""sportvu/scaling_report.py: first-pass results of the data-scaling study (sportvu/scaling.py).

Reads the court-replay scorecards reports/scaling/<game>__scal_<arm>.json and their per-observation errors
(data/sportvu/bench_obs/<game>__scal_<arm>.json), plus the V3 and r23 replays as reference points, and writes
reports/scaling/scaling_first_pass.{json,txt}:
  - per arm: position error p50 / p90 and near / far-third bias on gsw_phx_2016 (new game, training arena)
    and on CLE (cle_nyk_2015 + cle_gsw_2016 pooled; new arena), coverage
  - paired window-cluster bootstrap (95%) of each arm minus A_full, and the three contrasts:
      frames   B_f1 / B_f4 vs A_full          (more frames from the same windows)
      games    C_<game> vs C_all4, same 48 windows (one game vs four)
      arena    D_nocle vs A_full on CLE        (the old set's Cleveland frames removed)
Positive differences mean the first build is worse (larger error or larger |bias|).
"""
from __future__ import annotations
import json
import numpy as np
import config
from sportvu.bench import OBS_DIR
from sportvu.compare import _stats

CARDS = config.PROJECT_ROOT / "reports" / "scaling"
ARMS = ["A_full", "A_full_s2", "A_full_s3", "B_f1", "B_f4", "C_all4", "C_all4_s2", "C_all4_s3", "C_bkn", "C_cha", "C_ind", "C_sac",
        "D_nocle", "E_k0", "E_k3", "E_k9", "E_kall", "F_c4_s1", "F_c4_s2", "F_c4_s3", "F_full_s2", "F_full_s3"]
SEEDS = {"A_full": ["A_full", "A_full_s2", "A_full_s3"], "C_all4": ["C_all4", "C_all4_s2", "C_all4_s3"],
         "F_full(r23 recipe)": ["r23", "F_full_s2", "F_full_s3"], "F_c4(r23 recipe)": ["F_c4_s1", "F_c4_s2", "F_c4_s3"]}
REFS = {"V3": "v3_replay", "r23": "r23_replay"}
SETS = {"phx": ["gsw_phx_2016"], "CLE": ["cle_nyk_2015", "cle_gsw_2016"]}
REPS = 2000


def obs(build: str, games: list) -> dict:
    out = {}
    for g in games:
        p = OBS_DIR / ("%s__%s.json" % (g, build))
        if p.exists():
            for w, rows in json.loads(p.read_text()).items():
                out[w] = {(o[0], o[1]): o[2:] for o in rows}
    return out


def summary(build: str, games: list) -> dict:
    rows = [r for w in obs(build, games).values() for r in w.values()]
    s = _stats(rows)
    return {k: round(v, 2) for k, v in s.items()} | {"n": len(rows)}


def paired(a: str, b: str, games: list, seed: int = 0) -> dict:
    A, B = obs(a, games), obs(b, games)
    wins = sorted(w for w in set(A) & set(B) if set(A[w]) & set(B[w]))
    if not wins:
        return {}
    pa = {w: [A[w][k] for k in sorted(set(A[w]) & set(B[w]))] for w in wins}
    pb = {w: [B[w][k] for k in sorted(set(A[w]) & set(B[w]))] for w in wins}
    full_a, full_b = _stats(sum(pa.values(), [])), _stats(sum(pb.values(), []))
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in full_a}
    for _ in range(REPS):
        pick = rng.choice(len(wins), len(wins), replace=True)
        sa = _stats([o for i in pick for o in pa[wins[i]]]); sb = _stats([o for i in pick for o in pb[wins[i]]])
        for k in boot:
            boot[k].append(sa[k] - sb[k])
    out = {}
    for k, v in boot.items():
        v = np.array(v); v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[k] = {"diff": round(full_a[k] - full_b[k], 2), "ci95": [round(float(lo), 2), round(float(hi), 2)],
                  "verdict": "worse" if lo > 0 else ("better" if hi < 0 else "no difference")}
    return out


def main() -> None:
    rep = {"arms": {}, "contrasts": {}}
    for name in list(REFS) + ARMS:
        build = REFS.get(name, "scal_" + name)
        if not (OBS_DIR / ("gsw_phx_2016__%s.json" % build)).exists():
            continue
        rep["arms"][name] = {s: summary(build, gs) for s, gs in SETS.items()}
        cov = {}
        for g in sum(SETS.values(), []):
            p = (CARDS if name in ARMS else config.PROJECT_ROOT / "reports" / "scorecard") / ("%s__%s.json" % (g, build))
            if not p.exists():
                p = config.PROJECT_ROOT / "reports" / "scorecard" / ("%s__%s.json" % (g, {"V3": "v3", "r23": "r23"}.get(name, build)))
            if p.exists():
                cov[g] = json.loads(p.read_text())["metrics"]["coverage"]["coverage"]
        rep["arms"][name]["coverage"] = cov
    pairs = [("B_f1", "A_full"), ("B_f4", "A_full"), ("C_all4", "A_full"), ("D_nocle", "A_full"), ("A_full", "r23_ref")]
    pairs += [("C_" + g, "C_all4") for g in ("bkn", "cha", "ind", "sac")]
    pairs += [("A_full_s2", "A_full"), ("A_full_s3", "A_full"), ("C_all4_s2", "C_all4"), ("C_all4_s3", "C_all4"),
              ("E_k0", "E_kall"), ("E_k3", "E_kall"), ("E_k9", "E_kall"), ("E_kall", "A_full"),
              ("F_c4_s1", "r23_ref"), ("F_c4_s2", "r23_ref"), ("F_c4_s3", "r23_ref"), ("F_full_s2", "r23_ref"), ("F_full_s3", "r23_ref")]
    for a, b in pairs:
        if a not in rep["arms"] or (b != "r23_ref" and b not in rep["arms"]):
            continue
        ba = "scal_" + a
        bb = REFS["r23"] if b == "r23_ref" else "scal_" + b
        rep["contrasts"]["%s - %s" % (a, b if b != "r23_ref" else "r23")] = {s: paired(ba, bb, gs) for s, gs in SETS.items()}
    # training-seed spread: same data, 3 seeds; a single-run difference smaller than this range is not evidence
    rep["seed_spread"] = {}
    for base, runs in SEEDS.items():
        if all(r in rep["arms"] for r in runs):
            rep["seed_spread"][base] = {s: {k: [min(rep["arms"][r][s][k] for r in runs), max(rep["arms"][r][s][k] for r in runs)]
                                            for k in ("p50", "p90", "abs_near", "abs_far")} for s in SETS}
    rep["arena_curve"] = {a: rep["arms"][a]["CLE"] for a in ("E_k0", "E_k3", "E_k9", "E_kall") if a in rep["arms"]}
    (CARDS / "scaling_first_pass.json").write_text(json.dumps(rep, indent=1))
    L = ["DATA-SCALING STUDY, FIRST PASS: held-out position error from the exact court replay (sportvu.court_replay)",
         "phx = gsw_phx_2016 (new game, training arena); CLE = cle_nyk_2015 + cle_gsw_2016 pooled (new arena)", "",
         "%-8s | %-34s | %-34s | coverage" % ("arm", "phx p50 / p90 / near / far (ft)", "CLE p50 / p90 / near / far (ft)")]
    for name, r in rep["arms"].items():
        f = lambda s: "%5.2f / %5.2f / %+5.2f / %+5.2f" % (r[s]["p50"], r[s]["p90"], r[s]["abs_near"], r[s]["abs_far"]) if r[s].get("n") else "-"
        L.append("%-8s | %-34s | %-34s | %s" % (name, f("phx"), f("CLE"), " ".join("%.3f" % v for v in r["coverage"].values())))
    L += ["", "near/far columns are |median bias| toward the camera", "",
          "paired differences, first minus second (95% window-bootstrap interval); positive = first is worse:"]
    for k, v in rep["contrasts"].items():
        for s in SETS:
            if v[s]:
                L.append("  %-16s %-3s p50 %+5.2f [%+5.2f, %+5.2f] %-13s p90 %+5.2f [%+5.2f, %+5.2f] %s" % (
                    k, s, v[s]["p50"]["diff"], *v[s]["p50"]["ci95"], v[s]["p50"]["verdict"],
                    v[s]["p90"]["diff"], *v[s]["p90"]["ci95"], v[s]["p90"]["verdict"]))
    if rep["seed_spread"]:
        L += ["", "training-seed spread (same data, seeds 1..3), min..max:"]
        for base, v in rep["seed_spread"].items():
            for s_ in SETS:
                L.append("  %-7s %-3s p50 %.2f..%.2f  p90 %.2f..%.2f  |near| %.2f..%.2f  |far| %.2f..%.2f" % (
                    base, s_, *v[s_]["p50"], *v[s_]["p90"], *v[s_]["abs_near"], *v[s_]["abs_far"]))
    if rep["arena_curve"]:
        L += ["", "arena curve on CLE (old set limited to k non-Cleveland arenas; Cleveland never seen):"]
        for a, v in rep["arena_curve"].items():
            L.append("  %-7s p50 %.2f  p90 %.2f  |near| %.2f  |far| %.2f" % (a, v["p50"], v["p90"], v["abs_near"], v["abs_far"]))
    (CARDS / "scaling_first_pass.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
