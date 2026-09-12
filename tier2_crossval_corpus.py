#!/usr/bin/env python
"""
tier2_crossval_corpus.py — corpus-wide C2 offense cross-validation.

align_outcomes.py emits reports/tier2_crossval.json scoped to the clips of ONE
invocation, so a per-run report describes that run, not the corpus. Promoting
such a number to a project-wide claim is how CLAIMS.md A1 came to cite 91.1%
(n=642) — a run-11 total that no longer describes the outcomes on disk.

This script recomputes the same statistic over EVERY *_outcomes.json present,
using align_outcomes' own definition: among aligned possessions, the fraction
where the independent PBP actor team matches our predicted offense. It reads
only committed alignment output and re-derives nothing, so it cannot perturb
the state the join and credit are built on.

Output: reports/tier2_crossval_corpus.{json,txt}
"""
from __future__ import annotations

import json

import config
from fetch_pbp import PBP_DIR


def rate(records) -> dict:
    agree = sum(1 for r in records if r.get("offense_agrees") is True)
    disagree = sum(1 for r in records if r.get("offense_agrees") is False)
    unknown = sum(1 for r in records if r.get("offense_agrees") is None)
    checked = agree + disagree
    return {"agree": agree, "disagree": disagree, "unknown": unknown,
            "n_checked": checked,
            "agreement_rate": round(agree / checked, 3) if checked else None}


def main() -> None:
    files = sorted(PBP_DIR.glob("*_outcomes.json"))
    aligned, failed, statuses = [], 0, {}
    for f in files:
        for r in json.loads(f.read_text()):
            if r.get("status") != "aligned":
                failed += 1
                statuses[r.get("status")] = statuses.get(r.get("status"), 0) + 1
                continue
            aligned.append(r)

    out = {
        "scope": f"corpus-wide: all {len(files)} *_outcomes.json on disk",
        "clips": len(files),
        "possessions_aligned": len(aligned),
        "anchor_failures": failed,
        "failure_reasons": statuses,
        "offense_cross_validation": rate(aligned),
        # secondary slices — the join applies both of these filters before a
        # possession can reach credit, so they bound what the credit sample saw
        "non_duplicate": rate([r for r in aligned if "duplicate_of_span" not in r]),
        "overlap_ge_080": rate([r for r in aligned
                                if r.get("overlap_frac_of_span", 0) >= 0.8]),
        "note": "Same definition as align_outcomes.py, over every clip rather "
                "than one invocation's. reports/tier2_crossval.json is the "
                "PER-RUN artifact and is scoped to whichever clips that run "
                "processed — do not cite it as a corpus figure.",
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "tier2_crossval_corpus.json").write_text(json.dumps(out, indent=1))

    cv = out["offense_cross_validation"]
    lines = [
        f"TIER 2 C2 CROSS-VALIDATION — corpus-wide ({out['clips']} clips)",
        f"  possessions aligned   {out['possessions_aligned']}",
        f"  anchor failures       {out['anchor_failures']}  {statuses}",
        f"  agree / disagree      {cv['agree']} / {cv['disagree']}  (unknown {cv['unknown']})",
        f"  AGREEMENT RATE        {cv['agreement_rate']}  (n={cv['n_checked']})",
        "",
        f"  non-duplicate only    {out['non_duplicate']['agreement_rate']} "
        f"(n={out['non_duplicate']['n_checked']})",
        f"  overlap >= 0.80       {out['overlap_ge_080']['agreement_rate']} "
        f"(n={out['overlap_ge_080']['n_checked']})",
    ]
    (config.REPORTS_DIR / "tier2_crossval_corpus.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
