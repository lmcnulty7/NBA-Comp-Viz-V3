# CLAIMS.md — what this project will and will not claim

The definition of done (Phase 0 of the completion roadmap, 2026-07-16). Every
remaining task must serve a claim below; anything that serves no claim is out
of scope. A claim ships only with its evidence artifact and its caveats — the
project's differentiator is epistemic hygiene, and this file is its contract.

## Tier A — system & validation claims (evidence exists today)

| # | Claim | Evidence | Status |
|---|---|---|---|
| A1 | Vision-derived possession attributions agree with independent play-by-play ground truth **91.1% (n=642)**, with disagreements hard-excluded from all downstream stats by construction | tier2_join checks; PIPELINE.md stage 8 | SHIPPED |
| A2 | Every pipeline stage carries a held-out or human-labeled validation number (gate 98.7%; detection P.89/R.87; homography 0.30 ft median; teams 87.1%; possessions 96.5/91.2%; clock 100%-on-readable) | PIPELINE.md table + eval artifacts | SHIPPED (one gap: A2a) |
| A2a | …except **matchup assignment**, which has structural checks only | — | OPEN → Phase 2 closes it |
| A3 | Stale-artifact consumption is structurally impossible in the credit chain (content-fingerprint guard; refused a real stale join on first deployment) | tier2_credit guard + tests; DEVLOG 07-16 | SHIPPED |
| A4 | OCR read-rate saturates at 720p (30.8%→31.0% at 1080p, controlled A/B); added resolution buys tracking length (~3×) and team-call abstention (53%→11%) instead | DEVLOG 07-12; probe artifacts | SHIPPED |

## Tier B — team-level measurement claims (Phase 1 gates these)

| # | Claim | Requires | Status |
|---|---|---|---|
| B1 | Team-defense credit/100 reported **with cluster-bootstrapped 95% CIs** (by game), never as a point estimate | bootstrap in tier2_credit | SHIPPED 07-16 |
| B2 | Baselines are **leave-sample-out** and **context-matched** (each possession vs its own start-type norm — the audit showed 35% of the sample is live-start) | baseline rework | SHIPPED 07-16 |
| B3 | Funnel selection bias is **quantified**: +8.8/100 residual outcome-shape selection (FT-trip exclusion, made-shot enrichment); period/clock distributions clean. Cross-team comparisons use credit_rel (leave-bucket-out), which differences the common-mode offset out | reports/tier2_bias_audit.* | SHIPPED 07-16 |
| B4 | NEVER claim: "team X's defense is above/below league average." The sample is halfcourt set-cores from a nonrandom slice of games; the claim is always "vs. these opponents' own context-matched norms, on our surviving sample, with this CI" | discipline | STANDING |

## Tier C — player-level claims (capability demo ONLY; Phases 2+3 gate these)

| # | Claim | Requires | Status |
|---|---|---|---|
| C1 | Matchup (primary defender) assignment accuracy **X% vs. human labels (n≥150)** | Phase 2 labeling | OPEN |
| C2 | Jersey-OCR names **Y%** of joined possessions' primary-defender fragments at native 720p | Phase 3 batch + guards (min-reads ≥4, crop-color veto) | OPEN |
| C3 | Player-level credit tables are a CAPABILITY DEMONSTRATION: only GSW players appear in every game, so per-player n ≈ 20–50 — shipped with CIs and an explicit "not conclusions" banner, or not shipped | C1 + C2 + B-machinery | OPEN |

## Kill list (explicitly out of scope — deferred with reasons on record)

- Transition defense (sample is halfcourt set-cores; segmentation treats only set spans)
- Help-position / off-ball attentiveness metrics (unverifiable assumptions; no ball tracking — DEVLOG 07-07 scope decision)
- Ball tracking, shot-quality models
- Closeout tendency as a headline metric (directional only; inherits far-court homography tail)
- League-wide or era-normalized conclusions (2013–2017 pooled, GSW-centric sample)

## Standing caveats that ship with the report

1. **Gate operating point:** harvesting runs at threshold **0.35**, not the
   validated 0.70 (domain shift on unfamiliar broadcasts, DEVLOG 07-05). The
   98.7% number belongs to the 0.70 in-domain eval; the 0.35 point is
   protected downstream by possession-level structure + PBP cross-val, not by
   a frame-level eval of its own.
2. **Funnel yield ~58% span→join** and losses are not random (F3) — B3's audit
   quantifies this; until then no representativeness language.
3. **Identity = track fragment**, not player, everywhere upstream of jersey OCR.
4. **Data provenance:** broadcast video fetched from public YouTube uploads for
   research/education; no video redistributed; repo and report carry derived
   data and fair-use stills only.
