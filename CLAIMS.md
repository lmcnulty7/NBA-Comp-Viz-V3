# CLAIMS.md — what this project will and will not claim

The definition of done (Phase 0 of the completion roadmap, 2026-07-16). Every
remaining task must serve a claim below; anything that serves no claim is out
of scope. A claim ships only with its evidence artifact and its caveats — the
project's differentiator is epistemic hygiene, and this file is its contract.

## Stage 1 definition of done (ROADMAP.md, written 2026-10-03)

Targets, not claims. A row becomes a claim only when the scorecard measures it on held-out
footage; until then the V3 column is the baseline to beat. Status: CONFIRMED by Lucien on
2026-10-03 (ROADMAP gate after R0.3) with two changes: the court line row is reported, not
pass/fail, until the truth can resolve it, and the held-out era is reported with label-free
indicators only. The court stage is judged on position error and near-field bias.

| Metric | How measured | Target | V3 baseline, gsw_phx_2016 |
|---|---|---|---|
| Court line error | px between the projected template and the template under the SportVU truth H | reported. Goal <= 3 px on 95% of accepted wide frames; pass/fail only once the truth fit residual is <= 1.5 px median (the truth must be twice as fine as the goal) | not measurable (corrected 2026-10-04: the truth H fits the players' feet, not the painted court; reports/sportvu_truth_line_check.txt). The earlier 0.0% <= 3 px / p50 26.7 px compared against that truth and is withdrawn |
| Position error, visible players | pipeline court position vs SportVU, all matched players | p50 <= 2.0 ft, p90 <= 5.0 ft | p50 7.33 ft, p90 15.35 ft (n=11,292 players) |
| Near-field bias | median error toward the camera, near third vs far third | within +-0.5 ft | near third -1.02 ft, far third +1.67 ft |
| Coverage | share of live wide-shot seconds with accepted positions | reported; >= 50% of live play | 98.6% (838 of 850 live wide s; V3 has no acceptance rule) |
| Missed players | SportVU players in frame with no box within 2.5 ft | <= 5% | 25.1% (n=14,020; projected through the truth H away from the fitted feet, see the correction below) |
| Ghost boxes | boxes > 3 ft from any player, referees excluded by class | <= 5% | 15.1% non-referee (15.3% with referees; 280 of 12,637 boxes overlap a referee detection; same caveat as missed players) |
| Team labels | vs the SportVU team of the matched player | >= 95% | 81.8% (n=10,829) |
| Identity | id switches per possession vs SportVU player ids; jersey-read rate | reported; no target yet (720p ceiling) | 21.7 id switches per possession (56 shot-clock possessions); jersey-read rate not measurable (no OCR on the windows) |
| Generalisation | the same metrics on an arena never trained on; the held-out era (2013, no SportVU) gets label-free indicators only | held-out arena within 1.5x of the held-out game; era reported, not pass/fail | held-out arena CLE (R2.1): cle_nyk_2015 position p50 11.04 / p90 18.40 ft (1.51x / 1.20x of phx, not met), near +2.87 / far +7.61 ft; cle_gsw_2016 p50 9.51 / p90 17.14 ft (1.30x / 1.12x, met), near +1.60 / far +4.50 ft, coverage 38.5%. Era not measured yet |

V3 column: reports/scorecard/gsw_phx_2016__v3.{json,txt} (ROADMAP R0.3, 2026-10-03). Caveat:
testable frames only (1,447 of 9,500 rebuilt window frames, 15%, where a truth H could be fitted),
so the numbers are optimistic; the truth H passes through the pipeline's own boxes; a local
windowed rebuild, not the production artifacts; wide shots judged by gate v2, which saw phx frames
in training; the referee class is the V3 detector's own, its recall never measured; id switches
are counted on sparse samples (a lower bound).

Generalisation cell: reports/scorecard/cle_nyk_2015__v3.* and cle_gsw_2016__v3.* (ROADMAP R2.1,
2026-10-04), same bench and same B4 truth method as phx. Caveat: far fewer testable frames than phx
(485 of 6,393 and 158 of 3,380; V3 finds fewer than 6 boxes on 2,896 and 1,141 frames there), so the
CLE numbers are thinner and lean further toward frames V3 handles; cle_gsw_2016 has 830 s of SportVU
clock gaps (713 frames with no moment) and 4 of its 34 windows (25 s, short leftovers) failed to build
locally; gate v2, which defines the wide seconds, saw another CLE home game in training. The ratio is
per game against phx for the same build; R2.7 reports the arena as a whole.

Correction (2026-10-04, found at ROADMAP R1.3): the per-frame truth H (sportvu/truth.py) is an
8-parameter homography fitted to 6..10 feet bunched in one part of the court. Drawn over the frame it
sits about as far from the painted lines as V3's own H (A1 ridge distance median 39.2 px vs 45.8 px on
252 phx frames; worse than V3's on 34% of frames; reports/sportvu_truth_line_check.{json,txt}). It is
valid near the fitted feet only. Consequences: the court line row is not measurable with it (the
earlier 0.0% / 26.7 px and the 6.0 px "floor" are withdrawn); missed-player and ghost-box counts,
which project through it away from the fitted players, carry that error; position error and
near-field bias, which compare only players matched near the feet, stand; R1.3's court keypoint
labels cannot be projected through it. The scorecard's n differ from A8 because it re-matches every build's feet through
H_truth at 5 ft instead of reusing B4's stored pairs. The scorecard reproduces
row A8 (B5, reports/sportvu_phx.*) within 0.03 ft on position error.

Physical ceiling: one 720p camera gives about 1.5 ft median at best. Hawk-Eye (14 cameras, about
1 inch) is matched in output format and in honesty about error, not in accuracy, and the product
says so.

**Measurement protocol.**
- Held-out game: gsw_phx_2016 (2015-12-16, PHX at GSW, Oracle Arena; SportVU log
  12.16.2015.PHX.at.GSW).
- Held-out arena: CLE (Quicken Loans Arena), chosen 2026-10-04 at the [HUMAN] gate after ROADMAP
  R1.1 because OKC and NYK have no game with a usable broadcast (reports/sportvu_manifest.txt).
  Scored on 12.23.2015 NYK at CLE and 01.18.2016 GSW at CLE (the latter with 830 s of SportVU clock
  gaps). The whole arena is held out: no game played there enters training.
- Held-out era: gsw_nyk_curry54 (2013-02-27, GSW at NYK, the 2013 production game), untouched by
  training. The public SportVU logs cover 2015-16 only, so this game has no tracking truth: it is
  reported with the label-free indicators (qc/, track_qc.py) and is not a pass/fail row (decided
  2026-10-03).
- Nothing is trained, tuned, calibrated or thresholded on a held-out set. Thresholds come from the
  train arenas. Held-out sets are reported once per adoption decision. The splits live in
  `sportvu/splits.json` (ROADMAP R1.4) and every training script refuses held-out ids.
- Caveat (found at R1.4, 2026-10-04; reports/sportvu_splits.txt): gate v2 (models/trained_head_v2,
  the wide-frame gate) was trained before the splits existed on harvest frames that include 294
  frames of gsw_phx_2016, 299 of gsw_nyk_curry54 and 289 of gsw_cle_xmas16 (a CLE home game). It
  places no court lines and no players, but it picks the wide frames that the scorecard scores and
  counts for coverage, so wide-frame selection on those games is optimistic, and "untouched by
  training" above holds for every court and position model, not for gate v2. Builds are compared
  on the same cached frames, so adoption verdicts are unaffected. Its training scripts now refuse
  to run until those games are dropped.
- Every row reports its n, and beside it the untestable share (frames where no truth H fits) and the
  coverage. A rejected frame carries no positions, never wrong ones, and rejections are counted
  per rule per game.
- Adoption: a stage ships only if, on the held-out game AND the held-out arena scorecards, it
  improves every pass/fail row the stage touches and worsens no pass/fail row. Rows marked reported
  (court line error until its truth can resolve 3 px, identity, the held-out era) are printed
  beside the verdict but do not decide adoption. Coverage is a floor: it may fall but must stay
  >= 50% of live wide seconds.

**The single scorecard.** `python -m sportvu.bench <build_dir>` (ROADMAP R0.3) writes
`reports/scorecard/<game>__<build>.{json,txt}`; the V3 baseline is
`reports/scorecard/gsw_phx_2016__v3.{json,txt}`. Every number in this section comes from a file
under `reports/scorecard/`; row A8 keeps its B5 numbers until ROADMAP R5.3 rewrites it.

## Tier A — system & validation claims (evidence exists today)

| # | Claim | Evidence | Status |
|---|---|---|---|
| A1 | Vision-derived possession attributions agree with independent play-by-play ground truth **87.8% (n=990)**, with disagreements hard-excluded from all downstream stats by construction | reports/tier2_crossval_corpus.*; tier2_join checks | SHIPPED — **corrected 2026-09-11**: the prior 91.1% (n=642) was run-11's PER-RUN align report promoted to a corpus claim; recomputed corpus-wide it is 87.8% (869/990). Non-duplicate 89.6% (n=904); overlap≥0.80 88.0% (n=933) |
| A2 | Every pipeline stage carries a held-out or human-labeled validation number (gate 98.7%; detection P.89/R.87; homography 0.30 ft median; teams 87.1%; possessions 96.5/91.2%; clock 100%-on-readable) | PIPELINE.md table + eval artifacts | SHIPPED (one gap: A2a) |
| A2a | …except **matchup assignment**, which has structural checks only | — | OPEN → Phase 2 closes it |
| A3 | Stale-artifact consumption is structurally impossible in the credit chain (content-fingerprint guard; refused a real stale join on first deployment) | tier2_credit guard + tests; DEVLOG 07-16 | SHIPPED |
| A4 | OCR read-rate saturates at 720p (30.8%→31.0% at 1080p, controlled A/B); added resolution buys tracking length (~3×) and team-call abstention (53%→11%) instead | DEVLOG 07-12; probe artifacts | SHIPPED |
| A5 | Auto-label accuracy is **measured, not assumed** (LABEL_SCHEMA rule 4, 300 stratified accepted labels, human-judged): the **agreement band is 95.0% [88.8, 97.8] (n=100)** and is the usable dataset; the **Claude-adjudicated band measures 6.7–20.0% (n=200) and is NOT used for training**, along with the attributes riding on it | reports/label_audit.*; data/label_audit/labels.json | SHIPPED 2026-09-12 |
| A6 | The qualification gate protected only classes with an in-house baseline; classes exempted as "nothing to compare against" (rim/backboard/scorebug/ball + attributes) are exactly the ones that failed the audit — recorded as a design defect, not a footnote | reports/label_audit.*; DEVLOG 09-12 | SHIPPED 2026-09-12 |
| A7 | The judge's frame-level shot_type labels were audited before use (200 class-stratified frames, human): closeup-claimed 98.0% [89.5, 99.6] but **wide-claimed only 68.8% [57.9, 77.8]** — ~3 in 10 "wide" frames are closeups — so they are NOT used as gate training data for the wide class | reports/shot_type_audit.*; data/shot_type_audit/labels.json | SHIPPED 2026-10-02 |
| A8 (proposed 2026-10-03, FIX_PLAN B5) | On the held-out game gsw_phx_2016 against SportVU tracking (12.16.2015 PHX at GSW), on the frames where a truth homography could be fitted (1,447 of 9,500 rebuilt window frames, 15%): player position error **p50 7.3 ft, p90 15.4 ft** under the pipeline H, compressed toward mid-court (far players pulled 1.7 ft toward the camera, near players pushed 1.0 ft away); **25.1% of in-frame players have no box within 2.5 ft; 15.3% of boxes are more than 3 ft from any player (referees included)**; team label accuracy **81.8%** (n=11,240); pipeline primary man = SportVU nearest opponent **48.1%** (n=160, a proxy for C1, not the human-labeled matchup) | reports/sportvu_phx.*; reports/sportvu_truth_gsw_phx_2016.json; sportvu/ | PROPOSED, not a shipped claim. Correction 2026-10-04: the missed-player and ghost-box counts project through a truth H that is valid only near the fitted feet (see the Stage 1 section; reports/sportvu_truth_line_check.txt); position error stands. Caveats: testable frames only (the untestable 85% are where the pipeline H was too far off to match 6 players, so these numbers are optimistic); truth H derived through the pipeline's own boxes; local windowed rebuild, not the production artifacts (same code, models, h264 source); gate v2 saw phx frames (court, detector, teams, identity did not) |

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
- League-wide or era-normalized conclusions (2012–2017 pooled, GSW-centric sample)

## Standing caveats that ship with the report

1. **Gate operating point:** the possession harvest ran gate v1 at **0.70** — `harvest_driver.py` calls
   `build_trajectories.py --pregate`, which gates every frame at `thresholds.json["trained"]` (0.70 in every
   commit incl. the harvest commit b751d18). **Provenance corrected 2026-10-02:** docs from 07-05 to 10-02
   said the harvest ran at 0.35; that figure belongs to `label_factory.py` (the court LABEL factory), never to
   the possession harvest. **Measured in-domain (2026-10-02):** on all 6,159 harvest frames, human-verified
   (contact-sheet passes; 3,544 wide): at 0.70 the gate keeps 98.8% precision (37 FP, FP-rate 0.014) but
   drops **14.1% of truly wide frames** (recall 0.859, 500 FN) — the production error is lost yield, not
   contamination. At 0.35 it would have admitted 13.1% of non-wide frames (343; precision 0.912). The original
   objective (min FN s.t. FP-rate <= 0.10) picks 0.42 (recall 0.998, precision 0.933); max accuracy 0.975
   at 0.55. **Retrain-vs-recalibrate decided 2026-10-02** (leave-one-source-out, reports/gate_retrain_experiment.*):
   a head refit on the harvest frames + prototype train/val wins by the adoption rule (acc 0.970 vs 0.957,
   FP-rate 0.068 vs 0.098 at recall 0.998, worst source 0.948 vs 0.920, prototype test 0.975 within v1's
   interval) and ships as **gate v2** (`models/trained_head_v2.*`, threshold 0.33) for all FUTURE harvesting.
   Every published number was produced with v1 at 0.70 and stays so until the games are re-harvested
   under v2, after which the PBP canary and bias audit are re-run.
2. **Funnel yield ~58% span→join** and losses are not random (F3) — B3's audit
   quantifies this; until then no representativeness language.
3. **Identity = track fragment**, not player, everywhere upstream of jersey OCR.
4. **Data provenance:** broadcast video fetched from public YouTube uploads for
   research/education; no video redistributed; repo and report carry derived
   data and fair-use stills only.
