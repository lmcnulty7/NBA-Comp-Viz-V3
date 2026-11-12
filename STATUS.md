# STATUS.md: extension build, 2026-11-05 (unattended)

## Runbook progress

**WAITING ON YOU (2026-11-12): group 1 is frozen at tag `freeze-v1`. Review the freeze, then create
`gates/go-stage8` (`mkdir -p gates && touch gates/go-stage8`) to start the single held-out run of group 1.** What to
review: the "Frozen for the held-out run (freeze-v1)" section at the end of each group 1 page in docs/contract/
(xfg with its baseline, rebounds, matchups, gravity, ghosting, embeddings, epv, decisions, lineups, winprob, load),
and three choices the loop made that you may want to overrule before the run:
1. ghosting keeps the MLP stand-in, not the stage 6 torch GRU (the GRU was worse paired by game, 0.111 ft [0.085,
   0.138] at 1.0 s), although stage 6 says to replace the stand-in.
2. Two improvement-search variants were accepted and merged: winprob margin_over_sqrt_time and the rebounds hgb
   temperature (reports/archive/).
3. matchups is frozen with ridge although ridge is worse than the constant out of fold (its selection rule only
   compares raw with ridge); embeddings has no held-out score by design (held-out games are dropped before any
   feature), so stage 8 gives it none.
Stage 8 runs about 3 to 4 hours one lane at a time (epv about 21 GB and gravity about 15 GB peak footprint, both
finished on this Mac before). Nothing is running now.

Owner decisions 2026-11-11 (recorded here as asked):
- **Stage 4 hand labels are deferred** (the owner may label later or source labels elsewhere). The 8-game list
  and the commands stay below in "Stage 4 labelling (deferred)".
- **Lane groups** (docs/mac_runbook.md, "Lane groups"): group 1, the label-free lanes (xfg and its baseline,
  rebounds, matchups, gravity, ghosting, embeddings, epv, decisions, lineups, winprob, load), goes through
  stages 5 to 9 now and is frozen as freeze-v1 with its own single held-out run (gate gates/go-stage8). Group 2
  (actions, coverage, playcalling) waits for the labels and gets freeze-v2 and its own single held-out run
  (gate gates/go-stage8-v2). No lane is ever scored on held-out twice.
- **Betting skipped** (gates/odds-skip) and **NBA.com tables skipped** (gates/nbacom-skip). Both gate files were
  created by the loop on the owner's explicit instruction. Embeddings runs without the aggregate block;
  winprob's benchmark column stays empty with that reason stated.
- **Ghosting train memory:** fit on every 5th defender-step (1 Hz) of all training games, float32; evaluation
  unchanged at the 5 Hz roll-forward, streamed one game at a time; timed on 10 games first, then the full run
  if it fits overnight and in memory.

### Stage 4 labelling (deferred)

The 8 training games picked for labelling (16 teams; none held out): 10.29.2015.MEM.at.IND,
11.14.2015.DET.at.LAC, 11.17.2015.MIN.at.MIA, 11.30.2015.DAL.at.SAC, 12.19.2015.LAL.at.OKC, 12.21.2015.POR.at.ATL,
01.02.2016.DEN.at.GSW, 01.09.2016.WAS.at.ORL. Pass 1: `python3 -m analytics.label_tool --data "$DATA_ROOT" --game
<game> --sample 50 --seed 0 --port 8765`; pass 2 a week later with `--labeller-pass 2` (sample ranks 0, 10, 20,
30, 40 only); then gates/labels-done and `--kappa` (at least 0.7 for actions and coverages).

Stage 1 passed: PR #5 and PR #6 merged 2026-11-07. Stage 2 passed 2026-11-07. Two later decisions are open
(odds and NBA.com, under Open gates).

PRs #8 (stage 4a --skip-heldout), #9 (stage 9 results) and #10 (stage 6 torch models) merged 2026-11-11.

Updated 2026-11-11 by the autonomous loop (docs/mac_runbook.md).

- **Current stage:** 5 for group 1 (label-free lanes); stage 4 labels deferred, group 2 waits for them.
- **Label tool fix (2026-11-11):** `--labeller-pass 2` with `--sample` now lists only every tenth sample_rank
  (`--relabel-every`, default 10). Before, pass 2 listed the whole sample, and each `n` recorded a possession as
  visited by pass 2, so unrelabelled possessions would have entered kappa as (label, none). Tests in
  tests/test_label_tool.py.
- **Background jobs:** see the stage 5 group 1 entry below. Chain (one at a time): ghosting and
  lineups, winprob, both stage 6 GRU runs, the winprob table rerun
  the decisions rerun on the GRU EPV and the rebounds reruns are done. Nothing running.
- **Improvement search settled (2026-11-12):** 2 variants tried, both accepted (reports/archive/winprob.md,
  reports/archive/rebounds.md). Not searched, with the reason: xfg (calibrated, ECE 0.003, slope 1.00),
  embeddings (PCA clears its gate), gravity (the plan ships ridge), matchups and decisions (no signal is the
  finding; variants against the validation metric would be fishing), lineups (only 18 to 27 lineups recur
  across seasons, the limit is data), load (no failure the report points at), epv and ghosting (stage 6).
- **Stage 7 (freeze-v1) done 2026-11-12:** each group 1 contract page ends with "Frozen for the held-out run
  (freeze-v1)": the selected model, its settings, the rules and thresholds the held-out run applies, the
  out-of-fold headline and the stage 8 argv. Frozen models: xfg hgb_contest_prior (log loss 0.6294 [0.6277,
  0.6316]) and the baseline logistic (0.6665); rebounds hgb with temperature (1.6009 [1.5882, 1.6103]); matchups
  ridge (MSE 1.2854, worse than the constant 1.2842); gravity ridge (R2 0.416); ghosting MLP stand-in (1.0 s 3.657
  ft [3.635, 3.682]); embeddings PCA (5-NN 0.768 [0.720, 0.809]); epv torch GRU (MSE 1.2155 [1.2080, 1.2226]);
  decisions xfg_selected prior on the GRU EPV (no validated signal); lineups rapm (tie with plus-minus); winprob
  logistic with margin_over_sqrt_time (0.1468); load ridge and logistic (absence log loss 0.2764 [0.2644,
  0.2883]). Fixed before tagging: gravity and load wrote no tables without --out (OUT_ROOT now DATA_ROOT, tests);
  stale docstrings, help text and contract lines (train_xfg FULL_RUN_CMD paths, rebounds model text, epv L2 and
  keep-rule wording, decisions mean, ghosting, load and lineups held-out and drop wording); the rebounds report
  was regenerated so its model text names the temperature. Runbook: the stage 8 xfg baseline command passes the
  held-out shot file. Stage 8 dry run on the fixtures (the group 1 lane list without --skip-heldout, fixture
  output only): all 35 steps ok and each lane's held-out section written; epv with --deep torch through
  held-out scoring on the fixtures also ok. Known and left as is: winprob's hgb random_state is fixed at 0
  rather than --seed (the same at seed 0); xfg sets aside 18 held-out games and the tracking lanes 17
  (01.23.2016.CHI.at.CLE has shots but broken tracking); run_all passes no --bootstrap, so stage 8 uses each
  trainer's default of 200 as the stage 5 reports did.
- **Reports carried local paths (fixed 2026-11-12):** every committed lane report recorded the absolute DATA_ROOT,
  whose Google Drive folder name contains the Drive account address, and the repo's absolute path; the repo is
  public. analytics/report_paths.py now writes them as $DATA_ROOT, repo-relative and ~ (test
  tests/test_report_paths.py), run_all applies it to each lane report after every step, and the 12 committed
  reports are redacted. Not fixed: earlier commits still hold those paths, and the same account address
  appears in older committed files (ROADMAP.md, ROADMAP_LOG.md, the Colab scripts and notebooks); removing
  it from history needs a history rewrite, which is the owner's call.
- **Stage 6 ghosting (done 2026-11-12), deviation from the runbook wording:** the torch GRU (hidden 32) ran in 55 min,
  23 GB peak, and passes the lane's keep rule against ridge (1.0 s error 3.769 ft, minus ridge -0.773 [-0.800,
  -0.749]). But paired by game against the MLP stand-in it was to replace, it is worse: GRU minus stand-in 0.030 ft
  [0.029, 0.032] at 0.2 s and 0.111 ft [0.085, 0.138] at 1.0 s (604 games). Stage 6 says to replace the stand-in;
  the loop keeps the stand-in instead, by the same paired comparison on training folds that adopted epv's GRU, so
  the frozen ghosting model is the better of the two. Owner: say so before go-stage8 if you want the GRU frozen
  instead. Report reports/ghosting_torch.md; tables DATA_ROOT/stage6/ghosting_torch; run_all passes no --deep
  to ghosting. embeddings: gate shut, unchanged. Stage 6 is done.
- **Stage 6 epv (done 2026-11-12):** the torch GRU ran in 60 min, 21 GB peak. Same gate and keep rule: kept over hgb,
  0.0212 [0.0201, 0.0228]; MSE 1.2155 [1.2080, 1.2226] vs constant 1.2822 (lift 5.2 percent), calibration slope
  0.964. Paired by game against the MLP stand-in it replaces, MSE(stand-in) minus MSE(GRU) 0.0062 [0.0054, 0.0071]
  (604 games, 1,739,119 states). The GRU is the epv model: its tables moved to DATA_ROOT/analytics/epv (the MLP
  tables kept at DATA_ROOT/stage6/epv_mlp), reports/epv.md is the GRU report (the stand-in's is
  reports/epv_mlp_standin.md), run_all passes --deep torch to epv train on real data (DATA_TRAIN_ARGS, test),
  and decisions reruns on the GRU EPV.
- **Stage 6 plan (2026-11-12):** epv's gate opened, so `analytics.epv.train --deep torch --skip-heldout` runs
  with the contract's GRU (already in analytics/epv/deep.py, tests in tests/test_deep_models.py). Its report
  goes to reports/epv_torch.md and its tables to DATA_ROOT/stage6/epv_torch, so the MLP-run tables decisions
  read stay as they are until the keep rule decides. A synthetic benchmark put one epoch over the 1.4M
  training rows at about 3.4 s on one thread. embeddings' gate stayed shut (PCA 5-NN 0.768, at or above
  0.60): no change. ghosting's gate opened too (MLP kept over ridge): `analytics.ghosting.train --deep torch --skip-heldout` is
  queued after the epv GRU (report reports/ghosting_torch.md, tables DATA_ROOT/stage6/ghosting_torch, log
  logs/loop/stage6_ghosting_torch.log); a fixture run of it passed (0.45 GB).
- **Stage 5 phase 2 FAILED (2026-11-11 16:56):** `run_all.py --phase 2 --skip-heldout` was killed after about
  30 s with exit 137 (SIGKILL, out of memory on this 16 GB Mac) inside matchups validate, before any step log was
  written (logs/loop/stage5_phase2.log: "note: matchups depends on xfg, not selected; their outputs must already
  exist", then "exit=137"). Cause: analytics/matchups/validate.py load_games reads the frames of every game (621
  games, about 480 million frame rows) into one list before any work. gravity (validate.load_data_root) and
  ghosting (validate.iter_data_root) build the same all-games list. Audit of the other lanes: xfg, rebounds,
  load, embeddings (frames as a per-game callable) and epv (per-game paths, then 1 Hz states) stream already;
  coverage loads only the labelled games. actions (validate.py:410 loads every game into a list), decisions
  (validate appends every game's frames) and playcalling (load_data_root concatenates every frames file into one
  table; its glob sportvu_contract/**/*.parquet also picks up the _pbp files and the heldout folder) have the
  same defect and are fixed next, before phases 3 and 4. The lanes were only ever run on fixture seasons, so
  the tests never saw it.
- **Streaming fix merged (2026-11-11):** matchups (ea986ba), gravity (9ad28f4) and ghosting (df9d732) now read
  frames one game at a time, each with identical fixture results and a test that at most one game's frames is
  alive at a time. Also fixed: matchups train read the xFG predictions only from --out, which run_all never
  passes, so it would have used the location-only fallback and written nothing; OUT_ROOT now defaults to
  DATA_ROOT (ceea4c5, test). Still to fix: decisions (agent working), and for group 2 later, actions and
  playcalling. Ghosting train fit rule merged (6d397b0): every candidate fits on every 5th sampled step (1 Hz) of
  all training games, float32; evaluation unchanged at 5 Hz, rebuilt and scored one test game at a time (test:
  streamed evaluation equals the season-wide one). On fixtures the MLP is now dropped for ridge (only the fit
  subsample moved). 10 real games: train 46 to 65 s, 1.6 GiB; projected full season about 8 GiB and 40 to 60
  min (worst case about 3 h). Queued to run after epv and decisions (log logs/loop/stage5_ghosting.log). The
  --deep torch GRU path still needs about 18 GB at full season (stage 6 question).
- **Mac crash 2026-11-11 about 16:21:** it killed the first phase 1 run (PID 48469, started 16:17). By then
  xfg validate, features and train had finished ("ok" in logs/run_all/xfg_*.log; train wrote at 16:21:08) and
  rebounds validate had finished (16:21). Checked after the reboot: no process of the old run alive;
  `$DATA_ROOT/analytics/xfg/2015-16/xfg_predictions.parquet` reads whole (101,853 rows, 607 games, matching the
  train log; no nulls in the prediction columns); reports/xfg.md is complete (its last line matches the train
  log). xfg features writes no file without --out, so nothing else of xfg sits on Drive. So xfg was not rerun
  for the crash; only the missing steps were.
- **xfg defect found in the phase 1 review (fixed 2026-11-11):** player_id was null on every xfg row, because
  the basketball-reference shots table has no player ids and features.py kept the shots table's player_id
  instead of the resolved shooter (shot_frames.shooter_id, filled for about 94 percent of shots). The
  (player_id, zone) shooter prior of the selected model was therefore a zone prior for every shooter.
  docs/contract/xfg.md already defines player_id as the shooter; the code now follows it (test in
  tests/test_xfg_upgrade.py). The 16:17 xfg run (reports/xfg.md, xfg_predictions.parquet) is superseded: xfg
  validate, features and train rerun with --skip-heldout after the resume job, before phase 2 reads xfg.
  Done 16:55: player_id now filled on 99.1 percent of xfg_predictions rows (444 players). The superseded
  run's report and log were overwritten by the rerun, so the size of the change is not recorded.
- **Play-by-play gaps at the source:** eight 2021-10-27 games (home teams BOS, BRK, LAC, MIL, NOP, OKC, POR, TOR)
  fail with "no play-by-play events on the page", also on the --retry-failed rerun. The fetched page for
  202110270BOS has the pbp table with a caption and no rows, so basketball-reference has no events for them; the
  same date's ORL and PHO games parsed normally. Their box scores collected normally.

- **Open gates (not blocking yet):** odds for betting (put ODDS_API_KEY in ~/.zshrc and create gates/odds-ready,
  or create gates/odds-skip) and NBA.com tables (gates/nbacom-done or gates/nbacom-skip). Neither blocks anything
  before stage 5 phase 5 (betting) and the embeddings aggregates; the loop carries on meanwhile.
- **Branch:** the Mac checkout is on track-fix.

### Stage 3: Phase 0 on every tracked game (done 2026-11-07)

Commands as in the runbook, with `--bref-dir "$DATA_ROOT/collect"` (corrected, see below). Logs:
logs/loop/phase0.run1.log, phase0.log, phase0_rosters.log. Summary on Drive: DATA_ROOT/reports/phase0_summary.md.
Its totals include the 17 held-out games; the figures below are recomputed from phase0_manifest.json over the 604
training games only (rule 1).

| metric | value |
| --- | --- |
| games processed / failed / skipped | 621 / 0 / 5 (the 5 are source_broken in Step 1) |
| training games | 604 |
| shots | 101803 |
| share of shots with a release frame | 0.6220 |
| shooter identity agreement (shots with both ids) | 0.9538 |
| high-confidence shots | 34528 (0.3392 of shots) |
| misses / share with a rebound frame | 59762 / 0.8331 |
| rebounder identity agreement | 0.6902 |
| points check: possession points equal the final score | 537 of 604 |
| rosters with basketball-reference names (all 621 games) | 621 games; 12599 players with bref_slug; shot chart names matched 0.9950 |

- Points check: all 67 training mismatches undercount (possession points below the final score): 36 by 2, 11 by
  1, 8 by 3, 5 by 4, the rest 5 to 10. Likely points inside the 138 possessions dropped as shorter than a frame or
  across clock gaps; worth a look before the possession-based lanes in stage 5.
- Rebounder identity agreement (0.69) is low next to the shooter's (0.95); the rebound localiser's identity is
  weaker evidence than its frame.
- Fixed on the way (tests in each commit): transient Drive read errors retried (f10db36); a nameless player_id 0
  crashed name matching (f10db36); the runbook's `--bref-dir data/collect/bref_shots` held no pages, so rosters had
  no bref_slug until rebuilt with `--rebuild rosters --bref-dir "$DATA_ROOT/collect"` (06cfc7a).
- reports/sportvu_gaps.md: 626 games, 369 with a tracking gap over 10 s.

### Stage 3: audit sample (drawn 2026-11-09)

Pass mark approved (gates/audit-mark-approved, 2026-11-09 21:12) with the sample line unchanged, and PR #7 merged
first. `python3 -m analytics.audit_shots --data "$DATA_ROOT" --n 200 --seed 0` wrote
DATA_ROOT/analytics/audit/shots_sample.parquet (log: logs/loop/audit_sample.log).

| check | value |
| --- | --- |
| shots / games | 604 / 604 training games, one shot each |
| confidence_flag high / low / unmatched | 334 / 222 / 48 |
| held-out SportVU games in the sample (splits.json heldout_game, heldout_arena, heldout_era) | 0 |
| games at CLE | 0 (21 CLE road games, which are training games) |
| audit tests (tests/test_label_tool_audit.py and the audit_shots tests) | 51 passed; 53 after the 10-frame amendment, make test-fast 648 passed |
| audit tool smoke run on the real sample | page and shots 1, 301 and 604 served; sample file unchanged |

### Stage 5 group 1, phases 2 and 3 (2026-11-11, training folds only, held-out not scored)

| lane | time, peak memory footprint | selected | out-of-fold metric [95 percent game bootstrap] | notes |
|---|---|---|---|---|
| embeddings | 5 min, 2.1 GB | pca | 5-NN position accuracy 0.768 [0.720, 0.809] (majority 0.413) | autoencoder gate not reached; 399 of 448 player-seasons embedded; NBA.com block absent |
| matchups | 6 min, 7.2 GB | ridge | possession points MSE 1.2854 [1.2788, 1.2916] | ridge beats raw by 0.039 but is WORSE than the constant (1.2842; ridge minus constant 0.0012 [0.0002, 0.0020]): defender identity adds no out-of-fold predictive value; within-4-ft defended FG% 0.427 vs 0.454 beyond; assignment stable 82 percent of sampled frames; 368 of 447 defenders gated |
| gravity | 7 min, 14.5 GB | ridge (shipped by the plan; hgb diagnostic only) | MSE 28.12, R2 0.416 (hgb 22.71, R2 0.528) | top gravity 201939, 202691, 2594 (Curry, Thompson, Korver, by id) look right; the known-player check is fixture-only, so no claim; sign test p = 0.020 over 290 games |
| epv | 41 min (train 39), 19 GB | mlp (sequence stand-in) | MSE expected points 1.2218 [1.2137, 1.2294] vs constant 1.2822 (lift 4.7 percent) | beats hgb under paired bootstrap, so the stage 6 sequence-model gate is open; shuffled-target lift 0; 0-4 s bucket weakly calibrated (slope 0.68) |

| decisions | 6 min, 2.6 GB | xfg_selected prior (rule fixed in advance: xFG coverage 0.99 at least 0.50) | validation: mean decision quality per player-game vs on-court offensive rating, mean Spearman 0.009 [-0.009, 0.027] over 604 games, sign p 0.051 (location prior 0.003) | the decision-quality score does NOT track points on the floor: no validated signal; EPV hold option on 518,104 decisions (epv rerun 2026-11-12 wrote its tables; numbers identical to the first run). Rerun on the stage 6 GRU EPV (6 min, 2.3 GB): mean Spearman 0.012 [-0.008, 0.029], sign 317 of 603, p 0.222; still no validated signal |
| ghosting | 72 min (train 64), 9.2 GB | 3 MLP 64x64 on the 5-step window (GRU stand-in), fitted at 1 Hz on 7,147,100 float32 rows, scored at 5 Hz | 1.0 s error 3.657 ft [3.635, 3.682] vs ridge 4.541 and persistence 5.408; MLP minus ridge -0.884 [-0.909, -0.858] | deep candidate kept, so the stage 6 gate is open (torch GRU queued); graded deviation vs points allowed: 346 of 604 games positive, sign p 0.0002 |
| lineups | 2.3 min, 3.1 GB | rapm (mean out-of-season Spearman 0.109 vs plus_minus 0.105, rapm_synergy 0.087) | per-season Spearman intervals all span zero (for example rapm 0.222 [-0.170, 0.547] on 2022-23); only 18 to 27 gated lineups recur season to season | no candidate is distinguishable from raw plus-minus: the selection is a tie on the point estimate; 180,243 stints after validate (9,541 not 5v5 dropped), 53 shifted-period games dropped; the abbreviated-name player key issue is still open |
| winprob | 9 min, 5.4 GB | logistic (margin, time, possession, margin x sqrt time) | mean bucket Brier 0.1483 vs constant 0.2472 (hgb_iso 0.1486, level); log loss 0.4884 [0.4804, 0.4989] | under-confident in the last 4 minutes (slope 1.69 [1.60, 1.78], ECE 0.050) and over-confident at 24-12 (0.87): an improvement-search target; 53 shifted-period games dropped; spread missing on every game (odds skipped); benchmark empty (NBA.com skipped, stated in the report). Fixed (tests): it wrote no tables without --out (OUT_ROOT now defaults to DATA_ROOT) and its benchmark text said "unreachable"; train rerun queued after the ghosting GRU (logs/loop/stage5_winprob_rerun.log); improvement search variant 1 accepted and merged (margin_over_sqrt_time; paired selection gain 0.00149 [0.00132, 0.00169]): lane rerun 9 min, 6.0 GB, logistic selection 0.1468, bucket slopes 0.97 to 1.06 (last 4 minutes 1.065), log loss 0.4845 |
| load | 8 min, 1.7 GB | ridge (decline), logistic (absence) | decline MSE 0.4929 [0.4740, 0.5178] vs mean 0.5086; absence log loss 0.2764 [0.2644, 0.2883] vs base 0.3244, slope 0.99 | rest days show no clear effect on decline (back-to-back -0.48, one day -0.44, two plus -0.46 ft/s); the player ranking gate counts stints, not games, so two 2-game players lead it; no tracking aggregates (NBA.com skipped) |

lineups fixed (536a3c4, tests): run_all now runs stints before validate, and every step reads the stints the
stints step built; running it on real data also found shooting, personal and take fouls filed under the fouled
team in the basketball-reference play-by-play (the fouler landed in the opponent's lineup: 53 percent of stints
were 5v5) and starter names not matching the play-by-play (suffixes like Jr., and UTF-8 read as Latin-1); after
both fixes 95.5 percent of 2020-21 stints are 5v5. Open, not fixed: players are keyed by a hash of the
abbreviated play-by-play name, so players sharing an abbreviation (several "J. Green") share one key and
corrupt their RAPM; fixing it needs a player id or slug join the play-by-play lacks. Scratch real-data run: all
steps ok in 200 s, 2.6 GB. Full run queued after ghosting. winprob fixed (b2e92a9, tests): all 4,388 bad clocks
come from one collector bug (fetch_pbp.parse_events numbers periods by counting quarter header rows, and a page
that repeats a header shifts every later period up, so real 3rd and 4th quarters land in periods 5 and 6); 53 of
6,416 games carry it (a gap in their period numbers), 9 to 12 per season. The clock rule drops those games
whole and counts them in validate.json; validate fails only above 2 percent of a season (real data: at most 0.9
percent). Lineups drops the same games before building stints (shifted_period_games, test). The 2015-16
tracking-era play-by-play has no such game (checked: 0 of 604 training files with a period gap). Collector not
fixed and nothing re-collected: parse_events should read the period from the header row; the 53 games could
then be re-parsed from the cached pages without the network. Scratch real-data winprob run: ok in 787 s, 5.3 GB.
Was: winprob (4,388 play-by-play events
with a clock outside the period or no period). Each is on its own worktree branch with a test. winprob's trainer
sets the whole 2024-25 season aside (--heldout-season 2024-25); validate's heldout_rows 0 only counts row flags.
Also fixed: epv train wrote no tables on real data (out_root None without --out), so decisions would have run
without EPV (d71008b, test); epv train reruns now, then decisions (streaming fix merged, 323e979).

Memory: epv and gravity peaked above the 16 GB of RAM (memory compression) but finished; group 1 lanes run
one at a time. Next: decisions (after its streaming fix), lineups, winprob, load; ghosting after its fit rule.

### Stage 5 phase 1 (done 2026-11-11, training folds only, held-out not scored)

Steps: xfg validate, features, train (rerun 16:50 to 16:55 after the shooter fix), rebounds validate (16:21),
rebounds features and train (16:47 to 16:50, after the crash), xfg baseline (16:50). All ok. reports/run_all.md
(gitignored) covers the xfg rerun only; the rebounds steps are in logs/run_all/rebounds_*.log.

| lane | rows | selected | out-of-fold metric [95 percent game bootstrap] | notes |
|---|---|---|---|---|
| xfg | 101,853 attempts, 607 games (62,387 with contest features); 18 held-out games set aside | hgb_contest_prior | log loss 0.6294 [0.6277, 0.6316]; ECE 0.0029 (target under 0.02); slope 1.00 | beats location_logistic by 0.0370 [0.0354, 0.0385] and contest_logistic; shots: 92 out of range and 83 duplicates dropped; rosters: 2 null player names |
| rebounds | 49,288 misses, 604 games; 17 held-out games set aside | hgb | per-miss log loss 1.6234 [1.6130, 1.6304]; top-1 accuracy 0.4797 [0.4756, 0.4848] (plan target 0.40) | beats prior by 0.5146; about 18 percent of misses per game dropped (no clean rebound match); the top candidate's probability is under-confident (slope 1.69, ECE 0.109), a candidate for the improvement search; no --out, so rebound_predictions not written; improvement search variant 1 accepted and merged (hgb temperature fitted on inner folds, tau about 1.26: per-miss log loss 1.6009, paired -0.0225 [-0.0246, -0.0205], top-1 unchanged, ECE 0.109 to 0.041, slope 1.69 to 1.37); lane rerun reproduces it (1.6009 [1.5882, 1.6103], ECE 0.041, slope 1.37); predictions table now written (OUT_ROOT defaults to DATA_ROOT, test); memory fixed: features and train held every game's cleaned frames at once (43 GB peak footprint on the rerun; phase 1 was running when the Mac crashed on 2026-11-11), now one game at a time (2.0 GB, report identical; test) |
| xfg baseline | 102,024 attempts, 607 games | logistic (location) | log loss 0.6665 [0.6652, 0.6681]; slope 1.00; ECE 0.040 | constant model 0.6879; held-out shots live in a separate file the baseline is not given |

Looks wrong or worth a look: nothing blocking. The rebounds calibration and the 18 percent rebound drop are
noted for the improvement search; phase 2 (matchups, gravity, ghosting, embeddings) runs next.

### Stage 3 check (passed 2026-11-11)

`python3 -m analytics.audit_shots --summarise "$DATA_ROOT/analytics/audit/shots_sample.parquet" --report
reports/shot_audit.md`, run after gates/audit-done appeared at 15:14. All 604 rows have a verdict (555 with an
audited release frame, 49 marked bad).

| check | result | pass mark | verdict |
|---|---|---|---|
| pass_mark.share (high rows within 10 frames, bad = miss) | 0.9671 (323 of 334, 5 bad) | at least 0.90 | pass |
| high rows within 5 frames (information only) | 0.9611 (321 of 334) | not graded | |
| high rows within 2 frames (information only) | 0.9581 (320 of 334) | not graded | |
| low rows within 10 frames, audited only (reported) | 0.8895 (169 of 190, 32 bad) | not graded | |
| unmatched rows within 10 frames, audited only (reported) | 0.3333 (12 of 36, 12 bad) | not graded | |
| all audited rows within 10 frames | 0.9081 (504 of 555), mean abs diff 5.65 frames | not graded | |

The four localisation constants (apex search 2 s, rim radius 6 ft, flight 25 frames, high-confidence 0.6) are
recorded in docs/DATA_CONTRACT.md section 17 as validated for "shot event and shooter correct, release within
about 0.4 s", not for 2-frame release precision. The 2-frame figure above is the auditor's agreement with the
localiser on frames the animation shows only to about 10 frames, so it does not settle the open item below.
Readings for later stages: the high flag is the one to trust; low shots are usable with care; unmatched shots
(a third within 10 frames, a third bad) should stay out of anything that reads the release frame.

### Stage 3: audit pass mark (approved 2026-11-09, amended 2026-11-09 before scoring)

- **Pass mark:** among audited high-confidence shots (confidence_flag high), pass_mark.share, bad rows counted as
  misses: the detected release frame is within 10 frames (about 0.4 s) of the audited release frame on at least
  90 percent. This is `pass_mark.share` of `audit_shots --summarise` (analytics/audit_shots.py TOLERANCE_FRAMES = 10).
  The low and unmatched rows are reported, not graded.
- **What a pass validates:** "shot event and shooter correct, release within about 0.4 s". It does not validate
  2-frame release precision. DATA_CONTRACT.md section 17 says the same in its validation status.
- **Information only, not pass/fail:** the same share at 2 and 5 frames (0.08 s and 0.2 s), printed as
  `information only` lines and a "Tighter tolerances" table in the report.
- **Why amended:** the owner's audit labels confirm a release from the court animation only to about +-10 frames,
  so a 2-frame grade would measure the labels, not the localiser. Amended before any verdict was scored (1 of 604
  shots audited at the time). The x key now also covers a wrong shot or a wrong shooter under the yellow ring
  (docs/labeling_guide.md section 7), which is what makes the "shooter correct" part of the claim checkable.
- **If it fails:** the four localisation constants (apex search 2 s, rim radius 6 ft, flight 25 frames,
  high-confidence 0.6) are not recorded as validated; the loop writes what failed here and stops stage 3.

### Open item: 2-frame release precision (not a gate)

The audit cannot grade the release to 2 frames. Two independent checks, either of which can run later on
training games only; neither blocks a stage:

1. **Ball flight consistency, from the tracking alone.** For each high-confidence shot, fit a parabola to the
   ball's height (and a line to its horizontal position) over the free-flight frames between a few frames after
   release_frame and the apex, extrapolate back to the frame where the fitted ball is at the shooter's release
   height (or leaves the 2.5 ft radius around the shooter's tracked position), and compare that frame with
   release_frame. The fit uses the flight, not the hold-and-rise threshold that picks release_frame, so it is a
   separate estimate. Report the share within 2 and 5 frames and the bias. Cheap: no labels, every shot.
2. **Broadcast video of the SportVU-synced training games.** The four training games with a synced broadcast
   (gsw_bkn_2015, gsw_cha_2016, gsw_ind_2016, gsw_sac_2015 in sportvu/splits.json) let a labeller mark the release
   on the video frame, where the hand and ball are visible, then map it to a SportVU frame through the existing
   sync. First measure the sync's own error; it has to be well under 2 frames for this to grade 2-frame precision.
   About 50 shots would do. The held-out games (CLE and gsw_phx_2016) stay out.


- **Sample:** `python3 -m analytics.audit_shots --data "$DATA_ROOT" --n 200 --seed 0`, training games only, shots
  with a release frame. Note: the sampler takes n // games shots per game with at least 1, so with 604 training
  games `--n 200` draws one shot per game, **604 shots**, about three times the runbook's 3 hours (roughly half
  will be high confidence). To audit about 200, change this line to "200 games, one shot each" before creating the
  gate file and the loop adds a game cap to the sampler (with a test) before drawing.

### Stage 2 check (passed 2026-11-07)

Units per collector and season, read from each manifest under DATA_ROOT/collect. No collector had an empty unit;
the only failures are the eight play-by-play games above, rerun once with --retry-failed.

| season | schedule pages | positions | play-by-play ok / failed | box scores ok | schedule games |
| --- | --- | --- | --- | --- | --- |
| 2015-16 | 9 | 1 | | | 1316 |
| 2016-17 | | 1 | | | |
| 2020-21 | 8 | | 1165 / 0 | 1165 | 1171 |
| 2021-22 | 9 | | 1309 / 8 | 1317 | 1323 |
| 2022-23 | 9 | | 1314 / 0 | 1314 | 1320 |
| 2023-24 | 9 | | 1313 / 0 | 1313 | 1319 |
| 2024-25 | 9 | | 1315 / 0 | 1315 | 1321 |

- Play-by-play and box scores cover six fewer games a season than the schedule: the six play-in games, which those
  collectors do not request (default season types).
- Files present: collect/stints/playbyplay/<season>.parquet for 2020-21 to 2024-25 (526883, 593321, 599722,
  592078 and 601833 events); collect/stints/starters/<season>.parquet for the same seasons (10 starters a game:
  11650, 13170, 13140, 13130, 13150 rows); collect/positions.parquet (1070 rows, 2015-16 528 and 2016-17 542).
- Box scores also wrote the full player tables (players/<season>: 31394, 33649, 33834, 34677, 34940 rows).
- Logs: logs/loop/bref_schedule.log, positions.log, bref_seasons.run1.log, bref_seasons.log,
  bref_seasons.retry.log, bref_boxscores.log (gitignored, on the Mac).
- Optional, not collected: odds (paid key) and NBA.com tables. Both are open gates (above).

### Stage 0 (passed 2026-11-07)

DATA_ROOT is exported in ~/.zshrc and points at the Drive nba_harvest folder, which holds
sportvu_contract/manifest.json and collect/. The first run failed three ways; each was fixed with a test (PR #5):

| check | first run | after PR #5 |
| --- | --- | --- |
| make test-fast | could not run: no pytest, then no bs4, then test_makefile_targets_match_registry failed | 612 passed, 1 skipped |
| make all-fixtures | 46 of 47 steps OK; betting train refused --data without --out | 47 of 47 OK |

- requirements.txt now lists pytest, beautifulsoup4, pandas, scipy, threadpoolctl and requests. Running the
  stage 0 install command put pytest 9.2.0 and beautifulsoup4 4.15.0 into /usr/local/bin/python3. Use that
  python3, not /opt/anaconda3: anaconda's site-packages holds a stray `tests` package that shadows the repo's.
- The Makefile no longer takes the exported DATA_ROOT as its fixture root. Before the fix, `make fixtures` would
  have written synthetic data into the real Drive folder.

Goal: the code for every remaining part of the extension plan, on top of Phase 0 and the xFG location
baseline. Integration branch `analytics-phase0`, based on `track-fix`. Twenty-one lanes ran in parallel in
one shared working copy (worktrees did not fit the disk allowance): fifteen extensions and six
infrastructure lanes. Every lane was reviewed by a fresh cold reviewer, fixed, and re-checked; three
repo-wide cold reviews (leakage and held-out rules, interface consistency, code quality and tests) ran at
the end. The last round of fixes ran on Opus after the Fable usage limit was reached. Nothing here is a result: every number in every `reports/<ext>.md` is fixture output from the
synthetic generator, labelled as such in the file.

## Rules held

- Held-out games in `sportvu/splits.json` never fit or tune anything. Every trainer calls
  `analytics.eval.refuse_heldout` on its training game ids and `analytics.schema.assert_time_forward` on its
  feature frame; `tests/test_splits.py` lists every `analytics/*/train.py` and fails on any analytics file
  that fits without the guard. Folds follow one convention: >= 0 out-of-fold, -1 held-out, -2 in-sample. Held-out rows are set aside before any fit and scored once by the selected
  candidate only.
- No raw SportVU, play-by-play, NBA.com or odds data is committed. Fixture pages under
  `tests/fixtures/bref/pages/` are small hand-built excerpts. Nothing scrapes: ingestion tests mock every
  fetch and an autouse guard raises on any unpatched network call.
- Baselines first. A deep model stands in only where the plan names one (ghosting sequence model, EPV
  sequence model, embeddings autoencoder), as an sklearn MLP because torch is absent here, gated by a
  baseline comparison fixed in advance. On fixtures every gate kept the baseline.
- Fixture numbers are never presented as results. Reports say so on every numeric section.

## Assumptions taken without asking

- Shared working copy instead of worktrees; lanes owned disjoint files and ran no git; the lead committed
  by path after each lane's tests passed, with WIP snapshot commits in between (the stop hook required a
  clean tree).
- The twenty subagent cap meant the integration tester started after the first lane finished.
- Cross-lane path decisions, taken when producers and consumers disagreed: EPV writes
  `OUT_ROOT/analytics/epv/<season>/<game>.parquet` (states) and `epv_possessions.parquet`; actions writes
  `OUT_ROOT/analytics/actions/<season>/actions_detected/<game>.parquet`; coverage and playcalling read those;
  held-out rows of every producer sit under `heldout/<season>/`; hand labels are read only through
  `analytics.labels.read_labels` with `labeller_pass == 1` and `label_source == "hand"`; the starters table
  from basketball-reference box scores carries `side` and `player_slug` with null NBA ids, and the lineups
  builder keys on those.
- Gravity ships the ridge expectation and a zone-only shooting prior; the boosting candidate is a
  diagnostic (its reviewer showed a player-identifying prior lets boosting absorb the quantity measured).
- Betting and win probability share one closing-line rule (`analytics.betting.validate.closing_lines`
  then `join_games`): per-book latest pre-tip `is_close` snapshot, median over books, an 8 hour shift and a
  one-day tolerance on the date join.
- `analytics/possessions.py` now turns the season over in July, like the other Phase 0 modules (it said
  September).
- The HistGradientBoosting fits are pinned to one thread (threadpoolctl) because OpenMP spin-waits on the
  shared four-core box made fits hang; `--threads` raises it where offered.

## Lanes

| lane | files | tests | review | needs real data |
| --- | --- | --- | --- | --- |
| I1 fixtures | analytics/fixtures.py, docs/contract/fixtures.md | 24 | PASS, mediums fixed (league-stable ids, aggregates and stints share rosters, localiser confidence) | nothing |
| I2 ingestion | collect/bref_schedule.py, bref_seasons.py, bref_boxscores.py, bref_lineups.py, positions.py, drive.py; docs/contract/ingestion.md | 12 (plus 28 existing collect tests) | FAIL then fixed (starters table contract, status-code error classes, 17/min cap, tableless pages fail not empty, network guard) | basketball-reference pages: schedule 9 pages a season; shot charts, play-by-play and box scores 1 page a game (about 72 min a season each at 17 a minute); lineups 30 a season; positions 1 a season |
| I3 Phase 0 driver | analytics/run_phase0.py, audit_shots.py; docs/contract/phase0_driver.md; reports/phase0_driver.md | 19 | PASS, mediums fixed (subset-run resume, module paths, refusal test, --bref-dir fills bref_slug) | the Step 1 manifest and frames on Drive |
| I4 labeling tool | analytics/labels.py, label_tool.py, label_tool.html; docs/contract/labels.md; docs/labeling_guide.md | 29 | FAIL then PASS on re-review (tombstones, pass filter, visited record, held-out guard, server hardening) | about 27 hours of hand labelling on the Mac |
| I5 run_all | run_all.py, Makefile, analytics/registry.py, docs/run_order.md, pytest.ini | 18 | FAIL then fixed (--out and --seed to every step, quoted paths, Phase 0 layout, coverage edge, log capture) | nothing |
| I6 integration | analytics/interfaces.py, tests/test_integration.py, docs/contract/interfaces.md, reports/integration.md | 111 (fast part; slow part runs every lane on fixtures) | its loop found: rebounds features CLI, embeddings --folds, EPV output path, betting NaN RAPM join; all fixed | nothing |
| xfg | analytics/xfg/ | 13 (plus 7 baseline) | PASS, mediums fixed (selected candidate only on held-out, clock join tested and one-to-many, no event_id branch) | shot charts, shot frames for every tracked game, frames |
| rebounds | analytics/rebounds/ | 12 | PASS, mediums fixed (symmetric miss selection, frame-window tests) | rebound frames, shot frames, frames, rosters |
| matchups | analytics/matchups/ | 15 | PASS, mediums fixed (basket vote per team and period, one-game-per-defender notice, join tests, heldout column from xfg) | frames, possessions, rosters, shot frames, xfg output |
| gravity | analytics/gravity/ | 11 | FAIL then fixed (ridge shipped, zone-only prior, planted-pull recovery test, fallback section in the report) | frames, possessions, shot frames, rosters, shot charts |
| ghosting | analytics/ghosting/ | 9 | PASS, mediums fixed (output under analytics/ghosting, frame-stride cut) | frames, possessions; torch for the GRU that replaces the MLP stand-in |
| embeddings | analytics/embeddings/ | 14 | PASS, mediums fixed (positions file joined through rosters and collapsed to G/F/C, mirrored and gap tests) | frames, possessions, rosters, shot charts, NBA.com aggregates (blocked), positions table |
| epv | analytics/epv/ (states, validate, features, train, eda) | 19 | PASS, mediums fixed (MLP paired against the base candidate, in-sample MSE shown) | frames, play-by-play, possessions, rosters |
| actions | analytics/actions/ | 16 | FAIL then fixed (training restricted to visited possessions, one-to-one label matching, detections cast through analytics.labels) | hand labels under DATA_ROOT/labels, frames, possessions |
| coverage | analytics/coverage/ | 13 | PASS, mediums fixed (selected model only on held-out, by-construction note, unsure rows set aside, strict label reader, per-season team rates) | hand screen labels, frames, rosters, possessions, actions output |
| decisions | analytics/decisions/ | 19 | PASS, mediums fixed (basket per team and period, partial xfg coverage, held-out xfg path, mean quality statistic, drive in the option set) | possessions, shot frames, frames, epv and xfg outputs |
| playcalling | analytics/playcalling/ | 16 | FAIL then fixed (read_labels, action ids never overwritten, score-margin control, recovery test on a planted association) | possessions, labels, actions, coverage and epv outputs |
| lineups | analytics/lineups/ (stints, validate, features, train, eda) | 24 | FAIL then fixed (team ids never players, starters join through the games table, team_id from the output season, player_rapm_prior) | multi-season play-by-play, starters and games tables, bref lineups |
| winprob | analytics/winprob/ | 15 | FAIL then fixed (spread join delegated to betting, finals dropped before features, overtime tests, held-out season out of EDA) | multi-season play-by-play, games, odds |
| load | analytics/load/ | 12 | PASS, mediums fixed (smoothing never bridges a dropout, hand-built history test) | frames, manifest, rosters; aggregates optional |
| betting | analytics/betting/ | 19 | PASS, mediums fixed (per-book closing snapshot, time-forward test, team-name inference with votes, partial RAPM join imputed) | odds history (paid feed), games table, lineups output |

Every lane also has `notebooks/eda_<ext>.ipynb` (hand-written nbformat 4, tables only, no stored outputs),
`docs/contract/<ext>.md` and `reports/<ext>.md`. Each `validate`, `features` and `train` module exposes
`main(argv)` with `--fixtures | --data DATA_ROOT [--out OUT_ROOT] [--report ...] [--folds] [--seed]
[--bootstrap]`; `run_all.py` and the Makefile drive them in dependency order.

## Where it stands

`python3 -m pytest -q tests -m "not slow"` passes: 609 passed, 2 skipped (one needs the SportVU archives,
one a local tracking data file), in about 3 minutes. `python3 -m pytest -q tests -m slow` runs every lane's
validate, features and train on fixtures, then on a shared fixture DATA_ROOT, then through `run_all.py`,
checks the five interface tables, and scans every lane output for the held-out game outside `heldout/`.
On the final tree it passes: 63 passed in about 6 minutes. The per-lane detail is in
`reports/integration.md` (fixture output, not a result).

Open items, none blocking the Mac run: the odds feed and the NBA.com tables remain uncollected
(stats.nba.com drops non-browser clients, odds need an API key); the GRU, the EPV sequence model and the
embeddings autoencoder stand in as sklearn MLPs by default, with the torch models behind `--deep torch` on each
lane trainer for stage 6 to pass only where a gate opened (docs/contract/ghosting.md, epv.md, embeddings.md); `analytics.fixtures` keeps its
star on the floor all game, so the gravity on/off sign test runs on the lane's planted-pull generator; the
four shot-localisation constants (apex search 2 s, rim radius 6 ft, flight 25 frames, high-confidence 0.6)
are still to pin in the contract; the lineups lane keeps its own synthetic stints generator because the
shared one does not carry enough lineup signal for its recovery test.

## Final reviews

Three repo-wide cold reviews ran after every lane passed its own review. All three returned PASS with no
blocker; every medium they raised was fixed and the fixes were re-tested.

- **Leakage and held-out rules.** No fit, imputer, calibration, alpha choice, selection, gate or EDA fit
  touches a held-out game in any lane; every deep model is gated by a baseline comparison fixed in advance.
  Fixed: seven lanes printed held-out scores for non-selected candidates (now the selected candidate only,
  plus a labelled constant floor); in-sample rows wrote fold 0 (now -2, and consumers refuse anything but
  fold >= 0 on training rows); the xFG baseline's smoke run had scored the real held-out game (report
  regenerated without it, the smoke mode no longer scores held-out rows); the held-out arena rule is now
  part of `sportvu.splits.heldout_tokens`, so a lane that forgets the `.at.CLE` suffix still trips the guard;
  later seasons than the held-out one are dropped in win probability and betting; the integration test now
  scans every lane output for the real held-out id.
- **Interface consistency.** A full `run_all.py --fixtures --all` run finished 47 of 47 steps and all five
  interface files passed their checks. Fixed: EPV and decisions read the Phase 0 tables and the 1 Hz states
  from OUT_ROOT first, then DATA_ROOT; the registry and `docs/run_order.md` now name the collector tables
  the lanes actually read; decisions skips the xFG feature file; consumer lists in the interface registry.
- **Code quality and tests.** Hygiene held: no raw data, no network in tests, no em or en dashes. Fixed:
  helpers every lane had copied now live once in `analytics/eval.py` (held-out by id, thread pinning, fast
  calibration slope, interval and table formatting) and `analytics/schema.py` (season of a game, July rule);
  production code no longer imports `tests/` (synthetic generators a lane needs moved into the lane);
  dormant fallback generators deleted from `tests/_fx_*`; import-time thread environment writes removed;
  every fixture CLI accepts `--fixture-games`, `--fixture-possessions`, `--fixture-heldout-every`; report
  sections carrying numbers are each labelled fixture output, with no scratch paths; the integration test's
  tautology, environment gates and skip-on-missing-output were replaced by assertions; `run_all.py`
  re-raises Ctrl-C; the wall-clock test is marked slow; the slow EPV tests now take seconds.

## Next on the Mac (where the full data is)

`DATA_ROOT` is the parent of `sportvu_contract/` and `collect/` (the Drive `nba_harvest` folder). Quote it:
Drive paths have spaces. Full detail and time estimates are in `docs/run_order.md`.

1. Check the machine on fixtures: `make test-fast` then `make all-fixtures`.
2. Collectors, one at a time (17 requests a minute): `python -m collect.bref_schedule --seasons 2015-16 --out "$DATA_ROOT/collect"`, then `collect.bref_boxscores`, `collect.bref_lineups`, `collect.positions`, `collect.bref_seasons` (later seasons for lineups, winprob, betting), and `python -m collect.pbp --contract-dir "$DATA_ROOT/sportvu_contract" --br-only` for games whose play-by-play join failed.
3. Phase 0 over the whole manifest: `make phase0 DATA="$DATA_ROOT"` (runs `analytics.gaps` then `analytics.run_phase0 --data "$DATA_ROOT/sportvu_contract" --out "$DATA_ROOT" --steps rosters,shot_frames,rebound_frames,possessions`; add `--bref-dir "$DATA_ROOT/collect"` (the folder holding `raw/bref_shotchart/`) to fill `bref_slug`; `--dry-run` first; reruns resume).
4. Hand labels: `python -m analytics.label_tool --data "$DATA_ROOT" --game 11.14.2015.BKN.at.GSW --sample 100 --seed 0 --port 8765`, open `http://127.0.0.1:8765/`; relabel with `--labeller-pass 2`; check with `--kappa`.
5. Every extension in dependency order: `make all DATA="$DATA_ROOT"` (or one at a time, `make ext-xfg DATA="$DATA_ROOT"` and so on; `OUT=/path` moves outputs, logs and reports). Reports land in `reports/<ext>.md`, logs in `logs/run_all/`.
6. Optional: `ODDS_API_KEY=... python -m collect.extras --odds --seasons ...` before `make ext-betting`; NBA.com tables (`collect.shots`, `collect.tracking`, `collect.context`, `collect.stints`) from a network stats.nba.com accepts.
