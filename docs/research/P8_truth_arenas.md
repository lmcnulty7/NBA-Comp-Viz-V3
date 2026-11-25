# P8: Mac camera truth for NYK, OKC and POR, design before any build

Written 2026-11-25, before any build, for queue item 5 of docs/research/TRACKING_LEVELS.md. Design only: nothing in
this document has been built, and nothing here is a result. The three games are development games (D85 (3)); no
held-out game is read and no held-out number is computed (CLAUDE.md rule 4).

## 1. Why

Mac camera truth exists today for the four GSW games and hou_cle_2016 only. Level 1 step 7 (cross-arena accuracy on
development games) therefore has one non-GSW arena to tune and check on. nyk_bos_2016, okc_sac_2015 and por_okc_2016
would add three arenas (New York, Oklahoma City, Portland), five non-sealed arenas in all.

The method is the one the GSW games and hou_cle_2016 used: the frozen Phase A steps run through `tools/p8_frozen.py`
(split, moments, sync, direction, mirror, label, fit, rebuild), driven one job at a time by `tools/p8_night.py` under
the heavy-job lock (D85 (4), D140, D146). The only change is the platform: every build runs with the player detector
on CPU (`P8_DEVICE_DETECTOR=cpu`, court and gate unchanged), the reference platform of D175.

## 2. What exists for each video today

Checked 2026-11-25 from reports/research/p8_registry.json and the P8 workspace (read only).

| | nyk_bos_2016 | okc_sac_2015 | por_okc_2016 |
| --- | --- | --- | --- |
| SportVU game | 01.12.2016.BOS.at.NYK | 12.06.2015.SAC.at.OKC | 01.10.2016.OKC.at.POR |
| Identity | confirmed by automatic OCR (D130, D138 calibration run, P8_step0.md) | same | same (after the D138 parser fix for a spelled period) |
| sportvu/splits.json | not listed (the file lists only the R1.3 broadcast item set) | not listed | not listed |
| Guard | `refuse_cv_test` and `tools.p8_common.guard` pass on tag, SportVU id, video id and home team (run for this design) | pass | pass |
| D85 test set | not in it (home NYK, not MIA, TOR, LAC or LAL) | not in it | not in it |
| Whole video | 6,180 s | 6,811 s | 5,679 s |
| Sections | 11, h264 1280x720, 30000/1001 fps | 12, same format | 10, same format |
| SportVU moments | fetched | fetched | fetched |
| Scorebug layout | msg15_720, measured by hand on a frame on 2026-11-17 (before D128) | fso15_720, a registered layout (Fox Sports Ohio, same geometry); passes the frozen automatic check and the D152 layout finder | csn_nw15_720, measured by hand on 2026-11-17; period written as a word, read by `tools.p8_frozen.make_reader`; fails the frozen check (plausible share under 0.25) |
| OCR time maps | 11 of 11, done 2026-11-18 with msg15_720; 1,856 s of running clock mapped; s00 and s01 have 0 good reads (the layout misses the period there), so the first period has no time map | none | none |
| Direction builds | 4 of 11 sections, built on MPS 2026-11-18 | none | none |
| Label builds, fit, r23 rebuild | none (51 label windows planned, 1,787 s of video) | none | none |

"Training game" here means a development game under D85 (3): not a spent held-out game of sportvu/splits.json (the
guard's `is_heldout_game` check), not a D85 test game, video or arena. None of the three is in splits.json at all.

## 3. Two decisions for the lead before any build

1. **Hand-measured layouts (nyk_bos_2016, por_okc_2016).** D128 rules out hand labelling of any kind, and D152 ruled
   out measuring a new layout by hand for nop_gsw_2015. msg15_720 and csn_nw15_720 were measured on 2026-11-17, before
   D128, and nyk_bos_2016's time maps were made with msg15_720 the next day. Whether those two existing layouts may
   feed camera truth is not settled by any D line. Options: (a) accept the two pre-D128 layouts, recorded as such
   (all three games run); (b) refuse them (okc_sac_2015 runs alone, and the other two wait for a layout finder that
   passes its own validation, which D152's did not on these two broadcasts). This design runs under either; the order
   in section 7 says what each choice runs.
2. **Platform mix in development truth.** The existing Mac truth of the GSW games and hou_cle_2016 was built from MPS
   label builds, which carry the clip_boxes fault (D174) and fewer player boxes (P8_PLATFORM.md). Truth for the three
   new games would come from CPU-detector label builds. Cross-arena comparisons would then mix truth platforms. This
   design records the platform per game; rebuilding the existing games' truth on the reference platform (about 2 to 3
   lock hours a game for the label builds, by the rates below) is a separate queue item, not part of this one.

## 4. What is built, and how

Workspace: a fresh P8 root for reference-platform truth (`V3_P8_ROOT`, for example the P8 data folder's work_ref), as
`tools/p8_device.check_cached` requires ("use a fresh V3_P8_ROOT for another device setting"). It holds symbolic links
to the existing section videos and SportVU moments (read only, no copy). nyk_bos_2016's 11 time maps are linked too,
with their sync done markers copied, because the OCR ran on CPU (clock_reader keeps easyocr off MPS), so they do not
depend on the device. Its 4 MPS direction builds stay in the old workspace untouched and are built again in the new
one. The old workspace and every number scored in it stay as recorded.

Per video, in this order (each step a job of `tools/p8_night.py run --games <tag> --b4 none`, with
`P8_DEVICE_DETECTOR=cpu` and the fresh `V3_P8_ROOT` in the environment; each heavy job holds the lock for its own
duration only, and the device flag reaches every build through `tools.p8_device`'s proxy):

1. sweep, split (done or linked), moments (linked)
2. layout: skipped, the registry fixes a layout for all three (subject to decision 1)
3. sync, one job a section (okc_sac_2015 and por_okc_2016; nyk_bos_2016's are linked)
4. direction, one job a section: the 40 s direction windows built on the reference platform; then mirror (the frozen
   vote; it must resolve)
5. label, one job a section: every running span cut and built with gate v2 (V3 court model)
6. fit: the frozen camera position, per-frame camera truth and acceptance (autolabel phases B and C)
7. rebuild r23, one job a section: the scoring baseline build on the same windows (also needed for check C below)
8. scores: r23 and V3 on camera truth (the runner's SCORES); no B4 score (`--b4 none`, it is only for the gate games)

Every window writes its device record, which must show the detector on CPU; a window without it does not count.

## 5. Acceptance checks, fixed now

A game's truth is accepted for development use only when all three hold. The bars come from games already built, and
none of the three new games has been seen.

| Check | Bar | Why this bar |
| --- | --- | --- |
| A. Registration | camera fit status ok and median reprojection after refinement (`camera.reproj_px_median_after`) at most 10.0 px | accepted Mac fits: gsw_sac_2015 5.47, hou_cle_2016 6.40, gsw_bkn_2015 6.52, gsw_cha_2016 7.65 px; the fit D98 found broken, gsw_ind_2016 on MPS, 18.35 px; 10.0 sits above the worst good fit and well below the broken one |
| B. Coverage | at least 1,000 accepted camera-truth frames, and at least 1,500 s of running clock mapped by the time maps | accepted frames on good fits: 1,366 (gsw_cha_2016) to 4,427 (hou_cle_2016); the broken fit had 970. Mapped clock: 2,164 to 2,635 s on complete games; nyk_bos_2016 already has 1,856 s with a lost quarter |
| C. Offset stability (D166, the frozen offset rule) | between the V3 label build and the r23 rebuild of each window (windows with at least 20 usable frames), median absolute offset difference at most 0.60 s and share over 0.2 s at most 0.80 | the frozen rule's per-game values on development games: 0.44 to 0.60 s and 0.7273 to 0.7978 (P8_truth_v2.md); the bars are the worst development game, so a new game passes if it is no less stable than those already in use |

A game that fails A or C is reported apart and flagged, as gsw_ind_2016 is (D98), and is not used to tune anything. A
game that fails B is not used. No bar, constant or fit setting changes after a game's numbers are seen (D98); a later
change would be a new registration.

Check C needs code: `tools/p8_truth_v2.py measure` accepts only its four registered games. A small report-only
extension (or a separate acceptance tool that reuses its functions read only and writes into the fresh workspace) is
needed before the first fit, written and reviewed after this design. Checks A and B read the fit's own result file.

## 6. Expected lock hours (estimates from measured rates)

Rates measured on this Mac: CPU-detector builds 0.434 to 0.446 s a processed frame (M6, M9; 0.44 used); 8.62 to 8.87
processed frames a second of label-window video (gsw_bkn_2015, hou_cle_2016; 8.75 used); OCR 0.34 to 0.80 s a 1 Hz
read (nyk_bos_2016 0.34, hou_cle_2016 0.46, GSW games 0.72 to 0.88); fit about 0.15 h; scores about 0.2 h a game.
Label-window video for okc_sac_2015 and por_okc_2016 is estimated as 30 to 42 percent of the whole video (nyk_bos_2016
0.29 with a lost quarter, hou_cle_2016 0.41); direction windows as about 38 s a section (hou_cle_2016 457 s over 12).

| | nyk_bos_2016 | por_okc_2016 | okc_sac_2015 |
| --- | --- | --- | --- |
| Label-window video | 1,787 s (planned, 51 windows) | 1,700 to 2,390 s (estimate) | 2,040 to 2,860 s (estimate) |
| Sync (OCR) | 0 (linked) | 0.5 to 1.3 h | 0.6 to 1.5 h |
| Direction builds | 0.35 h | 0.4 h | 0.5 h |
| Label builds | 1.9 h | 1.8 to 2.6 h | 2.2 to 3.1 h |
| Fit | 0.15 h | 0.15 h | 0.15 h |
| Truth subtotal | about 2.4 h | about 2.9 to 4.4 h | about 3.5 to 5.2 h |
| r23 rebuild and scores | about 2.1 h | about 2.0 to 2.8 h | about 2.4 to 3.3 h |
| Total | about 4.5 h | about 4.9 to 7.2 h | about 5.9 to 8.5 h |

All three: about 8.8 to 12.0 lock hours for truth, about 15 to 20 with the r23 baselines. No Colab units.

## 7. Order

Smallest first, among the games decision 1 allows: nyk_bos_2016, then por_okc_2016, then okc_sac_2015. Under option
(b) of decision 1, only okc_sac_2015 runs. por_okc_2016 starts with a probe: the sync of its first two sections (about
0.2 h), because its word-period reader has never mapped a whole section; it goes on only if the probe passes the first
kill criterion's per-section form below. One game at a time; the next starts after the previous game's acceptance
checks are written.

## 8. Kill criteria (each checked at the earliest step that can show it)

- Before any job: the guard and refuse_cv_test fail, or the registry status is not confirmed: the game does not run.
- After sync: under 1,500 s of running clock mapped for the game (check B's bar), stop before any model build. For the
  por_okc_2016 probe: if neither of its first two sections maps any running clock, stop por_okc_2016.
- After direction: the frozen mirror vote does not resolve (status not ok): stop the game.
- During label: more than 10 percent of a game's windows fail to build (gate-empty windows are not failures), or a
  game's lock time passes 1.5 times its upper estimate in section 6: stop and diagnose before going on.
- After fit and rebuild: checks A, B and C of section 5 decide how the truth is used; a failure is recorded and nothing
  is retuned.

## 9. What this enables

- Level 1 step 7: cross-arena tuning and checking on development truth in five non-sealed arenas (GSW, HOU, NYK, OKC,
  POR) instead of two, with the D85 test arenas (MIA, TOR, LAC and LAL home games) untouched.
- Per-arena r23 baseline rows on the reference platform, beside hou_cle_2016's, so a change (E1R, E1b, a court model)
  can be read across arenas before the single D85 score.
- More arenas for the court model's SportVU-derived labels (Level 1 step 7's first item), from games it never trained
  on.

## 10. What the CUDA check's verdict would change

If CUDA matches CPU (docs/research/P8_cuda_check.md; its verdict covers the production player detector with the
frozen V3 court and gate models on the two D99 windows, on a T4, and it reports one window built with the r23 court
model beside the verdict), the label and rebuild builds, about 75 to 85 percent of the lock hours above, could run on
Colab instead of the Mac. The r23 rebuild is covered only by that one report-only window, so moving it would rest on
weaker evidence than moving the label builds. The OCR sync, the cuts, the fit and the scores would stay on the Mac (the fit reads
the window clips; OCR runs on CPU either way). The cost moves from lock hours to units and transfer: a game's label
clips are about 1.6 GB (hou_cle_2016's 93 clips), staged through Drive with `tools/p8_drive_xfer.py`. If CUDA matches
neither or is not repeatable, these builds stay on the Mac as planned here.

## 11. What is not done

No held-out game is read or scored, nothing is tuned on these games before they pass section 5, and the D85 test set is
not touched. No D line is written by this design; decisions 1 and 2 are the lead's.
