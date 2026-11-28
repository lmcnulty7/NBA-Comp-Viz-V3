# P8: the upper bound on the team row for label-level fixes (design, version 1, registered before any number)

Version 1, 2026-11-27. Report only, from stored files only: nothing is fitted, tuned, trained or built, no model runs
and no video is decoded. No Colab unit is used. No held-out game, no D85 test game and no test arena is read
(CLAUDE.md rule 4). hou_cle_2016 is not read; it stays held back for a team arm's single test (D203). No lock is taken
until this design is reviewed.

## 1. Question

The Level 2 team row is 0.834 pooled on gsw_cha_2016 and gsw_bkn_2015 (D202: 0.810 and 0.857), against a target of
0.95. D203 found 26 percent of team errors where another SportVU player is within 3 ft, with the rest spread out, and
did not separate tracking from colour assignment.

From stored files only: **what is the most the team row could rise through fixes to the team labels that need no new
model**, and which kind of fix carries that headroom?

## 2. How the build assigns team labels (read from the code)

- **Crops.** detect/reid.py TorsoCropCollector keeps a torso crop of each tracked box on every 3rd processed frame
  (config.REID_CROP_EVERY), at most 40 per fragment track (config.REID_MAX_CROPS).
- **Features.** detect/teams.py TeamClassifier builds one feature per crop: the CIELAB L quartiles and the median a
  and b of the floor-masked jersey patch. A crop that is mostly floor is dropped. A track's feature is the median of its
  crops' features, needing at least 2 valid crops (config.TEAM_MIN_CROPS).
- **Clusters.** k-means with k = 2 (config.TEAM_N_CLUSTERS) is fitted on the window's fragment tracks
  (build_trajectories.py, team_clf.fit(collector.crops)).
- **Labels.** After re-ID linking, each **canonical track** is assigned the nearest centroid from the pooled crops of
  its fragments (build_trajectories.py, team_by_canon = team_clf.assign(canon_crops)). A track abstains (no label) when
  its nearest-to-farthest centroid distance ratio exceeds 0.75 (config.TEAM_ABSTAIN_RATIO), or it has too few crops.
  The fragment labels also serve as a hard veto in linking (team_of).
- **Consequences:**
  - The label is **per canonical track**, constant over the window. There are no per-detection or per-frame labels.
  - So per-frame flicker within a track cannot occur, and that error kind is zero by construction. The tool asserts
    it: every row of a track carries the same label.
  - A per-track majority vote over the build's own labels is the current scheme, since each track has one label.
- **Scoring (D202, tools.p8_e3.window_team):** a team row is (canonical track, its label, the paired SportVU player's
  team) for every box paired one to one within 5 ft under the truth (sportvu.truth._match, B.MATCH_FT). Within a
  window, each label is mapped to its majority SportVU team, and a row is correct when its label's mapped team is the
  player's team. Abstained tracks give no rows.

## 3. Inputs, selection and the gate

- **The rows** are D203's: tools.p8_teamcause.window_rows. That is the scorer's own pairing recomputed with each row's
  features, on each game's current truth (multipool v1 in work_ms) and D202's composite build. It is called on every
  window of gsw_cha_2016 and gsw_bkn_2015.
- **The gate, per window:** the recomputed rows' count and majority score equal D202's score record (team n and
  majority), as D203's gate did. D202's input hashes and a complete cache (tools.p8_placecause.cache_gaps) are checked
  too. Any difference stops the run.
- **The scorer's identity pairing** (ID_MATCH_FT, 2.5 ft) gives each row's SportVU id where the pair is that close,
  as D203's switch flag used it.

## 4. Per track: purity and the split of errors (registered now)

For each canonical track c in a window:
- n_c is its rows;
- m_c is its rows on its majority SportVU team;
- e_c is its errors under the current labels and the scorer's majority mapping.

The purity is m_c / n_c. Each track's current errors split exactly, with no threshold:
- **Tracking part:** min(e_c, n_c - m_c). These are rows of the track's minority team. No single label per track can
  make them correct: they come from a switch or merge (or a pairing to the wrong player), which is tracking.
- **Colour part:** e_c - (tracking part). These are rows that relabelling the track to its majority team would fix,
  which is colour assignment.
- **Flicker part:** zero by construction (section 2), asserted.

Reported per game and pooled, as shares of team errors and as rates over team rows:
- each part;
- the tracks by purity (1.0, 0.9 to 1.0, 0.5 to 0.9), with their rows and errors;
- each part by D203's crowding bin (another SportVU player within 3 ft, 3 to 6 ft, over 6 ft).

## 5. Oracle bounds (registered now; each is a ceiling, not a forecast)

Each oracle replaces labels only. It keeps the scorer's pairing, the rows and the majority mapping, so the denominator
(team rows) is unchanged. Each is scored with tools.p8_e3.window_team, with the new labels taking the place of the old.

- **O1, a perfect per-track colour assignment:** each canonical track is labelled with its own majority SportVU team.
  This is the most any per-track colour fix can reach. It equals a track-level majority vote with the truth as the
  voter. In this build, a majority vote over the build's own labels changes nothing (section 2), so O1 is the
  track-level bound.
- **O2, split at identity switches, then perfect colour per segment:** each track is cut where its 2.5 ft identity
  pairing moves to another SportVU id and stays there for at least 1.0 s of the track's scored frames (a sustained
  switch). Each segment is then labelled with its own majority team.
  - A brief change shorter than 1.0 s (pairing noise in a crowd) does not cut. Cutting at every change of the paired
    id would make every segment pure by construction, since a row's team follows its id, and the bound would be
    trivially 1.0.
  - O2 - O1 is the most that splitting tracks at switches could add on top of perfect colour.
- **O3, colour with the current errors kept on mixed tracks (report only):** pure tracks (purity 1.0) relabelled to
  their team, mixed tracks left as they are. This isolates fixes that need no tracking change and no handling of mixed
  tracks.
- **Abstained tracks, report only:** O1 with abstained tracks also labelled (their majority team). This adds rows, so
  the denominator changes. It is reported apart and never compared with the rule.

The current row (the reproduction of D202) is reported beside each.

## 6. Registered reading table

A fix **deserves an arm** when its bound adds at least +0.03 to the team row on both games and pooled. The added
amounts are:
- the colour fix: O1 - current;
- the tracking fix: O2 - O1.

| pattern | condition | reading |
|---|---|---|
| colour headroom | O1 - current at least +0.03 on both games and pooled | a per-track colour assignment fix could carry at least 0.03; an arm on the team classifier (features, k, the abstain rule) deserves a design |
| tracking headroom | O2 - O1 at least +0.03 on both games and pooled | splitting tracks at switches could add at least 0.03 beyond perfect colour; an arm on track splitting or re-ID deserves a design |
| both | both hold | both are read, and the larger bound goes first |
| neither | neither holds | label-level fixes cannot reach +0.03; the team gap lies in what O2 leaves: the pairing itself (rows paired to the wrong player), brief swaps, or the truth |

Also reported against the 0.95 target: O2 and the gap left under it, 0.95 - O2.

Why +0.03: it is about a fifth of the gap to the target (0.95 - 0.834 = 0.116), and well above the window-level noise
of the team row (D202's pooled interval 0.818 to 0.849). The bounds are point values of oracles on the same rows; their
window-cluster intervals (2,000 resamples, seed 0) are reported beside them. The rule reads the points, since an
oracle has no sampling error beyond the rows themselves.

## 7. What it can and cannot tell

- The oracles use the truth (the SportVU team and id of each pair). They bound what a fix could reach on these rows;
  they do not show that any model reaches it.
- O2's 1.0 s rule is a convention, fixed now. Its sensitivity at 0.5 s and 2.0 s is reported, not ruled.
- Rows come from the scorer's pairing within 5 ft. A row paired to the wrong player in a crowd counts as tracking here,
  and no label fix can correct it.
- The truth frames are selected by the same r23 boxes (D202), and the team rows lean toward r23.
- Two games, one arena, one broadcaster.

## 8. Cost and guards

- One collect per game under the lock ("p8 teambound <game>", frozen CV python, V3_P8_ROOT set to work_ref). D203's
  stage B took about 2 lock minutes for both games, and this is the same row recomputation. So about 3 lock minutes
  with the manifests, cap 15 minutes per game. The report runs in system python.
- Guards on every game, SportVU id, video id and window: tools.p8_common.guard and tools.research_testset.refuse_cv_test.
  The tool refuses hou_cle_2016 before any file is read, and a test checks it.
- SHA-256 manifests of work_ref's and work_ms's labels, builds and cache before and after each step.
- Writes go only to work_ref/teambound/<game>.json and reports/research/p8_teambound.json. A test checks that these
  are the only writes.
- One chain, reported the moment it ends.

## 9. Version 1.1 (2026-11-27, after the design review of 4effde21, GO WITH CHANGES; before any code or number)

The review confirmed the code reading (section 2) and the exact split. Where this section differs from sections 1 to
8, it rules.

### 9.1 Reading table (replaces section 6's table)

The bar B_arm is set in 9.5. Gains are on the team row, per game, as point values, with intervals beside them.

| pattern | condition | reading |
|---|---|---|
| colour headroom | O1 - current at least B_arm on both games | a per-track colour fix could carry at least B_arm on both games; a colour arm deserves a design |
| tracking headroom | O2_swap - O1 at least B_arm on both games (9.2: swap cuts only) | splitting at sustained swaps could add at least B_arm beyond perfect colour on both games; a track-splitting or re-ID arm deserves a design |
| combined | O2_swap - current at least B_arm on both games, while neither row above holds on both | only colour and splitting together reach the bar; one arm would have to change both |
| split by kind | colour clears on one game and tracking on the other | the headroom is game-specific by kind; no single-kind arm is justified on both games, and a joint arm is the only one the bounds support |
| one game only | a kind clears on one game and nothing clears on the other | game-specific; it does not justify an arm alone |
| neither | O2_swap - current below B_arm on both games | the two label-level fixes measured here (per-track colour, splitting at sustained swaps) add less than B_arm on these rows; it claims nothing about other causes |

Stated before any number: the reviewer expects colour headroom on gsw_cha_2016 and tracking headroom on gsw_bkn_2015,
from D203's bins (tracks paired with a single id hold 0.050 of gsw_cha_2016's errors but 0.013 of gsw_bkn_2015's).
That is a prior, not a test.

### 9.2 Tagging O2's cuts (adds to section 5)

The scorer's pairing (sportvu.truth._match, lines 59 to 63) runs the Hungarian assignment before the 5 ft gate. A ghost
box can therefore push a near box onto the wrong player. Every O2 cut is tagged:
- **swap:** within the first 1.0 s of the new run, the old SportVU id is paired within 2.5 ft with another canonical
  track. The two players' tracks exchanged ids.
- **orphan:** in that span, the old id is paired with no track.
- **fragment boundary:** whether the cut falls where the canonical track's fragment tid changes (a re-ID link).

**Tracking headroom is read on swap cuts only (O2_swap).** O2 with all cuts is reported beside it.

Diagnostic, report only (the scorer stays frozen): the count of rows whose pair would change if the distances were
capped at 5 ft before the assignment (pairs beyond the gate given a large cost).

### 9.3 The per-bin split (replaces section 4's per-track min())

- **Tracking errors in a bin:** O1's leftover errors in that bin.
- **Colour errors in a bin:** current errors minus O1's errors in that bin. This can be negative in a bin.
- The per-track min() rule is dropped.
- m_c's majority team takes D203 8.5's tie rule: on a tie, the smaller SportVU team id.

### 9.4 Reachability (report only)

- **O1c:** each track is labelled by the true majority team of its rows on crop frames only, the frames the build's
  collector could take a crop from (detect/reid.py):
  - every 3rd processed frame of the window, by the sidecar's processed order;
  - at most the first 40 per fragment tid;
  - only a box whose inner jersey patch is at least 14 px on each side (width times 0.56, height times 0.47).

  A track with no scored row on a crop frame keeps its current label. This bounds what a perfect classifier could do
  from the evidence the build actually collects.
- **O1f:** perfect colour per fragment. Each fragment tid is labelled with its majority team, which bounds a colour fix
  applied before re-ID linking.

### 9.5 The bar (replaces section 6's +0.03)

- **Expected recovery fraction, registered:** f = 0.5. A practical fix is assumed to recover half of its oracle
  ceiling. For reference, E1b's ledger lesson found placement gains keeping about 0.3 at build level, so 0.5 is an
  optimistic assumption for colour, where the classifier is weak and simple.
- **The minimum detectable effect of a one-game test at hou_cle_2016's window count (93 windows):**
  1. For each fix, a per-window paired difference d_w = f x (oracle - current) is computed on each window of
     gsw_cha_2016 and gsw_bkn_2015.
  2. 93 windows are drawn with replacement from the pooled cha and bkn windows, 2,000 times (seed 0), and the
     row-weighted mean of d_w is taken each time.
  3. MDE = 2.8 x the standard deviation of those means: 5 percent two-sided, 80 percent power.
  - Only cha and bkn's per-window variability is used. hou_cle_2016 is never read.
- **B_arm = max(0.03, MDE / f) per fix.** The requirement is ceiling x f at least the MDE. When MDE / f exceeds 0.03,
  the bar is raised to it, and the record says so and why.

### 9.6 O2's rule, exact (replaces section 5's O2 wording)

- Time is real time: seconds = frame difference / fps, measured between rows that have an id (a 2.5 ft identity
  pair).
- A run is a maximal sequence of a track's id-rows with the same SportVU id. A run lasting at least L seconds, from its
  first to its last id-row, is **sustained**.
- **Cuts are placed at the first row of each sustained run that follows a sustained run of a different id.**
  - A-B-A returns: a sustained B between two A runs gives two cuts.
  - A short B (under L) gives none, and its rows stay in the A segment as noise.
- Rows with no id (pairs between 2.5 and 5 ft) are assigned to the segment that spans their frame.
- **L is 1.0 s for the rule.** The readings are also computed at L = 0.5, 0.75, 1.5 and 2.0 s. A reading that changes
  anywhere between 0.5 and 2.0 s is **fragile**, and a fragile reading cannot justify an arm alone.

### 9.7 The rows and the gate (replaces section 3's gate)

- A copy of tools.p8_teamcause.window_rows, extended to output each row's SportVU id and the box's fragment tid. It is
  copied, not edited, so D203's merged tool is unchanged.
- The gate, as D203 8.7, per window: the recomputed (canonical track, label, team) tuples equal score_window's own
  team_rows in order, and n, majority and fixed equal D202's record. D202's records are pinned by SHA-256 and run id.
  Otherwise nothing is reported.

### 9.8 Oracle labels and O3

- Every oracle encodes its labels as 0 and 1 by sorted SportVU team id within the window. Window_team's majority mapping
  then reads them as intended.
- O3 relabels each pure track to the build label that the window's current majority mapping sends to the track's
  team. Mixed tracks are left as they are.

### 9.9 Coverage and abstention

- Any later colour arm that changes abstention needs a coverage non-inferiority rule, since abstained tracks give no
  team rows. That is flagged now for such an arm.
- Reported here:
  - the abstained tracks, and among them those with fewer than 2 crop frames by 9.4's rule (crop-less);
  - the rows those tracks would have given.

### 9.10 Guards and outputs

- The tool refuses hou_cle_2016 explicitly, before any file is read, since tools.p8_common.guard does not refuse it. A
  test checks this.
- It never calls tools.p8_teamcause.cmd_stored, which reads hou_cle_2016's record.
- find_private runs on both outputs: work_ref/teambound/<game>.json and reports/research/p8_teambound.json.
- The date of version 1 is corrected to 2026-11-27.

### 9.11 Note on the code (2026-11-27, before any number)

- **The reading table's order of precedence**, as coded: both; colour headroom; tracking headroom; split by kind;
  combined; one game only; neither.
  - "Split by kind" comes before "combined": when colour clears on one game and tracking on the other, combined
    usually holds too, and "split by kind" says more.
  - "Combined" then covers the case where no single kind clears on a game but colour and splitting together do on
    both.
- **Abstained rows are left out of every oracle's denominator,** as section 5 states. The abstained view (O1 with
  abstained tracks labelled too, a larger denominator) is reported apart, with the counts of abstained tracks,
  crop-less ones and their rows.

### 9.12 Amendment (2026-11-27, after the code check of 23d51419, GO WITH CHANGES; before the run)

Where this differs from sections 1 to 9.11, it rules.

1. **The MDE takes no f.** As written in 9.5, f cancelled: the code multiplied each gain by f, then divided the MDE by
   f. Now:
   - d_w = oracle - current, without f;
   - 93 windows are drawn with replacement from **all** cha and bkn windows (an empty window adds zero rows), 2,000
     times;
   - MDE = 2.8 x the standard deviation of the row-weighted mean gain.
   - The bar stays B_arm = max(0.03, MDE / f), with f = 0.5. A test checks that f changes the bar.
2. **The reading table's last rows** (replacing 9.11's tail):
   - a combined clear on one game only reads "one game only", never "neither";
   - "neither" is read only when the combined part lies below its bar on both games;
   - anything else reads "unclassified".

   Tests cover each case.
3. **Fragility per kind.** Only the tracking and combined parts can be fragile, since O1 does not depend on L. Each is
   checked per game across L = 0.5, 0.75, 1.0, 1.5 and 2.0 s. **The bars are fixed at L = 1.0**, and the other L values
   reuse them.
4. **Write order and checks.** collect checks the manifests before writing anything. report refuses unless both game
   records:
   - read manifest_unchanged true;
   - come from the same clean head;
   - carry the pinned D202 hash.
5. **The watch** covers this game's pool windows' files only, as D203 did, so no hou_cle_2016 file is hashed. Per window:
   - builds/r23/<window>_*, label_build/<window>*, cache/scorecard/<window>.json and the truth's labels file;
   - per game, both truth_accept records.
6. **The capped-pairing diagnostic** counts the team rows whose box's partner differs when distances are capped before
   the assignment, and how many of those rows are errors. The scorer stays frozen.
7. **Report only:**
   - O2_all at L = 0, the outer bound (a cut at every id change);
   - the count of tracks with two or more ids and no sustained run at L = 1.0.
   - O2 is a **rule value**: it depends on L by construction.
8. **The two kinds of unlabelled track are counted apart:**
   - abstained: a team_by_track entry of None;
   - no entry at all in team_by_track.

   Each also gets its crop-less count.
9. **New tests:** O3 at row level (a pure track relabelled, a mixed one kept) and O1c on crop rows.

## 10. Result (2026-11-28, from 5a2e1cd8; record reports/research/p8_teambound.json)

**Run.** The fast suite ran first under the lock: 2,735 passed, 7 skipped, 65 deselected. Then one collect per game
under the lock: 54 s each, from the heavy-job lines of work_ref/logs/teambound.log. The report took 2 s.

- The gate (score_window's team_rows in order, and D202's n, majority and fixed) passed on every window.
- Each canonical track carried one label.
- The pool windows' files were unchanged.
- hou_cle_2016 was not read and stays unspent.

**The team row under each oracle** (team rows: 24,708 on gsw_cha_2016 and 24,966 on gsw_bkn_2015, unchanged across
oracles):

| | gsw_cha_2016 | gsw_bkn_2015 |
|---|---|---|
| current (reproduces D202) | 0.810 | 0.857 |
| O1, perfect per-track colour | 0.905 | 0.898 |
| O2_swap (rule value, L = 1.0 s) | 0.947 | 0.942 |
| O2_all (L = 1.0 s) | 0.952 | 0.949 |
| gap to 0.95 under O2_swap | 0.003 | 0.008 |

**Gains and bars.** f = 0.5. Each bar is max(0.03, MDE / f), with the MDE from 93 windows drawn from cha and bkn,
fixed at L = 1.0.

| part | gsw_cha_2016 | gsw_bkn_2015 | MDE | bar | clears |
|---|---|---|---|---|---|
| colour (O1 - current) | 0.095 [0.076, 0.117] | 0.041 [0.031, 0.052] | 0.025 | 0.050 | gsw_cha_2016 only |
| tracking (O2_swap - O1) | 0.042 [0.032, 0.053] | 0.044 [0.034, 0.053] | 0.013 | 0.030 | both games |
| combined (O2_swap - current) | 0.137 [0.115, 0.162] | 0.085 [0.072, 0.100] | 0.028 | 0.056 | both games |

**Reading, as registered: tracking headroom.**
- Tracking clears its bar on both games at the rule's L = 1.0 s. Colour clears on gsw_cha_2016 only.
- **But the tracking part is fragile on both games** (9.6 and 9.12 (3)). It clears at L = 0.5, 0.75, 1.0 and 1.5 s
  and falls below its bar at L = 2.0 s: 0.030 and 0.025 against 0.030. By 9.6, **a fragile reading cannot justify an
  arm on its own.**
- **The combined part is not fragile.** It clears on both games at every L from 0.5 to 2.0 s, its lowest value being
  0.066 on gsw_bkn_2015 at 2.0 s against 0.056.
- The table registers no arm for a fragile tracking reading. "Combined", as a pattern, applies only when neither single
  kind clears on both games, which is not the case at L = 1.0. So this item licenses no arm by itself. The combined
  part's robustness is reported as a fact, not a pattern.

**Report only:**

| | gsw_cha_2016 | gsw_bkn_2015 |
|---|---|---|
| O1c (crop frames only) | 0.894 (+0.083 over current) | 0.887 (+0.030) |
| O1f (perfect per fragment) | 0.921 | 0.910 |
| O3 (pure tracks relabelled) | 0.856 | 0.867 |
| O2_all at L = 0, the outer bound | 0.986 | 0.986 |
| O1 with abstained tracks labelled (larger denominator) | 0.905 | 0.897 |
| O2 cuts at L = 1.0: swap / orphan / on a fragment boundary | 172 / 19 / 51 | 163 / 24 / 36 |
| tracks with two or more ids and no sustained run | 86 | 81 |
| capped diagnostic: team rows whose partner changes; of them errors | 250; 97 | 227; 93 |
| unlabelled: abstained (crop-less) / no team entry (crop-less) | 70 (27) / 9 (9) | 62 (22) / 8 (8) |
| abstained rows | 823 | 747 |

- **Errors by crowding** (SportVU's nearest other player), split by 9.3 into tracking and colour:
  - under 3 ft: 1,040 errors (785 tracking, 255 colour) and 1,109 (872, 237);
  - 3 to 6 ft: 1,673 (905, 768) and 1,313 (1,023, 290);
  - over 6 ft: 1,978 (660, 1,318) and 1,142 (652, 490).
- **Errors by track purity:**
  - pure: 1,125 of 10,908 rows and 230 of 11,001;
  - 0.9 to 1.0: 772 of 6,082 and 376 of 5,569;
  - 0.5 to 0.9: 2,794 of 7,718 and 2,958 of 8,396.

**Post hoc, not ruled; no arm is proposed here beyond what the table licenses.**
- Under these oracles, label-level fixes come within 0.01 of 0.95 on both games (O2_swap 0.947 and 0.942).
- Most of gsw_cha_2016's colour headroom is reachable from the crops the build already collects: O1c recovers +0.083 of
  O1's +0.095. On gsw_bkn_2015 it recovers +0.030 of +0.041.
- The capped diagnostic shows the scorer's pairing order matters for about 1 percent of team rows (250 of 24,708 and
  227 of 24,966), about 40 percent of which are errors.
- The reviewer's prior (colour on gsw_cha_2016, tracking on gsw_bkn_2015) held for colour. Tracking cleared on both
  games, though fragilely.
