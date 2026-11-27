# P8: team errors by cause (registered 2026-11-27, before any number)

Status: design only, report only. No number in this document has been computed. Nothing is fitted, tuned or adopted;
no rule, threshold or model changes; no row here is a gate. Code, after review: tools/p8_teamcause.py with tests.

## 1. Question

On the development standard rows (D202; D193 (2)), team accuracy is 0.810 (gsw_cha_2016), 0.857 (gsw_bkn_2015), 0.834
pooled and 0.756 on hou_cle_2016, against a Level 2 target of 0.95. Which causes carry the team errors, and what would
each point to?

## 2. What "a team error" is (the scorer's own definition, unchanged)

tools.p8_scorecard.score_window pairs, on each scored frame, every box (referee boxes included) with a SportVU player
one to one within B.MATCH_FT (sportvu.truth._match), and keeps a team row (canonical track, the build's team label of
that track, the player's SportVU team) when the track has a label (team_by_track in the window's identity file; an
abstained track gives no row). tools.p8_e3.window_team scores the rows under the frozen majority mapping: within a
window, each label counts its majority SportVU team as correct. So a team error is a row whose label's majority team
in that window is not the player's team: a label that mixes both teams. A window whose two labels are swapped as a
whole scores as correct and is not an error here. The limits of D202 hold: the truth frames are selected by the same
r23 boxes (rows optimistic, biased toward r23), and multipool v1 is a better fit, not a truth shown better.

## 3. Inputs (all exist; nothing is rebuilt)

The D202 score records (the reference root's rebaseline/score__<game>__current.json, run 20261127T171848Z: per window
team n, majority and fixed); the composite build's identity files (team_by_track, idmap, team_colors_bgr, light_team,
identity.teams: silhouette, tracks per team, abstained tracks, frame balance, team_vetoed); the sidecars; each game's
current truth (gsw_cha_2016 and gsw_bkn_2015 multipool v1 in work_ms, hou_cle_2016 fixed CPU in the reference root);
the scorecard caches (referee detections, cut signal) in the reference root.

## 4. Stage A: what the stored summaries already decide (system python, no lock)

From the score records and identity files only, per game and pooled over gsw_cha_2016 and gsw_bkn_2015 (hou_cle_2016
apart, D153):
- the per-window team accuracy (majority over n) and how concentrated the errors are: the share of all team errors in
  the worst tenth of windows, and the count of windows below 0.5 (a label mixing the teams about evenly);
- the window-level diagnostics against window accuracy: the clustering's silhouette, the distance between the two team
  colours (team_colors_bgr, Euclidean in BGR and in CIELAB through OpenCV's conversion of the two colours), the number
  of abstained tracks, the frame balance, team_vetoed; reported as rank correlations (Spearman) and as errors by tercile
  of each diagnostic;
- the per-game medians of the colour distance and the silhouette.
These need no per-row data. Stage A also states what the stored summaries cannot decide (every cause of section 5) and
records whether majority equals fixed in every window (a two-label one-to-one mapping gives the same score).

## 5. Stage B: per-row causes (frozen CV python, under the lock)

Per window and current truth, the tool calls the pinned score_window (with the reference-root layout and the truth's
labels swapped in, as D202's path: install_paths on the reference root only) and, on the same scored frames,
recomputes the same pairs with sportvu.truth._match on the same inputs, so each team row gets its features. Gate first,
per window: the recomputed rows' count and their majority score must equal the score record's team n and majority
exactly (and score_window's own team_rows), else nothing is reported.

Features of each team row (all from stored inputs; none fitted):
- **which team:** the player's SportVU team (home or away) and whether its label is the window's light kit
  (light_team, mapped through the label's majority team);
- **court region:** the player's third (scorer's third_of: near, middle, far) and zone (scorer's zone_of);
- **crowding:** the distance from the matched box's foot to the nearest other box's foot (court ft, through H_truth),
  and the scorer's SportVU crowding (distance to the nearest other player); bins under 3 ft, 3 to 6 ft, over 6 ft;
- **frame position and box size:** the box's distance to the nearest image edge (px; bins under 50, 50 to 150, over
  150) and its height (px; terciles per game);
- **jersey-colour distance:** the window's distance between the two team colours (as stage A), per row by its window;
- **frames after cuts:** seconds since the last cut (the scorer's since_cut from the cached cut signal; bins under 1 s,
  1 to 3 s, over 3 s, and censored);
- **referees taken as players:** the matched box is a referee by the scorer's rule (sidecar class, or IoU 0.5 with a
  cached referee detection), or overlaps a cached referee detection at IoU 0.3 or more;
- **track age:** the row's frame minus the first frame of its canonical track in the window (seconds; bins under 1 s,
  1 to 3 s, over 3 s), for "early frames".

Reported per game, pooled over gsw_cha_2016 and gsw_bkn_2015, and hou_cle_2016 apart: for each feature and bin, rows,
errors, the error rate and the share of all team errors; each feature on its own (the features overlap and are not an
exclusive split; their correlation is reported as a table of pairwise rank correlations over rows).

## 6. Readings (fixed now)

A feature bin **carries** team errors when it holds at least 0.25 of a unit's team errors and its error rate is at
least 1.5 times the unit's overall error rate. For each carrying bin the report names what it points to:

| carrying bin | points to |
|---|---|
| low colour distance or low silhouette (stage A or B, lowest tercile) | the colour model per game or window: kits too close for this clustering |
| one team's rows (or the light kit's) | an asymmetric colour model (one kit confused with the court, the crowd or the referees) |
| crowding under 3 ft | mixed crops in crowds: crowding-aware assignment or a crop mask |
| near the image edge, or the smallest boxes | crop quality: cut-off players or too few pixels |
| far third, or a zone | distance from the camera (resolution), or a lighting region |
| under 1 s since a cut, or track age under 1 s | early frames: the label needs time to settle (temporal voting) |
| referee boxes | referees taken as players: a referee filter before team assignment |
| errors concentrated in a few windows (stage A: worst tenth holds at least half of the errors) | per-window failures of the clustering, to be read window by window in a later item |
If no bin carries errors, the report says the errors are spread and names the largest bin. A reading names a
direction for a later, separately registered arm; nothing here fits, tunes or adopts anything.

## 7. Cost and guards

Stage A: stored JSON only, system python, no lock, seconds. Stage B: score_window on the three games' current truths
(249 windows: 73, 83 and 93; the scorer's caches are complete for these truths after D202), frozen CV python, about 5 minutes,
inside the lock "p8 teamcause rows", cap 20 minutes; at the cap it stops and reports nothing. Writes: one output path
per stage, reports/research/p8_teamcause_stored.json (stage A) and reports/research/p8_teamcause.json (stage B), both
checked by find_private before writing; the tool opens every input read-only and writes nothing else (a test patches
the writers). Guards: tools.p8_common.guard and tools.research_testset.refuse_cv_test on every game, stem and SportVU
id; the manifests and content checks of D202 (labels, label_build and truth_accept in the roots read; time maps by
content) before and after stage B; V3_HEAVY_HELD required for stage B; the scorecard pin compared on every field.
Lessons applied: identity is tested on content (time maps by frames and spans); stage A states what the stored
summaries already decide before any rescore; each stage has one output path.
