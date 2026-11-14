# SCRIPT.md: the V3 script

Status: draft of 2026-11-14, written by Claude Code (teammate T3, PLAN.md D39) for approval under PLAN.md M4,
and revised the same day after an independent fact check and an edit. Copy goes into the site only after that
approval. This file replaces the old story script, which was written for another page and whose numbers were
prototype numbers; none of them is used here.

What it drives: the scroll-driven cold open on the home page (D25, M4) and, later, the interview slides at the
talk route of the private site (D35, M10). One scene is one scroll section and one slide.

Scene order (D25): headline, the clip, boxes on players, the court map, the truth overlay, the same models on two
sets of positions, the entry points.

## Rules

**Copy.** The on-screen copy of a scene is the quoted lines (the lines that start with `>`). A line headed
"Step" is one scroll step. A line headed "Caption" or "Fallback" says when it is shown. Plain words, short
sentences, reading grade 9 or lower, no jargon, no long dashes. The voice is the product's: no "I" and no "we" on
screen. The presenter notes are for the slides and may say "I".

**Labels.** Words that sit on the picture (bar names, badges, legends, readouts) are listed under "Labels" in
each scene, each in double quotes. A Visual block puts no other words on screen. Labels follow the same rules as
the copy and are checked by the same test.

**Numbers.** No number is typed into copy, labels, notes or visuals. Every number is a token in braces that names
a field of the site data, for example `{scorecard.adopted.heldout_game.position_error.p50_ft}`. The page prints
the digits the data file holds. The only change it may make is a change of unit that the token table states (a
share shown as a percentage), made by moving the decimal point in the stored digits. It never multiplies a stored
value and never rounds. The table at the end lists every token with the data file and the field it resolves to.
Copy holds no digit and no number word that stands for a measured value.

**Words kept out of the copy.** These may appear only in the evidence layer and in the presenter notes:
homography, reprojection, bootstrap, Hungarian, RANSAC, YOLO, BoT-SORT, CLIP, embedding, confidence interval, CI,
PBP, xFG, EPV, ghost and ghosting, held-out, out-of-fold, log loss, MSE, Brier, calibration, percentile, p50, p90,
median. "Ground truth" appears once, in the headline, because the headline is fixed by D17. SportVU is named once
in the copy, with a plain description, because the analytics results must name their source (D16).

**Never written, anywhere on the site.**
- That an average defense is the best, the optimal or the ideal one, or where a defender "should" have stood.
- That no model saw the demo game, without the sentence about the frame filter (scene 1).
- That position error passes, without the sentence about the margin of error around it (scene 4).
- That a scorecard value covers the whole game. It covers the scored stretches of the game (scene 4).
- That the model results on the site come from broadcast tracking. They come from the sensor data, and for the
  lineup and win probability models of the results page from play-by-play records (D16, `reports/results.md`).
- That the demo or the comparison in scene 5 is a test. Each is an illustration (D16, C6).
- That nothing was rebuilt. The models of scene 5 were built again from the locked code and used only when they
  gave back their stored predictions (D45, D49).

**Two builds of the site.** The private build plays short clips. The public build shows a still frame and a note
on how to ask for access (D21). A line marked "public build only" is a fixed caption under the still. It is on
screen for the whole scene and replaces nothing.

**When a value is missing.**
- If a summary in `demo/swap.json` says `available: false`, the step that uses it is replaced by the fallback
  line given in scene 5.
- If `error_ft` of the frame on screen is null, the readout of scene 4 shows the label listed for that case.
- If `p50_error_ft` of the scene is null, step 4.2 is replaced by the fallback line given in scene 4.
- If the site data is synthetic (`manifest.json`, `synthetic: true`) every number on the page is labelled
  fixture output, not a result.

**A scene's gap never stands alone.** Wherever the gap of a demo scene is on screen, the scorecard value of the
game is beside it (D50, which calls it the whole-game value). That value covers the scored stretches of the game,
not every minute of it, and the copy says so.

**Motion and keys.** Every scroll step can be reached with the keyboard. With reduced motion, each step shows its
final state as a still picture.

**Reading grade.** Flesch-Kincaid grade, computed by `tests/test_public_docs.py` with no outside package:
0.39 x (words per sentence) + 11.8 x (syllables per word) - 15.59. A token counts as one word of one syllable,
which is how a standard counter treats a numeral. The table near the end also gives the grade with every token
counted as four syllables, for each scene and for its hardest single line. The same test fails when a stated
grade no longer matches the copy, or when a scene or a single line other than the headline is above grade 9 on
either count.

## Scene 0. Headline

**Question.** What is this, in one line?

**Visual.** Full-width opening view. Private build: the first frame of the cold open clip, paused and dimmed.
Public build: the still frame of the same scene, dimmed. The headline sits over it, the lines of the second
step under it, then a scroll cue with no words. No number on screen. The three entry points of scene 6 are also
in the top navigation, so a reader who skips the cold open loses nothing.

**Labels.** None.

**Copy.**

Step 0.1 (the headline, fixed by D17)
> From broadcast pixels to possession value, measured against ground truth at every step.

Step 0.2
> A tool for reviewing basketball possessions, built on player tracking data. It opens with a question. Can a TV broadcast stand in for the tracking data that teams buy?

**Numbers.** None.

**Evidence layer.** None on this view. The decisions behind the wording: PLAN.md D17 (the headline), D22 (the
broadcast pipeline stands in for the tracking that teams buy from vendors), D13 (the demo comes first), D16 (the
demo is an illustration, so the copy asks a question and does not call the demo a test). Step 5.7 gives the
answer.

**Presenter notes.**
- Say what the product is and who uses it: coaching staff reviewing possessions (D23). The models are the product.
- Say why a broadcast demo opens it: teams buy tracking from vendors. I built the tracking side from TV video and
  scored it, to show the whole chain and to show how I measure.
- The headline has two halves. Both carry weight. The rows that fail are part of the story, not a footnote (D17).

**Reading grade.** 7.0

## Scene 1. The clip

**Question.** What does the raw material look like, and what is it checked against?

**Visual.** Private build: the cold open clip plays muted and loops, the raw broadcast with nothing drawn on it
(`clip` of the scene, inside the length caps of PLAN.md M2 and D50). Public build: the still frame of the scene
(`still`) in place of the video, with the caption below under it from the first step. The public build carries no
video file. A small marker with no words shows which moment of the play is on screen. It stays in the same corner
through scenes 1 to 5.

**Labels.** None. The caption of the public build is in the copy.

**Copy.**

Step 1.1
> This is an NBA game the way you see it on TV. A single camera, always moving.

Step 1.2
> Teams buy tracking data that says where every player stands, many times a second. This demo measures how close the TV picture alone can get.

Step 1.3
> There is an answer key. The league's own tracking system, SportVU, recorded this same game. This page calls that the sensor data.

Step 1.4
> These models learn from example footage. That is called training. The models that find the court and the players never trained on this game. A filter that runs first did. It picks out the wide camera shots, and some of its training frames came from this game.

Caption under the still (public build only, on screen from step 1.1, not a scroll step)
> You are looking at a still frame. The video clips sit on a private copy of this site. Ask for access to watch them.

**Numbers.** None on screen. The count of this game's frames in the filter's training set is not in the site
data. The evidence layer names the report that states it.

**Evidence layer.**
- `sportvu/splits.json`: the split that names this game as the held-out game. Every training script refuses the
  held-out ids.
- `reports/sportvu_splits.txt`: what each training script reads. The row for the frame gate states how many of
  its training frames come from this game. Every other row reads none. The report's own caveat: games are matched
  by name, so a training item that carries no game name cannot be traced.
- `CLAIMS.md`: the caveat on the gate (Stage 1 measurement protocol), and the note that this game shares its
  arena and season with the training games, so the second arena is the harder test (Court stage candidate r23).
- `demo/frames.json`: `sources` (the clip, the build and the SportVU log behind the scene) and `scene_rule` (how
  the scenes were chosen, by a rule that reads no position error, D50).

**Presenter notes.**
- Let the clip run once before talking.
- I say the filter caveat out loud here. The court model and the player detector never trained on this game.
  The frame gate did, so the choice of wide frames is kinder than a clean test would be.
- If asked why the public site has no clip: the footage is not mine to republish, so the clips stay on a private
  copy and the public site carries stills and derived data (D21, CLAIMS.md standing caveat 4).

**Reading grade.** 3.7

## Scene 2. Boxes on players

**Question.** Can it find the players?

**Visual.** The frame freezes. Boxes fade in, one for each tracked player (`boxes`). Then the boxes that matched
no sensor player switch to an outline (`status` is `unmatched` in the data). Three short bars appear under the
frame, named by the first three labels below. Each bar is drawn against its target line and carries the status
label. The site data holds no team guess for a box, so the team row is a number and is not drawn on the picture.

**Labels.**
- "players with no box" (the first bar)
- "boxes with no player" (the second bar)
- "team labels" (the third bar)
- "target" (the target line of each bar)
- "not met" (the status of each bar)

**Copy.**

Step 2.1
> First, find the players. A model draws a box around each one and follows it from frame to frame.

Step 2.2
> It is far from clean. On the frames that could be checked against the sensor data, it had no box for {scorecard.adopted.heldout_game.missed_players.rate} percent of the players in view. And, leaving out the referees, {scorecard.adopted.heldout_game.ghost_boxes.rate} percent of its boxes matched no player on the floor.

Step 2.3
> It also sorts the players into two teams by jersey color. For the players it found, that is right at most {scorecard.adopted.heldout_game.team_labels.accuracy} percent of the time.

Step 2.4
> None of these meets its target. They are shown here as measured.

**Numbers.**
- `{scorecard.adopted.heldout_game.missed_players.rate}`: share of sensor players in view with no box near them.
- `{scorecard.adopted.heldout_game.ghost_boxes.rate}`: share of boxes, referees left out, with no sensor player
  near. The report also holds the share with referees counted, which the copy does not use.
- `{scorecard.adopted.heldout_game.team_labels.accuracy}`: share of matched players with the right team label,
  under the best mapping of the two groups to the two teams for each stretch, so it is an upper bound. The copy
  says "at most".

**Evidence layer.**
- `reports/scorecard/gsw_phx_2016__r23.txt`: the rows missed players, ghost boxes and team labels, each with its
  target, its status and its n. "Ghost boxes" is the scorecard's name for boxes that match no player.
- The same file, caveats: these rows are scored on the frames with usable truth only. The counts of missed
  players and of unmatched boxes are placed through a truth mapping that is valid only near the fitted feet, so
  they carry that error. Referee boxes are left out by the detector's own referee class. Team accuracy uses the
  best mapping for each stretch of play, so it is an upper bound.
- `PLAN.md`, the out-of-scope line of the checklist: fixes to detection, teams and identity are not part of this
  build.

**Presenter notes.**
- These rows are the weak part and I show them first. Detection, team labels and identity are what I would fix
  next. They are out of scope for this build.
- The first two rows are the same for every court build, because every build scores the same boxes.
- Identity is reported without a target: at least
  {scorecard.adopted.heldout_game.identity.id_switches_per_possession} identity switches in a possession, counted
  on sparse frames, so the true count is higher. That is why the models downstream cannot read broadcast rows as
  they are.
- If asked how a box counts as matched: the scorecard's own distances, listed under `method` in `scorecard.json`.

**Reading grade.** 3.3

## Scene 3. The court map

**Question.** Where on the floor is each player?

**Visual.** Split view. Left: the frame with its boxes. Right: a top-down court drawn by the shared court
renderer. A dot drops from the foot of each box onto the map (`broadcast` points of the frame). When the clip
plays, or when the reader drags the time bar, the dots move. Public build: the still frame stays on the left and
the map alone plays through the sampled frames of the scene.

**Labels.** None.

**Copy.**

Step 3.1
> A box on a screen is not a spot on the floor. The camera pans and zooms, so the same pixel is a new place in every frame.

Step 3.2
> A second model finds the painted court lines in each frame. The lines show where the camera is pointing.

Step 3.3
> Then the spot where each player stands can be placed on a flat map of the court, measured in feet.

Step 3.4
> This step has no score of its own. The sensor data tracks players, not painted lines. So the map is judged by where it puts the players. That is next.

**Numbers.** None.

**Evidence layer.**
- `CLAIMS.md`, "Court stage candidate r23": what build r23 is and how it was adopted. PLAN.md D42: why the site
  shows r23 and lists the camera-model build beside it as a measured candidate.
- `reports/scorecard/gsw_phx_2016__r23.txt`, the row court line error: not measurable, with the reason. The truth
  mapping fits the players' feet, not the painted court.
- `CLAIMS.md`, Stage 1 definition of done: the court stage is judged on position error and near-field bias.

**Presenter notes.**
- Court mapping was the dominant error of the first version (ROADMAP.md). The camera never holds still, so the
  mapping is solved again for every frame.
- The court model in the demo is the adopted build. A second build with a camera model was measured and not
  adopted, because the two differ only on the test footage and picking the better one there would be a selection
  on test data (D42). It reads {scorecard.candidate.heldout_game.position_error.p50_ft} feet on the same row.
- Live: drag the time bar and watch the dots follow the play.

**Reading grade.** 2.6

## Scene 4. The truth overlay

**Question.** How far off are the positions?

**Visual.** The sensor positions for the same moment appear on the map in the truth colour (`truth` points). A
thin line joins each broadcast dot to the sensor dot of the same player. A sensor player with no broadcast dot
stays as a hollow dot. A legend names the two kinds of dot. A readout follows the play: the gap on the frame on
screen. The scorecard row of the game sits beside the readout from the first moment, with its name, its target
and its status label: the gap of the stretch is never on screen without the scorecard value beside it (D50). Two
more fixed figures follow: the typical gap of this stretch, and a thin bar that shows the checked frames as a
part of the frames of the scored stretches.

**Labels.**
- "broadcast" and "sensor data" (the legend)
- "gap in this frame: {frames.scene.frame.error_ft} feet" (the readout)
- "no matched player in this frame" (the readout, when `error_ft` of the frame is null)
- "all scored stretches of this game" (the name of the scorecard row)
- "target" (the target of the scorecard row)
- "met, not safely" (the status of the scorecard row)
- "frames checked" (the thin bar)

**Copy.**

Step 4.1
> Now the sensor data goes on the same map. Each line joins a broadcast dot to the sensor dot for the same player.

Step 4.2
> In this stretch of play, for the players that got a box, the typical gap is {frames.scene.p50_error_ft} feet. Across all the scored stretches of this game it is {scorecard.adopted.heldout_game.position_error.p50_ft} feet. Most gaps there are under {scorecard.adopted.heldout_game.position_error.p90_ft} feet.

Fallback for step 4.2, used when `p50_error_ft` of the scene is null
> Across all the scored stretches of this game, the typical gap is {scorecard.adopted.heldout_game.position_error.p50_ft} feet. That is for the players that got a box. Most gaps are under {scorecard.adopted.heldout_game.position_error.p90_ft} feet.

Step 4.3
> Both values for the scored stretches meet the target. But the margin of error around each one runs past the target, so neither is safely met.

Step 4.4
> The scored stretches cover only part of the game. And their numbers come from {scorecard.adopted.heldout_game.testable} of their {scorecard.adopted.heldout_game.truth_frames} frames. Those are the frames where the sensor data lined up with the picture. They lean toward the easier ones.

**Numbers.**
- `{frames.scene.frame.error_ft}`: the gap on the frame on screen, over the matched players of that frame.
- `{frames.scene.p50_error_ft}`: the typical gap in this stretch, over its {frames.scene.n_matched} matched
  pairs. The public build has no clip, so the copy says "stretch of play".
- `{scorecard.adopted.heldout_game.position_error.p50_ft}` and
  `{scorecard.adopted.heldout_game.position_error.p90_ft}`: the scorecard row of the held-out game, over its
  scored stretches. "Typical" is the median. "Most" is the ninetieth percentile.
- `{scorecard.adopted.heldout_game.testable}` of `{scorecard.adopted.heldout_game.truth_frames}`: frames with
  usable truth, of the frames of the scored stretches. The scorecard states the rest as a share:
  {scorecard.adopted.heldout_game.untestable_share}.

**Evidence layer.**
- `reports/scorecard/gsw_phx_2016__r23.txt`: the row position error with its target and n, and the lines under
  the title, which state how many frames have usable truth and why the others have none.
- The same scorecard, caveats, shown as text from `caveats` of the build in `scorecard.json`: the score is on a
  local rebuild of windows of running clock, not on the whole game and not on the production artifacts. The
  caveat states how many windows were scored. The coverage row of the same file gives their length in live
  seconds.
- `reports/r23_verification.txt` and `CLAIMS.md`, "Court stage candidate r23": the interval around each value of
  the row and the chance that it meets the target. The row is met on the point estimate only. The same two files
  say that the sensor-based truth has an error of its own, so the copy says "sensor dot", not "true spot".
- The same scorecard, caveats: the checked frames are the ones where the first version of the pipeline already
  matched enough players, so they lean toward frames it handled. The truth mapping is fitted through the
  pipeline's own boxes.
- `demo/frames.json`, `notes`: how the gap of the stretch is computed. The contract defines it as the median
  distance between matched pairs, so players the tracker lost are left out. It is not the scorecard's row, which
  has its own matching rule and uses only the frames with usable truth, so the two cannot be compared.
- `reports/scorecard/cle_nyk_2015__r23.txt` and `reports/scorecard/cle_gsw_2016__r23.txt`: the same row on the
  held-out arena, not met there.

**Presenter notes.**
- Two different numbers are on screen and I name both. The clip number is for this clip and only for the players
  the tracker found. The scorecard number is the claim, and it covers the scored stretches, not the whole game.
- I say "met on the point estimate", never "passes". The interval around the row crosses the target.
- On the second arena the same row is not met in either game. The results page shows both values beside this
  one. That is the harder test, because the demo game shares its arena and season with the training games. The
  arena stays inside the ratio that the generalisation row allows, which is a weaker claim.
- The court model is the best of several training seeds of its recipe, so the arena numbers are kinder than the
  recipe's average (ROADMAP_LOG.md; PLAN.md D52 says the site's scorecard carries this caveat).
- A bias remains toward the camera on the far side of the court:
  {scorecard.adopted.heldout_game.near_field_bias.far_third_ft} feet. That row is not met.
- If asked how this compares with arena systems: a single broadcast camera cannot match them. The comparison I
  claim is on output format and on honesty about error, not on accuracy (D2).

**Reading grade.** 3.2

## Scene 5. The same models on two sets of positions

**Question.** Does the position error change the answers?

**Visual.** Two courts side by side replay the cold open scene. Left: sensor positions. Right: broadcast
positions, with every filled-in player drawn hollow. Under the courts, the items of `demo/swap.json` for this
stretch, each as a pair of values with the gap between them: shot quality at the shot, possession value at each
moment, and the average-defense markers. A fixed badge marks the scene as an illustration. A bar shows the fill
share. The badge and the fill share stay on screen for the whole scene.

**Labels.**
- "Illustration" (the badge)
- "sensor positions" and "broadcast positions" (the names of the two courts)
- "filled in from sensor data" (the hollow players and the fill share bar)
- "shot quality", "possession value" and "average defender" (the names of the three rows of values)

**Copy.**

Step 5.1
> Positions only matter if the answers built on them hold up.

Step 5.2
> This is an illustration, not a test. The same models ran twice on {swap.n_windows} stretches of this game. Once on sensor positions. Once on broadcast positions. The models were locked before this run. Each one was built again from the same code and the same training games. It was used only if it gave back its saved answers. Nothing was changed to fit the broadcast.

Step 5.3
> Shot quality is the chance that a shot goes in. It moved by {swap.xfg.mean_abs_diff} percentage points on average. That is across {swap.xfg.n} shots.

Step 5.4
> Possession value is the points a possession is worth at each moment. It moved by {swap.epv.mean_abs_diff} points on average. That is across {swap.epv.n} moments.

Step 5.5
> Then the average defender. The spot where one would stand moved by {swap.ghosting.mean_abs_diff} feet on average. That is across {swap.ghosting.n} cases.

Step 5.6
> The broadcast tracker misses players, so the missing ones were filled in from the sensor data. That is {swap.fill_share} percent of all player positions. Boxes that matched no player were dropped. Names, teams and the ball come from the sensor data too. So these numbers show what position error alone does.

Step 5.7
> Can the broadcast stand in yet? No. It misses players and it has no ball. That is why the model results on this site come from sensor data or from play-by-play records, not from the broadcast.

Fallback for steps 5.3 to 5.5, used when the summary of that model says `available: false`
> This model did not give back its saved answers when it was built again. So it is left out of the comparison.

**Numbers.**
- `{swap.n_windows}`: the stretches of the demo game in the comparison.
- `{swap.xfg.mean_abs_diff}` over `{swap.xfg.n}` shots: mean absolute difference in shot quality.
- `{swap.epv.mean_abs_diff}` over `{swap.epv.n}` moments: mean absolute difference in possession value.
- `{swap.ghosting.mean_abs_diff}` over `{swap.ghosting.n}` cases: mean absolute difference of the average-defense
  output, in feet. What one case is (a defender at a moment, or a possession) is stated in `method` of the file.
- `{swap.fill_share}`: share of player positions that came from the sensor data because the tracker had no
  matched box there. The same share for one possession is in `possessions` of the same file.

**Evidence layer.**
- `demo/swap.json`: `method` (how positions were paired and filled, and that boxes with no matched player were
  dropped), `checks` (each rebuilt model gives every stored held-out prediction back within a fixed tolerance,
  and no outcome is read, D49), `match_ft` and `max_gap_s` (the matching distance, read from the scorecard, and
  the longest gap that is filled by a straight line, neither of them tuned, D50), `possessions` (the fill share
  of each possession) and `items` (every pair of values).
- `reports/xfg.md`, `reports/epv.md`, `reports/ghosting.md`: the frozen models and their own held-out results.
- PLAN.md D16 (an illustration of the end state, not a claim that the results come from broadcast tracking), D27
  (missed players filled from the sensor data and unmatched boxes dropped, so the comparison measures position
  error only), D45 (no fitted model was ever saved, so the model objects are rebuilt by rerunning the frozen
  code on training games, and used only when they reproduce the stored predictions), D24 (no pass or fail
  threshold was set for this comparison).
- `reports/results.md`, Data: the tracking models read SportVU positions. The lineup and win probability models
  read play-by-play of later seasons. `PLAN.md`, Tracking state: the broadcast rows lack the ball, a lasting
  player identity and a reliable team, and the frozen models read all three.

**Presenter notes.**
- The question a team would ask is not "how many feet off" but "does the answer change". This scene answers it
  for position error only.
- I say the limits before anyone asks. The tracker has no ball, no names and weak team labels, so those come from
  the sensor data, and {swap.fill_share} percent of the positions do too. A full replacement test needs those
  fixed first.
- "Locked" is the frozen tag. The models were not saved when they were fitted, so each was rebuilt from the
  frozen code and checked against its stored predictions before use. I do not say "nothing was retrained".
- The comparison never looks at what happened on the play. It compares the models' answers with each other, so it
  is not a second score on the test games.
- Ghosting here means where an average defender would stand. It is never the best position.
- Live: pick a possession and show its own fill share.

**Reading grade.** 3.5

## Scene 6. Where to go next

**Question.** What can I do with this?

**Visual.** Three cards in reading order, each a link with a small live preview. Review: a featured possession
playing on the court renderer. Prep: one player card. Results: the first rows of the results table with their
verdicts, a failed row among them.

**Labels.**
- "Review", "Prep" and "Results" (the titles of the three cards)

**Copy.**

Step 6.1
> Review a possession. Choose from {manifest.n_possessions} possessions. They come from {manifest.n_games} games that were kept out of model training. See shot quality, possession value over time, and where an average defense would have stood.

Step 6.2
> Prepare for a game. Look up a player: shot making, the pull on defenders, rebounding, the risk of missing the next game, and similar players.

Step 6.3
> See the results. Every model with its test numbers. The ones that failed are listed too.

**Numbers.**
- `{manifest.n_possessions}` and `{manifest.n_games}`: the held-out possessions and games in the review.
- On the prep card: `{players.n_shown}` players shown, `{players.n_hidden}` left out for too small a sample.

**Evidence layer.**
- `manifest.json`: the games, their split and the source of every data file.
- `sportvu/splits.json`: the held-out games. The review shows held-out games only, so every prediction in it is
  out of sample (C8).
- `players.json`, `rules` and `metrics`: the minimum sample of each player view. A model prediction shown for a
  player comes from out-of-fold rows. The descriptive tables (the pull on defenders, similar players, observed
  rates) come from training games and say so. No held-out row feeds a player view (D47, D51).
- A single prediction on the review page (one shot, one moment, one average-defender position) is a point
  estimate. The page shows the model's held-out result with its interval beside it (D53).
- `results.json`, copied from `reports/results.md`: every model with its held-out value, its interval and its
  verdict. Models that failed or found no signal stay in the table (C7, C10).

**Presenter notes.**
- Open the review on a featured possession and play it: shot quality at the release, possession value over time,
  the average-defense markers.
- On the results page, go to a failed row first. The decision-quality model failed its check, so it is shown as
  tested with no signal. Read its held-out value and its interval from the row.
- Every model number on these pages comes from sensor data or play-by-play. The broadcast pipeline is behind
  only the demo and the tracking scorecard.

**Reading grade.** 4.7

## Slide version (M10)

One slide for each scene, in the same order, with the same copy. The slide shows the last state of the scene's
visual. The slides for scenes 3, 4 and 5 hold the live court, so the play can be run and stopped while talking.
The presenter notes of each scene are the notes of its slide. Slides about how the project was built (D14) are not
written here. They can reuse the evidence layers above and `reports/results.md`.

## Reading grades

Computed by `tests/test_public_docs.py` from the quoted lines of each scene. "Upper" counts every token as four
syllables. "Hardest line" is the highest grade of a single quoted line of the scene, with tokens as one syllable
and then as four. The headline alone reads above grade 9 and is fixed by D17, so it is left out of that column.

| Scene | Words | Sentences | Grade | Upper | Hardest line |
| --- | --- | --- | --- | --- | --- |
| 0. Headline | 42 | 4 | 7.0 | 7.0 | 6.1 / 6.1 |
| 1. The clip | 135 | 15 | 3.7 | 3.7 | 6.7 / 6.7 |
| 2. Boxes on players | 101 | 9 | 3.3 | 4.3 | 4.6 / 6.2 |
| 3. The court map | 97 | 9 | 2.6 | 2.6 | 5.2 / 5.2 |
| 4. The truth overlay | 153 | 14 | 3.2 | 4.9 | 5.4 / 5.8 |
| 5. The same models on two sets of positions | 255 | 29 | 3.5 | 4.6 | 5.9 / 7.2 |
| 6. Where to go next | 75 | 9 | 4.7 | 5.6 | 6.3 / 7.2 |

## Tokens

How to read a token: the first part names the data file, the last part names the field, and the parts between
choose a row by a typed field of the contract. `adopted` and `candidate` are the `role` of a build, and
`heldout_game` is the `split` of a game. Both fields exist in the real data and in the synthetic data of
`export/sample.py`, so every token resolves on both. `scene` is the scene the cold open plays, which is the first
scene in `demo/frames.json`. `frame` is the frame on screen. Row keys and value names inside a scorecard cell are
the field names of the scorecard report (see "What the site data does not hold").

"Shown as" is the only change the page may make. "Copied from" is where the export takes the value. On the real
data the adopted build is `r23`, the candidate is `r23_cam` (PLAN.md D42) and the held-out game is
`gsw_phx_2016` (`sportvu/splits.json`).

| Token | Data file | Field | Shown as | Copied from |
| --- | --- | --- | --- | --- |
| `{scorecard.adopted.heldout_game.missed_players.rate}` | `scorecard.json` | `builds[role=adopted].rows[key=missed_players].cells[split=heldout_game].values[rate]` | share as a percentage | `reports/scorecard/gsw_phx_2016__r23.json`, `metrics.missed_players.rate` |
| `{scorecard.adopted.heldout_game.ghost_boxes.rate}` | `scorecard.json` | `builds[role=adopted].rows[key=ghost_boxes].cells[split=heldout_game].values[rate]` | share as a percentage | `reports/scorecard/gsw_phx_2016__r23.json`, `metrics.ghost_boxes.rate` |
| `{scorecard.adopted.heldout_game.team_labels.accuracy}` | `scorecard.json` | `builds[role=adopted].rows[key=team_labels].cells[split=heldout_game].values[accuracy]` | share as a percentage | `reports/scorecard/gsw_phx_2016__r23.json`, `metrics.team_labels.accuracy` |
| `{scorecard.adopted.heldout_game.identity.id_switches_per_possession}` | `scorecard.json` | `builds[role=adopted].rows[key=identity].cells[split=heldout_game].values[id_switches_per_possession]` | as stored | `reports/scorecard/gsw_phx_2016__r23.json`, `metrics.identity.id_switches_per_possession` |
| `{scorecard.adopted.heldout_game.position_error.p50_ft}` | `scorecard.json` | `builds[role=adopted].rows[key=position_error].cells[split=heldout_game].values[p50_ft]` | feet, as stored | `reports/scorecard/gsw_phx_2016__r23.json`, `metrics.position_error.p50_ft` |
| `{scorecard.adopted.heldout_game.position_error.p90_ft}` | `scorecard.json` | `builds[role=adopted].rows[key=position_error].cells[split=heldout_game].values[p90_ft]` | feet, as stored | `reports/scorecard/gsw_phx_2016__r23.json`, `metrics.position_error.p90_ft` |
| `{scorecard.adopted.heldout_game.near_field_bias.far_third_ft}` | `scorecard.json` | `builds[role=adopted].rows[key=near_field_bias].cells[split=heldout_game].values[far_third_ft]` | feet, as stored | `reports/scorecard/gsw_phx_2016__r23.json`, `metrics.near_field_bias.far_third_ft` |
| `{scorecard.candidate.heldout_game.position_error.p50_ft}` | `scorecard.json` | `builds[role=candidate].rows[key=position_error].cells[split=heldout_game].values[p50_ft]` | feet, as stored | `reports/scorecard/gsw_phx_2016__r23_cam.json`, `metrics.position_error.p50_ft` |
| `{scorecard.adopted.heldout_game.testable}` | `scorecard.json` | `builds[role=adopted].games[split=heldout_game].testable` | count, as stored | `reports/scorecard/gsw_phx_2016__r23.json`, `testable` |
| `{scorecard.adopted.heldout_game.truth_frames}` | `scorecard.json` | `builds[role=adopted].games[split=heldout_game].truth_frames` | count, as stored | `reports/scorecard/gsw_phx_2016__r23.json`, `truth_frames` |
| `{scorecard.adopted.heldout_game.untestable_share}` | `scorecard.json` | `builds[role=adopted].games[split=heldout_game].untestable_share` | share as a percentage | `reports/scorecard/gsw_phx_2016__r23.json`, `untestable_share` |
| `{frames.scene.p50_error_ft}` | `demo/frames.json` | `scenes[0].p50_error_ft` | feet, as stored | the demo bridge (PLAN.md M2), from the stored broadcast build and the SportVU log |
| `{frames.scene.n_matched}` | `demo/frames.json` | `scenes[0].n_matched` | count, as stored | the demo bridge (PLAN.md M2) |
| `{frames.scene.frame.error_ft}` | `demo/frames.json` | `scenes[0].frames[k].error_ft` | feet, as stored | the demo bridge (PLAN.md M2) |
| `{swap.n_windows}` | `demo/swap.json` | `n_windows` | count, as stored | the demo bridge (PLAN.md M2) |
| `{swap.fill_share}` | `demo/swap.json` | `fill_share` | share as a percentage | the demo bridge (PLAN.md M2) |
| `{swap.xfg.mean_abs_diff}` | `demo/swap.json` | `summaries[kind=xfg].mean_abs_diff` | probability as percentage points | the demo bridge (PLAN.md M2), frozen predict only |
| `{swap.xfg.n}` | `demo/swap.json` | `summaries[kind=xfg].n` | count, as stored | the demo bridge (PLAN.md M2) |
| `{swap.epv.mean_abs_diff}` | `demo/swap.json` | `summaries[kind=epv].mean_abs_diff` | points, as stored | the demo bridge (PLAN.md M2), frozen predict only |
| `{swap.epv.n}` | `demo/swap.json` | `summaries[kind=epv].n` | count, as stored | the demo bridge (PLAN.md M2) |
| `{swap.ghosting.mean_abs_diff}` | `demo/swap.json` | `summaries[kind=ghosting].mean_abs_diff` | feet, as stored | the demo bridge (PLAN.md M2), frozen predict only |
| `{swap.ghosting.n}` | `demo/swap.json` | `summaries[kind=ghosting].n` | count, as stored | the demo bridge (PLAN.md M2) |
| `{manifest.n_possessions}` | `manifest.json` | `n_possessions` | count, as stored | the site data export (PLAN.md M1) |
| `{manifest.n_games}` | `manifest.json` | `n_games` | count, as stored | the site data export (PLAN.md M1) |
| `{players.n_shown}` | `players.json` | `n_shown` | count, as stored | the site data export (PLAN.md M1) |
| `{players.n_hidden}` | `players.json` | `n_hidden` | count, as stored | the site data export (PLAN.md M1) |

"Share as a percentage" and "probability as percentage points" mean the stored digits with the decimal point
moved two places to the right: a stored 0.819 prints as 81.9. The page moves the point in the printed digits. It
never multiplies a stored value, because that can print a long tail of digits, and it never rounds.

## What the site data does not hold

The copy is written so that none of these is needed. Each would be a change to `export/schemas.py`, which is not
part of this draft.
- Row keys and value names of the scorecard. In the contract the key of a row is a free string and the values of
  a cell are an open dictionary, so the contract cannot catch a wrong name. The tokens rely on these row keys:
  `missed_players`, `ghost_boxes`, `team_labels`, `identity`, `position_error`, `near_field_bias`. They rely on
  these value names: `rate`, `accuracy`, `id_switches_per_possession`, `p50_ft`, `p90_ft`, `far_third_ft`. The
  scorecard reports use the same names and `export/sample.py` writes them. Until the contract declares them, the
  guard is `tests/test_public_docs.py`, which resolves every token on the synthetic data and reads every
  scorecard value from its report.
- A way to pick one game of the held-out arena. The scored arena games share one `split`, and the synthetic data
  has one arena game. So the presenter notes send the presenter to the results page for the arena values and
  hold no token for them. The same holds for the held-out value of one named model.
- The number of scored stretches behind the scorecard of a game. The scorecard's caveat states it, and the page
  shows that caveat as text (scene 4).
- The number of this game's frames in the frame gate's training set (scene 1). It is in
  `reports/sportvu_splits.txt`.
- The interval around the position error row and the chance that it meets the target (scene 4). They are in
  `reports/r23_verification.txt` and `CLAIMS.md`.
- The count of training seeds behind the adopted court model (scene 4, presenter notes). It is in
  `ROADMAP_LOG.md`.
- A team guess for each box, and the court lines the model found in each frame (scenes 2 and 3).
- A flag that names the scene the cold open plays. This draft uses the first scene in `demo/frames.json`.

## To check when the demo data exists

The demo bridge was not built when this draft was written, so these rest on the contract alone.
- The nouns of scene 5 ("shots", "moments", "cases") must match what one item of `demo/swap.json` is. The file
  states it in `method`. Step 5.5 reads the ghosting items as distances in feet between two average-defender
  positions. If the bridge compares another quantity, that sentence changes with it.
- The evidence text of scene 4 describes the gap in the clip as the contract defines it. If `notes` of
  `demo/frames.json` says more, the evidence layer shows that text.
- The page prints the digits a file stores. If the export stores a mean with more digits than a reader needs,
  the rounding belongs in the export, where the rule can be written down once, and not in the page.
- Step 5.2 says each model gave back its saved answers. If a check in `checks` of `demo/swap.json` did not pass,
  the summary of that model says `available: false` and the fallback line replaces its step.

## Decisions taken in this draft

For the lead to log or reverse.
- Voice: the product's voice on screen, no "I" and no "we". First person only in the presenter notes.
- The answer key is introduced in scene 1, so that scene 2 can give its measured rows as soon as the boxes
  appear. The headline says "at every step", so each step states its measurement, and scene 3 says that it has
  none of its own.
- Missed players, unmatched boxes and team labels are in scene 2, with the plain statement that none meets its
  target. The unmatched boxes are given with referees left out, and team accuracy as "at most", as the scorecard
  defines them.
- "Typical" stands for the median and "most" for the ninetieth percentile. The exact terms are in the numbers
  list of scene 4.
- The list of words kept out of the copy is the old story check's list plus the names of the models and metrics
  of this project.
- The comparison in scene 5 gives the mean absolute difference and n for each of the three models, the number of
  stretches, and the fill share (C6, D27). It is called an illustration in its first step that holds a number.
- The cold open gives the second arena and the near-field bias to the presenter notes and the results page, not
  to the on-screen copy.
- The gap of the stretch and the scorecard value of the game share one step, so neither build of the site can
  show the first without the second (D50).

Added in the revision of 2026-11-14, after the fact check and the edit.
- The cold open asks a question in step 0.2 ("Can a TV broadcast stand in") and answers it in step 5.7 ("No"),
  with the reason. Neither the demo nor the comparison is called a test (D16).
- The scorecard value is described as covering "all the scored stretches of this game", never the whole game,
  because the scorecard is on windows of running clock. D50's own name for it, the whole-game value, is kept
  only where D50 is cited.
- "Margin of error" and "neither is safely met" replace "the range" and "not a safe pass", which reads as a pass
  of the ball on a basketball page. The status label on the picture is "met, not safely".
- Step 5.2 says the models were locked, built again and used only if they gave back their saved answers (D45,
  D49). "Nothing was retrained" is no longer written.
- Step 5.7 names play-by-play records beside the sensor data, because two rows of the results page are built on
  play-by-play.
- "Training" is explained where it first appears (step 1.4).
- Tokens choose a row by a typed field of the contract (the role of a build, the split of a game), so every
  token resolves on the real data and on the synthetic data. The three presenter-note tokens that could not be
  chosen that way (the two arena games and the held-out value of the decision model) are replaced by words.
- Words on the picture are listed under "Labels" in each scene and checked like the copy.
- No single line other than the headline is above grade 9, with a token read as one syllable or as four, and
  the table states the hardest line of each scene.
