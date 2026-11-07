# Labeling guide: actions and ball-screen coverages

How to hand-label offensive actions and ball-screen coverages on SportVU possessions with
`analytics/label_tool.py`. The tables are defined in `docs/contract/labels.md`.

Start the tool on the Mac (frames and possessions are read in place, nothing is copied):

```
python -m analytics.label_tool --data DATA_ROOT --game 11.14.2015.BKN.at.GSW --sample 100 --seed 0 --port 8765
```

then open `http://127.0.0.1:8765/` in a browser. `DATA_ROOT` holds the frames parquet
(`<game>.parquet`, or under `frames/` or `sportvu_contract/`), the possessions parquet (under
`analytics/possessions/` or `possessions/`) and receives `labels/`. `--frames` and `--possessions`
point at files elsewhere. `--labeller-pass 2` is the relabel pass.

## 1. The seven actions

Label an action from the frame where its defining movement begins to the frame where the ball
handler's next decision is made (a shot, a pass, a pick-up or a reset). Actions of different types may
overlap (a drive out of a ball screen is two labels); two actions of the same type never overlap.

**ball_screen.** An offensive player (the screener) sets a stationary or near-stationary body between the
ball handler and the handler's defender, and the handler uses it: the handler dribbles within about 3 ft of
the screener's position and changes direction or speed because of it. Start at the screener's last step
before setting; end when the handler makes the next decision. A screen that is set but refused (the handler
drives away from it) is still a ball_screen when the screener was within 3 ft of the handler's defender.
Handler = the dribbler, screener = the player setting the screen. Slips (the screener leaves before contact)
count as ball_screen when the handler reacted to it.

**handoff.** A player holding or dribbling the ball gives it directly to a teammate who is moving past him
(dribble handoff or hand-back), with the giver's body acting as a screen. Start at the giver's first step
toward the receiver (or the receiver's first step toward the giver); end at the receiver's next decision.
Handler = the receiver, screener = the giver. A pass that travels through the air for more than a yard is
not a handoff.

**off_ball_screen.** A screen set for a player who does not have the ball (pin-down, flare, back screen,
cross screen). Start when the screener sets; end when the screened player receives the ball or the play
moves on. Handler = the player being screened for, screener = the screener. Do not label a screen that is
set but ignored by the cutter.

**isolation.** The ball handler attacks his defender one on one with the other four offensive players
cleared to the other side or along the perimeter, and no screener within 8 ft for the first second. Start at
the first attacking dribble (or the first jab from a hold); end at the next decision. Handler only. A post_up
is not an isolation.

**post_up.** An offensive player receives or dribbles the ball with his back to the basket inside or at the
edge of the paint (within about 12 ft of the rim) and holds position against his defender. Start at the
catch (or the first back-down dribble); end at the next decision. Handler = the post player. A face-up drive
from the elbow is a drive, not a post_up.

**drive.** The ball handler dribbles toward the rim from outside the paint and reaches the paint or the
restricted area with a clear intent to score or draw help, covering at least 8 ft toward the basket. Start at
the first dribble toward the rim; end at the shot, pass or pick-up. Handler only. A drive that follows a
ball_screen gets both labels.

**cut.** A player without the ball makes a sharp move toward the basket or the ball (backdoor, basket cut,
flash) that his defender has to react to, with no screen helping him. Start at the first step of the cut; end
when he receives the ball or stops. Handler = the cutter. Label only cuts the offence reacts to (the handler
looks at or passes to the cutter).

## 2. The seven coverages

A coverage describes how the two defenders involved in a ball_screen play it. Label from the moment the
screener arrives until the handler is past the screen. Set `confidence` to `unsure` when the two defenders
act differently from each other or the pictures are ambiguous; `unsure` labels are kept in the file but
`kappa` and the first training set (`read_labels(..., labeller_pass=1)` filtered to `confidence == "sure"`)
leave them out.

**switch.** The screener's defender takes the ball handler and the on-ball defender takes the screener, both
staying with their new man for at least a second after the screen. Decision rule: after the screen each
defender is nearest to the other's original man.

**drop.** The screener's defender stays back between the ball and the rim (below the level of the screen,
retreating toward the paint) while the on-ball defender fights over or under to recover; neither defender
leaves his original man. Decision rule: the screener's defender's position never rises above the screen.

**hedge_show.** The screener's defender steps out to the level of the ball for a moment (a show) to slow the
handler, then recovers to the screener, while the on-ball defender recovers to the handler. Decision rule: the
screener's defender comes to the handler's path and leaves within about a second without trapping.

**blitz_trap.** Both defenders commit to the ball handler at the screen with the intent to trap, leaving the
screener open until a rotation arrives. Decision rule: both defenders within 5 ft of the handler for more
than a second after the screen.

**ice.** On a side or wing screen the on-ball defender positions on the high side to push the handler toward
the baseline and away from the screen, while the screener's defender drops toward the baseline to contain.
Decision rule: the on-ball defender is on the screen side before the screen is used and the handler goes
toward the baseline.

**under.** The on-ball defender goes under the screen (between the screener and the basket) and the screener's
defender stays with the screener; neither switches. Decision rule: the on-ball defender passes between the
screener and the rim.

**over.** The on-ball defender goes over the top of the screen (on the ball side, fighting through contact)
and the screener's defender plays his man without a drop or a show. Decision rule: the on-ball defender passes
on the far side of the screener from the basket, and the screener's defender neither drops back nor shows.

When over or under is combined with a drop, label the screener's defender's behaviour (drop).

## 3. Keys and clicks

| key | action |
| --- | --- |
| `space` | play / pause |
| `left`, `right` | one frame back / forward; with `shift` ten frames |
| `home`, `end` | first / last frame of the possession |
| `[`, `]` | slower / faster (0.25x to 4x) |
| `1` to `7` | action type in the order ball_screen, handoff, off_ball_screen, isolation, post_up, drive, cut; in coverage mode the coverage in the order switch, drop, hedge_show, blitz_trap, ice, under, over |
| `s`, `e` | mark the start / end frame at the current frame |
| `h` then click a player | set the ball handler (in coverage mode: the on-ball defender) |
| `c` then click a player | set the screener (in coverage mode: the screener's defender) |
| click a player with no pending role | sets the handler (coverage mode: the on-ball defender) |
| `enter` | save the pending action (coverage mode: the pending coverage) |
| `esc` | clear the pending action, or leave coverage mode |
| `v` | coverage mode for the selected ball_screen (the last saved one when none is selected) |
| `u` | toggle sure / unsure |
| `j` | jump to the selected action's start frame |
| `n`, `p` | next / previous possession; the possession being left is recorded as visited by this pass, labels or none (also at the first and last possession of the list) |
| `f` | half court / full court |

Keys pressed with `ctrl`, `cmd` or `alt` are left to the browser. Opening a held-out game shows a
HELD-OUT banner: its labels are evaluation labels and tune nothing.

The label list on the right shows every action of the possession with its coverages; `select` picks an
action for coverage mode and jumps to its start, `delete` appends a tombstone (an action's coverages go
with it). Every save is validated by `analytics.labels.validate_labels`; a refused label shows the reason in
the status line and nothing is written.

## 4. Sampling plan

Possessions to label come from `analytics.labels.sample_possessions`, run by the tool with `--sample N
--seed S` so that the same N and S always give the same list:

1. training games only: a game id carrying a held-out token (`sportvu/splits.json`) is never sampled;
2. half-court possessions only: the ball crosses the half-court line and then spends at least 2 s in the
   front court (transition possessions and possessions that end before the ball crosses are skipped);
3. stratified by offensive team: the quota is split evenly across the two teams of the game;
4. at least half of each team's quota comes from possessions the rule-based screen candidate flags (a
   non-handler offensive player within 4 ft of the handler's nearest defender for 1 s), the rest from
   unflagged possessions, so the set holds both screens and non-screens.

Targets: 400 possessions for actions over at least 8 training games (50 per game, 25 per offensive team),
and 300 ball screens for coverages (the ball_screen actions of those possessions, topped up from further
sampled possessions of the same games when short).

## 5. Relabel and kappa

One week after the first pass, 10% of the labelled possessions (every tenth `sample_rank` of each game's
sample) are labelled again from scratch with `--labeller-pass 2`, without looking at the first pass. Then:

```
python -m analytics.label_tool --data DATA_ROOT --game G --kappa
python -m analytics.label_tool --data DATA_ROOT --kappa          # pooled over every game with labels
```

takes the possessions both passes visited (the visited record the page writes on next/previous; for a pass
without one, the possessions it labelled), pairs the two passes' actions inside each by temporal overlap (an
action only one pass found counts as `none`) and prints Cohen's kappa for `action_type` and, over the paired
ball screens, for `coverage` using `sure` labels only. Held-out games are skipped and counted. Both kappas
must reach 0.7. Below 0.7, the disagreements are reviewed against the definitions above,
the definitions are tightened, and the 10% is relabelled a third time before the full set is trusted; the
definitions, not the labels, are what gets edited.

## 6. Hours

| task | count | rate | hours |
| --- | --- | --- | --- |
| actions | 400 possessions | 2.5 min each | 16.7 |
| coverages | 300 screens | 1.5 min each | 7.5 |
| relabel (10% of both, a week later) and the kappa review | 40 possessions, 30 screens | | 3 |
| total | | | about 27 |

## 7. Shot audit mode (stage 3)

The stage 3 shot audit checks the shot localiser (`docs/DATA_CONTRACT.md` section 17) against the court
replay. The label tool walks the `analytics.audit_shots` sample shot by shot and writes each verdict
straight into the sample parquet that `audit_shots --summarise` reads; no CSV, no labels.

1. Draw the sample (training games only):

   ```
   python3 -m analytics.audit_shots --data "$DATA_ROOT" --n 200 --seed 0
   ```

   Drawing again over a sample that already holds verdicts is refused (`--force` overwrites it and loses
   them).

2. Audit it, then open `http://127.0.0.1:8765/`:

   ```
   python3 -m analytics.label_tool --data "$DATA_ROOT" --audit "$DATA_ROOT/analytics/audit/shots_sample.parquet"
   ```

   The page opens at the first unaudited shot (stop with Ctrl-C at any time and rerun to resume), with the
   court at the detected release frame and 4 s of frames either side. The shooter has a yellow ring; the
   strip under the court is the ball height (dashed line at 8 ft) with the detected release in green, your
   mark in purple and the apex dashed grey. The readout gives the offset from the detected frame, the ball
   height and the ball's distance to the shooter.

3. For each shot find the release: the last frame the ball is in the shooter's hands before it rises.
   Then one key saves the row and opens the next shot:

| key | action |
| --- | --- |
| `a` | accept: the detected frame is the release (`audited_release_frame` = the detected frame, `audited_ok` = true) |
| `m` | mark the current frame as the release |
| `enter` | save the marked frame (`audited_release_frame` = the mark, `audited_ok` = true) |
| `x` | bad: not a shot here, or the ball or shooter cannot be followed (`audited_release_frame` empty, `audited_ok` = false); add a note |
| `left`, `right` | one frame back / forward; with `shift` five frames (the possession labelling view steps ten) |
| `r` | back to the detected release frame |
| `esc` | clear the mark (in the note field: leave the field) |
| `t` | type a note; `enter` or `esc` leaves the field, and the note is saved with the next `a`, `enter` or `x` |
| `n`, `p` | next / previous shot without saving (to revisit or change a verdict) |
| `space`, `[`, `]` | play / pause, slower / faster |
| `home`, `end`, `f` | first / last served frame, half court / full court |

   A release off by more than 2 frames is outside the pass mark, so place the mark on the exact frame.
   Use `x` only when there is no release to place: a high-confidence shot marked bad counts as a miss in
   the pass-mark statistic. Every verdict is written to the parquet at once (atomically; each session
   keeps the state before its first write as `shots_sample.parquet.<UTC time>.bak`). A later verdict on
   the same shot replaces the earlier one. The tool refuses a sample that holds a held-out game. The
   sample can stay in the Drive folder: the tool notices an outside change by the file's content, so Drive
   touching the file time is harmless; run one audit session per sample at a time.

4. Summarise and compare with the pass mark in STATUS.md:

   ```
   python3 -m analytics.audit_shots --summarise "$DATA_ROOT/analytics/audit/shots_sample.parquet" --report reports/shot_audit.md
   ```

   Compare the `pass-mark statistic` line (`pass_mark.share`: high-confidence rows within 2 frames, bad
   rows counted as misses) with the pass mark. The older agreement line leaves bad rows out and is for
   reference. The column semantics are in `docs/contract/phase0_driver.md` (Audit sample).
