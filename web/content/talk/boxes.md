---
scene: boxes
visual: boxes
---
# Can it find the players?

First, find the players. A model draws a box around each one and follows it from frame to frame.

It is far from clean. On the frames that could be checked against the sensor data, it had no box for {scorecard.adopted.heldout_game.missed_players.rate} percent of the players in view. And, leaving out the referees, {scorecard.adopted.heldout_game.ghost_boxes.rate} percent of its boxes matched no player on the floor.

It also sorts the players into two teams by jersey color. For the players it found, that is right at most {scorecard.adopted.heldout_game.team_labels.accuracy} percent of the time.

None of these meets its target. They are shown here as measured.

## Notes

- These rows are the weak part and I show them first. Detection, team labels and identity are what I would fix next. They are out of scope for this build.
- The first two rows are the same for every court build, because every build scores the same boxes.
- Identity is reported without a target: at least {scorecard.adopted.heldout_game.identity.id_switches_per_possession} identity switches in a possession, counted on sparse frames, so the true count is higher. That is why the models downstream cannot read broadcast rows as they are.
- If asked how a box counts as matched: the scorecard's own distances, listed under method in scorecard.json.
