---
scene: swap
visual: swap
---
# Does the position error change the answers?

Positions only matter if the answers built on them hold up.

This is an illustration, not a test. The same models ran twice on {swap.n_windows} stretches of this game. Once on sensor positions. Once on broadcast positions. The models were locked before this run. Each one was built again from the same code and the same training games. It was used only if it gave back its saved answers. Nothing was changed to fit the broadcast.

Shot quality is the chance that a shot goes in. It moved by {swap.xfg.mean_abs_diff} percentage points on average. That is across {swap.xfg.n} shots.

Otherwise: This model did not give back its saved answers when it was built again. So it is left out of the comparison.

Possession value is the points a possession is worth at each moment. It moved by {swap.epv.mean_abs_diff} points on average. That is across {swap.epv.n} moments.

Otherwise: This model did not give back its saved answers when it was built again. So it is left out of the comparison.

Then the average defender. The spot where one would stand moved by {swap.ghosting.mean_abs_diff} feet on average. That is across {swap.ghosting.n} cases.

Otherwise: This model did not give back its saved answers when it was built again. So it is left out of the comparison.

The broadcast tracker misses players, so the missing ones were filled in from the sensor data. That is {swap.fill_share} percent of all player positions. Boxes that matched no player were dropped. Names, teams and the ball come from the sensor data too. So these numbers show what position error alone does.

Can the broadcast stand in yet? No. It misses players and it has no ball. That is why the model results on this site come from sensor data or from play-by-play records, not from the broadcast.

## Notes

- The question a team would ask is not "how many feet off" but "does the answer change". This scene answers it for position error only.
- I say the limits before anyone asks. The tracker has no ball, no names and weak team labels, so those come from the sensor data, and {swap.fill_share} percent of the positions do too. A full replacement test needs those fixed first.
- "Locked" is the frozen tag. The models were not saved when they were fitted, so each was rebuilt from the frozen code and checked against its stored predictions before use. I do not say "nothing was retrained".
- The comparison never looks at what happened on the play. It compares the models' answers with each other, so it is not a second score on the test games.
- Ghosting here means where an average defender would stand, and never more than that.
- Live: pick a possession and show its own fill share.
