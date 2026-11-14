---
scene: truth
visual: truth
---
# How far off are the positions?

Now the sensor data goes on the same map. Each line joins a broadcast dot to the sensor dot for the same player.

In this stretch of play, for the players that got a box, the typical gap is {frames.scene.p50_error_ft} feet. Across all the scored stretches of this game it is {scorecard.adopted.heldout_game.position_error.p50_ft} feet. Most gaps there are under {scorecard.adopted.heldout_game.position_error.p90_ft} feet.

Otherwise: Across all the scored stretches of this game, the typical gap is {scorecard.adopted.heldout_game.position_error.p50_ft} feet. That is for the players that got a box. Most gaps are under {scorecard.adopted.heldout_game.position_error.p90_ft} feet.

Both values for the scored stretches meet the target. But the margin of error around each one runs past the target, so neither is safely met.

The scored stretches cover only part of the game. And their numbers come from {scorecard.adopted.heldout_game.testable} of their {scorecard.adopted.heldout_game.truth_frames} frames. Those are the frames where the sensor data lined up with the picture. They lean toward the easier ones.

## Notes

- Two different numbers are on screen and I name both. The clip number is for this clip and only for the players the tracker found. The scorecard number is the claim, and it covers the scored stretches, not the whole game.
- I say "met on the point estimate", never "passes". The interval around the row crosses the target.
- On the second arena the same row is not met in either game. The results page shows both values beside this one. That is the harder test, because the demo game shares its arena and season with the training games. The arena stays inside the ratio that the generalisation row allows, which is a weaker claim.
- The court model is the best of several training seeds of its recipe, so the arena numbers are kinder than the recipe's average (ROADMAP_LOG.md; the site's scorecard carries this caveat).
- A bias remains toward the camera on the far side of the court: {scorecard.adopted.heldout_game.near_field_bias.far_third_ft} feet. That row is not met.
- If asked how this compares with arena systems: a single broadcast camera cannot match them. The comparison I claim is on output format and on honesty about error, not on accuracy.
