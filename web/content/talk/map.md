---
scene: map
visual: map
---
# Where on the floor is each player?

A box on a screen is not a spot on the floor. The camera pans and zooms, so the same pixel is a new place in every frame.

A second model finds the painted court lines in each frame. The lines show where the camera is pointing.

Then the spot where each player stands can be placed on a flat map of the court, measured in feet.

This step has no score of its own. The sensor data tracks players, not painted lines. So the map is judged by where it puts the players. That is next.

## Notes

- Court mapping was the dominant error of the first version (ROADMAP.md). The camera never holds still, so the mapping is solved again for every frame.
- The court model in the demo is the adopted build. A second build with a camera model was measured and not adopted, because the two differ only on the test footage and picking the better one there would be a selection on test data. It reads {scorecard.candidate.heldout_game.position_error.p50_ft} feet on the same row.
- Live: drag the time bar and watch the dots follow the play.
