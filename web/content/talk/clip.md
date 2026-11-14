---
scene: clip
visual: clip
---
# What does the raw material look like, and what is it checked against?

This is an NBA game the way you see it on TV. A single camera, always moving.

Teams buy tracking data that says where every player stands, many times a second. This demo measures how close the TV picture alone can get.

There is an answer key. The league's own tracking system, SportVU, recorded this same game. This page calls that the sensor data.

These models learn from example footage. That is called training. The models that find the court and the players never trained on this game. A filter that runs first did. It picks out the wide camera shots, and some of its training frames came from this game.

## Notes

- Let the clip run once before talking.
- I say the filter caveat out loud here. The court model and the player detector never trained on this game. The frame gate did, so the choice of wide frames is kinder than a clean test would be.
- If asked why the public site has no clip: the footage is not mine to republish, so the clips stay on a private copy and the public site carries stills and derived data (CLAIMS.md, standing caveat four).
