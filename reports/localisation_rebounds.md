# Rebound localisation (DATA_CONTRACT section 18)

`analytics/rebound_frames.py` locates, for every missed shot, the rim frame (the ball falls below 10 ft within
6 ft of a basket after rising above 10 ft earlier in the window) and the rebound frame (the first later frame
where a player is within 2 ft of a ball under 7 ft), and compares that player with the basketball-reference
rebounder. Every threshold is fixed in the module and none was tuned on any game. One line per `run`, with the
`validate()` numbers.

| run (UTC) | game | split | n_misses | n_rebound_found | share_rebound_found | n_identity_known | identity_agreement | share_high | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-05 07:36 | 12.16.2015.PHX.at.GSW | heldout_game | 95 | 83 | 0.874 | 80 | 0.725 | 0.600 | held-out game, used as a smoke test only, no parameter tuned on it |

## Smoke test on the held-out game (2026-11-05)

12.16.2015.PHX.at.GSW is the held-out game (`heldout_game` in sportvu/splits.json) and the only game in this
checkout with both frames and play-by-play. It was run once, as a smoke test only, with no parameter tuned on
it: every threshold is the value written in `analytics/rebound_frames.py` before the run, and nothing was
changed after reading the numbers.

- n_misses 95 (92 `shot_missed` plus 3 trip-ending `ft_missed`), n_rebound_found 83 (share 0.874); the 12
  misses without a rebound frame are exactly the 12 without a rim frame.
- n_identity_known 80 (frame-level rebounder and play-by-play rebounder both present), identity_agreement 0.725.
- confidence_flag: high 57 (share_high 0.600), low 23, unmatched 15 (5 with no play-by-play rebounder, all of them
  "Team" rebounds, plus 10 more with no rebound frame; 2 of the 5 also lack a rebound frame). Every miss had a rebound event within 6 s on the play-by-play.
- Median rim-to-rebound gap on the located misses: 12 frames (about 0.5 s at 25 Hz).

The output parquet (`analytics/rebound_frames/heldout/2015-16/12.16.2015.PHX.at.GSW.parquet`) is derived from
tracking data and is not committed.
