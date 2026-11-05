# Player style embeddings (analytics/embeddings/train.py) (fixture output, not a result)

Written 2026-11-05T09:42:24Z by `python -m analytics.embeddings.train --fixtures --local-fixtures --report reports/embeddings.md --seed 0 --bootstrap 200`.

**Every number below is fixture output, not a result**: the inputs are synthetic games from tests/_fx_embeddings.py with planted archetypes. The real run is the Mac command at the end.

## Inputs

Source: fixtures (tests/_fx_embeddings, local generator), written under a temporary directory.
Training games (6): 11.01.2015.BKN.at.GSW, 11.04.2015.SAC.at.BKN, 11.07.2015.CHA.at.SAC, 11.13.2015.GSW.at.PHX, 11.16.2015.SAC.at.GSW, 11.19.2015.CHA.at.BKN.
Held-out material set aside before any feature: 2 frame file(s) under heldout/, 0 game id(s) dropped by sportvu/splits.json or the CLE arena (none), 0 shot row(s) flagged held out, 0 game(s) without a possessions table (none).
Guards: `analytics.eval.refuse_heldout` on the training game ids, `analytics.schema.assert_time_forward` on the feature frame.
NBA.com aggregates: present. Positions table: present.
Player-seasons: 50, of which 44 sufficient (at least 20 possessions) and 6 insufficient; the contract gate is 300, lowered for the short fixture games.
Features used (38): avg_speed_ftps, dist_per_poss_ft, zone_near_left, zone_near_right, zone_mid_left, zone_mid_right, zone_far_left, zone_far_right, ball_share, nearest_teammate_ft, offball_ball_dist_ft, shot_share_restricted_area, shot_share_paint_non_ra, shot_share_mid_range, shot_share_corner_3, shot_share_above_break_3, shot_share_backcourt, shot_share_3pt, attempts_per_poss, agg_drives, agg_touches, agg_passes_made, agg_catch_shoot_fga, agg_pull_up_fga, agg_deflections, agg_contested_shots, agg_screen_assists, pt_isolation_freq, pt_transition_freq, pt_prballhandler_freq, pt_prrollman_freq, pt_postup_freq, pt_spotup_freq, pt_handoff_freq, pt_cut_freq, pt_offscreen_freq, pt_offrebound_freq, pt_misc_freq.

## Candidates (fixture output, not a result)

Protocol: median imputation and standardisation fit on the sufficient rows; PCA to 8 components; the autoencoder (MLPRegressor (32, 8, 32), tanh, bottleneck 8) only when PCA's 5-NN accuracy is under 0.60, kept only when it beats PCA. Intervals: 2.5 and 97.5 percentiles of 200 bootstrap resamples of players (seed 0). The imputer, scaler and embedding are fit once on all sufficient rows, unsupervised; only the ridge fit and the k-NN vote are out-of-sample.

| method | 5-NN position accuracy | majority baseline | next-season R squared (ridge) | mean baseline R squared | note |
| --- | --- | --- | --- | --- | --- |
| pca | 1.000 [1.000, 1.000] | 0.409 [0.364, 0.568] | 0.887 [0.804, 0.937] | -0.110 [-0.264, -0.079] | explained variance 0.62, 0.15, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02 |
| autoencoder | not trained | | | | gate not reached |

5-NN: leave-one-out over 44 labelled player-seasons, majority class G. Next season: leaguedashptstats.drives (2016-17), 44 player(s), out-of-fold under 5 grouped folds by player_id.

## Decision

Chosen: **pca**. PCA 5-NN accuracy 1.000 is at or above the 0.60 gate: the autoencoder is not trained.
The gate compares the 5-NN position accuracy reported in the candidate table above (leave-one-out point estimate).

## Outputs

Tables (docs/contract/embeddings.md): player_season_features (50 rows), player_embeddings (44 rows), embedding_neighbours (220 rows).
Written under a temporary directory (no --out given).

## Assumptions (decided without asking; also in the module docstrings)

- The attacked basket of a (game, period, team) is the side where the ball spends that team's offensive frames; court zones are shares of the player's OFFENSIVE frames in a 3 by 2 grid of that half, backcourt frames in the denominator only.
- The handler is the player nearest the ball within 3 ft while the ball is under 8 ft; speed uses the section 4 gap rule (dt <= 2 s) and drops steps over 30 ft/s.
- Shots are restricted to the tracked training games (attempts per possession needs the tracking possessions); a null basketball-reference shooter id is resolved through the rosters by player_slug = bref_slug; a table without zone_basic gets zones derived from court_x, court_y, shot_type.
- NBA.com aggregates are matched by metric name under the listed endpoints; play-type frequency is poss_pct, or a poss count normalised over the play types; a missing play type is 0, a missing metric NaN (median-imputed at fit). The 2016-17 file supplies the next-season target only.
- Positions (player_id, season, position in G, F, C) are a needed input at collect/positions.parquet or collect/tracking/positions/<season>.parquet; without them the 5-NN metric is n/a and PCA stands (the gate cannot be evaluated).
- Insufficient rows (under 20 possessions; 300 in the contract) are written to player_season_features but never fitted, embedded or scored.
- The NBA.com season aggregates and the positions table are season-level and include the held-out games' minutes; no held-out row is fitted, but those season totals are not held-out-free. There is no flag to drop the aggregate block yet: a tracked-era run without it is the run without collect/tracking/aggregates present (the block is then NaN and unused).
- The shared fixtures (analytics.fixtures) key their positions and aggregates by other player ids than their games, so on them both metrics read n/a; the lane's local generator (tests/_fx_embeddings.py, 5 Hz, planted G/F/C archetypes, aligned ids) is what exercises the metrics. Fixture output is never a result.

## Mac command

On the owner's Mac, with the full 2015-16 tracking games, possessions, shots, the NBA.com aggregates when collected and collect/positions.parquet (player_id, season, position):

    python -m analytics.embeddings.train --data /path/to/DATA_ROOT --report reports/embeddings.md --out /path/to/DATA_ROOT --seed 0 --bootstrap 1000

The numbers above are fixture output, not a result.
