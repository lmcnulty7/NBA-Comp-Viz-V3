# Shared fixtures: `analytics/fixtures.py` (lane I1)

Deterministic synthetic data in the shapes of `docs/DATA_CONTRACT.md`, so every analytics lane tests
without raw data. Everything derives from `numpy.random.RandomState(seed)`: the same call gives the same
bytes. No network, no raw data, no file is read. Tests: `tests/test_fixtures.py`.

```
from analytics import fixtures
g = fixtures.synth_game(seed=0)                       # 0.4 to 1.3 s for 40 possessions on the shared box
league = fixtures.synth_league(0)                      # six codes, fixed team ids and 7-man rosters
games = fixtures.synth_season(0, n_games=8, heldout_every=0)       # one league, passed to every game
agg, positions = fixtures.synth_aggregates(0)                      # or players=<rosters frame>
st = fixtures.synth_stints(0)                          # st["stints"], st["playbyplay"], st["games"], st["truth"]; or players=
odds = fixtures.synth_odds(0, st["games"])
labels = fixtures.synth_labels(0, g)                   # labels["actions"], labels["screens"]
manifest = fixtures.write_fixture_root(tmp_path, seed=0, n_games=6, heldout_every=3)
python -m analytics.fixtures --out DIR [--seed 0] [--games 6] [--heldout-every 3] [--possessions 40] [--stint-games 20] [--agg-players 60]
```

Id families (never overlapping):

| family | team ids | player ids |
| --- | --- | --- |
| a game drawn on its own (`synth_game` without `teams`/`rosters`) | `1610612701 + 2 * seed` home, `+ 1` away | `9000 + 20 * seed + i`, i in 0..13 (home first) |
| a league (`synth_league(seed)`; `synth_season`, `write_fixture_root`) | `1610612800 + LEAGUE_CODES.index(code)`: fixed per code, the same in every seed | `500000 + 200 * seed + 10 * index + k`, k in 0..6 |
| `synth_aggregates` without `players` | `1610612700 + i % 4` | `200000 + 100 * seed + i` |
| `synth_stints` without `players` | `1610612720 + t`, codes `TEAM_CODES[6:12]` | `300000 + 100 * seed + i` |

No modulus anywhere, so ids are unique across seeds (seed 2500 still gives distinct ids). With
`write_fixture_root`, and whenever `players` (a section 16 rosters frame) is passed, the aggregates and the
stints use the games' players and team ids, so every family of the root shares one population.

## `synth_game(seed, n_possessions=40, heldout=False, sportvu_game=None, teams=None, rosters=None) -> dict`

Keys: `frames` (section 4), `events` (section 5 plus the v1.1 NBA.com columns), `shots` (section 10 shape),
`rosters` (16), `shot_frames` (17), `rebound_frames` (18), `possessions` (19), `states_1hz` (20), plus
`meta`, `truth` and `manifest_entry`. The five contract tables pass `analytics.schema.validate` and carry
`analytics.schema.cast` dtypes (pandas string dtype, `Int32` where nullable, float32). `frames` and
`events` carry the `sportvu/colab_contract.py` dtypes (the strings are plain object columns).

Identity. The game id is `MM.DD.YYYY.AAA.at.BBB` with synthetic three-letter codes from
`fixtures.TEAM_CODES` (none is a real NBA or basketball-reference code), dated November 2015 to March 2016,
season `2015-16`. `heldout=True` with no id uses the real held-out id `12.16.2015.PHX.at.GSW`
(`fixtures.HELDOUT_GAME`): team codes PHX and GSW, `split` `heldout_game`, `heldout` True in `shots`, `meta`
and `manifest_entry`. The split always follows the real routing (`sportvu.colab_contract.split_of` with
`sportvu/splits.json`, mirrored by `fixtures.split_for(id)`): `heldout_game` for the real id,
`heldout_arena` for any game whose home is `CLE` (`fixtures.HELDOUT_ARENA`), `train` otherwise. Note
`split_of` returns `None` for a synthetic training id (it is not listed in `splits.json`) where `split_for`
and the manifest say `train`; both agree on held out or not, which is what the routing uses. A `heldout`
flag that disagrees with the routing of an explicit id raises `ValueError`, so a manifest never marks a
game held out that the localisers and the driver would route to the training folder
(`test_write_fixture_root` checks every manifest entry against `split_of`).

Teams and rosters. With `teams` (code -> team id) and `rosters` (code -> seven dicts with `player_id`,
`player`, `jersey`), both from `synth_league`, the game uses those ids and names (a missing code raises
`ValueError`); without them it draws its own, see the id families above. Team ids are `1610612701 + 2 * (seed % 50)` (home)
and the next integer (away); player ids `9000 + 20 * (seed % 1000) + i`, home players first. Fourteen
players per game (seven a side): slots 0 to 4 start, slot 5 replaces slot 4 for period 2, returns for
period 3, slot 6 replaces slot 3 for period 4 (`fixtures.LINEUPS`, `fixtures.SUBS`), each change logged as a
`substitution` event ("F. Six enters the game for E. Five", `player1_id` out, `player2_id` in) at 720.0 s of
the new period. Names are "Alpha One" style, unique last names, play-by-play "A. One"; `rosters` matches
every player `exact` with a synthetic `bref_slug`.

Court and clock. 94 x 50 ft, baskets at (5.25, 25) and (88.75, 25) (`fixtures.BASKETS`). The home team
attacks the right basket (88.75) in periods 1 and 2 and the left one in 3 and 4. Period 1 and 4 open with
the home offence, 2 and 3 with the away offence; within a period the offence alternates every possession.
Frames run at 25 Hz, ids `0..N-1` contiguous across the game, the game clock counting down from 720 s in
each of four periods; `shot_clock_s` is 24 minus the elapsed time of the possession (reset on an offensive
rebound), never null. `ts_ms` is monotone. Possessions are spread `n_possessions // 4` per period (the
remainder to the first periods); each lasts about 5 to 21 s of clock (mean 12 s). Players are re-placed at
random at every period start, so only the period boundaries and the gap frame show a player step over
30 ft/s; inside a period players stay under 32 ft/s and the ball never jumps more than 3 ft a frame.

The gap. Exactly one 3.0 s clock jump per game, inside the set-up phase of possession
`fixtures.GAP_POSSESSION(n_possessions) == n_possessions // 3` (period 2 for 40 possessions). That
possession has `gap == True` in `possessions`; `manifest_entry["gaps_over_2s"]` records it as
`{"count": 1, "seconds": 3.0, "by_quarter": {"1": [], "2": [[from_clock, to_clock]], "3": [], "4": []}}`,
the shape `analytics.gaps.game_gaps` reads. Frame ids stay contiguous (the frames are missing, not the ids),
so `n_frames == end_frame - start_frame + 1` everywhere. `states_1hz` holds no row for the seconds inside
the gap.

Physics (what the localisers rely on):

- Carried ball: within 1 ft of the handler, 0.8 ft toward the basket, z between 1 and 4 ft when dribbling,
  5 ft when held. Passes travel 0.4 to 0.7 s in a straight line with z under 7 ft. The ball never jumps
  more than 3 ft between consecutive frames of a period (the gap frame excepted); players never exceed
  32 ft/s.
- Shot: the shooter stands still while the ball rises 4 to 7.5 ft over 8 frames (the gather); the last of
  them is `release_frame` (ball within 2.5 ft horizontally of the shooter, below 8 ft, rising for the next
  five frames; the shooter is the nearest player to the ball, nobody else within 1.6 ft). The flight is a
  parabola from 7.5 ft to exactly 10 ft at the rim centre over 1.1 s (layup), 1.2 s (mid-range) or 1.3 s
  (three), apex 13.7 to 15.6 ft. `apex_frame` is the highest ball z, `outcome_frame` the first frame after
  it with z under 10 ft within 6 ft of a basket (as `analytics.shot_frames` finds them). The play-by-play
  shot event sits at `outcome_frame` with its clock. `confidence` is the section 17 localiser's own formula,
  `1 - min(1, |apex_frame - release_frame - 25| / 25)` times 1 for a matched identity (0.64 to 0.72 here,
  so every flag is `high`), `identity_match` True; no row is `unmatched` or `low` and no frame column is -1.
- Made shot: the ball drops through the net at the rim over 0.4 s, rests on the baseline 2 ft from the
  basket until the other team's point guard (slot 1) walks to it; then the next possession's set-up begins.
- Miss: the ball bounces off the rim (peak under 11.5 ft) to a landing point 6.5 to 9 ft from the basket
  where the rebounder (slot 3 or 4 of either team, offensive with probability 0.25) waits, reaching 6.5 ft
  in his hands at `rebound_frame`; everyone else is kept 3.5 ft away. `rim_frame == outcome_frame` of the
  shot and the rebounder is the first player within 2 ft of a ball under 7 ft after it (as
  `analytics.rebound_frames` finds him); the `rebound` event sits at `rebound_frame`. `confidence` is the
  section 18 formula `1 - min(1, (rebound_frame - rim_frame) / 75)` (0.6 to 0.8, every flag `high`);
  `rebounder_id`, `pbp_rebounder_id` and `rebound_event` are never null and there are no team rebounds.
  An offensive rebound
  holds 2 to 2.5 s and then puts the ball back (a layup); a defensive one starts the next possession with
  the rebounder as handler.
- Turnover (probability 0.12 per possession): a pass is intercepted by the receiver's defender who steps
  into the lane; the event is "Turnover by A. One (bad pass; steal by F. Six)" at the catch frame and the
  stealer brings the ball up.
- Defence: man to man by slot (slot k guards slot k), between the man and the basket, 3 ft from the
  handler, 5.5 ft from others, with a first-order lag (18 percent a frame, capped at 22 ft/s) and 0.35 ft
  noise. Possession ends are reflected exactly by `analytics.possessions.segment` (every column agrees on
  seeds 0 to 5; `tests/test_fixtures.py` checks the invariants without importing the other modules).

Hidden effects (the signal downstream models should recover):

- Gravity: slot 0 of each team is the star (`truth["star_player_ids"]`): guarded 2 ft tight instead of
  5.5, and every other defender is shifted 12 percent of the way toward him. Mean nearest-defender
  distance is about 2 ft for the star against 4.7 ft for the rest.
- Shot quality: `p(make) = BASE_MAKE[class] + 0.025 * min(nearest_defender_ft, 8) + skill[shooter]` with
  `BASE_MAKE` layup 0.55, mid 0.36, three 0.30, `skill` uniform in (-0.06, 0.06) plus 0.08 for the star
  (`truth["skill"]`). Shot classes: layup 1.5 to 3.5 ft, mid-range 9 to 19 ft, three 24.5 to 27 ft above the
  break; 3PT iff the class is three. `shots.loc_x, loc_y` are tenths of a foot from the basket (`loc_y`
  toward half court, `loc_x` across, page right positive for the attacking side); `court_x, court_y` the
  shooter's court position. The SportVU-join columns of section 10 are present and null.
- Actions: every possession carries one action (`truth["actions"]`, also `synth_labels`), `action_id` its
  index within the game: `ball_screen`
  0.42, `isolation` 0.12, `handoff`, `off_ball_screen`, `drive` 0.10 each, `post_up`, `cut` 0.08 each.
  `start_frame`/`end_frame` span the action's frames inside the possession; `ball_handler_id` is the handler
  at the end of the action and `screener_id` the screener (or the handoff giver), null otherwise.
- Screens: a `ball_screen` action spans four phases from `start_frame`: the approach (1.0 s: the handler
  comes to 24 ft on the wing and the screener, slot 4, to the shoulder of the handler's defender; first
  contact within 4 ft about 0.84 s after `start_frame`), the set (1.0 s: the screener still within 2.5 ft of
  the handler's defender, so within 4 ft for at least 25 consecutive frames), the drive (1.2 s) and the
  recovery (0.8 s, to `end_frame`). During the drive the two defenders move by the coverage (uniform over
  the seven classes): `switch` swaps the two assignments for
  the rest of the possession; `drop` sends the screener's defender to 7 ft from the rim; `hedge_show` has
  him step to 2 ft off the handler's path for 0.6 s then recover to the roller; `blitz_trap` puts both
  defenders within 2.5 ft of the handler; `ice` has the on-ball defender jump to the screen side and the
  handler drive the opposite way (toward the sideline and baseline) while the big drops; `under` takes the
  on-ball defender between the screener and the basket, `over` across the handler's side. A rule on
  distances at the end of the drive (the test's `rule`) scores well above the majority class.

`truth` also holds `gap_possession`, `gap_spans`, `make_model`. `meta` holds `sportvu_game`, `gameid`
(`00215` + five digits of the seed), `date`, `season`, `home`, `away`, `home_team_id`, `away_team_id`,
`heldout`, `split`, `final_score` ("away-home"), `seed`, `n_possessions`. `manifest_entry` is the Step 1
manifest entry of section 6 (`status` `ok`, `pbp_joined` True, `parquet` paths with `heldout/` when held out).

Events. `kind` in `shot_made`, `shot_missed`, `rebound`, `turnover`, `substitution`; `team` `home` or `away`;
`score` running "away-home"; `frame` always joined (`frame_clock_diff_s` 0); `source` `fixture`, `endpoint`
`synth_game`; `event_type` 1 made, 2 missed, 4 rebound, 5 turnover, 8 substitution; `action_type` 1 jump
shot, 5 layup, 2 bad pass; `player1_id` the shooter, rebounder, turnover player or the player going out,
`player2_id` the assister (a made shot that followed a pass), the stealer or the player coming in. No free
throws, fouls or period_end events (possession `end_kind` is one of `shot_made`, `shot_missed`,
`turnover`).

## `synth_league(seed, n_teams=6) -> dict`

`{"seed", "codes", "teams", "rosters"}`: `n_teams` codes drawn from `TEAM_CODES` plus the three held-out
codes PHX, GSW and CLE (always present, so held-out games have league rosters too), `teams` mapping every
code to its fixed team id (`LEAGUE_TEAM_ID_BASE + LEAGUE_CODES.index(code)`, identical in every seed) and
`rosters` mapping every code to seven players (ids `500000 + 200 * seed + 10 * index + k`, unique last
names across the league: `One`..`Twentysix`, then `Oneson`, `Twoson`, ...). Roster slot k is the same role
in every game: 0 the star, 1 the point guard, 2 a wing, 3 and 4 bigs, 5 and 6 reserves.

## `synth_season(seed, n_games=8, heldout_every=0, n_possessions=40, league=None) -> list[dict]`

One league for the whole season (`synth_league(seed)` unless given), so a code maps to one team id and one
roster in every game; `meta["league_seed"]` names it. Game k uses seed `seed * 1000 + k`; ids are distinct
within the season. With `heldout_every = k > 0`, games
k, 2k, ... are held out: the first of them carries `12.16.2015.PHX.at.GSW` (`heldout_game`), later ones a
game at the held-out arena, `MM.DD.YYYY.AAA.at.CLE` with a synthetic away code (`heldout_arena`), so the
real routing holds every one of them out and ids never repeat. A game's content does not depend on the
held-out rule, only its id.

## `synth_aggregates(seed, n_players=60, seasons=("2015-16", "2016-17"), players=None) -> (aggregates, positions)`

Section 11 long form: `player_id, player_name, team_id, team_abbreviation, season, season_type, endpoint,
split_name, split_value, pt_measure_type, player_or_team, play_type, type_grouping, measure_type, metric,
value, heldout, source, fetched_at`. Endpoints and metrics: `leaguedashptstats` (`drives`, `touches`,
`time_of_poss`, `avg_speed`, `dist_miles`, `catch_shoot_fga`, `pull_up_fga`, `reb`, `contested_reb`, `gp`,
`min`, with `pt_measure_type`), `leaguedashplayerptshot` (`fga`, `fg_pct`, `efg_pct` per closest-defender
range in `split_value`, `split_name` `closest_defender`), `synergyplaytypes` (`ppp`, `poss`, `fg_pct` per
`play_type`), `leaguehustlestatsplayer` (`screen_assists`, `deflections`, `contested_shots`, `box_outs`,
`loose_balls_recovered`, `charges_drawn`). The hidden style factor is one of three archetypes per player
(`aggregates.attrs["archetype"]`): guards drive and pull up most, bigs rebound, set screens and shoot best
at the rim, wings in between; `fg_pct` rises with defender distance for guards and wings. `positions`
(`player_id, season, position`) maps guard to G, wing to F, big to C exactly. Without `players` the
archetype is random and the ids are the aggregates family above; with `players` (a rosters frame, as
`write_fixture_root` passes the season's concatenated rosters) the players, names, team ids and codes are
the frame's, `n_players` is ignored and the archetype follows the roster slot (`fixtures.SLOT_ARCHETYPE`:
slots 0 and 1 guard, 2 wing, 3 and 4 big, reserves 5 big and 6 wing), so the aggregates describe the same
people who appear in the frames.

## `synth_stints(seed, n_games=20, seasons=("2020-21",), n_players_per_team=9, players=None) -> dict`

`stints` (section 12 columns, `endpoint` `synth_stints`), `playbyplay` (section 5 plus `game_id`, `frame`
-1, `split`, `heldout`, `season_type`, `game_date`; `period_start`, `substitution` with `player1_id` out and
`player2_id` in, `shot_made` scoring events with a running score, `period_end`), `games` (`game_id, season,
game_date, home_team_id, away_team_id, home_team, away_team` (codes), `home_score, away_score, home_win,
spread_home` nullable Float32) and `truth`. Without `players`: six teams (ids 1610612720 to 25, codes
`TEAM_CODES[6:12]`), nine players each; with `players` (a rosters frame) its teams, codes and players
(`n_players_per_team` ignored). Game dates walk forward from November 1 by 1 to 3 days per game, so ids are
chronological and short rests occur. Lineups of five sorted ascending, a stint boundary every 60 to 240 s
over 2880 s; the starting five is the first five of the roster all season and comes back at half of the
boundaries (otherwise one player is swapped, every change logged as a substitution), so starting lineups
accumulate well over 100 possessions. Hidden effects: per-player
`off_rating` and `def_rating` (normal, sd 0.03 points per possession), a lineup interaction (`pair_effect`:
the first two players of each team add 0.12 points per possession when both are on the floor), home court
0.02. Points per stint are Poisson with that rate over `duration / 14.5` possessions; stint points sum to the
game score; `spread_home` is minus the roster strength difference times 40 plus 2.5, rounded to halves,
null for about one game in ten.

## `synth_odds(seed, games) -> DataFrame`

Section 14 columns. Three books (`fixtures.BOOKS`), markets `h2h`, `spreads`, `totals`, two snapshots per
game: the open at 14:00 UTC the day before and the close at 01:30 UTC on the game date (`is_close` True),
both before the 02:00 UTC tip (`commence_time`). Home spread line equal to `spread_home` plus noise (sd 1.0
open, 0.5 close), away line the negative, prices -110 on spreads and totals, moneylines from a logistic of
the spread with a 4.5 percent vig. `game_key` is `YYYY-MM-DD_AwayName_HomeName` with names from
`fixtures.TEAM_NAMES` (spaces removed); `home_team`, `away_team` are the full names, taken from the games
frame's `home_team`/`away_team` codes when present (a code without a name falls back to the code itself).

## `synth_labels(seed, game) -> dict(actions, screens)`

Exactly the `docs/contract/labels.md` shapes (`analytics.labels.ACTIONS_SCHEMA`, `SCREENS_SCHEMA`;
`fixtures.ACTION_COLUMNS`, `fixtures.SCREEN_COLUMNS`). `actions`: `sportvu_game, possession, action_id,
action_type, start_frame, end_frame, ball_handler_id, screener_id (Int32), label_source "synthetic",
labeller_pass 1 (int16), labelled_at` (the fixture timestamp `2026-01-01T00:00:00Z`). `screens`:
`sportvu_game, possession, screen_id, action_id, coverage, on_ball_defender_id, screener_defender_id,
label_source, confidence "sure", labeller_pass, labelled_at`, one row per `ball_screen` action, `action_id`
the action's id and `screen_id` unique within the game. They are the simulator's own record
(`game["truth"]`), so every screen is physically present in the frames as described above; `seed` is
accepted for API symmetry. `analytics.labels.read_labels(root, game)` and `validate_labels` accept the
files `write_fixture_root` writes.

## `write_fixture_root(root, seed=0, n_games=6, heldout_every=3, n_possessions=40, n_stint_games=20, n_agg_players=60) -> manifest`

The DATA_ROOT layout the lanes read; returns (and writes) the Step 1 manifest. Note the Step 1 files
(frames, play-by-play, `manifest.json`) are NOT at `root` itself: they sit under `root/sportvu_contract`,
with `collect/`, `analytics/` and `labels/` beside that directory. `fixtures.fixture_paths(root)` returns
these locations as strings (`root`, `sportvu`, `manifest`, `analytics`, `collect`, `labels`) and the returned
manifest carries the same dict under `manifest["paths"]`, so a caller that needs `manifest.json` reads
`manifest["paths"]["manifest"]` (or `fixture_paths(root)["manifest"]`):

```
DATA_ROOT/sportvu_contract/<game>.parquet, <game>_pbp.parquet, heldout/<game>.parquet, heldout/<game>_pbp.parquet
DATA_ROOT/sportvu_contract/manifest.json          section 6 shape, status ok, gaps_over_2s per game, heldout list,
                                                   split and heldout per game as sportvu.colab_contract.split_of gives them, plus paths
DATA_ROOT/collect/bref_shots/shotchart/2015-16.parquet  and  heldout/shotchart/2015-16.parquet
DATA_ROOT/analytics/<table>/2015-16/<game>.parquet  and  <table>/heldout/2015-16/<game>.parquet
                                                   for rosters, shot_frames, rebound_frames, possessions, states_1hz
DATA_ROOT/collect/tracking/aggregates/<season>.parquet   and   tracking/positions/<season>.parquet
DATA_ROOT/collect/stints/stints/<season>.parquet, playbyplay/<season>.parquet, games/<season>.parquet
DATA_ROOT/collect/extras/odds/<season>.parquet
DATA_ROOT/labels/actions/<game>.jsonl, labels/screens/<game>.jsonl   (one JSON object per line)
```

The games come from `synth_season(seed, ...)` (one league); `collect/tracking/aggregates` and the stints,
games and odds are built on the season's concatenated rosters (`players=`), so `n_agg_players` is unused by
the root. Parquet is written with `DataFrame.to_parquet`; strings read back as the pandas string dtype
(under pandas 3 the default string dtype of every frame). With the defaults (6 games, every third held out,
20 stint games) the root takes 5 to 12 s to write depending on load; the tests use smaller sizes and the
whole test file runs in 6 to 20 s.

## Assumptions taken where the brief was silent

- The clock runs continuously within a period (no stoppage after a made basket); each period holds only
  as many seconds of frames as its possessions need, so a period does not reach 0.0 s.
- No free throws, fouls, jump balls or period_end events; the last possession of a period ends on its
  shot or turnover and the period's frames end 0.6 s later.
- Substitutions happen only at period starts, so the frames roster is seven a side and ten on the floor.
- The held-out game is synthetic too (synthetic team ids, the real id and codes), so nothing real is
  ever written.
- `synth_season` keeps ids unique by giving a second held-out slot a game at the held-out arena (home
  `CLE`) instead of repeating the real id; `split_for` mirrors `sportvu/splits.json` rather than reading it,
  so the fixtures stay file-free (the test `test_split_matches_real_routing` checks the two agree).
