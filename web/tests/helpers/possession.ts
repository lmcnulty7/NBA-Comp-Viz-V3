import { SCHEMA_VERSION, type DemoFrame, type Possession } from "@/lib/types/contract";

/** A small possession made for tests: two attackers, two defenders, four frames. Not a result. */
export function tinyPossession(overrides: Partial<Possession> = {}): Possession {
  return {
    schema_version: SCHEMA_VERSION,
    game_id: "00.00.0000.AAA.at.BBB",
    game_slug: "test-game",
    possession: 7,
    period: 1,
    hz: 5,
    frame0: 100,
    t: [0, 0.2, 0.4, 0.6],
    game_clock_s: [700, 699.8, 699.6, null],
    shot_clock_s: [24, 23.8, 23.6, 23.4],
    offense: "AAA",
    defense: "BBB",
    attack_x: 88.75,
    result: "shot_made",
    points: 2,
    gap: false,
    players: [
      { player_id: 1, name: "Attacker One", jersey: "4", team: "AAA", offense: true, x: [60, 61, 62, 63], y: [20, 21, 22, 23] },
      { player_id: 2, name: "Attacker Two", jersey: null, team: "AAA", offense: true, x: [70, 70, null, 70], y: [30, 30, 30, 30] },
      { player_id: 11, name: "Defender One", jersey: "12", team: "BBB", offense: false, x: [64, 65, 66, 67], y: [21, 22, 23, 24] },
      { player_id: 12, name: "Defender Two", jersey: "30", team: "BBB", offense: false, x: [74, 74, 74, 74], y: [31, 31, 31, null] },
    ],
    ball: { x: [61, 62, 63, null], y: [20, 21, 22, 23], z: [4, 5, null, 6] },
    epv: { t: [0], value: [1.01] },
    ghosts: {
      label: "average defense",
      horizon_s: 1,
      dropped_points: 0,
      mean_deviation_ft: 2.5,
      tracks: [
        { player_id: 11, x: [null, 66, 67, 68], y: [null, 22.5, 23.5, 24.5] },
        { player_id: 12, x: [null, 75, 75, 75], y: [null, 30, 30, 30] },
      ],
    },
    shots: [
      {
        event: 3,
        t: 0.41,
        t_source: "release_frame",
        x: 62,
        y: 22,
        location_source: "tracking_at_release",
        shooter_id: 1,
        shooter: "Attacker One",
        made: true,
        three: false,
        xfg: 0.5,
        has_contest: true,
      },
    ],
    notes: [],
    sources: [{ path: "tests/helpers/possession.ts", what: "made for a test, not a result", rows: "not_split" }],
    ...overrides,
  };
}

export function tinyDemoFrame(): DemoFrame {
  return {
    t: 0.5,
    boxes: [],
    broadcast: [
      { player_id: 1, team: "AAA", x: 60, y: 20 },
      { player_id: 11, team: "BBB", x: 64, y: 21 },
      { player_id: null, team: null, x: 10, y: 10 },
    ],
    truth: [
      { player_id: 1, team: "AAA", x: 61.5, y: 20.5 },
      { player_id: 99, team: "BBB", x: 80, y: 40 },
    ],
    error_ft: 1.6,
  };
}
