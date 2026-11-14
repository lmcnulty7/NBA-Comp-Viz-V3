import { expect, test } from "@playwright/test";
import { domainOf } from "@/components/charts/IntervalBar";
import { metricEstimate, normalise, searchPlayers, sortPlayers } from "@/components/prep/sort";
import type { Estimate, PlayerView } from "@/lib/types/contract";

const est = (value: number): Estimate => ({ value, lo: value - 1, hi: value + 1, interval: "test", point_estimate: false });

function player(id: number, name: string, extra: { shot?: number | null; gravity?: number; games?: number; team?: string; absence?: number | null } = {}): PlayerView {
  return {
    player_id: id,
    slug: `p-${id}`,
    name,
    teams: [extra.team ?? "AAA"],
    position: "G",
    games: extra.games ?? 10,
    shot_making: {
      shown: extra.shot != null,
      attempts: 200,
      min_attempts: 100,
      over_expected_per_100: extra.shot != null ? est(extra.shot) : null,
    },
    gravity: { shown: extra.gravity != null, possessions: 400, min_possessions: 300, games: 10, gravity_ft: extra.gravity != null ? est(extra.gravity) : null },
    rebounding: { shown: false, chances: 10, min_chances: 300 },
    absence_risk: {
      shown: true,
      games: 20,
      min_games: 10,
      absent: 1,
      kind: extra.absence === null ? "observed_only" : "model_out_of_fold",
      model_probability: extra.absence === null ? null : est(extra.absence ?? 0.1),
      observed_rate: est(0.05),
    },
    similar: { shown: false, available: true, possessions: 10, min_possessions: 300, neighbours: [] },
  };
}

const players = [
  player(1, "Zed Zulu", { shot: 2, gravity: 0.5, games: 30 }),
  player(2, "Ann Able", { shot: null, gravity: 1.2, games: 12, team: "BBB" }),
  player(3, "Luka Dončić", { shot: 5, games: 30, absence: null }),
  player(4, "Bob Baker", { shot: -1, gravity: 0.1, games: 50 }),
];

test("search finds a name or a team, ignores case and accents, and keeps everyone for an empty query", () => {
  expect(searchPlayers(players, "").length).toBe(4);
  expect(searchPlayers(players, "doncic").map((p) => p.player_id)).toEqual([3]);
  expect(searchPlayers(players, "  ZED ").map((p) => p.player_id)).toEqual([1]);
  expect(searchPlayers(players, "bbb").map((p) => p.player_id)).toEqual([2]);
  expect(searchPlayers(players, "ann zulu")).toEqual([]);
  expect(normalise("Dončić")).toBe("doncic");
});

test("sort by name, games or a metric; players below the minimum come last either way", () => {
  expect(sortPlayers(players, "name").map((p) => p.player_id)).toEqual([2, 4, 3, 1]);
  expect(sortPlayers(players, "name", true).map((p) => p.player_id)).toEqual([1, 3, 4, 2]);
  expect(sortPlayers(players, "games").map((p) => p.player_id)).toEqual([4, 3, 1, 2]);
  expect(sortPlayers(players, "shot_making").map((p) => p.player_id)).toEqual([3, 1, 4, 2]);
  expect(sortPlayers(players, "shot_making", true).map((p) => p.player_id)).toEqual([4, 1, 3, 2]);
  expect(sortPlayers(players, "gravity").map((p) => p.player_id)).toEqual([2, 1, 4, 3]);
  // The model probability only: a player with the observed share alone is not ranked by it.
  expect(sortPlayers(players, "absence_risk").at(-1)!.player_id).toBe(3);
  expect(metricEstimate(players[2]!, "absence_risk")).toBeNull();
  expect(metricEstimate(players[1]!, "shot_making")).toBeNull();
  // Sorting never changes the list it was given.
  expect(players.map((p) => p.player_id)).toEqual([1, 2, 3, 4]);
});

test("the interval axis holds every interval and the reference", () => {
  const [lo, hi] = domainOf([est(2), est(-1)], [0]);
  expect(lo).toBeLessThan(-2);
  expect(hi).toBeGreaterThan(3);
  expect(domainOf([])).toEqual([0, 1]);
});
