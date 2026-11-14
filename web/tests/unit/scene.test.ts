import { expect, test } from "@playwright/test";
import {
  EMPTY_SCENE,
  clampFrame,
  defaultFrame,
  frameAtTime,
  frameCount,
  lastShotSpot,
  sceneFromDemoFrame,
  sceneFromPossession,
  untrackedPlayers,
} from "@/components/court/scene";
import { tinyDemoFrame, tinyPossession } from "../helpers/possession";

test("a frame of a possession becomes players by role, a ball and ghosts linked to their defenders", () => {
  const scene = sceneFromPossession(tinyPossession(), 1);
  expect(scene.players.map((player) => [player.id, player.role, player.x, player.y, player.label])).toEqual([
    [1, "offense", 61, 21, "4"],
    [2, "offense", 70, 30, null],
    [11, "defense", 65, 22, "12"],
    [12, "defense", 74, 31, "30"],
  ]);
  expect(scene.ball).toEqual({ x: 62, y: 21, z: 5 });
  expect(scene.ghosts).toEqual([
    { id: 11, x: 66, y: 22.5, from: { x: 65, y: 22 } },
    { id: 12, x: 75, y: 30, from: { x: 74, y: 31 } },
  ]);
  expect(scene.truth).toEqual([]);
  expect(scene.markers).toEqual([]);
});

test("a missing value leaves the player, the ball or the ghost out of the scene", () => {
  const possession = tinyPossession();
  expect(sceneFromPossession(possession, 0).ghosts).toEqual([]);
  const third = sceneFromPossession(possession, 2);
  expect(third.players.map((player) => player.id)).toEqual([1, 11, 12]);
  expect(untrackedPlayers(possession, 2).map((track) => track.name)).toEqual(["Attacker Two"]);
  expect(third.ball).toEqual({ x: 63, y: 22, z: null });
  const last = sceneFromPossession(possession, 3);
  expect(last.ball).toBeNull();
  expect(last.players.map((player) => player.id)).toEqual([1, 2, 11]);
  // The ghost of a defender with no row keeps its place, with no link to draw.
  expect(last.ghosts.find((ghost) => ghost.id === 12)?.from).toBeNull();
  expect(untrackedPlayers(possession, 0)).toEqual([]);
});

test("a possession with no ghosts gives a scene with none", () => {
  expect(sceneFromPossession(tinyPossession({ ghosts: null }), 1).ghosts).toEqual([]);
});

test("the frame is clamped to the possession", () => {
  const possession = tinyPossession();
  expect(frameCount(possession)).toBe(4);
  expect([clampFrame(possession, -3), clampFrame(possession, 2.4), clampFrame(possession, 99)]).toEqual([0, 2, 3]);
  expect(clampFrame(possession, Number.NaN)).toBe(0);
  expect(sceneFromPossession(possession, 99).players.map((player) => player.id)).toEqual([1, 2, 11]);
});

test("a possession opens on the release of its last timed shot", () => {
  const possession = tinyPossession();
  expect(frameAtTime(possession.t, 0.41)).toBe(2);
  expect(frameAtTime(possession.t, -5)).toBe(0);
  expect(frameAtTime(possession.t, 50)).toBe(3);
  expect(defaultFrame(possession)).toBe(2);
  expect(lastShotSpot(possession)).toEqual({ x: 62, y: 22 });
  const untimed = tinyPossession({ shots: [{ ...possession.shots[0]!, t: null, x: null, t_source: "none" }] });
  expect(defaultFrame(untimed)).toBe(0);
  expect(lastShotSpot(untimed)).toBeNull();
  expect(defaultFrame(tinyPossession({ shots: [] }))).toBe(0);
});

test("markers pass through to the scene", () => {
  const markers = [{ kind: "spot", x: 62, y: 22 } as const, { kind: "player", id: 1 } as const];
  expect(sceneFromPossession(tinyPossession(), 1, { markers }).markers).toEqual(markers);
});

test("a demo frame becomes tracked players with the sensor positions as the truth overlay", () => {
  const neutral = sceneFromDemoFrame(tinyDemoFrame());
  expect(neutral.players.map((player) => player.role)).toEqual(["neutral", "neutral", "neutral"]);
  expect(neutral.ball).toBeNull();
  expect(neutral.truth).toEqual([
    { id: 1, x: 61.5, y: 20.5, from: { x: 60, y: 20 } },
    { id: 99, x: 80, y: 40, from: null },
  ]);
  const byTeam = sceneFromDemoFrame(tinyDemoFrame(), { offenseTeam: "AAA" });
  expect(byTeam.players.map((player) => player.role)).toEqual(["offense", "defense", "neutral"]);
});

test("the empty scene draws nothing but the court", () => {
  expect(EMPTY_SCENE).toEqual({ players: [], ball: null, ghosts: [], truth: [], markers: [] });
});
