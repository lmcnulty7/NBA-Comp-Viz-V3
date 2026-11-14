import { expect, test } from "@playwright/test";
import { sceneFromPossession } from "@/components/court/scene";
import {
  SPEEDS,
  advance,
  endTime,
  epvAt,
  frameAtOrBefore,
  sceneAtTime,
  speedRate,
  timeOfFrame,
} from "@/components/review/timeline";
import { tinyPossession } from "../helpers/possession";

test("a time maps to the last frame at or before it", () => {
  const times = [0, 0.2, 0.4, 0.6];
  expect(frameAtOrBefore(times, -1)).toBe(0);
  expect(frameAtOrBefore(times, 0)).toBe(0);
  expect(frameAtOrBefore(times, 0.19)).toBe(0);
  expect(frameAtOrBefore(times, 0.2)).toBe(1);
  expect(frameAtOrBefore(times, 0.59)).toBe(2);
  expect(frameAtOrBefore(times, 0.6)).toBe(3);
  expect(frameAtOrBefore(times, 99)).toBe(3);
  expect(frameAtOrBefore([], 1)).toBe(0);
  expect(frameAtOrBefore([0], 5)).toBe(0);
  expect(frameAtOrBefore(times, Number.NaN)).toBe(0);
});

test("the end time and the time of a frame come from the possession's own clock", () => {
  const possession = tinyPossession();
  expect(endTime(possession)).toBe(0.6);
  expect(timeOfFrame(possession, 2)).toBe(0.4);
  expect(timeOfFrame(possession, 99)).toBe(0.6);
  expect(endTime(tinyPossession({ t: [0] }))).toBe(0);
  expect(endTime(tinyPossession({ t: [] }))).toBe(0);
});

test("on a sample, and without tweening, the scene is exactly the frame's scene", () => {
  const possession = tinyPossession();
  for (let frame = 0; frame < possession.t.length; frame += 1) {
    const time = possession.t[frame]!;
    expect(sceneAtTime(possession, time, { tween: true })).toEqual(sceneFromPossession(possession, frame));
    expect(sceneAtTime(possession, time + 0.1, { tween: false })).toEqual(sceneFromPossession(possession, frame));
  }
});

test("between samples a tweened scene sits on the line, and a missing side falls back to the nearer sample", () => {
  const possession = tinyPossession();
  const scene = sceneAtTime(possession, 0.1, { tween: true });
  const one = scene.players.find((player) => player.id === 1)!;
  expect(one.x).toBeCloseTo(60.5, 6);
  expect(one.y).toBeCloseTo(20.5, 6);
  expect(one.label).toBe("4");
  // The ball has x at frames 0 and 1, so it moves; z is 4 then 5.
  expect(scene.ball!.x).toBeCloseTo(61.5, 6);
  expect(scene.ball!.z).toBeCloseTo(4.5, 6);
  // Ghost 11 has no value at frame 0: before the middle it is left out, after it the frame 1 value is used.
  const early = sceneAtTime(possession, 0.05, { tween: true });
  expect(early.ghosts.find((ghost) => ghost.id === 11)).toBeUndefined();
  const later = sceneAtTime(possession, 0.15, { tween: true });
  expect(later.ghosts.find((ghost) => ghost.id === 11)).toMatchObject({ x: 66, y: 22.5 });
  // Attacker 2 has no x at frame 2: between frames 1 and 2 he stays at frame 1 until the middle.
  const gap = sceneAtTime(possession, 0.25, { tween: true });
  expect(gap.players.find((player) => player.id === 2)).toMatchObject({ x: 70, y: 30 });
  const gone = sceneAtTime(possession, 0.35, { tween: true });
  expect(gone.players.find((player) => player.id === 2)).toBeUndefined();
});

test("markers pass through, tweened or not", () => {
  const possession = tinyPossession();
  const markers = [{ kind: "spot" as const, x: 10, y: 20 }];
  expect(sceneAtTime(possession, 0.1, { tween: true, markers }).markers).toEqual(markers);
  expect(sceneAtTime(possession, 0.1, { tween: false, markers }).markers).toEqual(markers);
});

test("a one-frame possession gives its only frame at any time", () => {
  const possession = tinyPossession({
    t: [0],
    game_clock_s: [700],
    shot_clock_s: [24],
    players: tinyPossession().players.map((track) => ({ ...track, x: [track.x[0] ?? null], y: [track.y[0] ?? null] })),
    ball: { x: [61], y: [20], z: [4] },
    ghosts: null,
  });
  expect(sceneAtTime(possession, 3, { tween: true })).toEqual(sceneFromPossession(possession, 0));
});

test("EPV at a time is on the line between states and flat outside the curve", () => {
  const curve = { t: [0, 1, 2], value: [1, 2, 0.5] };
  expect(epvAt(curve, -1)).toBe(1);
  expect(epvAt(curve, 0)).toBe(1);
  expect(epvAt(curve, 0.5)).toBeCloseTo(1.5, 9);
  expect(epvAt(curve, 1.5)).toBeCloseTo(1.25, 9);
  expect(epvAt(curve, 2)).toBe(0.5);
  expect(epvAt(curve, 9)).toBe(0.5);
  expect(epvAt({ t: [3], value: [0.9] }, 0)).toBe(0.9);
  expect(epvAt(null, 1)).toBeNull();
  expect(epvAt({ t: [], value: [] }, 1)).toBeNull();
});

test("the playback clock advances by wall time times the speed, caps a long pause and stops at the end", () => {
  expect(advance(1, 0.1, 2, 10)).toEqual({ time: 1.2, done: false });
  expect(advance(1, 5, 1, 10)).toEqual({ time: 1.25, done: false });
  expect(advance(9.9, 0.2, 1, 10)).toEqual({ time: 10, done: true });
  expect(advance(1, -1, 1, 10)).toEqual({ time: 1, done: false });
  expect(SPEEDS.map((speed) => speed.key)).toEqual(["half", "normal", "double"]);
  expect(speedRate("half")).toBe(0.5);
  expect(speedRate("double")).toBe(2);
});
