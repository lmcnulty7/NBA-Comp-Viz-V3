/**
 * A court scene: what the renderer draws at one moment, in feet. Pure functions turn contract data into a
 * scene (a possession at a frame, a demo frame); the renderer knows nothing about the contract.
 */
import type { DemoFrame, PlayerTrack, Possession } from "@/lib/types/contract";
import type { Point } from "./geometry";

export type PlayerRole = "offense" | "defense" | "neutral";

export interface ScenePlayer {
  id: number | null;
  x: number;
  y: number;
  role: PlayerRole;
  /** Short text drawn inside the disc when it is large enough: the jersey number. */
  label: string | null;
  name: string | null;
  team: string | null;
}

export interface SceneBall {
  x: number;
  y: number;
  /** Height above the floor in feet, when the data has it. */
  z: number | null;
}

/** Where the model expects a defender to stand. `from` is the real position of the same defender. */
export interface SceneGhost {
  id: number;
  x: number;
  y: number;
  from: Point | null;
}

/** A sensor position shown over a tracked one. `from` is the tracked position of the same player. */
export interface SceneTruth {
  id: number | null;
  x: number;
  y: number;
  from: Point | null;
}

/** A highlighted player or spot: the marker that follows one thing across views. */
export type SceneMarker = { kind: "player"; id: number } | { kind: "spot"; x: number; y: number };

export interface CourtScene {
  players: ScenePlayer[];
  ball: SceneBall | null;
  ghosts: SceneGhost[];
  truth: SceneTruth[];
  markers: SceneMarker[];
}

export const EMPTY_SCENE: CourtScene = { players: [], ball: null, ghosts: [], truth: [], markers: [] };

export function frameCount(possession: Possession): number {
  return possession.t.length;
}

/** A frame index inside the possession: whole, and between the first and the last frame. */
export function clampFrame(possession: Possession, frame: number): number {
  const last = Math.max(0, frameCount(possession) - 1);
  if (!Number.isFinite(frame)) return 0;
  return Math.min(last, Math.max(0, Math.round(frame)));
}

/** The index of the frame closest to time `t` (seconds from the start of the possession). */
export function frameAtTime(times: readonly number[], t: number): number {
  let best = 0;
  let bestGap = Number.POSITIVE_INFINITY;
  times.forEach((time, index) => {
    const gap = Math.abs(time - t);
    if (gap < bestGap) {
      best = index;
      bestGap = gap;
    }
  });
  return best;
}

/** The frame a possession opens on: the release of its last timed shot, else the first frame. */
export function defaultFrame(possession: Possession): number {
  for (let index = possession.shots.length - 1; index >= 0; index -= 1) {
    const time = possession.shots[index]?.t;
    if (time !== null && time !== undefined) return frameAtTime(possession.t, time);
  }
  return 0;
}

/** Where the last located shot of the possession was taken, or null. */
export function lastShotSpot(possession: Possession): Point | null {
  for (let index = possession.shots.length - 1; index >= 0; index -= 1) {
    const shot = possession.shots[index];
    if (shot && shot.x != null && shot.y != null) return { x: shot.x, y: shot.y };
  }
  return null;
}

function at(values: readonly (number | null)[], frame: number): number | null {
  const value = values[frame];
  return value === undefined ? null : value;
}

/** Players with no tracking row at this frame. They are left out of the scene and named beside it. */
export function untrackedPlayers(possession: Possession, frame: number): PlayerTrack[] {
  return possession.players.filter((track) => at(track.x, frame) === null || at(track.y, frame) === null);
}

export interface PossessionSceneOptions {
  markers?: SceneMarker[];
}

/** One frame of a possession as a scene. A player or ghost with no value at this frame is left out. */
export function sceneFromPossession(
  possession: Possession,
  frame: number,
  options: PossessionSceneOptions = {},
): CourtScene {
  const index = clampFrame(possession, frame);
  const players: ScenePlayer[] = [];
  const positions = new Map<number, Point>();
  for (const track of possession.players) {
    const x = at(track.x, index);
    const y = at(track.y, index);
    if (x === null || y === null) continue;
    positions.set(track.player_id, { x, y });
    players.push({
      id: track.player_id,
      x,
      y,
      role: track.offense ? "offense" : "defense",
      label: track.jersey ?? null,
      name: track.name,
      team: track.team,
    });
  }
  const ghosts: SceneGhost[] = [];
  for (const track of possession.ghosts?.tracks ?? []) {
    const x = at(track.x, index);
    const y = at(track.y, index);
    if (x === null || y === null) continue;
    ghosts.push({ id: track.player_id, x, y, from: positions.get(track.player_id) ?? null });
  }
  const ballX = at(possession.ball.x, index);
  const ballY = at(possession.ball.y, index);
  return {
    players,
    ball: ballX === null || ballY === null ? null : { x: ballX, y: ballY, z: at(possession.ball.z, index) },
    ghosts,
    truth: [],
    markers: options.markers ?? [],
  };
}

export interface DemoSceneOptions {
  /** The team on offense, when known. Without it every tracked player is drawn in the neutral colour. */
  offenseTeam?: string | null;
  markers?: SceneMarker[];
}

/**
 * One demo frame as a scene: the broadcast positions as players, the sensor positions as the truth overlay,
 * each linked to the broadcast position of the same player.
 */
export function sceneFromDemoFrame(frame: DemoFrame, options: DemoSceneOptions = {}): CourtScene {
  const tracked = new Map<number, Point>();
  const players: ScenePlayer[] = frame.broadcast.map((point) => {
    if (point.player_id != null) tracked.set(point.player_id, { x: point.x, y: point.y });
    let role: PlayerRole = "neutral";
    if (options.offenseTeam && point.team) role = point.team === options.offenseTeam ? "offense" : "defense";
    return {
      id: point.player_id ?? null,
      x: point.x,
      y: point.y,
      role,
      label: null,
      name: null,
      team: point.team ?? null,
    };
  });
  const truth: SceneTruth[] = frame.truth.map((point) => ({
    id: point.player_id ?? null,
    x: point.x,
    y: point.y,
    from: point.player_id != null ? (tracked.get(point.player_id) ?? null) : null,
  }));
  return { players, ball: null, ghosts: [], truth, markers: options.markers ?? [] };
}
