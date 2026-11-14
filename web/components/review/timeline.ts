/**
 * Time inside a possession, for playback. Pure functions: a time in seconds from the start of the possession
 * becomes a frame index, a court scene, or a point on the EPV curve. Nothing here reads the DOM.
 *
 * The tracking is sampled five times a second. While the possession plays, the scene between two samples is
 * drawn on the straight line between them (a display choice, so motion is smooth); a paused court always shows
 * a sampled frame exactly, through sceneFromPossession.
 */
import { clampFrame, frameCount, sceneFromPossession, type CourtScene, type SceneMarker } from "@/components/court/scene";
import type { EpvCurve, Possession } from "@/lib/types/contract";

/** The time of the last frame, in seconds from the start. Zero for a possession with one frame or none. */
export function endTime(possession: Possession): number {
  const last = possession.t[possession.t.length - 1];
  return last !== undefined && Number.isFinite(last) ? Math.max(0, last) : 0;
}

/** The index of the last frame at or before `time`. Times are increasing; a time before the first is frame 0. */
export function frameAtOrBefore(times: readonly number[], time: number): number {
  if (times.length === 0 || !Number.isFinite(time)) return 0;
  let lo = 0;
  let hi = times.length - 1;
  if (time >= (times[hi] ?? 0)) return hi;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if ((times[mid] ?? 0) <= time) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/** The time of a frame, clamped into the possession. */
export function timeOfFrame(possession: Possession, frame: number): number {
  if (frameCount(possession) === 0) return 0;
  return possession.t[clampFrame(possession, frame)] ?? 0;
}

function lerp(a: number, b: number, share: number): number {
  return a + (b - a) * share;
}

/** A value between two samples. When either side is missing, the nearer sample's value, which may be null. */
function between(
  values: readonly (number | null)[],
  frame: number,
  share: number,
): number | null {
  const a = values[frame] ?? null;
  const b = values[frame + 1] ?? null;
  if (a !== null && b !== null) return lerp(a, b, share);
  return share < 0.5 ? a : b;
}

export interface TimeSceneOptions {
  /** Draw positions between samples. False shows the sampled frame at or before the time. */
  tween?: boolean;
  markers?: SceneMarker[];
}

/**
 * The court at a time. Without tweening, or exactly on a sample, it is the scene of that frame. With tweening,
 * players, ball and ghosts sit on the line between the two samples around the time.
 */
export function sceneAtTime(possession: Possession, time: number, options: TimeSceneOptions = {}): CourtScene {
  const frame = frameAtOrBefore(possession.t, time);
  const base = sceneFromPossession(possession, frame, { markers: options.markers ?? [] });
  const t0 = possession.t[frame];
  const t1 = possession.t[frame + 1];
  if (!options.tween || t0 === undefined || t1 === undefined || !(t1 > t0) || time <= t0) return base;
  const share = Math.min(1, (time - t0) / (t1 - t0));
  const tweened = {
    ...base,
    players: [] as CourtScene["players"],
    ghosts: [] as CourtScene["ghosts"],
  };
  const positions = new Map<number, { x: number; y: number }>();
  const labels = new Map(base.players.map((player) => [player.id, player]));
  for (const track of possession.players) {
    const x = between(track.x, frame, share);
    const y = between(track.y, frame, share);
    if (x === null || y === null) continue;
    positions.set(track.player_id, { x, y });
    const known = labels.get(track.player_id);
    tweened.players.push({
      id: track.player_id,
      x,
      y,
      role: track.offense ? "offense" : "defense",
      label: known?.label ?? track.jersey ?? null,
      name: track.name,
      team: track.team,
    });
  }
  for (const track of possession.ghosts?.tracks ?? []) {
    const x = between(track.x, frame, share);
    const y = between(track.y, frame, share);
    if (x === null || y === null) continue;
    tweened.ghosts.push({ id: track.player_id, x, y, from: positions.get(track.player_id) ?? null });
  }
  const bx = between(possession.ball.x, frame, share);
  const by = between(possession.ball.y, frame, share);
  tweened.ball = bx === null || by === null ? null : { x: bx, y: by, z: between(possession.ball.z, frame, share) };
  return tweened;
}

/** The EPV value at a time: on the line between the two states around it, held flat outside the curve. */
export function epvAt(curve: EpvCurve | null | undefined, time: number): number | null {
  if (!curve || curve.t.length === 0 || curve.value.length === 0) return null;
  const n = Math.min(curve.t.length, curve.value.length);
  const first = curve.t[0] ?? 0;
  if (time <= first) return curve.value[0] ?? null;
  const last = curve.t[n - 1] ?? 0;
  if (time >= last) return curve.value[n - 1] ?? null;
  const k = frameAtOrBefore(curve.t.slice(0, n), time);
  const t0 = curve.t[k] ?? 0;
  const t1 = curve.t[k + 1] ?? t0;
  const v0 = curve.value[k] ?? 0;
  const v1 = curve.value[k + 1] ?? v0;
  return t1 > t0 ? lerp(v0, v1, (time - t0) / (t1 - t0)) : v0;
}

/** The playback speeds offered, as multiples of real time. Their names live in web/content. */
export const SPEEDS = [
  { key: "half", rate: 0.5 },
  { key: "normal", rate: 1 },
  { key: "double", rate: 2 },
] as const;

export type SpeedKey = (typeof SPEEDS)[number]["key"];

export function speedRate(key: SpeedKey): number {
  return SPEEDS.find((speed) => speed.key === key)?.rate ?? 1;
}

/**
 * One step of the playback clock: the next time after `elapsed` seconds of wall time at `rate`. Returns the
 * time and whether playback reached the end. A long pause between two animation frames (a hidden tab) is
 * capped, so the court never jumps by more than a quarter of a second.
 */
export function advance(time: number, elapsed: number, rate: number, end: number): { time: number; done: boolean } {
  const step = Math.min(Math.max(0, elapsed), 0.25) * rate;
  const next = time + step;
  if (next >= end) return { time: end, done: true };
  return { time: next, done: false };
}
