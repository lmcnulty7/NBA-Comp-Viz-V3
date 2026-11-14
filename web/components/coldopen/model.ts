/**
 * What the cold open (and the slides that reuse it) needs from the site data, worked out once on the server.
 * Pure: the page passes the data in, and the components receive plain props. No number is computed here; values
 * are copied or turned into text by lib/tokens.ts.
 */
import type { CourtScene } from "@/components/court/scene";
import { frameAtTime, sceneFromDemoFrame } from "@/components/court/scene";
import { copy, fill } from "@/lib/copy";
import { renderLine, showValue, type TokenData } from "@/lib/tokens";
import type {
  DemoFrame,
  DemoScene,
  DemoSwap,
  LaneResult,
  Scorecard,
  ScorecardCell,
  ScorecardRow,
  SwapItem,
} from "@/lib/types/contract";

const ui = copy.home.ui;
const open = copy.home.coldOpen;

/** The scene the cold open plays: the first scene of demo/frames.json (docs/SCRIPT.md, Tokens). */
export function storyScene(data: Partial<TokenData>): DemoScene | null {
  return data.frames?.scenes[0] ?? null;
}

/** The frame closest to the still of the scene: the frame the boxes are drawn on. */
export function stillFrameIndex(scene: DemoScene): number {
  return frameAtTime(
    scene.frames.map((frame) => frame.t),
    scene.still_t,
  );
}

/** The broadcast positions of a frame on the court; the sensor positions when `truth` is on. */
export function broadcastCourt(frame: DemoFrame | undefined, truth: boolean): CourtScene {
  if (!frame) return { players: [], ball: null, ghosts: [], truth: [], markers: [] };
  const scene = sceneFromDemoFrame(frame);
  return truth ? scene : { ...scene, truth: [] };
}

/** The sensor positions of a frame as players: the left court of the comparison. */
export function sensorCourt(frame: DemoFrame | undefined): CourtScene {
  if (!frame) return { players: [], ball: null, ghosts: [], truth: [], markers: [] };
  const scene = sceneFromDemoFrame({ ...frame, broadcast: frame.truth, truth: [] });
  return scene;
}

/**
 * The positions the comparison ran on (D27): the matched broadcast points, and every sensor player the
 * broadcast has no position for drawn hollow at the sensor position (the players filled in from the sensor
 * data). A broadcast point that matched no player (player_id null) is left out, as the comparison drops it.
 */
export function filledCourt(frame: DemoFrame | undefined): CourtScene {
  if (!frame) return { players: [], ball: null, ghosts: [], truth: [], markers: [] };
  const matched = frame.broadcast.filter((point) => point.player_id != null);
  const tracked = new Set(matched.map((point) => point.player_id));
  const scene = sceneFromDemoFrame({ ...frame, broadcast: matched, truth: [] });
  const filled = frame.truth
    .filter((point) => point.player_id == null || !tracked.has(point.player_id))
    .map((point) => ({ id: point.player_id ?? null, x: point.x, y: point.y, from: null }));
  return { ...scene, truth: filled };
}

// ------------------------------------------------------------------------------------------- scorecard bars

export type Tone = "pass" | "fail" | "warn" | "plain";

export interface Bar {
  key: string;
  label: string;
  /** The value as the page shows it: the stored share as a percentage. */
  value: string;
  /** The bar length, 0 to 1. Only the drawing reads it. */
  share: number;
  target: string;
  /** Where the target line sits, 0 to 1, when the target text names one share. */
  targetShare: number | null;
  status: string;
  tone: Tone;
}

function heldoutCell(row: ScorecardRow | undefined): ScorecardCell | undefined {
  const cells = row?.cells.filter((cell) => cell.split === "heldout_game") ?? [];
  return cells.length === 1 ? cells[0] : undefined;
}

/** A target such as "<= 5%" or ">= 95%" read as the share it names; null for any other wording. */
export function targetShareOf(target: string): number | null {
  const match = /^\s*(?:<=|>=|<|>|≤|≥)\s*([0-9]+(?:\.[0-9]+)?)\s*%/.exec(target);
  return match ? Number(match[1]) / 100 : null;
}

export function statusText(status: ScorecardCell["status"], metLabel: string): { text: string; tone: Tone } {
  const labels = copy.results.scorecard.status;
  if (status === "met") return { text: metLabel, tone: "pass" };
  if (status === "not met") return { text: open.boxes.labels.notMet, tone: "fail" };
  return { text: labels[status], tone: "plain" };
}

/** The three rows of scene 2 (missed players, unmatched boxes, team labels) for the held-out game. */
export function scorecardBars(scorecard: Scorecard): Bar[] {
  const adopted = scorecard.builds.find((build) => build.role === "adopted");
  const rows: [string, string, string][] = [
    ["missed_players", "rate", open.boxes.labels.missed],
    ["ghost_boxes", "rate", open.boxes.labels.ghosts],
    ["team_labels", "accuracy", open.boxes.labels.teams],
  ];
  const bars: Bar[] = [];
  for (const [key, field, label] of rows) {
    const row = adopted?.rows.find((candidate) => candidate.key === key);
    const cell = heldoutCell(row);
    const value = cell?.values[field];
    if (!row || !cell || value === undefined) continue;
    const status = statusText(cell.status, copy.results.scorecard.status.met);
    bars.push({
      key,
      label,
      value: fill(ui.percent, { value: showValue(value, "share as a percentage") }),
      share: Math.min(1, Math.max(0, value)),
      target: row.target,
      targetShare: targetShareOf(row.target),
      status: status.text,
      tone: status.tone,
    });
  }
  return bars;
}

// ------------------------------------------------------------------------------------- the position row

export interface PositionRow {
  name: string;
  values: string;
  target: string;
  status: string;
  tone: Tone;
  /** The typical gap of the stretch, as text, or null when the data has none. */
  stretch: string | null;
  checked: { text: string; share: number } | null;
}

/** The scorecard row of scene 4, with the gap of the stretch beside it (D50). */
export function positionRow(data: Partial<TokenData>): PositionRow | null {
  const adopted = data.scorecard?.builds.find((build) => build.role === "adopted");
  const row = adopted?.rows.find((candidate) => candidate.key === "position_error");
  const cell = heldoutCell(row);
  if (!row || !cell) return null;
  const typical = renderLine("{scorecard.adopted.heldout_game.position_error.p50_ft}", null, data);
  const most = renderLine("{scorecard.adopted.heldout_game.position_error.p90_ft}", null, data);
  const stretch = renderLine("{frames.scene.p50_error_ft}", null, data);
  const game = adopted?.games.filter((candidate) => candidate.split === "heldout_game");
  const checkedText = renderLine(
    fill(copy.home.ui.ofCount, {
      a: "{scorecard.adopted.heldout_game.testable}",
      b: "{scorecard.adopted.heldout_game.truth_frames}",
    }),
    null,
    data,
  );
  const status = statusText(cell.status, open.truth.labels.metNotSafely);
  const one = game?.length === 1 ? game[0] : undefined;
  return {
    name: open.truth.labels.row,
    values:
      typical && most ? fill(ui.feetPair, { a: typical, b: most }) : fill(ui.feet, { value: typical ?? most ?? "" }),
    target: row.target,
    status: status.text,
    tone: status.tone,
    stretch: stretch ? fill(ui.feet, { value: stretch }) : null,
    checked:
      one && checkedText && one.truth_frames > 0
        ? { text: checkedText, share: Math.min(1, one.testable / one.truth_frames) }
        : null,
  };
}

/** The readout of scene 4 for every frame of the scene, so the browser only has to pick one. */
export function gapReadouts(data: Partial<TokenData>): string[] {
  const scene = storyScene(data);
  if (!scene) return [];
  return scene.frames.map(
    (_frame, k) => renderLine(open.truth.labels.gap, null, data, { frame: k }) ?? open.truth.labels.noGap,
  );
}

// ------------------------------------------------------------------------------------- the comparison

export interface PairRow {
  key: string;
  kind: SwapItem["kind"];
  label: string;
  sensor: string;
  broadcast: string;
  gap: string;
}

function itemValue(kind: SwapItem["kind"], value: number, gap: boolean): string {
  if (kind === "xfg") {
    return gap
      ? fill(ui.percentPoints, { value: showValue(value, "probability as percentage points") })
      : fill(ui.percent, { value: showValue(value, "share as a percentage") });
  }
  if (kind === "epv") return fill(ui.points, { value: showValue(value, "points, as stored") });
  return fill(ui.feet, { value: showValue(value, "feet, as stored") });
}

/** The items of demo/swap.json in the window of the scene, each as a pair of values and the gap between them. */
export function pairRows(swap: DemoSwap | null, scene: DemoScene | null): PairRow[] {
  if (!swap || !scene) return [];
  const available = new Set(swap.summaries.filter((summary) => summary.available).map((summary) => summary.kind));
  const order: SwapItem["kind"][] = ["xfg", "epv", "ghosting"];
  return swap.items
    .filter((item) => item.window === scene.window && available.has(item.kind))
    .sort((a, b) => order.indexOf(a.kind) - order.indexOf(b.kind))
    .map((item) => ({
      key: `${item.kind}|${item.key}`,
      kind: item.kind,
      label: open.swap.labels[item.kind],
      // The average-defender item is a shift, not a position (D58): the spot the model gives from sensor
      // positions is the reference, and the gap is how far the spot moved on broadcast positions.
      sensor: item.kind === "ghosting" ? ui.pairReference : itemValue(item.kind, item.sensor, false),
      broadcast: item.kind === "ghosting" ? ui.pairMoved : itemValue(item.kind, item.broadcast, false),
      gap: itemValue(item.kind, item.abs_diff, true),
    }));
}

// ----------------------------------------------------------------------------------------- entry cards

/** The first rows of the results table for the results card: the first three lanes, with a failed one among them. */
export function previewLanes(lanes: LaneResult[], count = 3): LaneResult[] {
  const first = lanes.slice(0, count);
  if (first.some((lane) => lane.verdict_class === "failed")) return first;
  const failed = lanes.find((lane) => lane.verdict_class === "failed");
  return failed ? [...first.slice(0, count - 1), failed] : first;
}
