/**
 * Court geometry: feet to canvas pixels, and the court lines in feet. Pure functions, no DOM.
 *
 * The transform (transform version 1) is stated a second time in Python, in export/court_reference.py, and a
 * test compares the two within half a pixel. Change one and the other must change with it.
 *
 * Court positions are in feet in the contract frame (export/schemas.py): x from 0 to 94 along the court, y from
 * 0 to 50 across it, rims at (5.25, 25) and (88.75, 25). The canvas has a width and a height in CSS pixels, its
 * origin at the top left, x to the right and y down. Court x maps to canvas x and court y to canvas y: no flip,
 * no rotation.
 *
 * 1. Region. The full view shows x in [0, 94]. The half view shows the half the offense attacks: x in [0, 47]
 *    when attackX is below 47, otherwise x in [47, 94]. Both views show y in [0, 50].
 * 2. Margin. The region is widened by MARGIN_FT (3 ft) on each of its four sides, so a player who steps out of
 *    bounds is still on the canvas.
 * 3. Scale. s = min(width / (regionWidth + 2 * margin), height / (regionHeight + 2 * margin)), in pixels per
 *    foot. The widened region is as large as it can be while staying whole, with one scale on both axes.
 * 4. Centring. The widened region is centred in the canvas:
 *    left = (width - s * (regionWidth + 2 * margin)) / 2 + s * margin, and the same for top with the heights.
 * 5. Position. px = left + (x - regionX0) * s and py = top + (y - regionY0) * s. Nothing is rounded.
 *
 * Every step is linear in the canvas size, so a canvas k times larger gives positions k times larger. The
 * renderer works in CSS pixels and lets the canvas scale them to the backing store (the device pixel ratio).
 */

export const TRANSFORM_VERSION = 1;

export const COURT_LENGTH_FT = 94;
export const COURT_WIDTH_FT = 50;
export const RIM_X_FT = [5.25, 88.75] as const;
export const RIM_Y_FT = 25;
export const HALF_X_FT = COURT_LENGTH_FT / 2;
export const MARGIN_FT = 3;

export type CourtView = "full" | "half";

export interface CourtRegion {
  x0: number;
  y0: number;
  width: number;
  height: number;
}

export interface CourtTransform {
  view: CourtView;
  /** Canvas size in CSS pixels. */
  width: number;
  height: number;
  /** The court region shown, in feet. */
  x0: number;
  y0: number;
  regionWidthFt: number;
  regionHeightFt: number;
  marginFt: number;
  /** Pixels per foot. */
  scale: number;
  /** Canvas position of the region's corner (x0, y0). */
  left: number;
  top: number;
}

export interface Point {
  x: number;
  y: number;
}

/** The court region a view shows, in feet (step 1). */
export function courtRegion(view: CourtView, attackX?: number | null): CourtRegion {
  if (view === "full") {
    return { x0: 0, y0: 0, width: COURT_LENGTH_FT, height: COURT_WIDTH_FT };
  }
  if (attackX === undefined || attackX === null) {
    throw new Error("the half view needs attackX, the x of the rim the offense attacks");
  }
  return { x0: attackX < HALF_X_FT ? 0 : HALF_X_FT, y0: 0, width: HALF_X_FT, height: COURT_WIDTH_FT };
}

/** Width over height of a view with its margin: the shape a canvas should have to show it with no empty band. */
export function viewAspectRatio(view: CourtView): number {
  const region = courtRegion(view, RIM_X_FT[0]);
  return (region.width + 2 * MARGIN_FT) / (region.height + 2 * MARGIN_FT);
}

/** The transform for a canvas of `width` by `height` CSS pixels (steps 1 to 4). */
export function courtTransform(
  width: number,
  height: number,
  view: CourtView = "full",
  attackX?: number | null,
): CourtTransform {
  if (!(width > 0 && height > 0)) {
    throw new Error("the canvas width and height must be positive");
  }
  const region = courtRegion(view, attackX);
  const wideWidth = region.width + 2 * MARGIN_FT;
  const wideHeight = region.height + 2 * MARGIN_FT;
  const scale = Math.min(width / wideWidth, height / wideHeight);
  return {
    view,
    width,
    height,
    x0: region.x0,
    y0: region.y0,
    regionWidthFt: region.width,
    regionHeightFt: region.height,
    marginFt: MARGIN_FT,
    scale,
    left: (width - scale * wideWidth) / 2 + scale * MARGIN_FT,
    top: (height - scale * wideHeight) / 2 + scale * MARGIN_FT,
  };
}

/** Feet to canvas pixels (step 5). */
export function toPx(transform: CourtTransform, x: number, y: number): Point {
  return {
    x: transform.left + (x - transform.x0) * transform.scale,
    y: transform.top + (y - transform.y0) * transform.scale,
  };
}

/** Canvas pixels back to feet, for pointer input. */
export function toFeet(transform: CourtTransform, px: number, py: number): Point {
  return {
    x: transform.x0 + (px - transform.left) / transform.scale,
    y: transform.y0 + (py - transform.top) / transform.scale,
  };
}

/** The canvas rectangle of the widened region: everything outside it is the empty band of a canvas with another shape. */
export function viewRect(transform: CourtTransform): { x: number; y: number; width: number; height: number } {
  const margin = transform.marginFt * transform.scale;
  return {
    x: transform.left - margin,
    y: transform.top - margin,
    width: (transform.regionWidthFt + 2 * transform.marginFt) * transform.scale,
    height: (transform.regionHeightFt + 2 * transform.marginFt) * transform.scale,
  };
}

/** Named court points in feet. The court reference in Python reports the same ones in pixels. */
export const LANDMARKS_FT: Record<string, Point> = {
  rim_left: { x: RIM_X_FT[0], y: RIM_Y_FT },
  rim_right: { x: RIM_X_FT[1], y: RIM_Y_FT },
  centre: { x: HALF_X_FT, y: COURT_WIDTH_FT / 2 },
  corner_x0_y0: { x: 0, y: 0 },
  corner_x94_y0: { x: COURT_LENGTH_FT, y: 0 },
  corner_x94_y50: { x: COURT_LENGTH_FT, y: COURT_WIDTH_FT },
  corner_x0_y50: { x: 0, y: COURT_WIDTH_FT },
};

/** The landmarks in canvas pixels. */
export function landmarksPx(transform: CourtTransform): Record<string, Point> {
  const out: Record<string, Point> = {};
  for (const [name, point] of Object.entries(LANDMARKS_FT)) {
    out[name] = toPx(transform, point.x, point.y);
  }
  return out;
}

// ------------------------------------------------------------------------------------------- court lines (feet)

/** NBA court dimensions in feet. */
export const LINES = {
  laneDepth: 19, // baseline to the free-throw line
  laneHalfWidth: 8, // the lane is 16 ft wide
  freeThrowCircleRadius: 6,
  centreCircleRadius: 6,
  centreInnerRadius: 2,
  backboardX: 4, // the backboard is 4 ft from the baseline
  backboardHalfWidth: 3, // and 6 ft wide
  rimRadius: 0.75, // an 18 inch rim
  restrictedRadius: 4,
  threeRadius: 23.75,
  threeCornerY: 3, // the corner three is 3 ft from the sideline, 22 ft from the rim
} as const;

export type LineWeight = "boundary" | "line" | "detail";

export type CourtPrimitive =
  | { kind: "line"; x1: number; y1: number; x2: number; y2: number; weight: LineWeight }
  /** An arc drawn clockwise on the canvas from `start` to `end` (radians, 0 pointing along +x, y down). */
  | { kind: "arc"; cx: number; cy: number; r: number; start: number; end: number; weight: LineWeight; dashed: boolean };

/** Where the corner three meets the arc, as a distance from the baseline. */
export const THREE_BREAK_X_FT =
  RIM_X_FT[0] + Math.sqrt(LINES.threeRadius ** 2 - (RIM_Y_FT - LINES.threeCornerY) ** 2);

/** The angle, seen from the rim, at which the three-point arc meets the corner line. */
export const THREE_BREAK_ANGLE = Math.asin((RIM_Y_FT - LINES.threeCornerY) / LINES.threeRadius);

function leftHalf(): CourtPrimitive[] {
  const rimX = RIM_X_FT[0];
  const laneTop = RIM_Y_FT - LINES.laneHalfWidth;
  const laneBottom = RIM_Y_FT + LINES.laneHalfWidth;
  const farY = COURT_WIDTH_FT - LINES.threeCornerY;
  const half = Math.PI / 2;
  return [
    // the lane and the free-throw line
    { kind: "line", x1: 0, y1: laneTop, x2: LINES.laneDepth, y2: laneTop, weight: "line" },
    { kind: "line", x1: LINES.laneDepth, y1: laneTop, x2: LINES.laneDepth, y2: laneBottom, weight: "line" },
    { kind: "line", x1: LINES.laneDepth, y1: laneBottom, x2: 0, y2: laneBottom, weight: "line" },
    // the free-throw circle: solid toward the centre of the court, dashed inside the lane
    { kind: "arc", cx: LINES.laneDepth, cy: RIM_Y_FT, r: LINES.freeThrowCircleRadius, start: -half, end: half, weight: "line", dashed: false },
    { kind: "arc", cx: LINES.laneDepth, cy: RIM_Y_FT, r: LINES.freeThrowCircleRadius, start: half, end: 3 * half, weight: "detail", dashed: true },
    // backboard, rim and restricted area
    { kind: "line", x1: LINES.backboardX, y1: RIM_Y_FT - LINES.backboardHalfWidth, x2: LINES.backboardX, y2: RIM_Y_FT + LINES.backboardHalfWidth, weight: "line" },
    { kind: "arc", cx: rimX, cy: RIM_Y_FT, r: LINES.rimRadius, start: 0, end: 2 * Math.PI, weight: "line", dashed: false },
    { kind: "arc", cx: rimX, cy: RIM_Y_FT, r: LINES.restrictedRadius, start: -half, end: half, weight: "detail", dashed: false },
    { kind: "line", x1: LINES.backboardX, y1: RIM_Y_FT - LINES.restrictedRadius, x2: rimX, y2: RIM_Y_FT - LINES.restrictedRadius, weight: "detail" },
    { kind: "line", x1: LINES.backboardX, y1: RIM_Y_FT + LINES.restrictedRadius, x2: rimX, y2: RIM_Y_FT + LINES.restrictedRadius, weight: "detail" },
    // the three-point line: two corner lines and the arc between them
    { kind: "line", x1: 0, y1: LINES.threeCornerY, x2: THREE_BREAK_X_FT, y2: LINES.threeCornerY, weight: "line" },
    { kind: "line", x1: 0, y1: farY, x2: THREE_BREAK_X_FT, y2: farY, weight: "line" },
    { kind: "arc", cx: rimX, cy: RIM_Y_FT, r: LINES.threeRadius, start: -THREE_BREAK_ANGLE, end: THREE_BREAK_ANGLE, weight: "line", dashed: false },
  ];
}

/** The same primitive on the other half of the court (x becomes 94 - x). */
function mirrored(primitive: CourtPrimitive): CourtPrimitive {
  if (primitive.kind === "line") {
    return { ...primitive, x1: COURT_LENGTH_FT - primitive.x1, x2: COURT_LENGTH_FT - primitive.x2 };
  }
  // A clockwise arc from start to end becomes a clockwise arc from (pi - end) to (pi - start).
  return {
    ...primitive,
    cx: COURT_LENGTH_FT - primitive.cx,
    start: Math.PI - primitive.end,
    end: Math.PI - primitive.start,
  };
}

/** Every line of the court, in feet: the boundary, the centre line and circles, and both ends. */
export function courtLines(): CourtPrimitive[] {
  const left = leftHalf();
  const centreY = COURT_WIDTH_FT / 2;
  return [
    { kind: "line", x1: 0, y1: 0, x2: COURT_LENGTH_FT, y2: 0, weight: "boundary" },
    { kind: "line", x1: COURT_LENGTH_FT, y1: 0, x2: COURT_LENGTH_FT, y2: COURT_WIDTH_FT, weight: "boundary" },
    { kind: "line", x1: COURT_LENGTH_FT, y1: COURT_WIDTH_FT, x2: 0, y2: COURT_WIDTH_FT, weight: "boundary" },
    { kind: "line", x1: 0, y1: COURT_WIDTH_FT, x2: 0, y2: 0, weight: "boundary" },
    { kind: "line", x1: HALF_X_FT, y1: 0, x2: HALF_X_FT, y2: COURT_WIDTH_FT, weight: "line" },
    { kind: "arc", cx: HALF_X_FT, cy: centreY, r: LINES.centreCircleRadius, start: 0, end: 2 * Math.PI, weight: "line", dashed: false },
    { kind: "arc", cx: HALF_X_FT, cy: centreY, r: LINES.centreInnerRadius, start: 0, end: 2 * Math.PI, weight: "detail", dashed: false },
    ...left,
    ...left.map(mirrored),
  ];
}

/** The end point of an arc primitive at one of its two angles, in feet. */
export function arcPoint(arc: Extract<CourtPrimitive, { kind: "arc" }>, angle: number): Point {
  return { x: arc.cx + arc.r * Math.cos(angle), y: arc.cy + arc.r * Math.sin(angle) };
}

// ------------------------------------------------------------------------------------------- mark sizes (pixels)

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

/** Player discs are drawn larger than a real player so they read on a phone, within fixed pixel bounds. */
export function playerRadiusPx(scale: number): number {
  return clamp(1.3 * scale, 4.5, 13);
}

export function ballRadiusPx(scale: number): number {
  return clamp(0.65 * scale, 2.5, 6);
}

/** A jersey number fits inside a player disc from this radius up. */
export const LABEL_MIN_RADIUS_PX = 8;

/** Line widths in CSS pixels: thin marks at every size. */
export function lineWidthPx(weight: LineWeight, scale: number): number {
  const base = clamp(scale * 0.16, 1, 1.75);
  if (weight === "boundary") return base * 1.25;
  if (weight === "detail") return Math.max(0.75, base * 0.75);
  return base;
}
