import { expect, test } from "@playwright/test";
import {
  COURT_LENGTH_FT,
  COURT_WIDTH_FT,
  HALF_X_FT,
  LINES,
  MARGIN_FT,
  RIM_X_FT,
  RIM_Y_FT,
  THREE_BREAK_ANGLE,
  THREE_BREAK_X_FT,
  arcPoint,
  ballRadiusPx,
  courtLines,
  courtRegion,
  courtTransform,
  landmarksPx,
  lineWidthPx,
  playerRadiusPx,
  toFeet,
  toPx,
  viewAspectRatio,
  viewRect,
  type CourtPrimitive,
} from "@/components/court/geometry";

// The same numbers as tests/test_export_court_reference.py, so both statements of the transform are pinned
// to one set of known values.
test("full view on a canvas of the same shape", () => {
  const t = courtTransform(1000, 560);
  expect([t.scale, t.left, t.top]).toEqual([10, 30, 30]);
  expect(toPx(t, 0, 0)).toEqual({ x: 30, y: 30 });
  expect(toPx(t, 94, 50)).toEqual({ x: 970, y: 530 });
  expect(toPx(t, RIM_X_FT[0], RIM_Y_FT)).toEqual({ x: 82.5, y: 280 });
  expect(toPx(t, RIM_X_FT[1], RIM_Y_FT)).toEqual({ x: 917.5, y: 280 });
  expect(toPx(t, 47, 25)).toEqual({ x: 500, y: 280 });
});

for (const [width, height, scale, left, top] of [
  [1000, 1000, 10, 30, 250],
  [2000, 560, 10, 530, 30],
  [358, 200.48, 3.58, 10.74, 10.74],
] as const) {
  test(`the region is centred and never stretched on ${width} by ${height}`, () => {
    const t = courtTransform(width, height);
    expect(t.scale).toBeCloseTo(scale, 9);
    expect(t.left).toBeCloseTo(left, 9);
    expect(t.top).toBeCloseTo(top, 9);
    const centre = toPx(t, 47, 25);
    expect(centre.x).toBeCloseTo(width / 2, 9);
    expect(centre.y).toBeCloseTo(height / 2, 9);
  });
}

test("half view shows the half the offense attacks", () => {
  const right = courtTransform(530, 560, "half", RIM_X_FT[1]);
  expect([right.x0, right.regionWidthFt, right.scale]).toEqual([47, 47, 10]);
  expect(toPx(right, 47, 0)).toEqual({ x: 30, y: 30 });
  expect(toPx(right, 94, 50)).toEqual({ x: 500, y: 530 });
  const left = courtTransform(530, 560, "half", RIM_X_FT[0]);
  expect([left.x0, left.scale]).toEqual([0, 10]);
  expect(toPx(left, RIM_X_FT[0], RIM_Y_FT)).toEqual({ x: 82.5, y: 280 });
  expect(courtRegion("half", 46.9).x0).toBe(0);
  expect(courtRegion("half", 47).x0).toBe(HALF_X_FT);
});

test("bad inputs are refused", () => {
  expect(() => courtTransform(0, 100)).toThrow();
  expect(() => courtTransform(100, -1)).toThrow();
  expect(() => courtTransform(Number.NaN, 100)).toThrow();
  expect(() => courtTransform(100, 100, "half")).toThrow(/attackX/);
});

test("a larger canvas scales every position, so device pixels are CSS pixels times the ratio", () => {
  for (const view of ["full", "half"] as const) {
    const small = courtTransform(358, 200, view, 88.75);
    const large = courtTransform(358 * 3, 200 * 3, view, 88.75);
    for (const [x, y] of [[0, 0], [5.25, 25], [61.3, 12.9], [94, 50], [-2, 53]] as const) {
      const a = toPx(small, x, y);
      const b = toPx(large, x, y);
      expect(b.x).toBeCloseTo(3 * a.x, 9);
      expect(b.y).toBeCloseTo(3 * a.y, 9);
    }
  }
});

test("pixels map back to feet", () => {
  const t = courtTransform(733, 411, "half", 5.25);
  for (const [x, y] of [[0, 0], [12.4, 33.3], [47, 50], [-1.5, 51]] as const) {
    const px = toPx(t, x, y);
    const back = toFeet(t, px.x, px.y);
    expect(back.x).toBeCloseTo(x, 9);
    expect(back.y).toBeCloseTo(y, 9);
  }
});

test("a canvas with the shape of the view has no empty band", () => {
  expect(viewAspectRatio("full")).toBeCloseTo((94 + 2 * MARGIN_FT) / (50 + 2 * MARGIN_FT), 12);
  expect(viewAspectRatio("half")).toBeCloseTo((47 + 2 * MARGIN_FT) / (50 + 2 * MARGIN_FT), 12);
  for (const view of ["full", "half"] as const) {
    const width = 640;
    const height = width / viewAspectRatio(view);
    const rect = viewRect(courtTransform(width, height, view, 5.25));
    expect(rect.x).toBeCloseTo(0, 9);
    expect(rect.y).toBeCloseTo(0, 9);
    expect(rect.width).toBeCloseTo(width, 9);
    expect(rect.height).toBeCloseTo(height, 9);
  }
  // On another shape the widened region is centred, with a band on two sides.
  const band = viewRect(courtTransform(1000, 1000));
  expect([band.x, band.y, band.width, band.height]).toEqual([0, 220, 1000, 560]);
});

test("landmarks are the rims, the centre and the four corners", () => {
  const marks = landmarksPx(courtTransform(1000, 560));
  expect(Object.keys(marks).sort()).toEqual(
    ["centre", "corner_x0_y0", "corner_x0_y50", "corner_x94_y0", "corner_x94_y50", "rim_left", "rim_right"].sort(),
  );
  expect(marks.rim_left).toEqual({ x: 82.5, y: 280 });
  expect(marks.corner_x94_y50).toEqual({ x: 970, y: 530 });
});

// ------------------------------------------------------------------------------------------------ court lines

const lines = courtLines();
type Arc = Extract<CourtPrimitive, { kind: "arc" }>;
const arcs = lines.filter((primitive): primitive is Arc => primitive.kind === "arc");
const straight = lines.filter((primitive) => primitive.kind === "line");

test("the court is 94 by 50 feet with rims 5.25 feet from each baseline", () => {
  expect([COURT_LENGTH_FT, COURT_WIDTH_FT, RIM_Y_FT]).toEqual([94, 50, 25]);
  expect(RIM_X_FT).toEqual([5.25, 88.75]);
  expect(RIM_X_FT[0] + RIM_X_FT[1]).toBe(COURT_LENGTH_FT);
  const rims = arcs.filter((arc) => arc.r === LINES.rimRadius);
  expect(rims.map((arc) => [arc.cx, arc.cy]).sort()).toEqual([[5.25, 25], [88.75, 25]]);
  const boundary = straight.filter((line) => line.weight === "boundary");
  expect(boundary).toHaveLength(4);
});

test("the three-point line is 22 feet in the corners and 23.75 feet on the arc, and its parts meet", () => {
  expect(THREE_BREAK_X_FT).toBeCloseTo(5.25 + Math.sqrt(23.75 ** 2 - 22 ** 2), 12);
  expect(THREE_BREAK_X_FT).toBeCloseTo(14.1978, 3);
  const threes = arcs.filter((arc) => arc.r === LINES.threeRadius);
  expect(threes).toHaveLength(2);
  for (const arc of threes) {
    // The arc is centred on a rim, and both of its ends sit on a corner line, 3 feet from a sideline.
    expect(RIM_X_FT).toContain(arc.cx);
    expect(arc.cy).toBe(RIM_Y_FT);
    for (const angle of [arc.start, arc.end]) {
      const end = arcPoint(arc, angle);
      expect(Math.min(Math.abs(end.y - 3), Math.abs(end.y - 47))).toBeLessThan(1e-9);
      const breakX = arc.cx < HALF_X_FT ? THREE_BREAK_X_FT : COURT_LENGTH_FT - THREE_BREAK_X_FT;
      expect(end.x).toBeCloseTo(breakX, 9);
      const joins = straight.some(
        (line) => Math.abs(line.x2 - end.x) < 1e-9 && Math.abs(line.y2 - end.y) < 1e-9,
      );
      expect(joins).toBe(true);
    }
    // The arc bulges toward the centre of the court.
    const middle = arcPoint(arc, (arc.start + arc.end) / 2);
    expect(Math.abs(middle.x - HALF_X_FT)).toBeLessThan(Math.abs(arc.cx - HALF_X_FT));
    expect(middle.y).toBeCloseTo(25, 9);
  }
  expect(THREE_BREAK_ANGLE).toBeCloseTo(Math.asin(22 / 23.75), 12);
});

test("the lane is 16 feet wide and 19 feet deep, with a 6 foot free-throw circle", () => {
  const laneEnds = straight.filter((line) => line.x1 === line.x2 && (line.x1 === 19 || line.x1 === 75));
  expect(laneEnds.map((line) => [line.x1, Math.min(line.y1, line.y2), Math.max(line.y1, line.y2)]).sort()).toEqual([
    [19, 17, 33],
    [75, 17, 33],
  ]);
  const circles = arcs.filter((arc) => arc.r === LINES.freeThrowCircleRadius && arc.cx !== HALF_X_FT);
  expect(new Set(circles.map((arc) => arc.cx))).toEqual(new Set([19, 75]));
  // Each free-throw circle is one solid half and one dashed half.
  for (const cx of [19, 75]) {
    const halves = circles.filter((arc) => arc.cx === cx);
    expect(halves.map((arc) => arc.dashed).sort()).toEqual([false, true]);
    for (const half of halves) expect(half.end - half.start).toBeCloseTo(Math.PI, 12);
    // The dashed half is the one inside the lane, toward the baseline.
    const dashed = halves.find((arc) => arc.dashed)!;
    const middle = arcPoint(dashed, (dashed.start + dashed.end) / 2);
    expect(Math.abs(middle.x - HALF_X_FT)).toBeGreaterThan(Math.abs(cx - HALF_X_FT));
  }
});

test("the two ends of the court mirror each other", () => {
  const key = (primitive: CourtPrimitive, flip: boolean): string => {
    const x = (value: number) => (flip ? COURT_LENGTH_FT - value : value).toFixed(6);
    if (primitive.kind === "line") {
      const ends = [`${x(primitive.x1)},${primitive.y1.toFixed(6)}`, `${x(primitive.x2)},${primitive.y2.toFixed(6)}`];
      return `line ${ends.sort().join(" ")}`;
    }
    // A full circle is named by its centre and its radius; any other arc by the points at its two ends as well.
    if (primitive.end - primitive.start >= 2 * Math.PI - 1e-9) {
      return `circle ${x(primitive.cx)} ${primitive.cy} ${primitive.r} ${primitive.dashed}`;
    }
    const ends = [primitive.start, primitive.end]
      .map((angle) => arcPoint(primitive, angle))
      .map((point) => `${x(point.x)},${point.y.toFixed(6)}`);
    return `arc ${x(primitive.cx)} ${primitive.r} ${ends.sort().join(" ")} ${primitive.dashed}`;
  };
  const plain = lines.map((primitive) => key(primitive, false)).sort();
  const flipped = lines.map((primitive) => key(primitive, true)).sort();
  expect(flipped).toEqual(plain);
});

test("every line stays on the court", () => {
  for (const primitive of lines) {
    const points =
      primitive.kind === "line"
        ? [{ x: primitive.x1, y: primitive.y1 }, { x: primitive.x2, y: primitive.y2 }]
        : Array.from({ length: 33 }, (_, step) =>
            arcPoint(primitive, primitive.start + ((primitive.end - primitive.start) * step) / 32),
          );
    for (const point of points) {
      expect(point.x).toBeGreaterThanOrEqual(-1e-9);
      expect(point.x).toBeLessThanOrEqual(COURT_LENGTH_FT + 1e-9);
      expect(point.y).toBeGreaterThanOrEqual(-1e-9);
      expect(point.y).toBeLessThanOrEqual(COURT_WIDTH_FT + 1e-9);
    }
  }
});

test("marks keep a readable size on a phone and do not balloon on a wide screen", () => {
  expect(playerRadiusPx(1)).toBe(4.5);
  expect(playerRadiusPx(3.58)).toBeCloseTo(4.654, 3);
  expect(playerRadiusPx(8)).toBeCloseTo(10.4, 9);
  expect(playerRadiusPx(40)).toBe(13);
  expect(ballRadiusPx(1)).toBe(2.5);
  expect(ballRadiusPx(40)).toBe(6);
  for (const scale of [1, 3.58, 11, 40]) {
    expect(ballRadiusPx(scale)).toBeLessThan(playerRadiusPx(scale));
    expect(lineWidthPx("detail", scale)).toBeLessThanOrEqual(lineWidthPx("line", scale));
    expect(lineWidthPx("line", scale)).toBeLessThanOrEqual(lineWidthPx("boundary", scale));
    expect(lineWidthPx("boundary", scale)).toBeLessThanOrEqual(2.2);
  }
});
