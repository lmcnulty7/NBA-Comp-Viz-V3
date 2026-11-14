import { expect, test, type Locator, type Page } from "@playwright/test";
import type { CourtCanvasElement, CourtState } from "@/components/court/Court";
import type { DrawnMark } from "@/components/court/draw";
import { copy } from "@/lib/copy";
import { build, courtCase, tokenChannels } from "./site";

/**
 * The court renderer against the Python court reference (export/court_reference.py): the browser draws one
 * possession of the data root at one frame, and every position the renderer used must equal the position the
 * reference computes for the same canvas size, within half a pixel. This is the snapshot test of PLAN.md M3.
 * It runs on whatever data root the site was built from, so it covers a real possession when SITE_DATA_DIR
 * points at real data.
 */

const HALF_PIXEL = 0.5;

// The public site holds no player position (PLAN.md D60), so it draws no possession to compare: these tests run on
// the private build. The still court of the public site is tested in review.spec.ts.
test.skip(build.variant === "public", "the public site draws no possession (PLAN.md D60); this runs on the private build");

interface ReferencePoint {
  x_ft: number | null;
  y_ft: number | null;
  px: number | null;
  py: number | null;
}

interface Reference {
  transform_version: number;
  frame: number;
  n_frames: number;
  transform: { view: string; width: number; height: number; scale: number; left: number; top: number; x0: number };
  players: (ReferencePoint & { player_id: number; offense: boolean })[];
  ball: ReferencePoint;
  ghosts: (ReferencePoint & { player_id: number })[];
  landmarks: Record<string, ReferencePoint>;
}

async function pythonReference(file: string, frame: number, width: number, height: number, view: string): Promise<Reference> {
  const { runPythonModule } = (await import("../../scripts/lib/python.mjs")) as unknown as {
    runPythonModule(module: string, args: string[]): { status: number; stdout: string; stderr: string };
  };
  const result = runPythonModule("export.court_reference", [
    "--possession", file, "--frame", String(frame), "--width", String(width), "--height", String(height), "--view", view,
  ]);
  if (result.status !== 0) throw new Error(`export.court_reference failed: ${result.stderr}`);
  return JSON.parse(result.stdout) as Reference;
}

interface Snapshot {
  width: number;
  height: number;
  ratio: number;
  canvasWidth: number;
  canvasHeight: number;
  state: CourtState;
}

async function snapshot(canvas: Locator): Promise<Snapshot> {
  return canvas.evaluate((element) => {
    const rect = element.getBoundingClientRect();
    const court = element as CourtCanvasElement;
    return {
      width: rect.width,
      height: rect.height,
      ratio: window.devicePixelRatio,
      canvasWidth: court.width,
      canvasHeight: court.height,
      state: court.courtState as CourtState,
    };
  });
}

async function openPossession(page: Page, frame: number): Promise<Locator> {
  const target = courtCase();
  await page.goto(`review/${target.gameSlug}/${target.possession}/?frame=${frame}`);
  const canvas = page.locator('canvas[data-court][data-ready="true"]');
  await expect(canvas).toBeVisible();
  await expect(page.getByTestId("possession-stage")).toHaveAttribute("aria-busy", "false");
  await expect(page.getByTestId("frame-value")).toContainText(
    copy.review.possession.court.frameValue.split(",")[0]!.replace("{frame}", String(frame + 1)).replace("{total}", String(target.frames)),
  );
  // Wait for the fonts, so the last draw is the settled one.
  await page.evaluate(() => document.fonts.ready);
  return canvas;
}

function near(actual: { x: number; y: number }, expected: ReferencePoint, what: string): void {
  expect(expected.px, `${what} has no reference position`).not.toBeNull();
  expect(Math.abs(actual.x - (expected.px as number)), `${what} x`).toBeLessThanOrEqual(HALF_PIXEL);
  expect(Math.abs(actual.y - (expected.py as number)), `${what} y`).toBeLessThanOrEqual(HALF_PIXEL);
}

function compare(shot: Snapshot, reference: Reference): void {
  const { state } = shot;
  expect(reference.transform_version).toBe(1);
  // The renderer drew on the canvas the page shows: its size is the size of the element.
  expect(Math.abs(state.transform.width - shot.width)).toBeLessThanOrEqual(HALF_PIXEL);
  expect(Math.abs(state.transform.height - shot.height)).toBeLessThanOrEqual(HALF_PIXEL);
  expect(state.transform.view).toBe(reference.transform.view);
  expect(state.transform.x0).toBe(reference.transform.x0);
  expect(state.transform.scale).toBeCloseTo(reference.transform.scale, 3);

  const byKind = (kind: DrawnMark["kind"]) => state.marks.filter((mark) => mark.kind === kind);
  const tracked = reference.players.filter((player) => player.px !== null);
  expect(tracked.length, "the reference frame has no tracked player").toBeGreaterThan(0);
  expect(byKind("player")).toHaveLength(tracked.length);
  for (const player of tracked) {
    const mark = byKind("player").find((candidate) => candidate.id === player.player_id);
    expect(mark, `player ${player.player_id} was not drawn`).toBeDefined();
    near(mark!, player, `player ${player.player_id}`);
    expect(mark!.role).toBe(player.offense ? "offense" : "defense");
  }
  const ball = byKind("ball");
  if (reference.ball.px === null) {
    expect(ball).toHaveLength(0);
  } else {
    expect(ball).toHaveLength(1);
    near(ball[0]!, reference.ball, "ball");
  }
  const ghosts = reference.ghosts.filter((ghost) => ghost.px !== null);
  expect(byKind("ghost")).toHaveLength(ghosts.length);
  for (const ghost of ghosts) {
    const mark = byKind("ghost").find((candidate) => candidate.id === ghost.player_id);
    expect(mark, `ghost ${ghost.player_id} was not drawn`).toBeDefined();
    near(mark!, ghost, `ghost ${ghost.player_id}`);
  }
  expect(Object.keys(state.landmarks).sort()).toEqual(Object.keys(reference.landmarks).sort());
  for (const [name, point] of Object.entries(reference.landmarks)) near(state.landmarks[name]!, point, name);
}

test("the court draws a possession where the Python reference says, in the full and the half view", async ({ page }) => {
  const target = courtCase();
  const frame = Math.floor(target.frames / 2);
  const canvas = await openPossession(page, frame);

  const full = await snapshot(canvas);
  compare(full, await pythonReference(target.file, frame, full.width, full.height, "full"));
  expect(full.state.transform.view).toBe("full");

  await page.getByRole("radio", { name: copy.review.possession.court.half }).check();
  await expect(page.locator('canvas[data-court="half"]')).toBeVisible();
  await expect
    .poll(async () => (await snapshot(page.locator("canvas[data-court]"))).state.transform.view)
    .toBe("half");
  const half = await snapshot(page.locator("canvas[data-court]"));
  expect(half.height).toBeGreaterThan(full.height);
  compare(half, await pythonReference(target.file, frame, half.width, half.height, "half"));
});

test("the reference holds at the first and the last frame too", async ({ page }) => {
  const target = courtCase();
  for (const frame of [0, target.frames - 1]) {
    const canvas = await openPossession(page, frame);
    const shot = await snapshot(canvas);
    compare(shot, await pythonReference(target.file, frame, shot.width, shot.height, "full"));
  }
});

test("the backing store follows the device pixel ratio and the ink is where the marks are", async ({ page }) => {
  const target = courtCase();
  const canvas = await openPossession(page, Math.floor(target.frames / 2));
  const shot = await snapshot(canvas);

  expect(shot.canvasWidth).toBe(Math.round(shot.width * shot.ratio));
  expect(shot.canvasHeight).toBe(Math.round(shot.height * shot.ratio));
  expect(shot.state.backingWidth).toBe(shot.canvasWidth);
  expect(shot.state.pixelRatioX).toBeCloseTo(shot.ratio, 1);
  // The canvas has the shape of the view, so the court fills it with no empty band.
  expect(shot.width / shot.height).toBeCloseTo(100 / 56, 1);

  // Read the pixels the canvas really holds: the ground in a corner, and the colour of the role on each disc.
  const players = shot.state.marks.filter((mark) => mark.kind === "player");
  const probes = players.map((mark) => {
    // A point inside the disc: its centre, or above the jersey number when the disc is large enough to hold one.
    const point = { x: mark.x, y: mark.r >= 8 ? mark.y - mark.r * 0.72 : mark.y };
    const later = shot.state.marks.slice(shot.state.marks.indexOf(mark) + 1);
    const covered = later.some((other) => Math.hypot(other.x - point.x, other.y - point.y) <= other.r + 2.5);
    return { ...point, role: mark.role, covered };
  });
  const pixels = await canvas.evaluate(
    (element, input) => {
      const context = (element as HTMLCanvasElement).getContext("2d")!;
      const read = (x: number, y: number) =>
        Array.from(context.getImageData(Math.round(x * input.ratioX), Math.round(y * input.ratioY), 1, 1).data.slice(0, 3));
      return { corner: read(2, 2), discs: input.points.map((point) => read(point.x, point.y)) };
    },
    { ratioX: shot.state.pixelRatioX, ratioY: shot.state.pixelRatioY, points: probes },
  );
  const close = (a: number[], b: number[]) => a.every((channel, index) => Math.abs(channel - b[index]!) <= 6);
  expect(close(pixels.corner, await tokenChannels("light", "--ground")), `corner is ${pixels.corner}`).toBe(true);
  const colours = { offense: await tokenChannels("light", "--offense"), defense: await tokenChannels("light", "--defense") };
  let checked = 0;
  probes.forEach((probe, index) => {
    if (probe.covered || (probe.role !== "offense" && probe.role !== "defense")) return;
    checked += 1;
    expect(close(pixels.discs[index]!, colours[probe.role]), `a ${probe.role} disc is ${pixels.discs[index]}`).toBe(true);
  });
  expect(checked, "too few discs were clear of other marks to check").toBeGreaterThanOrEqual(3);
});

test("the frame slider moves the court with the keyboard", async ({ page }) => {
  const target = courtCase();
  const frame = Math.floor(target.frames / 2);
  const canvas = await openPossession(page, frame);
  const before = await snapshot(canvas);
  const slider = page.getByRole("slider", { name: copy.review.possession.court.frame });
  await expect(slider).toHaveValue(String(frame));
  await slider.focus();
  await page.keyboard.press("ArrowRight");
  await expect(slider).toHaveValue(String(frame + 1));
  await expect.poll(async () => (await snapshot(canvas)).state.draws).toBeGreaterThan(before.state.draws);
  const after = await snapshot(canvas);
  compare(after, await pythonReference(target.file, frame + 1, after.width, after.height, "full"));
  await page.keyboard.press("Home");
  await expect(slider).toHaveValue("0");
});

test("the court has a text alternative: a name, and the positions as a table", async ({ page }) => {
  const target = courtCase();
  const canvas = await openPossession(page, 0);
  await expect(canvas).toHaveAttribute("role", "img");
  const name = await canvas.getAttribute("aria-label");
  expect(name).toContain(String(target.possession));
  expect(name!.length).toBeGreaterThan(20);
  const details = page.locator("details", { hasText: copy.review.possession.positions.title });
  await details.locator("summary").click();
  const reference = await pythonReference(target.file, 0, 1000, 560, "full");
  await expect(details.locator("tbody tr")).toHaveCount(reference.players.length + 1);
  await expect(page.locator("canvas[data-court]")).toHaveCount(1);
  expect(build.variant).toMatch(/^(public|private)$/);
});
