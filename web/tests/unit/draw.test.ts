import { expect, test } from "@playwright/test";
import { DEFAULT_LAYERS, drawCourtLines, drawScene, renderCourt } from "@/components/court/draw";
import { LINES, ballRadiusPx, courtLines, courtTransform, playerRadiusPx, toPx } from "@/components/court/geometry";
import { sceneFromDemoFrame, sceneFromPossession } from "@/components/court/scene";
import { tinyDemoFrame, tinyPossession } from "../helpers/possession";
import { RecordingContext, TEST_PALETTE } from "../helpers/recording-context";

const WIDTH = 1000;
const HEIGHT = 560;

test("every mark is drawn at the transformed position of its player, ghost or ball", () => {
  const ctx = new RecordingContext();
  const transform = courtTransform(WIDTH, HEIGHT);
  const possession = tinyPossession();
  const scene = sceneFromPossession(possession, 1);
  const marks = drawScene(ctx, transform, scene, TEST_PALETTE);

  const radius = playerRadiusPx(transform.scale);
  const players = marks.filter((mark) => mark.kind === "player");
  expect(players.map((mark) => mark.id).sort()).toEqual([1, 11, 12, 2]);
  for (const mark of players) {
    const player = scene.players.find((candidate) => candidate.id === mark.id)!;
    expect({ x: mark.x, y: mark.y }).toEqual(toPx(transform, player.x, player.y));
    expect(mark.r).toBe(radius);
    expect(mark.role).toBe(player.role);
    // The disc itself: an arc at that centre, filled in the colour of the role.
    const arc = ctx.calls.find(
      (call) => call.name === "arc" && call.args[0] === mark.x && call.args[1] === mark.y && call.fillStyle === player.role,
    );
    expect(arc?.args[2]).toBe(radius);
  }
  const ball = marks.find((mark) => mark.kind === "ball")!;
  expect({ x: ball.x, y: ball.y }).toEqual(toPx(transform, 62, 21));
  expect(ball.r).toBe(ballRadiusPx(transform.scale));
  const ghosts = marks.filter((mark) => mark.kind === "ghost");
  expect(ghosts.map((mark) => [mark.id, mark.x, mark.y])).toEqual([
    [11, toPx(transform, 66, 22.5).x, toPx(transform, 66, 22.5).y],
    [12, toPx(transform, 75, 30).x, toPx(transform, 75, 30).y],
  ]);
});

test("defenders are drawn before attackers and the ball is drawn last", () => {
  const ctx = new RecordingContext();
  const marks = drawScene(ctx, courtTransform(WIDTH, HEIGHT), sceneFromPossession(tinyPossession(), 1), TEST_PALETTE);
  const order = marks.map((mark) => (mark.kind === "player" ? mark.role : mark.kind));
  expect(order).toEqual(["ghost", "ghost", "defense", "defense", "offense", "offense", "ball"]);
});

test("a ghost is a dashed ring in the ghost colour, linked to its real defender", () => {
  const ctx = new RecordingContext();
  const transform = courtTransform(WIDTH, HEIGHT);
  drawScene(ctx, transform, sceneFromPossession(tinyPossession(), 1), TEST_PALETTE);
  const centre = toPx(transform, 66, 22.5);
  const ring = ctx.calls.find((call) => call.name === "arc" && call.args[0] === centre.x && call.args[1] === centre.y)!;
  expect(ring.strokeStyle).toBe("ghost");
  expect(ring.dash.length).toBe(2);
  const from = toPx(transform, 65, 22);
  const link = ctx.calls.findIndex((call) => call.name === "moveTo" && call.args[0] === from.x && call.args[1] === from.y);
  expect(ctx.calls[link + 1]).toMatchObject({ name: "lineTo", args: [centre.x, centre.y], strokeStyle: "ghost", dash: [] });
});

test("layers switch the ghosts, the truth overlay, the labels and the markers off", () => {
  const transform = courtTransform(WIDTH, HEIGHT);
  const scene = sceneFromPossession(tinyPossession(), 1, { markers: [{ kind: "player", id: 1 }, { kind: "spot", x: 62, y: 22 }] });
  const all = drawScene(new RecordingContext(), transform, scene, TEST_PALETTE);
  expect(all.filter((mark) => mark.kind === "marker").map((mark) => mark.id)).toEqual([1, null]);
  const bare = new RecordingContext();
  const marks = drawScene(bare, transform, scene, TEST_PALETTE, { ghosts: false, truth: false, labels: false, markers: false });
  expect(marks.map((mark) => mark.kind)).toEqual(["player", "player", "player", "player", "ball"]);
  expect(bare.named("fillText")).toHaveLength(0);
  expect(bare.calls.some((call) => call.strokeStyle === "ghost" || call.strokeStyle === "marker")).toBe(false);
  expect(DEFAULT_LAYERS).toEqual({ ghosts: true, truth: true, labels: true, markers: true });
});

test("a marker follows its player and is skipped when the player is not in the scene", () => {
  const transform = courtTransform(WIDTH, HEIGHT);
  const scene = sceneFromPossession(tinyPossession(), 1, { markers: [{ kind: "player", id: 11 }, { kind: "player", id: 404 }] });
  const markers = drawScene(new RecordingContext(), transform, scene, TEST_PALETTE).filter((mark) => mark.kind === "marker");
  expect(markers).toHaveLength(1);
  expect({ x: markers[0]!.x, y: markers[0]!.y }).toEqual(toPx(transform, 65, 22));
  expect(markers[0]!.r).toBeGreaterThan(playerRadiusPx(transform.scale));
});

test("jersey numbers are drawn only when the disc is large enough to hold them", () => {
  const scene = sceneFromPossession(tinyPossession(), 1);
  const large = new RecordingContext();
  drawScene(large, courtTransform(WIDTH, HEIGHT), scene, TEST_PALETTE);
  const texts = large.named("fillText");
  expect(texts.map((call) => call.args[0]).sort()).toEqual(["12", "30", "4"]);
  expect(texts.find((call) => call.args[0] === "4")?.fillStyle).toBe("offense-on");
  expect(texts.find((call) => call.args[0] === "12")?.fillStyle).toBe("defense-on");
  expect(texts[0]?.font).toContain("Test Sans");
  const small = new RecordingContext();
  drawScene(small, courtTransform(358, 200.48), scene, TEST_PALETTE);
  expect(small.named("fillText")).toHaveLength(0);
});

test("the truth overlay marks each sensor position and links it to the tracked one", () => {
  const ctx = new RecordingContext();
  const transform = courtTransform(WIDTH, HEIGHT);
  const marks = drawScene(ctx, transform, sceneFromDemoFrame(tinyDemoFrame()), TEST_PALETTE);
  const truth = marks.filter((mark) => mark.kind === "truth");
  expect(truth.map((mark) => mark.id)).toEqual([1, 99]);
  expect({ x: truth[0]!.x, y: truth[0]!.y }).toEqual(toPx(transform, 61.5, 20.5));
  const from = toPx(transform, 60, 20);
  const link = ctx.calls.findIndex((call) => call.name === "moveTo" && call.args[0] === from.x && call.args[1] === from.y);
  expect(ctx.calls[link + 1]).toMatchObject({ name: "lineTo", args: [truth[0]!.x, truth[0]!.y], strokeStyle: "truth" });
  expect(marks.filter((mark) => mark.kind === "player").every((mark) => mark.role === "neutral")).toBe(true);
});

test("the court lines are drawn thin, in one colour, with the dashed halves dashed", () => {
  const ctx = new RecordingContext();
  const transform = courtTransform(WIDTH, HEIGHT);
  drawCourtLines(ctx, transform, TEST_PALETTE);
  const primitives = courtLines();
  expect(ctx.named("stroke")).toHaveLength(primitives.length);
  expect(ctx.named("stroke").every((call) => call.strokeStyle === "line" && call.lineWidth <= 2.2)).toBe(true);
  const arcs = ctx.named("arc");
  expect(arcs).toHaveLength(primitives.filter((primitive) => primitive.kind === "arc").length);
  // Both three-point arcs: centred on a rim, with a radius of 23.75 ft at this scale.
  const threes = arcs.filter((call) => call.args[2] === LINES.threeRadius * transform.scale);
  expect(threes.map((call) => [call.args[0], call.args[1]])).toEqual([
    [toPx(transform, 5.25, 25).x, 280],
    [toPx(transform, 88.75, 25).x, 280],
  ]);
  expect(arcs.filter((call) => call.dash.length > 0)).toHaveLength(2);
});

test("renderCourt scales to the backing store, paints the ground, clips to the view and reports what it drew", () => {
  const ctx = new RecordingContext();
  const scene = sceneFromPossession(tinyPossession(), 1);
  const result = renderCourt(ctx, {
    width: 530,
    height: 560,
    pixelRatioX: 2,
    pixelRatioY: 2,
    view: "half",
    attackX: 88.75,
    scene,
    palette: TEST_PALETTE,
  });
  expect(ctx.calls[0]).toMatchObject({ name: "setTransform", args: [2, 0, 0, 2, 0, 0] });
  expect(ctx.named("fillRect")[0]).toMatchObject({ args: [0, 0, 530, 560], fillStyle: "ground" });
  expect(ctx.named("rect")[0]?.args).toEqual([0, 0, 530, 560]);
  expect(ctx.named("clip")).toHaveLength(1);
  expect(result.transform).toEqual(courtTransform(530, 560, "half", 88.75));
  expect(result.transform.x0).toBe(47);
  expect(result.marks.filter((mark) => mark.kind === "player")).toHaveLength(4);
  expect(result.landmarks.rim_right).toEqual(toPx(result.transform, 88.75, 25));
  // Positions are CSS pixels: the same whatever the pixel ratio.
  const again = renderCourt(new RecordingContext(), {
    width: 530,
    height: 560,
    pixelRatioX: 1,
    pixelRatioY: 1,
    view: "half",
    attackX: 88.75,
    scene,
    palette: TEST_PALETTE,
  });
  expect(again.marks).toEqual(result.marks);
});
