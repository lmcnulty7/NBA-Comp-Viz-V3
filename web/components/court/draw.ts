/**
 * Drawing a court scene on a 2D canvas context. The functions here take a context and plain data, and return
 * the marks they drew with their positions in CSS pixels, so a test can compare them with the Python court
 * reference (export/court_reference.py). No DOM access: the React component in Court.tsx owns the canvas.
 */
import {
  LABEL_MIN_RADIUS_PX,
  ballRadiusPx,
  courtLines,
  courtTransform,
  landmarksPx,
  lineWidthPx,
  playerRadiusPx,
  toPx,
  viewRect,
  type CourtTransform,
  type CourtView,
  type Point,
} from "./geometry";
import type { CourtScene, PlayerRole } from "./scene";

/** The part of CanvasRenderingContext2D the renderer uses. A test passes a recording stand-in. */
export interface Ctx2D {
  fillStyle: string | CanvasGradient | CanvasPattern;
  strokeStyle: string | CanvasGradient | CanvasPattern;
  lineWidth: number;
  lineCap: CanvasLineCap;
  lineJoin: CanvasLineJoin;
  font: string;
  textAlign: CanvasTextAlign;
  textBaseline: CanvasTextBaseline;
  save(): void;
  restore(): void;
  setTransform(a: number, b: number, c: number, d: number, e: number, f: number): void;
  clearRect(x: number, y: number, w: number, h: number): void;
  fillRect(x: number, y: number, w: number, h: number): void;
  beginPath(): void;
  closePath(): void;
  rect(x: number, y: number, w: number, h: number): void;
  clip(): void;
  moveTo(x: number, y: number): void;
  lineTo(x: number, y: number): void;
  arc(x: number, y: number, radius: number, startAngle: number, endAngle: number, counterclockwise?: boolean): void;
  fill(): void;
  stroke(): void;
  setLineDash(segments: number[]): void;
  fillText(text: string, x: number, y: number): void;
}

/** Colours and the label font, read from the design tokens (app/styles/tokens.css) by the component. */
export interface CourtPalette {
  ground: string;
  line: string;
  offense: string;
  offenseOn: string;
  defense: string;
  defenseOn: string;
  neutral: string;
  neutralOn: string;
  ghost: string;
  truth: string;
  ball: string;
  marker: string;
  fontFamily: string;
}

export interface CourtLayers {
  ghosts: boolean;
  truth: boolean;
  labels: boolean;
  markers: boolean;
}

export const DEFAULT_LAYERS: CourtLayers = { ghosts: true, truth: true, labels: true, markers: true };

export type MarkKind = "player" | "ball" | "ghost" | "truth" | "marker";

/** One thing the renderer drew: what it is, whose it is, and its centre and radius in CSS pixels. */
export interface DrawnMark {
  kind: MarkKind;
  id: number | null;
  role: PlayerRole | null;
  x: number;
  y: number;
  r: number;
}

const FULL_TURN = 2 * Math.PI;

function fillOf(role: PlayerRole, palette: CourtPalette): string {
  if (role === "offense") return palette.offense;
  if (role === "defense") return palette.defense;
  return palette.neutral;
}

function textOn(role: PlayerRole, palette: CourtPalette): string {
  if (role === "offense") return palette.offenseOn;
  if (role === "defense") return palette.defenseOn;
  return palette.neutralOn;
}

function segment(ctx: Ctx2D, from: Point, to: Point): void {
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
}

/** The court lines, thin and in one colour. */
export function drawCourtLines(ctx: Ctx2D, transform: CourtTransform, palette: CourtPalette): void {
  ctx.save();
  ctx.strokeStyle = palette.line;
  ctx.lineCap = "butt";
  ctx.lineJoin = "miter";
  for (const primitive of courtLines()) {
    ctx.lineWidth = lineWidthPx(primitive.weight, transform.scale);
    if (primitive.kind === "line") {
      ctx.setLineDash([]);
      segment(ctx, toPx(transform, primitive.x1, primitive.y1), toPx(transform, primitive.x2, primitive.y2));
      continue;
    }
    const centre = toPx(transform, primitive.cx, primitive.cy);
    const dash = Math.max(2, transform.scale * 0.9);
    ctx.setLineDash(primitive.dashed ? [dash, dash] : []);
    ctx.beginPath();
    ctx.arc(centre.x, centre.y, primitive.r * transform.scale, primitive.start, primitive.end);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.restore();
}

/** The players, the ball and the overlays of a scene. Returns every mark drawn, in drawing order. */
export function drawScene(
  ctx: Ctx2D,
  transform: CourtTransform,
  scene: CourtScene,
  palette: CourtPalette,
  layers: CourtLayers = DEFAULT_LAYERS,
): DrawnMark[] {
  const marks: DrawnMark[] = [];
  const radius = playerRadiusPx(transform.scale);
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  if (layers.ghosts) {
    for (const ghost of scene.ghosts) {
      const centre = toPx(transform, ghost.x, ghost.y);
      ctx.strokeStyle = palette.ghost;
      if (ghost.from) {
        ctx.setLineDash([]);
        ctx.lineWidth = 1;
        segment(ctx, toPx(transform, ghost.from.x, ghost.from.y), centre);
      }
      ctx.setLineDash([3, 2.5]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(centre.x, centre.y, radius, 0, FULL_TURN);
      ctx.stroke();
      marks.push({ kind: "ghost", id: ghost.id, role: "defense", x: centre.x, y: centre.y, r: radius });
    }
    ctx.setLineDash([]);
  }

  // Defenders first, then attackers, so the side with the ball stays on top where discs overlap.
  const order: PlayerRole[] = ["neutral", "defense", "offense"];
  const players = [...scene.players].sort((a, b) => order.indexOf(a.role) - order.indexOf(b.role));
  for (const player of players) {
    const centre = toPx(transform, player.x, player.y);
    ctx.fillStyle = fillOf(player.role, palette);
    ctx.beginPath();
    ctx.arc(centre.x, centre.y, radius, 0, FULL_TURN);
    ctx.fill();
    ctx.strokeStyle = palette.ground;
    ctx.lineWidth = 1.25;
    ctx.stroke();
    if (layers.labels && player.label && radius >= LABEL_MIN_RADIUS_PX) {
      ctx.fillStyle = textOn(player.role, palette);
      ctx.font = `600 ${Math.round(radius * 1.05)}px ${palette.fontFamily}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(player.label, centre.x, centre.y + radius * 0.06);
    }
    marks.push({ kind: "player", id: player.id, role: player.role, x: centre.x, y: centre.y, r: radius });
  }

  if (layers.truth) {
    const arm = radius * 0.55;
    for (const point of scene.truth) {
      const centre = toPx(transform, point.x, point.y);
      ctx.strokeStyle = palette.truth;
      ctx.lineWidth = 1.5;
      if (point.from) segment(ctx, toPx(transform, point.from.x, point.from.y), centre);
      ctx.beginPath();
      ctx.arc(centre.x, centre.y, radius * 0.8, 0, FULL_TURN);
      ctx.stroke();
      segment(ctx, { x: centre.x - arm, y: centre.y }, { x: centre.x + arm, y: centre.y });
      segment(ctx, { x: centre.x, y: centre.y - arm }, { x: centre.x, y: centre.y + arm });
      marks.push({ kind: "truth", id: point.id, role: null, x: centre.x, y: centre.y, r: radius * 0.8 });
    }
  }

  if (layers.markers) {
    for (const marker of scene.markers) {
      let centre: Point | null = null;
      let id: number | null = null;
      if (marker.kind === "spot") {
        centre = toPx(transform, marker.x, marker.y);
      } else {
        const player = scene.players.find((candidate) => candidate.id === marker.id);
        if (player) {
          centre = toPx(transform, player.x, player.y);
          id = marker.id;
        }
      }
      if (!centre) continue;
      const ring = radius + 4;
      ctx.strokeStyle = palette.marker;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(centre.x, centre.y, ring, 0, FULL_TURN);
      ctx.stroke();
      if (marker.kind === "spot") {
        ctx.fillStyle = palette.marker;
        ctx.beginPath();
        ctx.arc(centre.x, centre.y, 2, 0, FULL_TURN);
        ctx.fill();
      }
      marks.push({ kind: "marker", id, role: null, x: centre.x, y: centre.y, r: ring });
    }
  }

  if (scene.ball) {
    const centre = toPx(transform, scene.ball.x, scene.ball.y);
    const ballRadius = ballRadiusPx(transform.scale);
    ctx.fillStyle = palette.ball;
    ctx.beginPath();
    ctx.arc(centre.x, centre.y, ballRadius, 0, FULL_TURN);
    ctx.fill();
    ctx.strokeStyle = palette.ground;
    ctx.lineWidth = 1.25;
    ctx.stroke();
    marks.push({ kind: "ball", id: null, role: null, x: centre.x, y: centre.y, r: ballRadius });
  }

  ctx.restore();
  return marks;
}

export interface RenderInput {
  /** Canvas size in CSS pixels. */
  width: number;
  height: number;
  /** Backing-store pixels per CSS pixel on each axis (the device pixel ratio, after rounding the store size). */
  pixelRatioX: number;
  pixelRatioY: number;
  view: CourtView;
  attackX?: number | null;
  scene: CourtScene;
  palette: CourtPalette;
  layers?: CourtLayers;
}

export interface RenderResult {
  transform: CourtTransform;
  marks: DrawnMark[];
  landmarks: Record<string, Point>;
}

/** Draw one whole picture: ground, court lines, scene. Everything is in CSS pixels; the context scales. */
export function renderCourt(ctx: Ctx2D, input: RenderInput): RenderResult {
  const transform = courtTransform(input.width, input.height, input.view, input.attackX);
  ctx.setTransform(input.pixelRatioX, 0, 0, input.pixelRatioY, 0, 0);
  ctx.clearRect(0, 0, input.width, input.height);
  ctx.fillStyle = input.palette.ground;
  ctx.fillRect(0, 0, input.width, input.height);
  ctx.save();
  const visible = viewRect(transform);
  ctx.beginPath();
  ctx.rect(visible.x, visible.y, visible.width, visible.height);
  ctx.clip();
  drawCourtLines(ctx, transform, input.palette);
  const marks = drawScene(ctx, transform, input.scene, input.palette, input.layers ?? DEFAULT_LAYERS);
  ctx.restore();
  return { transform, marks, landmarks: landmarksPx(transform) };
}
