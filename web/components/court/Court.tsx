"use client";

import { useEffect, useRef } from "react";
import {
  DEFAULT_LAYERS,
  renderCourt,
  type CourtLayers,
  type CourtPalette,
  type RenderResult,
} from "./draw";
import { viewAspectRatio, type CourtView } from "./geometry";
import type { CourtScene } from "./scene";
import styles from "./court.module.css";

/**
 * The one court renderer of the site: a canvas that draws a scene (players by role, ball, ghost defenders,
 * a truth overlay, a marker) on accurate court lines. The possession review, the demo and the slides all use
 * it. Geometry lives in geometry.ts, drawing in draw.ts; this component owns the canvas element.
 *
 * It keeps the backing store at the size of the canvas times the device pixel ratio, and redraws when the
 * scene, the size, the pixel ratio, the colour theme or the fonts change. Nothing in it animates.
 */
export interface CourtProps {
  scene: CourtScene;
  view?: CourtView;
  /** The x of the rim the offense attacks. The half view shows that half. */
  attackX?: number | null;
  layers?: Partial<CourtLayers>;
  /** The accessible name of the picture. */
  label: string;
  className?: string;
}

/** What the last draw did. Tests read it from the canvas element to compare with the Python court reference. */
export interface CourtState extends RenderResult {
  /** Backing-store size in device pixels. */
  backingWidth: number;
  backingHeight: number;
  pixelRatioX: number;
  pixelRatioY: number;
  draws: number;
}

export type CourtCanvasElement = HTMLCanvasElement & { courtState?: CourtState };

function readPalette(element: Element): CourtPalette {
  const style = getComputedStyle(element);
  const token = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
  return {
    ground: token("--ground", "#ffffff"),
    line: token("--court-line", "#8c8c8c"),
    offense: token("--offense", "#d9731a"),
    offenseOn: token("--offense-on", "#121212"),
    defense: token("--defense", "#2f6bc4"),
    defenseOn: token("--defense-on", "#ffffff"),
    neutral: token("--muted", "#666666"),
    neutralOn: token("--ground", "#ffffff"),
    ghost: token("--ghost", "#7c8aa5"),
    truth: token("--truth", "#1b7a52"),
    ball: token("--ball", "#121212"),
    marker: token("--accent", "#be2490"),
    fontFamily: token("--sans", "sans-serif"),
  };
}

export function Court({ scene, view = "full", attackX = null, layers, label, className }: CourtProps) {
  const canvasRef = useRef<CourtCanvasElement>(null);
  const redraw = useRef<(() => void) | null>(null);
  const draws = useRef(0);

  const ghosts = layers?.ghosts ?? DEFAULT_LAYERS.ghosts;
  const truth = layers?.truth ?? DEFAULT_LAYERS.truth;
  const labels = layers?.labels ?? DEFAULT_LAYERS.labels;
  const markers = layers?.markers ?? DEFAULT_LAYERS.markers;

  // Draw whenever what is shown changes.
  useEffect(() => {
    const draw = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      if (!(rect.width > 0 && rect.height > 0)) return;
      const ratio = window.devicePixelRatio || 1;
      const backingWidth = Math.max(1, Math.round(rect.width * ratio));
      const backingHeight = Math.max(1, Math.round(rect.height * ratio));
      if (canvas.width !== backingWidth) canvas.width = backingWidth;
      if (canvas.height !== backingHeight) canvas.height = backingHeight;
      const context = canvas.getContext("2d");
      if (!context) return;
      const pixelRatioX = backingWidth / rect.width;
      const pixelRatioY = backingHeight / rect.height;
      const result = renderCourt(context, {
        width: rect.width,
        height: rect.height,
        pixelRatioX,
        pixelRatioY,
        view,
        attackX,
        scene,
        palette: readPalette(canvas),
        layers: { ghosts, truth, labels, markers },
      });
      draws.current += 1;
      canvas.courtState = { ...result, backingWidth, backingHeight, pixelRatioX, pixelRatioY, draws: draws.current };
      if (canvas.dataset.ready !== "true") canvas.dataset.ready = "true";
    };
    redraw.current = draw;
    draw();
  }, [scene, view, attackX, ghosts, truth, labels, markers]);

  // Redraw when the canvas is resized, the pixel ratio or the colour theme changes, or a font arrives.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const again = () => redraw.current?.();

    const resize = new ResizeObserver(again);
    resize.observe(canvas);

    const scheme = window.matchMedia("(prefers-color-scheme: dark)");
    scheme.addEventListener("change", again);
    const theme = new MutationObserver(again);
    theme.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class"] });

    let density: MediaQueryList | null = null;
    const watchDensity = () => {
      density?.removeEventListener("change", onDensity);
      density = window.matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`);
      density.addEventListener("change", onDensity);
    };
    const onDensity = () => {
      again();
      watchDensity();
    };
    watchDensity();

    const fonts = document.fonts;
    fonts?.addEventListener?.("loadingdone", again);
    fonts?.ready.then(again).catch(() => undefined);

    return () => {
      resize.disconnect();
      scheme.removeEventListener("change", again);
      theme.disconnect();
      density?.removeEventListener("change", onDensity);
      fonts?.removeEventListener?.("loadingdone", again);
    };
  }, []);

  return (
    <div
      className={className ? `${styles.frame} ${className}` : styles.frame}
      style={{ aspectRatio: String(viewAspectRatio(view)) }}
    >
      <canvas ref={canvasRef} className={styles.canvas} role="img" aria-label={label} data-court={view} />
    </div>
  );
}
