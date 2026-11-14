import { useId } from "react";
import {
  COURT_LENGTH_FT,
  COURT_WIDTH_FT,
  HALF_X_FT,
  MARGIN_FT,
  arcPoint,
  courtLines,
  type CourtPrimitive,
} from "./geometry";

/**
 * The court as a line drawing in SVG, built from the same primitives the canvas renderer draws. It is the
 * recurring motif of the site: decoration, with no data in it, so it is hidden from assistive technology.
 */

const FULL_TURN = 2 * Math.PI;

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function pathOf(primitive: CourtPrimitive): string {
  if (primitive.kind === "line") {
    return `M${primitive.x1} ${primitive.y1}L${primitive.x2} ${primitive.y2}`;
  }
  const { cx, cy, r } = primitive;
  const sweep = primitive.end - primitive.start;
  if (sweep >= FULL_TURN - 1e-9) {
    return `M${cx - r} ${cy}A${r} ${r} 0 1 1 ${cx + r} ${cy}A${r} ${r} 0 1 1 ${cx - r} ${cy}`;
  }
  const from = arcPoint(primitive, primitive.start);
  const to = arcPoint(primitive, primitive.end);
  const large = sweep > Math.PI ? 1 : 0;
  return `M${round(from.x)} ${round(from.y)}A${r} ${r} 0 ${large} 1 ${round(to.x)} ${round(to.y)}`;
}

function leftmost(primitive: CourtPrimitive): number {
  return primitive.kind === "line" ? Math.min(primitive.x1, primitive.x2) : primitive.cx - primitive.r;
}

export interface CourtMotifProps {
  /** The whole court, or the left half of it. */
  half?: boolean;
  className?: string;
}

export function CourtMotif({ half = false, className }: CourtMotifProps) {
  const clipId = useId();
  const width = half ? HALF_X_FT : COURT_LENGTH_FT;
  const lines = courtLines()
    .filter((primitive) => primitive.kind === "line" || !primitive.dashed)
    // The half court keeps what starts in the left half; the clip trims the lines that cross the middle.
    .filter((primitive) => !half || leftmost(primitive) <= HALF_X_FT);
  const viewBox = `${-MARGIN_FT} ${-MARGIN_FT} ${width + 2 * MARGIN_FT} ${COURT_WIDTH_FT + 2 * MARGIN_FT}`;
  return (
    <svg className={className} viewBox={viewBox} aria-hidden="true" focusable="false" fill="none">
      <defs>
        <clipPath id={clipId}>
          <rect x={-MARGIN_FT} y={-MARGIN_FT} width={width + MARGIN_FT} height={COURT_WIDTH_FT + 2 * MARGIN_FT} />
        </clipPath>
      </defs>
      <g clipPath={half ? `url(#${clipId})` : undefined} stroke="currentColor" strokeWidth="1">
        {lines.map((primitive, index) => (
          <path key={index} d={pathOf(primitive)} vectorEffect="non-scaling-stroke" />
        ))}
      </g>
    </svg>
  );
}

/** A small mark beside the site name: a court outline with its half-court line and centre circle. */
export function CourtGlyph() {
  return (
    <svg viewBox="0 0 24 14" aria-hidden="true" focusable="false" fill="none" stroke="currentColor" strokeWidth="1.3">
      <rect x="0.65" y="0.65" width="22.7" height="12.7" />
      <path d="M12 0.65V13.35" />
      <circle cx="12" cy="7" r="2.6" />
    </svg>
  );
}
