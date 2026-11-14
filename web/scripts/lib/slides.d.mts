// Types of slides.mjs for the TypeScript side of the site (lib/talk.ts) and the tests.
export type Visual = "none" | "still" | "clip" | "boxes" | "map" | "truth" | "swap" | "next" | "results";
export type Block = { kind: "p"; text: string; fallback?: string } | { kind: "list"; items: string[] };
export interface ParsedSlide {
  meta: { scene?: string; visual: Visual; [key: string]: string | undefined };
  title: string;
  blocks: Block[];
  notes: string[];
}
export const VISUALS: Visual[];
export function parseSlide(text: string, name?: string): ParsedSlide;
export function onScreenLines(slide: ParsedSlide): string[];
