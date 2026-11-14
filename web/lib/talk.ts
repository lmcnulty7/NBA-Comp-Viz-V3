/**
 * The interview slides (PLAN.md M10, D35), read from web/content/talk while the site is built. The order and the
 * words of the controls are in deck.json; each slide is a Markdown file in the format of scripts/lib/slides.mjs.
 * Tokens in a slide are filled from the site data (lib/tokens.ts); an unknown token fails the build.
 *
 * Only server code may import this module: it reads the disk.
 */
import { readFileSync } from "node:fs";
import path from "node:path";
import deck from "@/content/talk/deck.json";
import { parseSlide, type ParsedSlide } from "@/scripts/lib/slides.mjs";
import { renderLine, type TokenData } from "@/lib/tokens";

export type SlideVisual = "none" | "still" | "clip" | "boxes" | "map" | "truth" | "swap" | "next" | "results";
export type SlideScene = "headline" | "clip" | "boxes" | "map" | "truth" | "swap" | "next";

export type SlideBlock = { kind: "p"; text: string } | { kind: "list"; items: string[] };

export interface Slide {
  id: string;
  /** 1 for the first slide. */
  number: number;
  title: string;
  visual: SlideVisual;
  scene: SlideScene | null;
  blocks: SlideBlock[];
  notes: string[];
}

const SCENES: SlideScene[] = ["headline", "clip", "boxes", "map", "truth", "swap", "next"];
const TALK_DIR = path.join(process.cwd(), "content", "talk");

export const talkCopy = deck;

/** The slide ids, in order. */
export function slideIds(): string[] {
  return [...deck.slides];
}

/** One slide file, parsed but not yet filled. */
export function readSlide(id: string): ParsedSlide {
  if (!deck.slides.includes(id)) throw new Error(`talk: no slide ${JSON.stringify(id)} in content/talk/deck.json`);
  const text = readFileSync(path.join(TALK_DIR, `${id}.md`), "utf8");
  return parseSlide(text, `content/talk/${id}.md`);
}

function line(text: string, fallback: string | undefined, data: TokenData): string {
  return renderLine(text, fallback, data) ?? deck.ui.missing;
}

/** One slide with every token filled from the data. */
export function loadSlide(id: string, data: TokenData): Slide {
  const parsed = readSlide(id);
  const scene = parsed.meta.scene ?? null;
  if (scene !== null && !SCENES.includes(scene as SlideScene)) {
    throw new Error(`content/talk/${id}.md: scene ${JSON.stringify(scene)} is not a scene of docs/SCRIPT.md`);
  }
  return {
    id,
    number: deck.slides.indexOf(id) + 1,
    title: line(parsed.title, undefined, data),
    visual: parsed.meta.visual,
    scene: scene as SlideScene | null,
    blocks: parsed.blocks.map((block) =>
      block.kind === "p"
        ? { kind: "p", text: line(block.text, block.fallback, data) }
        : { kind: "list", items: block.items.map((item) => line(item, undefined, data)) },
    ),
    notes: parsed.notes.map((note) => line(note, undefined, data)),
  };
}
