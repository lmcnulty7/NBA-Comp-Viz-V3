import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import type { Manifest, PlayersFile, PossessionIndex } from "@/lib/types/contract";

/** What the browser tests know about the built site: read from web/out, never from the sources. */

export const WEB_DIR = path.resolve(__dirname, "..", "..");
export const OUT_DIR = path.join(WEB_DIR, "out");

export interface BuildInfo {
  variant: "public" | "private";
  basePath: string;
  synthetic: boolean;
  dataBuild: string | null;
}

function readJson<T>(relative: string): T {
  return JSON.parse(readFileSync(path.join(OUT_DIR, relative), "utf8")) as T;
}

export const build = readJson<BuildInfo>("build-info.json");
export const manifest = readJson<Manifest>("data/manifest.json");
export const players = readJson<PlayersFile>("data/players.json");

export function possessionIndex(gameSlug: string): PossessionIndex {
  return readJson<PossessionIndex>(`data/games/${gameSlug}/possessions.json`);
}

/** Every page of the built site, as a path relative to the base path ("" is the home page). */
export function builtRoutes(): string[] {
  const routes: string[] = [];
  const walk = (dir: string, prefix: string) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (!entry.isDirectory() || entry.name.startsWith("_") || entry.name === "data" || entry.name === "clips") continue;
      const relative = `${prefix}${entry.name}/`;
      if (existsSync(path.join(dir, entry.name, "index.html"))) routes.push(relative);
      walk(path.join(dir, entry.name), relative);
    }
  };
  if (existsSync(path.join(OUT_DIR, "index.html"))) routes.push("");
  walk(OUT_DIR, "");
  return routes.sort();
}

/**
 * The pages a test run visits. Every fixed page, and from each family of generated pages (the possessions
 * of a game, the players) the first two, the last one and every featured possession, so a run stays short
 * when the data root holds thousands of possessions.
 */
export function sampledRoutes(): string[] {
  const all = builtRoutes();
  const keep = new Set<string>();
  const families = new Map<string, string[]>();
  for (const route of all) {
    const parts = route.split("/").filter(Boolean);
    const isPossession = parts.length === 3 && parts[0] === "review";
    const isPlayer = parts.length === 2 && parts[0] === "prep";
    if (!isPossession && !isPlayer) {
      keep.add(route);
      continue;
    }
    const family = parts.slice(0, -1).join("/");
    families.set(family, [...(families.get(family) ?? []), route]);
  }
  for (const [family, members] of families) {
    const ordered = [...members].sort((a, b) => {
      const tail = (route: string) => route.split("/").filter(Boolean).pop() ?? "";
      return Number(tail(a)) - Number(tail(b)) || tail(a).localeCompare(tail(b));
    });
    for (const route of [...ordered.slice(0, 2), ...ordered.slice(-1)]) keep.add(route);
    const parts = family.split("/");
    if (parts[0] === "review" && parts[1]) {
      for (const row of possessionIndex(parts[1]).possessions) {
        if (row.featured) keep.add(`review/${parts[1]}/${row.possession}/`);
      }
    }
  }
  return [...keep].filter((route) => all.includes(route)).sort();
}

/** The first game of the data root and the possession the court tests draw: a featured one when there is one. */
export function courtCase(): { gameSlug: string; possession: number; frames: number; file: string } {
  const game = manifest.games[0];
  if (!game) throw new Error("the data root has no game");
  const index = possessionIndex(game.slug);
  const withGhosts = index.possessions.filter((row) => row.has_ghosts && row.n_frames > 20);
  const row = withGhosts.find((candidate) => candidate.featured) ?? withGhosts[0] ?? index.possessions[0];
  if (!row) throw new Error(`game ${game.slug} has no possession`);
  return {
    gameSlug: game.slug,
    possession: row.possession,
    frames: row.n_frames,
    file: path.join(OUT_DIR, "data", "games", game.slug, `${row.possession}.json`),
  };
}

interface TokensModule {
  themes(css?: string): { light: Map<string, string>; dark: Map<string, string> };
  resolve(tokens: Map<string, string>, name: string): string;
  parseHex(value: string): number[];
}

/** The value of a design token in a theme, as "rgb(r, g, b)", the way a browser reports a colour. */
export async function tokenRgb(theme: "light" | "dark", name: string): Promise<string> {
  const tokens = (await import("../../scripts/lib/tokens.mjs")) as unknown as TokensModule;
  const [r, g, b] = tokens.parseHex(tokens.resolve(tokens.themes()[theme], name));
  return `rgb(${r}, ${g}, ${b})`;
}

export async function tokenChannels(theme: "light" | "dark", name: string): Promise<number[]> {
  const tokens = (await import("../../scripts/lib/tokens.mjs")) as unknown as TokensModule;
  return tokens.parseHex(tokens.resolve(tokens.themes()[theme], name));
}
