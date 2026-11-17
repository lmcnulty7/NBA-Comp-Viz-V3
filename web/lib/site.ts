/**
 * Build-time facts about the site: the base path it is served under and the variant (public or private).
 * Both are inlined when the site is built (next.config.ts), so they are the same on the server and in the
 * browser.
 */

export type Variant = "public" | "private";

/** "" at the root of a host, or a sub-path such as the GitHub Pages repository path. Never ends in a slash. */
export const BASE_PATH: string = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export const VARIANT: Variant = process.env.NEXT_PUBLIC_VARIANT === "private" ? "private" : "public";

/** The URL of a file the site serves from public/ (data files, stills), with the base path in front. */
export function assetUrl(path: string): string {
  return `${BASE_PATH}/${path.replace(/^\/+/, "")}`;
}

/**
 * The URL of a private clip. The private build copies the clips folder to clips/ in its output; the public
 * build has no clips, so this returns null there and the caller shows the still.
 */
export function clipUrl(fileName: string): string | null {
  return VARIANT === "private" ? assetUrl(`clips/${fileName}`) : null;
}

/** Every route of the site, in one place. Links use the game slug, never the dotted game id. */
export const routes = {
  home: "/",
  review: "/review/",
  game: (gameSlug: string) => `/review/${gameSlug}/`,
  possession: (gameSlug: string, possession: number) => `/review/${gameSlug}/${possession}/`,
  prep: "/prep/",
  player: (playerSlug: string) => `/prep/${playerSlug}/`,
  results: "/results/",
  methods: "/methods/",
} as const;
