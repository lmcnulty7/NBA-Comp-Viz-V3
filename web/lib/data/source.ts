/**
 * The one way the site gets its data. Pages and components call a DataSource; they never fetch a URL or read
 * a file themselves. The interface names what the site needs, not where it comes from, so a live service can
 * implement it later and nothing above this file changes (PLAN.md, Site structure).
 *
 * StaticSource is the implementation for the static site: it reads the JSON files of the data root through a
 * reader function. On the server the reader opens files (lib/data/server.ts); in the browser it fetches them
 * (lib/data/browser.ts).
 */
import {
  SCHEMA_VERSION,
  type DemoFrames,
  type DemoSwap,
  type Manifest,
  type PlayersFile,
  type Possession,
  type PossessionIndex,
  type Results,
  type Scorecard,
} from "@/lib/types/contract";

export interface DataSource {
  /** The entry point: build id, the games, and whether the data is synthetic fixture output. */
  manifest(): Promise<Manifest>;
  /** The possession list of one game. */
  possessionIndex(gameSlug: string): Promise<PossessionIndex>;
  /** One possession: positions, ball, ghost defenders, EPV curve and shots. */
  possession(gameSlug: string, possession: number): Promise<Possession>;
  /** Player views for the prep tab. */
  players(): Promise<PlayersFile>;
  /** The tracking scorecard. */
  scorecard(): Promise<Scorecard>;
  /** The model results table. */
  results(): Promise<Results>;
  /** The demo comparison, or null when this data build has none. */
  demoSwap(): Promise<DemoSwap | null>;
  /** The demo scenes, or null when this data build has none. */
  demoFrames(): Promise<DemoFrames | null>;
}

export type DataErrorReason = "missing" | "invalid" | "version" | "unreachable" | "bad-request";

/** A data request that failed, with the thing that was asked for and why. */
export class DataError extends Error {
  readonly what: string;
  readonly reason: DataErrorReason;

  constructor(what: string, reason: DataErrorReason, detail: string) {
    super(`${what}: ${detail}`);
    this.name = "DataError";
    this.what = what;
    this.reason = reason;
  }
}

/**
 * Reads one JSON document by its path under the data root. Resolves to undefined when there is no such
 * document, and rejects when it exists and cannot be read.
 */
export type ReadJson = (relativePath: string) => Promise<unknown>;

const SLUG = /^[a-z0-9][a-z0-9-]*$/;

/** Paths under the data root, as the contract names them (CONTRACT_FILES in lib/types/contract.ts). */
export const dataPaths = {
  manifest: "manifest.json",
  possessionIndex: (gameSlug: string) => `games/${gameSlug}/possessions.json`,
  possession: (gameSlug: string, possession: number) => `games/${gameSlug}/${possession}.json`,
  players: "players.json",
  scorecard: "scorecard.json",
  results: "results.json",
  demoSwap: "demo/swap.json",
  demoFrames: "demo/frames.json",
} as const;

/** How many possession documents a source keeps in memory. The fixed documents are always kept. */
const POSSESSION_CACHE_SIZE = 32;

export class StaticSource implements DataSource {
  private readonly read: ReadJson;
  private readonly fixed = new Map<string, Promise<unknown>>();
  private readonly possessions = new Map<string, Promise<unknown>>();

  constructor(read: ReadJson) {
    this.read = read;
  }

  manifest(): Promise<Manifest> {
    return this.required<Manifest>(dataPaths.manifest, this.fixed);
  }

  possessionIndex(gameSlug: string): Promise<PossessionIndex> {
    assertSlug(gameSlug);
    return this.required<PossessionIndex>(dataPaths.possessionIndex(gameSlug), this.fixed);
  }

  possession(gameSlug: string, possession: number): Promise<Possession> {
    assertSlug(gameSlug);
    if (!Number.isInteger(possession) || possession < 0) {
      throw new DataError(`possession ${possession}`, "bad-request", "a possession is a whole number");
    }
    const path = dataPaths.possession(gameSlug, possession);
    const cached = this.possessions.get(path);
    if (cached) {
      // Move to the newest end, so the least recently used document is the one dropped.
      this.possessions.delete(path);
      this.possessions.set(path, cached);
      return cached as Promise<Possession>;
    }
    const loaded = this.required<Possession>(path, this.possessions);
    for (const key of this.possessions.keys()) {
      if (this.possessions.size <= POSSESSION_CACHE_SIZE) break;
      this.possessions.delete(key);
    }
    return loaded;
  }

  players(): Promise<PlayersFile> {
    return this.required<PlayersFile>(dataPaths.players, this.fixed);
  }

  scorecard(): Promise<Scorecard> {
    return this.required<Scorecard>(dataPaths.scorecard, this.fixed);
  }

  results(): Promise<Results> {
    return this.required<Results>(dataPaths.results, this.fixed);
  }

  demoSwap(): Promise<DemoSwap | null> {
    return this.optional<DemoSwap>(dataPaths.demoSwap);
  }

  demoFrames(): Promise<DemoFrames | null> {
    return this.optional<DemoFrames>(dataPaths.demoFrames);
  }

  private load(path: string, cache: Map<string, Promise<unknown>>): Promise<unknown> {
    let pending = cache.get(path);
    if (!pending) {
      pending = this.read(path).then((document) => {
        if (document !== undefined) assertVersion(path, document);
        return document;
      });
      cache.set(path, pending);
      // A failed read is not remembered: the next call tries again.
      pending.catch(() => {
        if (cache.get(path) === pending) cache.delete(path);
      });
    }
    return pending;
  }

  private async required<T>(path: string, cache: Map<string, Promise<unknown>>): Promise<T> {
    const document = await this.load(path, cache);
    if (document === undefined) {
      cache.delete(path);
      throw new DataError(path, "missing", "the data root has no such file");
    }
    return document as T;
  }

  private async optional<T>(path: string): Promise<T | null> {
    const document = await this.load(path, this.fixed);
    return document === undefined ? null : (document as T);
  }
}

function assertSlug(gameSlug: string): void {
  if (!SLUG.test(gameSlug)) {
    throw new DataError(`game ${JSON.stringify(gameSlug)}`, "bad-request", "not a game slug");
  }
}

function assertVersion(path: string, document: unknown): void {
  const version =
    typeof document === "object" && document !== null
      ? (document as { schema_version?: unknown }).schema_version
      : undefined;
  if (version !== SCHEMA_VERSION) {
    throw new DataError(
      path,
      "version",
      `schema version ${JSON.stringify(version)} is not the ${SCHEMA_VERSION} this site was built for`,
    );
  }
}
