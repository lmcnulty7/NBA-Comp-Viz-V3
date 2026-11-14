import { expect, test } from "@playwright/test";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { sceneFromPossession } from "@/components/court/scene";
import { groupSources } from "@/components/evidence/groups";
import { metricEstimate, SORT_KEYS } from "@/components/prep/sort";
import { endTime, epvAt, frameAtOrBefore, sceneAtTime } from "@/components/review/timeline";
import { buildTour } from "@/components/review/tour";
import { StaticSource } from "@/lib/data/source";

/**
 * Loads every possession of the data root through the same DataSource the site uses, and runs every pure
 * function the review page runs on it (PLAN.md M5: "a script loads every possession's data without error").
 * It also checks C8 on the data: the review covers held-out games only, and no possession reads a training or
 * out-of-fold row. The data root is SITE_DATA_DIR, else web/data, so the same test runs on the real data:
 *
 *   SITE_DATA_DIR=<data root> npx playwright test --config playwright.unit.config.ts review-data
 */
const WEB_DIR = path.resolve(__dirname, "..", "..");
const named = (process.env.SITE_DATA_DIR ?? "").trim();
const ROOT = named ? path.resolve(WEB_DIR, named) : path.join(WEB_DIR, "data");

async function readRoot(relative: string): Promise<unknown> {
  const file = path.join(ROOT, relative);
  if (!existsSync(file)) return undefined;
  return JSON.parse(await readFile(file, "utf8")) as unknown;
}

const HELDOUT_ROWS = new Set(["heldout", "not_split"]);

test.describe.configure({ timeout: 30 * 60_000 });

test("every possession of the data root loads and plays, and the review reads held-out rows only", async () => {
  test.skip(!existsSync(path.join(ROOT, "manifest.json")), "no data root");
  const source = new StaticSource(readRoot);
  const manifest = await source.manifest();
  let possessions = 0;
  let frames = 0;
  const kinds = { oneFrame: 0, noEpv: 0, noGhosts: 0, longest: 0 };
  const games = [];
  for (const game of manifest.games) {
    // C8: a game in the review is a held-out game.
    expect(["heldout_game", "heldout_arena"], game.slug).toContain(game.split);
    const index = await source.possessionIndex(game.slug);
    expect(index.game.slug).toBe(game.slug);
    expect(index.possessions.length, game.slug).toBe(game.n_possessions);
    for (const item of index.sources) expect(HELDOUT_ROWS.has(item.rows), `${game.slug} index reads ${item.rows} rows`).toBe(true);
    games.push({ slug: game.slug, label: game.label, possessions: index.possessions });

    for (const row of index.possessions) {
      const where = `${game.slug}/${row.possession}`;
      const detail = await source.possession(game.slug, row.possession);
      possessions += 1;
      const n = detail.t.length;
      frames += n;
      expect(detail.game_slug, where).toBe(game.slug);
      expect(detail.possession, where).toBe(row.possession);
      expect(n, where).toBe(row.n_frames);
      expect(n, where).toBeGreaterThan(0);
      for (const track of detail.players) {
        expect(track.x.length, where).toBe(n);
        expect(track.y.length, where).toBe(n);
      }
      for (const track of detail.ghosts?.tracks ?? []) {
        expect(track.x.length, where).toBe(n);
        expect(track.y.length, where).toBe(n);
      }
      expect(detail.ball.x.length, where).toBe(n);
      // C8: every source of a possession is a held-out table or a report; no training or out-of-fold row.
      for (const item of detail.sources) expect(HELDOUT_ROWS.has(item.rows), `${where} reads ${item.rows} rows`).toBe(true);
      const groups = groupSources(detail.sources);
      expect(Object.values(groups).flat().length, where).toBe(detail.sources.length);
      for (const shot of detail.shots) {
        if (shot.xfg != null) {
          expect(shot.xfg, where).toBeGreaterThanOrEqual(0);
          expect(shot.xfg, where).toBeLessThanOrEqual(1);
        }
      }

      // Everything playback runs, at every frame and between frames.
      const end = endTime(detail);
      expect(Number.isFinite(end), where).toBe(true);
      for (let frame = 0; frame < n; frame += 1) {
        const time = detail.t[frame]!;
        expect(frameAtOrBefore(detail.t, time), where).toBe(frame);
        const exact = sceneAtTime(detail, time, { tween: true });
        expect(exact, where).toEqual(sceneFromPossession(detail, frame));
        const mid = sceneAtTime(detail, time + 0.1, { tween: true });
        for (const player of mid.players) expect(Number.isFinite(player.x) && Number.isFinite(player.y), where).toBe(true);
      }
      if (detail.epv) {
        expect(detail.epv.t.length, where).toBe(detail.epv.value.length);
        for (const time of [0, end / 2, end]) expect(Number.isFinite(epvAt(detail.epv, time) ?? Number.NaN), where).toBe(true);
      } else {
        kinds.noEpv += 1;
      }
      if (!detail.ghosts) kinds.noGhosts += 1;
      if (n === 1) kinds.oneFrame += 1;
      kinds.longest = Math.max(kinds.longest, end);
    }
  }
  expect(possessions).toBe(manifest.n_possessions);

  // The tour: every stop is a featured possession of a game in the build.
  const stops = buildTour(games);
  for (const stop of stops) {
    const game = games.find((candidate) => candidate.slug === stop.gameSlug)!;
    expect(game.possessions.find((row) => row.possession === stop.possession)?.featured).toBe(true);
  }

  // The prep tab: every metric a player page shows has an interval or says it is a point estimate.
  const players = await source.players();
  expect(players.n_shown + players.n_hidden).toBe(players.n_players);
  for (const player of players.players) {
    for (const key of SORT_KEYS) {
      const estimate = metricEstimate(player, key);
      if (!estimate) continue;
      const hasInterval = !estimate.point_estimate && estimate.lo != null && estimate.hi != null;
      expect(hasInterval || estimate.point_estimate, `${player.slug} ${key}`).toBe(true);
    }
  }

  test.info().annotations.push({
    type: "data root",
    description: `${manifest.build_id}: ${manifest.games.length} games, ${possessions} possessions, ${frames} frames, ${stops.length} tour stops; one-frame ${kinds.oneFrame}, no EPV ${kinds.noEpv}, no ghosts ${kinds.noGhosts}, longest ${kinds.longest.toFixed(1)} s`,
  });
  console.log(test.info().annotations.at(-1)?.description);
});
