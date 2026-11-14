import { expect, test } from "@playwright/test";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { sceneFromPossession } from "@/components/court/scene";
import { groupSources } from "@/components/evidence/groups";
import { metricEstimate, SORT_KEYS } from "@/components/prep/sort";
import { endTime, epvStateAt, frameAtOrBefore, sceneAtTime } from "@/components/review/timeline";
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

// Checks collect their problems in a list and assert once: an expect per frame would make the run far slower
// than the work it checks on thousands of possessions.
test("every possession of the data root loads and plays, and the review reads held-out rows only", async () => {
  test.skip(!existsSync(path.join(ROOT, "manifest.json")), "no data root");
  const source = new StaticSource(readRoot);
  const manifest = await source.manifest();
  const problems: string[] = [];
  const check = (ok: boolean, what: string) => {
    if (!ok && problems.length < 50) problems.push(what);
  };
  let possessions = 0;
  let frames = 0;
  const kinds = { oneFrame: 0, noEpv: 0, noGhosts: 0, longest: 0 };
  const games = [];
  for (const game of manifest.games) {
    // C8: a game in the review is a held-out game.
    check(game.split === "heldout_game" || game.split === "heldout_arena", `${game.slug} is ${game.split}`);
    const index = await source.possessionIndex(game.slug);
    check(index.game.slug === game.slug, `${game.slug}: index of ${index.game.slug}`);
    check(index.possessions.length === game.n_possessions, `${game.slug}: ${index.possessions.length} possessions listed`);
    for (const item of index.sources) check(HELDOUT_ROWS.has(item.rows), `${game.slug} index reads ${item.rows} rows`);
    games.push({ slug: game.slug, label: game.label, possessions: index.possessions });

    for (const row of index.possessions) {
      const where = `${game.slug}/${row.possession}`;
      const detail = await source.possession(game.slug, row.possession);
      possessions += 1;
      const n = detail.t.length;
      frames += n;
      check(detail.game_slug === game.slug && detail.possession === row.possession, `${where}: wrong file`);
      check(n === row.n_frames && n > 0, `${where}: ${n} frames, index says ${row.n_frames}`);
      for (const track of [...detail.players, ...(detail.ghosts?.tracks ?? []), detail.ball]) {
        check(track.x.length === n && track.y.length === n, `${where}: a track is not one value per frame`);
      }
      // C8: every source of a possession is a held-out table or a report; no training or out-of-fold row.
      for (const item of detail.sources) check(HELDOUT_ROWS.has(item.rows), `${where} reads ${item.rows} rows`);
      const groups = groupSources(detail.sources);
      check(Object.values(groups).flat().length === detail.sources.length, `${where}: a source was dropped`);
      for (const shot of detail.shots) {
        if (shot.xfg != null) check(shot.xfg >= 0 && shot.xfg <= 1, `${where}: xFG ${shot.xfg}`);
      }

      // Everything playback runs, at every frame and between frames.
      const end = endTime(detail);
      check(Number.isFinite(end), `${where}: no end time`);
      for (let frame = 0; frame < n; frame += 1) {
        const time = detail.t[frame]!;
        check(frameAtOrBefore(detail.t, time) === frame, `${where}: frame ${frame} is not found by its time`);
        const exact = sceneAtTime(detail, time, { tween: true });
        check(
          JSON.stringify(exact) === JSON.stringify(sceneFromPossession(detail, frame)),
          `${where}: frame ${frame} is not drawn exactly`,
        );
        const mid = sceneAtTime(detail, time + 0.1, { tween: true });
        for (const player of mid.players) check(Number.isFinite(player.x) && Number.isFinite(player.y), `${where}: a tweened position`);
      }
      if (detail.epv) {
        check(detail.epv.t.length === detail.epv.value.length, `${where}: EPV lengths differ`);
        for (const time of [0, end / 2, end]) {
          const state = epvStateAt(detail.epv, time, end);
          check(state !== null && detail.epv.value[state.index] === state.value, `${where}: EPV at ${time} is not a stored state`);
        }
        // At the end of playback the readout shows the last stored state, even one stored after the last frame.
        check(epvStateAt(detail.epv, end, end)?.index === Math.min(detail.epv.t.length, detail.epv.value.length) - 1, `${where}: EPV at the end`);
      } else {
        kinds.noEpv += 1;
      }
      if (!detail.ghosts) kinds.noGhosts += 1;
      if (n === 1) kinds.oneFrame += 1;
      kinds.longest = Math.max(kinds.longest, end);
    }
  }
  check(possessions === manifest.n_possessions, `${possessions} possessions loaded, manifest says ${manifest.n_possessions}`);

  // The tour: every stop is a featured possession of a game in the build.
  const stops = buildTour(games);
  for (const stop of stops) {
    const game = games.find((candidate) => candidate.slug === stop.gameSlug);
    check(game?.possessions.find((row) => row.possession === stop.possession)?.featured === true, `tour stop ${stop.gameSlug}/${stop.possession}`);
  }

  // The prep tab: every metric a player page shows has an interval or says it is a point estimate.
  const players = await source.players();
  check(players.n_shown + players.n_hidden === players.n_players, "players: shown and hidden do not add up");
  for (const player of players.players) {
    for (const key of SORT_KEYS) {
      const estimate = metricEstimate(player, key);
      if (!estimate) continue;
      const hasInterval = !estimate.point_estimate && estimate.lo != null && estimate.hi != null;
      check(hasInterval || estimate.point_estimate, `${player.slug} ${key}: neither an interval nor a point estimate`);
    }
  }

  const summary = `${manifest.build_id}: ${manifest.games.length} games, ${possessions} possessions, ${frames} frames, ${stops.length} tour stops; one-frame ${kinds.oneFrame}, no EPV ${kinds.noEpv}, no ghosts ${kinds.noGhosts}, longest ${kinds.longest.toFixed(1)} s; ${players.players.length} player pages, ${players.n_hidden} hidden`;
  test.info().annotations.push({ type: "data root", description: summary });
  console.log(summary);
  expect(problems).toEqual([]);
});
