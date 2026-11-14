import { expect, test } from "@playwright/test";
import { findLane } from "@/components/charts/LaneMetric";
import { decimalsInText, linear, niceDomain, ticks } from "@/components/charts/scale";
import { groupSources, rowsRead } from "@/components/evidence/groups";
import {
  NO_FILTER,
  fifths,
  filterFromQuery,
  filterPossessions,
  filterToQuery,
  resultsPresent,
} from "@/components/review/filters";
import { buildTour, stopIndex } from "@/components/review/tour";
import { SCHEMA_VERSION, type PossessionSummary, type Results, type Source } from "@/lib/types/contract";

function row(possession: number, extra: Partial<PossessionSummary> = {}): PossessionSummary {
  return {
    possession,
    period: 1,
    start_clock_s: 700,
    end_clock_s: 690,
    duration_s: 10,
    offense: "AAA",
    defense: "BBB",
    result: "shot_made",
    points: 2,
    gap: false,
    n_frames: 50,
    xfg: 0.4,
    xfg_bucket: 2,
    epv_start: 1,
    epv_end: 1.1,
    has_ghosts: true,
    featured: false,
    featured_by: null,
    featured_cell: null,
    demo: false,
    ...extra,
  };
}

const rows = [
  row(1, { result: "shot_made", xfg_bucket: 1, featured: true, featured_by: "rule", featured_cell: "shot_made fifth 1" }),
  row(2, { result: "shot_missed", xfg_bucket: 5 }),
  row(3, { result: "turnover", xfg: null, xfg_bucket: null }),
  row(4, { result: "shot_missed", xfg_bucket: 1, featured: true, featured_by: "hand", featured_cell: "shot_missed fifth 1" }),
];

test("each filter keeps only what it names, and they combine", () => {
  expect(filterPossessions(rows, NO_FILTER).map((r) => r.possession)).toEqual([1, 2, 3, 4]);
  expect(filterPossessions(rows, { ...NO_FILTER, result: "shot_missed" }).map((r) => r.possession)).toEqual([2, 4]);
  expect(filterPossessions(rows, { ...NO_FILTER, fifth: "1" }).map((r) => r.possession)).toEqual([1, 4]);
  expect(filterPossessions(rows, { ...NO_FILTER, fifth: "none" }).map((r) => r.possession)).toEqual([3]);
  expect(filterPossessions(rows, { ...NO_FILTER, featuredOnly: true }).map((r) => r.possession)).toEqual([1, 4]);
  expect(
    filterPossessions(rows, { result: "shot_missed", fifth: "1", featuredOnly: true }).map((r) => r.possession),
  ).toEqual([4]);
  expect(resultsPresent(rows)).toEqual(["shot_made", "shot_missed", "turnover"]);
});

test("the filter survives a trip through the address, and unknown values are ignored", () => {
  const filter = { result: "turnover" as const, fifth: "3" as const, featuredOnly: true };
  expect(filterFromQuery(filterToQuery(filter))).toEqual(filter);
  expect(filterToQuery(NO_FILTER)).toBe("");
  expect(filterFromQuery("?result=dunk&fifth=12&featured=yes")).toEqual(NO_FILTER);
});

test("the fifths take their ranges from the index edges", () => {
  expect(fifths([0.3, 0.42, 0.5, 0.6])).toEqual([
    { fifth: 1, lo: null, hi: 0.3 },
    { fifth: 2, lo: 0.3, hi: 0.42 },
    { fifth: 3, lo: 0.42, hi: 0.5 },
    { fifth: 4, lo: 0.5, hi: 0.6 },
    { fifth: 5, lo: 0.6, hi: null },
  ]);
});

test("the tour visits made then missed shots, low xFG to high, across games, and keeps who picked each", () => {
  const stops = buildTour([
    { slug: "game-a", label: "A", possessions: rows },
    {
      slug: "game-b",
      label: "B",
      possessions: [
        row(9, { result: "shot_made", xfg_bucket: 1, featured: true, featured_by: "rule", featured_cell: "shot_made fifth 1" }),
        row(7, { result: "shot_made", xfg_bucket: 4, featured: true, featured_by: "rule", featured_cell: "shot_made fifth 4" }),
        row(8, { result: "turnover", xfg: null, xfg_bucket: null, featured: true, featured_by: "hand" }),
      ],
    },
  ]);
  expect(stops.map((stop) => `${stop.gameSlug}/${stop.possession}`)).toEqual([
    "game-a/1",
    "game-b/9",
    "game-b/7",
    "game-a/4",
    "game-b/8",
  ]);
  expect(stops.map((stop) => stop.featuredBy)).toEqual(["rule", "rule", "rule", "hand", "hand"]);
  expect(stopIndex(stops, "game-b", 7)).toBe(2);
  expect(stopIndex(stops, "game-a", 2)).toBe(-1);
  expect(buildTour([])).toEqual([]);
});

const src = (path: string, rows: Source["rows"] = "heldout"): Source => ({ path, what: "w", rows });

test("sources sort by the value they feed, and none is dropped", () => {
  const sources = [
    src("$DATA_ROOT/sportvu_contract/heldout/g.parquet"),
    src("$DATA_ROOT/analytics/possessions/heldout/2015-16/g.parquet"),
    src("$DATA_ROOT/analytics/rosters/heldout/2015-16/g.parquet"),
    src("analytics/decisions/features.py", "not_split"),
    src("$DATA_ROOT/analytics/epv/heldout/2015-16/g.parquet"),
    src("reports/epv.md", "not_split"),
    src("$DATA_ROOT/analytics/ghosting/heldout/2015-16/ghost_frames_g.parquet"),
    src("reports/ghosting.md", "not_split"),
    src("$DATA_ROOT/analytics/xfg/2015-16/xfg_predictions.parquet"),
    src("$DATA_ROOT/analytics/shot_frames/heldout/2015-16/g.parquet"),
    src("$DATA_ROOT/sportvu_contract/heldout/g_pbp.parquet"),
    src("$DATA_ROOT/collect/bref_shots/heldout/shotchart/2015-16.parquet"),
    src("reports/xfg.md", "not_split"),
    src("export/sample.py", "not_split"),
  ];
  const groups = groupSources(sources);
  expect(groups.positions.map((s) => s.path)).toEqual(sources.slice(0, 4).map((s) => s.path));
  expect(groups.epv.map((s) => s.path)).toEqual(sources.slice(4, 6).map((s) => s.path));
  expect(groups.ghosts.map((s) => s.path)).toEqual(sources.slice(6, 8).map((s) => s.path));
  expect(groups.shots.map((s) => s.path)).toEqual(sources.slice(8, 13).map((s) => s.path));
  expect(groups.other.map((s) => s.path)).toEqual(["export/sample.py"]);
  const total = Object.values(groups).reduce((sum, list) => sum + list.length, 0);
  expect(total).toBe(sources.length);
  expect(rowsRead(sources)).toEqual(["heldout", "not_split"]);
});

test("a lane is found by its key, and a missing one is null", () => {
  const results = { schema_version: SCHEMA_VERSION, lanes: [{ lane: "xfg" }] } as unknown as Results;
  expect(findLane(results, "xfg")).toMatchObject({ lane: "xfg" });
  expect(findLane(results, "epv")).toBeNull();
  expect(findLane(null, "xfg")).toBeNull();
});

test("chart scales: a linear map, round ticks, a widened domain, and decimals read from a report's text", () => {
  const x = linear([0, 10], [100, 200]);
  expect(x(5)).toBe(150);
  expect(linear([3, 3], [0, 10])(3)).toBe(5);
  expect(ticks(0, 1, 4)).toEqual([0, 0.5, 1]);
  expect(ticks(0, 1, 6)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
  expect(ticks(0, 12.4, 5)).toEqual([0, 5, 10]);
  expect(niceDomain(0.12, 1.37, 3)).toEqual([0, 1.5]);
  expect(niceDomain(1, 1)[0]).toBeLessThan(1);
  expect(decimalsInText("0.6232 [0.6132, 0.6327]")).toBe(4);
  expect(decimalsInText("3.417 ft")).toBe(3);
  expect(decimalsInText("12 ft")).toBe(0);
  expect(decimalsInText("none", 2)).toBe(2);
});
