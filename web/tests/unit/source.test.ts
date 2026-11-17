import { expect, test } from "@playwright/test";
import { DataError, StaticSource, dataPaths, type DataSource } from "@/lib/data/source";
import { CONTRACT_FILES, SCHEMA_VERSION } from "@/lib/types/contract";

function reader(files: Record<string, unknown>) {
  const asked: string[] = [];
  const read = async (path: string) => {
    asked.push(path);
    return files[path];
  };
  return { asked, read };
}

const versioned = (extra: Record<string, unknown> = {}) => ({ schema_version: SCHEMA_VERSION, ...extra });

test("each method reads the file the contract names for it", async () => {
  const files: Record<string, unknown> = {
    "manifest.json": versioned({ build_id: "b" }),
    "games/fixture-game-a/possessions.json": versioned({ possessions: [] }),
    "games/fixture-game-a/3.json": versioned({ possession: 3 }),
    "players.json": versioned(),
    "scorecard.json": versioned(),
    "results.json": versioned(),
    "demo/swap.json": versioned(),
    "demo/frames.json": versioned(),
  };
  const { asked, read } = reader(files);
  const source: DataSource = new StaticSource(read);
  expect(await source.manifest()).toMatchObject({ build_id: "b" });
  await source.possessionIndex("fixture-game-a");
  expect(await source.possession("fixture-game-a", 3)).toMatchObject({ possession: 3 });
  await source.players();
  await source.scorecard();
  await source.results();
  expect(await source.demoSwap()).not.toBeNull();
  expect(await source.demoFrames()).not.toBeNull();
  expect(asked).toEqual(Object.keys(files));
  // The paths are the contract's own patterns with the slug and the possession filled in.
  const patterns = Object.keys(CONTRACT_FILES);
  expect(patterns).toContain(dataPaths.manifest);
  expect(patterns).toContain(dataPaths.possessionIndex("<slug>"));
  expect(patterns).toContain(dataPaths.possession("<slug>", 0).replace("/0.json", "/<possession>.json"));
  for (const fixed of [dataPaths.players, dataPaths.scorecard, dataPaths.results, dataPaths.demoSwap, dataPaths.demoFrames]) {
    expect(patterns).toContain(fixed);
  }
});

test("a document is read once and then remembered", async () => {
  const { asked, read } = reader({ "manifest.json": versioned(), "games/a/1.json": versioned() });
  const source = new StaticSource(read);
  const [first, second] = await Promise.all([source.manifest(), source.manifest()]);
  expect(first).toBe(second);
  await source.possession("a", 1);
  await source.possession("a", 1);
  expect(asked).toEqual(["manifest.json", "games/a/1.json"]);
});

test("a missing file is an error for a required document and null for the demo", async () => {
  const { read } = reader({});
  const source = new StaticSource(read);
  await expect(source.manifest()).rejects.toMatchObject({ name: "DataError", reason: "missing", what: "manifest.json" });
  await expect(source.possession("a", 1)).rejects.toBeInstanceOf(DataError);
  expect(await source.demoSwap()).toBeNull();
  expect(await source.demoFrames()).toBeNull();
});

test("a failed read is tried again on the next call", async () => {
  let attempts = 0;
  const source = new StaticSource(async () => {
    attempts += 1;
    if (attempts === 1) throw new DataError("manifest.json", "unreachable", "offline");
    return versioned();
  });
  await expect(source.manifest()).rejects.toMatchObject({ reason: "unreachable" });
  await expect(source.manifest()).resolves.toMatchObject({ schema_version: SCHEMA_VERSION });
  expect(attempts).toBe(2);
});

test("data written for another schema version is refused", async () => {
  const { read } = reader({ "manifest.json": { schema_version: "0.9.0" }, "players.json": { players: [] } });
  const source = new StaticSource(read);
  await expect(source.manifest()).rejects.toMatchObject({ reason: "version" });
  await expect(source.players()).rejects.toMatchObject({ reason: "version" });
});

test("a game is named by its slug, and a possession by a whole number", async () => {
  const { asked, read } = reader({});
  const source = new StaticSource(read);
  for (const bad of ["12.16.2015.PHX.at.GSW", "../secrets", "Fixture", "", "a/b"]) {
    expect(() => source.possessionIndex(bad)).toThrow(DataError);
    expect(() => source.possession(bad, 1)).toThrow(DataError);
  }
  for (const bad of [-1, 1.5, Number.NaN]) {
    expect(() => source.possession("fixture-game-a", bad)).toThrow(DataError);
  }
  expect(asked).toEqual([]);
});

test("only the most recent possessions are kept in memory", async () => {
  const asked: string[] = [];
  const source = new StaticSource(async (path) => {
    asked.push(path);
    return versioned();
  });
  for (let number = 0; number < 40; number += 1) await source.possession("a", number);
  await source.possession("a", 39); // still remembered
  await source.possession("a", 0); // dropped, so it is read again
  expect(asked.filter((path) => path === "games/a/39.json")).toHaveLength(1);
  expect(asked.filter((path) => path === "games/a/0.json")).toHaveLength(2);
});

test("a data root of an earlier minor version reads, a later or another major one does not", async () => {
  const { read } = reader({
    "manifest.json": { schema_version: "1.1.0" },
    "players.json": { schema_version: "1.3.0" },
    "results.json": { schema_version: "2.0.0" },
  });
  const source = new StaticSource(read);
  await expect(source.manifest()).resolves.toMatchObject({ schema_version: "1.1.0" });
  await expect(source.players()).rejects.toMatchObject({ reason: "version" });
  await expect(source.results()).rejects.toMatchObject({ reason: "version" });
});

test("machine-written texts are optional", async () => {
  const { read } = reader({ "genai/scouting.json": versioned({ kind: "scouting", items: [] }) });
  const source = new StaticSource(read);
  await expect(source.texts("scouting")).resolves.toMatchObject({ kind: "scouting" });
  await expect(source.texts("possession")).resolves.toBeNull();
});
