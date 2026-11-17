import { expect, test } from "@playwright/test";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { python, runScript, sampleRoot } from "../helpers/scripts";

/**
 * The data check has two halves. validate-data.mjs checks the files with ajv against the JSON Schema, scans
 * for private strings and checks what the site's routes need. With --python it then runs
 * `python3 -m export.schemas --validate`, which holds the cross-field rules of the contract. Each case below
 * breaks a synthetic data root in one way and says which half must refuse it.
 */

type Json = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

function edit(root: string, rel: string, change: (doc: Json) => void): void {
  const file = path.join(root, rel);
  const doc = JSON.parse(readFileSync(file, "utf8")) as Json;
  change(doc);
  writeFileSync(file, `${JSON.stringify(doc)}\n`);
}

/** Bring the manifest entry of a file back in line with the file, so only the planted problem is left. */
function resyncManifest(root: string, rel: string): void {
  const bytes = readFileSync(path.join(root, rel));
  edit(root, "manifest.json", (manifest) => {
    const entry = manifest.files.find((candidate: Json) => candidate.path === rel);
    entry.bytes = bytes.length;
    entry.sha256 = createHash("sha256").update(bytes).digest("hex");
  });
}

/** Change a file and keep the manifest in step with it. */
function change(root: string, rel: string, mutate: (doc: Json) => void): void {
  edit(root, rel, mutate);
  resyncManifest(root, rel);
}

const validate = (root: string, ...args: string[]) => runScript("validate-data.mjs", ["--root", root, ...args]);

async function pythonValidate(root: string, ...args: string[]) {
  const { runPythonModule } = await python();
  return runPythonModule("export.schemas", ["--validate", root, ...args]);
}

// The probes below look like local paths, cloud mounts and email addresses. They are put together at run
// time, so this file itself holds none and passes the hygiene check with no exception.
const SLASH = "/";
const rooted = (...parts: string[]) => SLASH + parts.join(SLASH);
const USERS = rooted("Users", "");
const HOME = rooted("home", "");
const EMAIL = ["someone", "example.com"].join("@");
const CLOUD = ["Cloud", "Storage"].join("");
const GDRIVE = ["Google", "Drive"].join("");
const MY_DRIVE = ["My", "Drive"].join(" ");
const WEB = ["https:", "", "example.org"].join(SLASH);

interface Case {
  name: string;
  /** What validate-data.mjs prints on its own, or null when only the Python validator can see the problem. */
  web: RegExp | string | null;
  /** False for a rule only the site needs, which the Python contract does not have. */
  python?: boolean;
  break(root: string): void;
}

const CASES: Case[] = [
  {
    name: "a file the contract does not know",
    web: "notes.json: not a file the contract knows",
    break: (root) => writeFileSync(path.join(root, "notes.json"), "{}"),
  },
  {
    name: "a hidden file that is not an operating system dropping",
    web: "games/.hidden.json: not a file the contract knows",
    break: (root) => writeFileSync(path.join(root, "games", ".hidden.json"), "{}"),
  },
  {
    name: "a dotted game id used as a folder",
    web: "games/12.16.2015.PHX.at.GSW/1.json: not a file the contract knows",
    break: (root) => {
      mkdirSync(path.join(root, "games", "12.16.2015.PHX.at.GSW"));
      writeFileSync(path.join(root, "games", "12.16.2015.PHX.at.GSW", "1.json"), "{}");
    },
  },
  {
    name: "a missing fixed file",
    web: "results.json: missing",
    break: (root) => rmSync(path.join(root, "results.json")),
  },
  {
    name: "an unknown field",
    web: /players\.json: PlayersFile: .*additional properties.*"surprise"/,
    break: (root) =>
      change(root, "players.json", (doc) => {
        doc.surprise = 1;
      }),
  },
  {
    name: "a local path in a string",
    web: `scorecard.json: private string ${JSON.stringify(USERS)}`,
    break: (root) =>
      change(root, "scorecard.json", (doc) => {
        doc.decision = `written at ${USERS}someone/project`;
      }),
  },
  {
    name: "a local path with escaped slashes",
    web: `results.json: private string ${JSON.stringify(HOME)}`,
    break: (root) => {
      const file = path.join(root, "results.json");
      const escaped = `${HOME}someone/x`.replaceAll(SLASH, `\\${SLASH}`);
      writeFileSync(file, readFileSync(file, "utf8").replace("Synthetic fixture table", `Read from ${escaped}`));
      resyncManifest(root, "results.json");
    },
  },
  {
    name: "an email address",
    web: `players.json: private string ${JSON.stringify(EMAIL)}`,
    break: (root) =>
      change(root, "players.json", (doc) => {
        doc.window = `ask ${EMAIL}`;
      }),
  },
  {
    name: "a cloud drive mount",
    web: `demo/swap.json: private string ${JSON.stringify(CLOUD)}`,
    break: (root) =>
      change(root, "demo/swap.json", (doc) => {
        doc.method = [`Library/${CLOUD}/Provider-account/data`];
      }),
  },
  {
    name: "a ghost label that is not the average defense",
    web: /games\/fixture-game-a\/1\.json: Possession: \/ghosts/,
    break: (root) =>
      change(root, "games/fixture-game-a/1.json", (doc) => {
        doc.ghosts.label = "best defense";
      }),
  },
  {
    name: "a probability above one",
    web: /games\/fixture-game-a\/possessions\.json: PossessionIndex: \/possessions\/0\/xfg/,
    break: (root) =>
      change(root, "games/fixture-game-a/possessions.json", (doc) => {
        doc.possessions[0].xfg = 1.5;
      }),
  },
  {
    name: "a file with no source",
    web: /scorecard\.json: Scorecard: \/sources must NOT have fewer than 1 items/,
    break: (root) =>
      change(root, "scorecard.json", (doc) => {
        doc.sources = [];
      }),
  },
  {
    name: "a source that does not say which rows it read",
    web: /results\.json: Results: \/sources\/0 must have required property 'rows'/,
    break: (root) =>
      change(root, "results.json", (doc) => {
        delete doc.sources[0].rows;
      }),
  },
  {
    name: "a number that is not finite",
    web: "games/fixture-game-b/2.json: not valid JSON",
    break: (root) => {
      const file = path.join(root, "games/fixture-game-b/2.json");
      const text = readFileSync(file, "utf8");
      expect(text).toContain('"hz":5.0');
      writeFileSync(file, text.replace('"hz":5.0', '"hz":NaN'));
      resyncManifest(root, "games/fixture-game-b/2.json");
    },
  },
  {
    name: "a possession file at a path its content does not name",
    web: "games/fixture-game-a/9.json: holds possession 1 of fixture-game-a",
    break: (root) =>
      renameSync(path.join(root, "games/fixture-game-a/1.json"), path.join(root, "games/fixture-game-a/9.json")),
  },
  {
    name: "a file changed after the manifest was written",
    web: "manifest.json: results.json does not match the files on disk",
    break: (root) =>
      edit(root, "results.json", (doc) => {
        doc.notes = ["Changed after the manifest was written."];
      }),
  },
  {
    name: "another schema version",
    web: /players\.json: PlayersFile: \/schema_version/,
    break: (root) =>
      change(root, "players.json", (doc) => {
        doc.schema_version = "1.0.0";
      }),
  },
  // Rules the JSON Schema cannot carry: the Python validator is the one that refuses these.
  {
    name: "tracks that are not aligned",
    web: null,
    break: (root) =>
      change(root, "games/fixture-game-a/1.json", (doc) => {
        doc.players[0].x.pop();
      }),
  },
  {
    name: "an estimate with an interval that says it is a point estimate",
    web: null,
    break: (root) =>
      change(root, "players.json", (doc) => {
        doc.players[0].gravity.gravity_ft.point_estimate = true;
      }),
  },
  {
    name: "a value shown below its minimum sample",
    web: null,
    break: (root) =>
      change(root, "players.json", (doc) => {
        doc.players[0].shot_making.attempts = 5;
      }),
  },
  {
    name: "a paired difference whose flag contradicts its interval",
    web: null,
    break: (root) =>
      change(root, "scorecard.json", (doc) => {
        doc.paired[0].excludes_zero = false;
      }),
  },
  {
    name: "a test whose status the results do not support",
    web: null,
    break: (root) =>
      change(root, "methods.json", (doc) => {
        doc.lanes[0].experiments[1].status = "adopted";
      }),
  },
  {
    name: "a methods page number of ours that is not a number",
    web: null,
    break: (root) =>
      change(root, "methods.json", (doc) => {
        doc.lanes[0].benchmarks[0].ours[0].value = "about 0.63";
      }),
  },
  {
    name: "a scorecard cell whose status contradicts its pass flag",
    web: null,
    break: (root) =>
      change(root, "scorecard.json", (doc) => {
        doc.builds[0].rows[0].cells[0].passed = false;
      }),
  },
  {
    name: "two adopted builds",
    web: null,
    break: (root) =>
      change(root, "scorecard.json", (doc) => {
        doc.builds[1].role = "adopted";
      }),
  },
  {
    name: "the decisions lane shown as a model that works",
    web: null,
    break: (root) =>
      change(root, "results.json", (doc) => {
        doc.lanes[0].lane = "decisions";
      }),
  },
  {
    name: "a real source inside a data root marked synthetic",
    web: null,
    break: (root) =>
      change(root, "results.json", (doc) => {
        doc.sources[0].path = "reports/results.md";
      }),
  },
  // Rules only the site has: its routes are built from the indexes.
  {
    name: "a possession in the index with no file (a site rule)",
    web: "games/fixture-game-a/possessions.json: possession 77 has no file",
    python: false,
    break: (root) =>
      change(root, "games/fixture-game-a/possessions.json", (doc) => {
        doc.possessions.push({ ...doc.possessions[1], possession: 77 });
      }),
  },
  {
    name: "two players with one slug (a site rule)",
    web: "players.json: a player slug is listed twice",
    python: false,
    break: (root) =>
      change(root, "players.json", (doc) => {
        doc.players[1].slug = doc.players[0].slug;
      }),
  },
];

test("the synthetic data root is valid for both halves", async () => {
  const root = await sampleRoot();
  const web = validate(root);
  expect(web.output).toContain("21 file(s) valid against the JSON Schema");
  expect(web.status).toBe(0);
  const both = validate(root, "--python");
  expect(both.output).toContain("python: data root is valid");
  expect(both.status).toBe(0);
});

for (const item of CASES) {
  test(`the data check refuses ${item.name}`, async () => {
    const root = await sampleRoot();
    item.break(root);
    const web = validate(root);
    if (item.web === null) {
      expect(web.status, `the JSON Schema cannot see this, yet validate-data.mjs said:\n${web.output}`).toBe(0);
    } else {
      if (typeof item.web === "string") expect(web.output).toContain(item.web);
      else expect(web.output).toMatch(item.web);
      expect(web.status).toBe(1);
    }
    const py = await pythonValidate(root);
    if (item.python === false) {
      expect(py.status, "a site rule the Python contract does not have").toBe(0);
    } else {
      expect(py.status, `python3 -m export.schemas --validate must refuse this:\n${py.stdout}`).toBe(1);
    }
    // The check the site runs (npm run data:validate, npm run check) refuses every one of them.
    expect(validate(root, "--python").status).toBe(1);
  });
}

test("a file the operating system drops is not part of the data root", async () => {
  const root = await sampleRoot();
  writeFileSync(path.join(root, ".DS_Store"), "x");
  writeFileSync(path.join(root, "games", "fixture-game-a", ".DS_Store"), "x");
  expect(validate(root, "--python").status).toBe(0);
});

test("only a demo file may be missing, and only when it is allowed to be", async () => {
  const root = await sampleRoot();
  rmSync(path.join(root, "demo/swap.json"));
  edit(root, "manifest.json", (manifest) => {
    manifest.files = manifest.files.filter((entry: Json) => entry.path !== "demo/swap.json");
  });
  expect(validate(root).output).toContain("demo/swap.json: missing");
  expect(validate(root, "--python").status).toBe(1);
  const allowed = validate(root, "--allow-missing", "demo/swap.json", "--python");
  expect(allowed.status, allowed.output).toBe(0);
  expect(runScript("validate-data.mjs", ["--root", root, "--python"], { SITE_DATA_ALLOW_MISSING: "demo/swap.json" }).status).toBe(0);
  // No other file may be waved through.
  const refused = validate(root, "--allow-missing", "results.json");
  expect(refused.status).toBe(2);
  expect(refused.output).toContain("may be missing, not results.json");
  expect((await pythonValidate(root, "--allow-missing", "results.json")).status).not.toBe(0);
});

test("a folder with no data root is refused", () => {
  const missing = validate(path.join(__dirname, "no-such-folder"));
  expect(missing.status).toBe(1);
  expect(missing.output).toContain("no data root");
});

test("the private-string scan finds what find_private in export/schemas.py finds", async () => {
  const probes = [
    "reports/xfg.md",
    "$DATA_ROOT/analytics/xfg/2015-16/xfg_predictions.parquet",
    "12.16.2015.PHX.at.GSW",
    "p50 1.92 ft at 5 Hz",
    `${USERS}someone/project`,
    `see ${HOME}someone and ${rooted("private", "tmp", "x")} and ${rooted("var", "folders", "ab")}`,
    `docs${USERS}guide.md`,
    `a.b${rooted("tmp", "c")}`,
    rooted("tmp", "scratch"),
    rooted("Volumes", "Disk", "data"),
    rooted("content", "drive", "data"),
    rooted("opt", "anaconda3", "bin", "python"),
    rooted("root", "x"),
    ["C:", "Users", "someone"].join("\\"),
    `~${rooted("Developer", "x")}`,
    `$~${SLASH}x and a~${SLASH}b`,
    EMAIL,
    `${["first.last+tag", "mail.example.org"].join("@")} and ${["other", "example.io"].join("@")}`,
    `Library/${CLOUD}/${GDRIVE}-account/${MY_DRIVE}/x`,
    MY_DRIVE.replace(" ", ""),
    `caf${String.fromCodePoint(0xe9)}${"@"}example.com`,
    `${USERS}escaped`.replaceAll(SLASH, `\\${SLASH}`),
    // A web address is not a local path, whatever its folders are called.
    `${WEB}${HOME}page`,
    `read "${WEB}${rooted("tmp", "x")}" first`,
    `${WEB}${USERS}x and ${USERS}y`,
    `(${WEB}${rooted("private", "x")})${rooted("private", "y")}`,
  ];
  const contract = (await import("../../scripts/lib/contract.mjs")) as unknown as {
    privateScanner(): (text: string) => string[];
  };
  const findPrivate = contract.privateScanner();
  const web = probes.map((probe) => findPrivate(probe));
  const { runPython } = await python();
  const result = runPython([
    "-c",
    "import json, sys; from export import schemas as S; print(json.dumps([S.find_private(p) for p in json.loads(sys.argv[1])]))",
    JSON.stringify(probes),
  ]);
  expect(result.status, result.stderr).toBe(0);
  expect(web).toEqual(JSON.parse(result.stdout));
  // And the scan is not empty-handed: these probes are private, those are not.
  expect(web.slice(0, 4).every((found) => found.length === 0)).toBe(true);
  expect(web[4]).toEqual([USERS]);
  expect(web[16]).toEqual([EMAIL]);
  expect(web[22]).toEqual([]);
  expect(web[24]).toEqual([USERS]);
});

test("prepare-data copies the contract files and fails clearly when there is no data", async () => {
  const root = await sampleRoot();
  writeFileSync(path.join(root, "notes.txt"), "not a contract file");
  writeFileSync(path.join(root, ".DS_Store"), "x");
  const target = path.join(root, "..", `${path.basename(root)}-prepared`);
  const done = runScript("prepare-data.mjs", ["--to", target], { SITE_DATA_DIR: root });
  expect(done.status, done.output).toBe(0);
  expect(done.output).toContain("21 file(s)");
  expect(done.output).toContain("notes.txt is not a contract file and was not copied");
  expect(existsSync(path.join(target, "games/fixture-game-a/1.json"))).toBe(true);
  expect(existsSync(path.join(target, "notes.txt"))).toBe(false);
  expect(existsSync(path.join(target, ".DS_Store"))).toBe(false);

  const empty = path.join(root, "..", `${path.basename(root)}-empty`);
  mkdirSync(empty);
  const failed = runScript("prepare-data.mjs", ["--to", target], { SITE_DATA_DIR: empty });
  expect(failed.status).toBe(1);
  expect(failed.output).toContain("no site data");
  expect(failed.output).toContain("python3 -m export.sample --out web/data");
  // A failed run leaves the last prepared copy alone.
  expect(existsSync(path.join(target, "manifest.json"))).toBe(true);
});
