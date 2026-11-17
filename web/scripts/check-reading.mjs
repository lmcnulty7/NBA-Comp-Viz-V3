// Reading grade and plain words on the cold open and the slides (PLAN.md M4 and M10, docs/SCRIPT.md).
//
// The copy of the cold open lives in web/content/home.json (coldOpen), the slides in web/content/talk. This
// check fails when
//   - a scene of the cold open, or one of its lines other than the headline, reads above grade 9, with a token
//     counted as one syllable or as four (scripts/lib/readability.mjs, the rule of tests/test_public_docs.py);
//   - a slide, or one of its on-screen lines other than the headline, reads above grade 9 the same way;
//   - a line or a label on screen uses a word kept out of the copy (homography, xFG, held-out, median, ...) or
//     makes a claim the site never makes (best, optimal, should, passes, proves);
//   - the copy of a scene, or its labels, is not the approved copy of docs/SCRIPT.md, line for line;
//   - a slide of a scene does not carry the lines of that scene;
//   - copy uses a token that the token table of docs/SCRIPT.md does not list;
//   - talk/deck.json is missing or lists no slide;
//   - a machine-written text of the data root (genai/*.json, PLAN.md P3) breaks a rule of site copy other than the
//     no-digit rule: grade, plain words, claims, dashes, one paragraph (scripts/lib/generated-text.mjs).
// Presenter notes are left out of the grade and the word list: SCRIPT.md lets them name the methods.
//
//   node scripts/check-reading.mjs [--content <dir>] [--script <SCRIPT.md>] [--data <data root>]
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { checkGeneratedTexts } from "./lib/generated-text.mjs";
import { CONTENT_DIR, dataRoot, shown } from "./lib/paths.mjs";
import { TOKEN_RE, fkGrade, plainWordProblems, round1 } from "./lib/readability.mjs";
import { SCRIPT_FILE, scriptScenes, scriptTokens } from "./lib/script-doc.mjs";
import { onScreenLines, parseSlide } from "./lib/slides.mjs";

export const MAX_GRADE = 9;
// The scenes of the cold open in web/content/home.json, in the order of docs/SCRIPT.md.
export const SCENE_KEYS = ["headline", "clip", "boxes", "map", "truth", "swap", "next"];

function grades(lines) {
  const text = lines.join("\n\n");
  return { grade: round1(fkGrade(text)), upper: round1(fkGrade(text, 4)) };
}

function gradeProblems(where, lines, headline) {
  const problems = [];
  const all = grades(lines);
  if (all.grade > MAX_GRADE || all.upper > MAX_GRADE) {
    problems.push(`${where}: reads at grade ${all.grade} (upper ${all.upper}), above ${MAX_GRADE}`);
  }
  for (const line of lines) {
    if (line === headline) continue;
    const one = round1(fkGrade(line));
    const upper = round1(fkGrade(line, 4));
    if (one > MAX_GRADE || upper > MAX_GRADE) {
      problems.push(`${where}: one line reads at grade ${one} (upper ${upper}): "${line.slice(0, 80)}"`);
    }
  }
  return { problems, ...all };
}

function sameSet(a, b) {
  const left = new Set(a);
  const right = new Set(b);
  return {
    missing: [...right].filter((line) => !left.has(line)),
    extra: [...left].filter((line) => !right.has(line)),
  };
}

function strings(value) {
  if (typeof value === "string") return [value];
  if (Array.isArray(value)) return value.flatMap(strings);
  if (value && typeof value === "object") return Object.values(value).flatMap(strings);
  return [];
}

/** The copy of one scene of web/content/home.json: its on-screen lines and its labels. */
export function sceneCopy(scene) {
  const lines = [];
  if (scene.title) lines.push(scene.title);
  lines.push(...(scene.steps ?? []));
  if (scene.caption) lines.push(scene.caption);
  if (scene.fallback) lines.push(scene.fallback);
  return { lines, labels: strings(scene.labels ?? {}) };
}

export function checkReading({ contentDir = CONTENT_DIR, scriptFile = SCRIPT_FILE, dataDir = dataRoot() } = {}) {
  const problems = [];
  const report = [];
  const scriptText = readFileSync(scriptFile, "utf8");
  const script = scriptScenes(scriptText);
  const tokens = scriptTokens(scriptText);
  const home = JSON.parse(readFileSync(path.join(contentDir, "home.json"), "utf8"));
  const coldOpen = home.coldOpen ?? {};
  const headline = coldOpen.headline?.title;

  const knownTokens = (where, line) => {
    for (const match of line.matchAll(TOKEN_RE)) {
      if (!tokens.has(match[1])) problems.push(`${where}: {${match[1]}} is not in the token table of docs/SCRIPT.md`);
    }
  };

  // The cold open, scene by scene.
  if (Object.keys(coldOpen).join() !== SCENE_KEYS.join()) {
    problems.push(`home.json coldOpen: the scenes are ${Object.keys(coldOpen).join(", ")}, not ${SCENE_KEYS.join(", ")}`);
  }
  SCENE_KEYS.forEach((key, number) => {
    const scene = coldOpen[key];
    const approved = script.find((candidate) => candidate.number === number);
    const where = `home.json coldOpen.${key}`;
    if (!scene || !approved) {
      problems.push(`${where}: missing (scene ${number} of docs/SCRIPT.md)`);
      return;
    }
    const { lines, labels } = sceneCopy(scene);
    const copyDiff = sameSet(lines, approved.copy);
    for (const line of copyDiff.missing) problems.push(`${where}: lacks the approved line "${line.slice(0, 80)}"`);
    for (const line of copyDiff.extra) problems.push(`${where}: "${line.slice(0, 80)}" is not a line of scene ${number}`);
    const labelDiff = sameSet(labels, approved.labels);
    for (const text of labelDiff.missing) problems.push(`${where}.labels: lacks the approved label "${text}"`);
    for (const text of labelDiff.extra) problems.push(`${where}.labels: "${text}" is not a label of scene ${number}`);
    const graded = gradeProblems(where, lines, headline);
    problems.push(...graded.problems);
    for (const line of [...lines, ...labels]) {
      knownTokens(where, line);
      for (const found of plainWordProblems(line, { allowGroundTruth: line === headline })) {
        problems.push(`${where}: ${found} in "${line.slice(0, 80)}"`);
      }
    }
    report.push({ where: `scene ${number} (${key})`, grade: graded.grade, upper: graded.upper });
  });

  // The words around the pictures.
  for (const line of strings(home.ui ?? {})) {
    knownTokens("home.json ui", line);
    for (const found of plainWordProblems(line)) problems.push(`home.json ui: ${found} in "${line}"`);
    if (/[.!?]$/.test(line)) {
      const one = round1(fkGrade(line));
      if (one > MAX_GRADE) problems.push(`home.json ui: "${line}" reads at grade ${one}`);
    }
  }

  // The slides.
  // The deck is required: a missing deck would leave every slide unchecked, so it fails instead.
  const deckFile = path.join(contentDir, "talk", "deck.json");
  if (!existsSync(deckFile)) {
    problems.push("talk/deck.json: missing, so no slide could be checked");
  } else {
    const deck = JSON.parse(readFileSync(deckFile, "utf8"));
    if (!Array.isArray(deck.slides) || deck.slides.length === 0) problems.push("talk/deck.json: lists no slide");
    const caption = coldOpen.clip?.caption;
    for (const id of deck.slides ?? []) {
      const file = path.join(contentDir, "talk", `${id}.md`);
      const where = `talk/${id}.md`;
      if (!existsSync(file)) {
        problems.push(`${where}: listed in deck.json and missing`);
        continue;
      }
      let slide;
      try {
        slide = parseSlide(readFileSync(file, "utf8"), where);
      } catch (error) {
        problems.push(error.message);
        continue;
      }
      const lines = onScreenLines(slide);
      const graded = gradeProblems(where, lines, headline);
      problems.push(...graded.problems);
      for (const line of [...lines, ...slide.notes]) knownTokens(where, line);
      for (const line of lines) {
        for (const found of plainWordProblems(line, { allowGroundTruth: line === headline })) {
          problems.push(`${where}: ${found} in "${line.slice(0, 80)}"`);
        }
      }
      const scene = slide.meta.scene;
      if (scene) {
        const number = SCENE_KEYS.indexOf(scene);
        const approved = script.find((candidate) => candidate.number === number);
        if (!approved) {
          problems.push(`${where}: scene ${JSON.stringify(scene)} is not a scene of docs/SCRIPT.md`);
        } else {
          const shownLines = lines.filter((line) => line !== slide.title || line === headline);
          const diff = sameSet(shownLines, approved.copy.filter((line) => line !== caption));
          for (const line of diff.missing) problems.push(`${where}: lacks the line of scene ${number} "${line.slice(0, 80)}"`);
          for (const line of diff.extra) problems.push(`${where}: "${line.slice(0, 80)}" is not a line of scene ${number}`);
        }
      }
      report.push({ where, grade: graded.grade, upper: graded.upper });
    }
  }

  // The machine-written texts of the data root.
  const generated = checkGeneratedTexts(dataDir);
  problems.push(...generated.problems);
  return { problems, report, generated: generated.texts };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const flag = (name) => {
    const index = process.argv.indexOf(name);
    return index > 0 ? path.resolve(process.argv[index + 1]) : undefined;
  };
  const contentDir = flag("--content") ?? CONTENT_DIR;
  try {
    const dataDir = flag("--data") ?? dataRoot();
    const { problems, report, generated } = checkReading({ contentDir, scriptFile: flag("--script") ?? SCRIPT_FILE, dataDir });
    for (const row of report) console.log(`reading: ${row.where.padEnd(24)} grade ${row.grade}, upper ${row.upper}`);
    console.log(`reading: ${generated} machine-written text(s) in ${shown(dataDir)} checked`);
    for (const line of problems) console.error(`reading: ${line}`);
    if (problems.length) {
      console.error(`reading: ${problems.length} problem(s) in ${shown(contentDir)}`);
      process.exitCode = 1;
    } else {
      console.log(`reading: the cold open and the slides read at grade ${MAX_GRADE} or lower, in plain words`);
    }
  } catch (error) {
    console.error(`reading: ${error.message}`);
    process.exitCode = 1;
  }
}
