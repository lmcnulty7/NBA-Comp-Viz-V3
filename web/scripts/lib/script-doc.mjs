// docs/SCRIPT.md, read the way tests/test_public_docs.py reads it: the scenes with their on-screen copy (the
// quoted lines) and their labels (the double-quoted strings under Labels), and the token table.
import { readFileSync } from "node:fs";
import path from "node:path";
import { REPO_ROOT } from "./paths.mjs";

export const SCRIPT_FILE = path.join(REPO_ROOT, "docs", "SCRIPT.md");

const SCENE_RE = /^## Scene (\d+)\. (.+)$/gm;

function blockOf(body, label) {
  const start = body.indexOf(`**${label}.**`);
  if (start === -1) return "";
  const rest = body.slice(start + label.length + 5);
  const following = /^\*\*[A-Z][A-Za-z ]+\.\*\*/m.exec(rest);
  return following ? rest.slice(0, following.index) : rest;
}

/** The scenes of the script: number, title, copy (quoted lines), labels and presenter notes. */
export function scriptScenes(text = readFileSync(SCRIPT_FILE, "utf8")) {
  const starts = [...text.matchAll(SCENE_RE)];
  return starts.map((match, index) => {
    const from = match.index + match[0].length;
    let end;
    if (index + 1 < starts.length) end = starts[index + 1].index;
    else {
      const following = /^## /m.exec(text.slice(from));
      end = following ? from + following.index : text.length;
    }
    const body = text.slice(from, end);
    const labels = [];
    for (const line of blockOf(body, "Labels").split("\n")) {
      if (!line.startsWith("- ")) continue;
      for (const found of line.split(" (")[0].matchAll(/"([^"]+)"/g)) labels.push(found[1]);
    }
    const notes = blockOf(body, "Presenter notes")
      .split(/\n(?=- )/)
      .map((item) => item.replace(/^- /, "").split(/\s+/).filter(Boolean).join(" "))
      .filter(Boolean);
    return {
      number: Number(match[1]),
      title: match[2],
      copy: body.split("\n").filter((line) => line.startsWith(">")).map((line) => line.slice(1).trim()),
      labels,
      notes,
    };
  });
}

/** The token table: token -> { file, field, shownAs, copiedFrom }. */
export function scriptTokens(text = readFileSync(SCRIPT_FILE, "utf8")) {
  const section = text.split("\n## Tokens\n")[1]?.split("\n## ")[0] ?? "";
  const table = new Map();
  for (const line of section.split("\n")) {
    const cells = line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
    if (cells.length === 5 && cells[0].startsWith("`{")) {
      const token = cells[0].replace(/`/g, "").slice(1, -1);
      table.set(token, {
        file: cells[1].replace(/`/g, ""),
        field: cells[2].replace(/`/g, ""),
        shownAs: cells[3],
        copiedFrom: cells[4],
      });
    }
  }
  return table;
}
