// The slide format of web/content/talk/*.md (PLAN.md M10, D35). Pure: text in, a slide out, so the site
// (lib/talk.ts) and the reading check (scripts/check-reading.mjs) read a slide the same way.
//
//   ---
//   scene: truth          the cold-open scene the slide shows (docs/SCRIPT.md), when it shows one
//   visual: truth         what sits beside the copy: none, still, clip, boxes, map, truth, swap, next, results
//   ---
//   # The title
//
//   A paragraph of on-screen copy. Tokens such as {swap.fill_share} are filled from the site data.
//
//   Otherwise: the line shown in place of the paragraph above when one of its tokens has no value.
//
//   - a bullet of on-screen copy
//
//   ## Notes
//
//   - a presenter note, shown only in the presenter view

export const VISUALS = ["none", "still", "clip", "boxes", "map", "truth", "swap", "next", "results"];
const FALLBACK = "Otherwise: ";

function joinLines(lines) {
  return lines.map((line) => line.trim()).filter(Boolean).join(" ");
}

function blocksOf(text) {
  return text
    .split(/\n\s*\n/)
    .map((block) => block.split("\n").filter((line) => line.trim() !== ""))
    .filter((lines) => lines.length > 0);
}

function bulletsOf(lines) {
  const items = [];
  for (const line of lines) {
    if (line.startsWith("- ")) items.push(line.slice(2).trim());
    else if (items.length) items[items.length - 1] = `${items[items.length - 1]} ${line.trim()}`;
    else throw new Error(`a list starts with a line that is not a bullet: ${JSON.stringify(line)}`);
  }
  return items;
}

/** One slide file as { meta, title, blocks, notes }. Throws on anything the format does not allow. */
export function parseSlide(text, name = "slide") {
  const where = (message) => new Error(`${name}: ${message}`);
  const meta = {};
  let body = text.replace(/\r\n/g, "\n");
  if (body.startsWith("---\n")) {
    const end = body.indexOf("\n---\n", 4);
    if (end < 0) throw where("the front block has no closing ---");
    for (const line of body.slice(4, end).split("\n")) {
      if (!line.trim()) continue;
      const match = /^([a-z]+):\s*(.*)$/.exec(line.trim());
      if (!match) throw where(`cannot read the front line ${JSON.stringify(line)}`);
      meta[match[1]] = match[2];
    }
    body = body.slice(end + 5);
  }
  const visual = meta.visual ?? "none";
  if (!VISUALS.includes(visual)) throw where(`visual ${JSON.stringify(visual)} is not one of ${VISUALS.join(", ")}`);
  meta.visual = visual;

  const [copyPart, ...rest] = body.split(/^## Notes\s*$/m);
  if (rest.length > 1) throw where("more than one notes section");
  let title = null;
  const blocks = [];
  for (const lines of blocksOf(copyPart ?? "")) {
    const first = lines[0];
    if (first.startsWith("# ")) {
      if (title !== null) throw where("more than one title");
      if (lines.length > 1) throw where("a title is one line");
      title = first.slice(2).trim();
    } else if (first.startsWith("#")) {
      throw where(`only the title and the notes may be headings: ${JSON.stringify(first)}`);
    } else if (first.startsWith("- ")) {
      blocks.push({ kind: "list", items: bulletsOf(lines) });
    } else if (first.startsWith(FALLBACK)) {
      const previous = blocks[blocks.length - 1];
      if (!previous || previous.kind !== "p" || previous.fallback) {
        throw where("a fallback line must follow the paragraph it replaces");
      }
      previous.fallback = joinLines(lines).slice(FALLBACK.length);
    } else {
      blocks.push({ kind: "p", text: joinLines(lines) });
    }
  }
  if (title === null) throw where("no title (a line that starts with # )");
  const notesPart = rest[0] ?? "";
  const notes = [];
  for (const lines of blocksOf(notesPart)) {
    if (!lines[0].startsWith("- ")) throw where("presenter notes are bullets");
    notes.push(...bulletsOf(lines));
  }
  return { meta, title, blocks, notes };
}

/** Every line of a slide that is on screen: the title, the paragraphs, their fallbacks and the bullets. */
export function onScreenLines(slide) {
  const lines = [slide.title];
  for (const block of slide.blocks) {
    if (block.kind === "p") {
      lines.push(block.text);
      if (block.fallback) lines.push(block.fallback);
    } else {
      lines.push(...block.items);
    }
  }
  return lines;
}
