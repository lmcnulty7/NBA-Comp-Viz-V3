import { expect, test } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { runScript, tempDir } from "../helpers/scripts";

// The methods number check (scripts/check-methods.mjs): every number on the built methods page is a number of
// methods.json, and every lane and test of the data is on the page.

interface MethodsCheck {
  numbersIn(text: string): string[];
  dataNumbers(document: unknown): Set<string>;
  mainText(html: string): string | null;
  methodsProblems(html: string, document: unknown): string[];
}

const load = async () => (await import("../../scripts/check-methods.mjs")) as unknown as MethodsCheck;
const MINUS = String.fromCodePoint(0x2212);

const DATA = {
  schema_version: "1.2.0",
  lanes: [
    {
      lane: "xfg",
      notes: ["Metres converted from feet by us (1 ft = 0.3048 m)."],
      benchmarks: [{ ours: [{ value: "-0.014", lo: "+0.0012", hi: "0.0020" }], published: "log loss -25,474.93 (Table 1, p. 81)" }],
      experiments: [{ id: "xfg-E1", rows: 101853, diff: -0.00052, threshold: 0.0005 }],
    },
  ],
};

function page(main: string, outside = ""): string {
  return `<html><body><header>Data build site-df7fd5d2cb9a, schema 1.2.0${outside}</header><main id="main">${main}</main>` +
    `<footer>Data schema 9.9.9</footer><script>self.__next_f.push([1,"77777"])</script></body></html>`;
}

const GOOD = page(
  `<section id="lane-xfg"><h2>Shot quality</h2><span class="num">${MINUS}0.014</span> <span>0.0012 to 0.0020</span>` +
    `<p>log loss -25,474.93 (Table 1, p. 81)</p><p>Metres converted from feet by us (1 ft = 0.3048 m).</p>` +
    `<li id="xfg-E1"><span>xfg-E1</span> <span>101,853 shots</span> <span>${MINUS}0.00052</span> <span>+0.0005</span>` +
    ` <code>reports/scorecard/gsw_phx_2016__r23.txt</code> <code>reports/research/xfg-E1-dev.json</code> p50 p90</li>` +
    `<script>var a = 4242;</script></section>`,
);

test("numbers are read the way the data writes them: minus, plus and thousands commas", async () => {
  const { numbersIn } = await load();
  expect(numbersIn(`${MINUS}0.014 +0.0012 101,853 81.9% p50 xfg-E1 gsw_phx_2016 2015-16 (Table 1, p. 81)`)).toEqual([
    "-0.014", "0.0012", "101853", "81.9%", "2015", "16", "1", "81",
  ]);
});

test("a page whose numbers all come from methods.json passes; header, footer and scripts are not read", async () => {
  const { methodsProblems, mainText } = await load();
  expect(mainText(GOOD)).not.toContain("4242");
  expect(methodsProblems(GOOD, DATA)).toEqual([]);
});

for (const [name, html, message] of [
  ["a typed number", GOOD.replace("Shot quality", "Shot quality, 17 games"), "the number 17 on the page is not in methods.json"],
  ["a value rounded by the page", GOOD.replace(`${MINUS}0.00052`, `${MINUS}0.0005`), "the number -0.0005 on the page"],
  ["a value changed by the page", GOOD.replace("0.0012 to", "0.0013 to"), "the number 0.0013"],
  ["a lane left out", GOOD.replace('id="lane-xfg"', 'id="lane-other"'), "lane xfg of methods.json is not on the page"],
  ["a test left out", GOOD.replace('id="xfg-E1"', 'id="other"'), "experiment xfg-E1 of methods.json is not on the page"],
  ["no main element", "<html><body>0.014</body></html>", "has no main element"],
] as const) {
  test(`the methods check sees ${name}`, async () => {
    const { methodsProblems } = await load();
    expect(methodsProblems(html, DATA).join("\n")).toContain(message);
  });
}

test("the command fails on a built page with a number the data does not hold, and passes on a clean one", () => {
  const out = tempDir("methods-out");
  mkdirSync(path.join(out, "methods"), { recursive: true });
  mkdirSync(path.join(out, "data"), { recursive: true });
  writeFileSync(path.join(out, "data", "methods.json"), JSON.stringify(DATA));
  writeFileSync(path.join(out, "methods", "index.html"), GOOD);
  const good = runScript("check-methods.mjs", ["--dir", out]);
  expect(good.status, good.output).toBe(0);
  writeFileSync(path.join(out, "methods", "index.html"), GOOD.replace("Shot quality", "Shot quality 0.6295"));
  const bad = runScript("check-methods.mjs", ["--dir", out]);
  expect(bad.status).toBe(1);
  expect(bad.output).toContain("the number 0.6295 on the page is not in methods.json");
  const empty = runScript("check-methods.mjs", ["--dir", tempDir("methods-none")]);
  expect(empty.status).toBe(1);
  expect(empty.output).toContain("build the site first");
});
