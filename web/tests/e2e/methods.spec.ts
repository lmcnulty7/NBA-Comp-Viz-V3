import { expect, test } from "@playwright/test";
import { signedText } from "@/components/results/DataValue";
import { copy } from "@/lib/copy";
import { numberText } from "@/lib/format";
import { repoFileUrl } from "@/lib/repo";
import { walkField, type ShownAs } from "@/lib/tokens";
import type { MethodsFile } from "@/lib/types/contract";

// The methods and literature page (PLAN.md D72), read against the methods.json the built site serves.

interface Shown {
  field: string;
  shown: string;
  signed: boolean;
  text: string;
}

async function methodsData(request: import("@playwright/test").APIRequestContext): Promise<MethodsFile> {
  const served = await request.get("data/methods.json");
  expect(served.status()).toBe(200);
  return (await served.json()) as MethodsFile;
}

test("every value on the methods page equals methods.json", async ({ page }) => {
  const methods = await methodsData(page.request);
  await page.goto("methods/");
  const shown = await page.locator('main [data-file="methods.json"]').evaluateAll((nodes) =>
    nodes.map((node) => {
      const element = node as HTMLElement;
      return {
        field: element.dataset.field ?? "",
        shown: element.dataset.shown ?? "",
        signed: element.dataset.signed === "true",
        text: element.textContent ?? "",
      };
    }),
  );
  expect(shown.length).toBeGreaterThan(20);
  const wrong: string[] = [];
  for (const item of shown as Shown[]) {
    const value = walkField(methods, item.field);
    const want =
      item.shown === "text"
        ? String(value)
        : item.shown === "number text"
          ? numberText(String(value))
          : signedText(value as number, item.shown as ShownAs, item.signed);
    if (item.text !== want) wrong.push(`${item.field}: shows ${JSON.stringify(item.text)}, the file holds ${JSON.stringify(want)}`);
  }
  expect(wrong).toEqual([]);
});

test("every lane, paper, benchmark row and test of methods.json is on the page, with its state", async ({ page }) => {
  const methods = await methodsData(page.request);
  await page.goto("methods/");
  await expect(page.locator("h1")).toHaveText(copy.methods.title);
  await expect(page.locator("section[data-lane]")).toHaveCount(methods.lanes.length);
  for (const lane of methods.lanes) {
    const section = page.locator(`section[data-lane="${lane.lane}"]`);
    const papers = section.locator(`a[data-field^="lanes[lane=${lane.lane}].foundations["][data-field$=".cite"]`);
    await expect(papers).toHaveCount(lane.foundations.length);
    for (const [index, paper] of lane.foundations.entries()) await expect(papers.nth(index)).toHaveAttribute("href", paper.url);
    const rows = section.locator("li[data-kind]");
    await expect(rows).toHaveCount(lane.benchmarks.length);
    for (const [index, row] of lane.benchmarks.entries()) {
      await expect(rows.nth(index)).toHaveAttribute("data-kind", row.comparability_class);
      await expect(rows.nth(index)).toContainText(copy.methods.kind[row.comparability_class]);
      // A number of ours shows its interval, or says it is a point estimate or a setting.
      for (const [at, value] of row.ours.entries()) {
        const note = rows.nth(index).locator(`[data-field="lanes[lane=${lane.lane}].benchmarks[${index}].ours[${at}].value"] + .note`);
        if (value.lo == null) {
          await expect(note).toHaveText(value.kind === "setting" ? copy.methods.lane.setting : copy.methods.lane.point);
        } else {
          await expect(note).toHaveAttribute("data-interval", "true");
        }
        const url = repoFileUrl(value.source);
        if (url) await expect(rows.nth(index).locator(`a[data-report="${value.source}"]`).first()).toHaveAttribute("href", url);
      }
    }
    for (const experiment of lane.experiments) {
      const item = section.locator(`li[data-experiment="${experiment.id}"]`);
      await expect(item).toHaveAttribute("data-status", experiment.status);
      await expect(item).toContainText(copy.methods.status[experiment.status]);
    }
  }
  const ran = methods.lanes.flatMap((lane) => lane.experiments.filter((e) => e.status !== "pre_registered"));
  await expect(page.locator("[data-result]")).toHaveCount(ran.length);
  if (ran.length === 0) await expect(page.getByTestId("no-results")).toHaveText(copy.methods.results.none);
  for (const experiment of ran) {
    const result = page.locator(`[data-result="${experiment.id}"]`);
    await expect(result.locator("[data-stage]")).toHaveCount((experiment.development ? 1 : 0) + (experiment.test ? 1 : 0));
    if (experiment.status === "development_met_test_pending") {
      await expect(result.getByTestId("test-pending")).toHaveText(copy.methods.results.pending);
    }
    const protocol = experiment.protocol;
    await expect(result.locator("[data-protocol]")).toHaveCount(protocol ? 1 : 0);
    if (protocol) {
      // Each statement shows its reading, and its valid range marked post hoc or the words point estimate. The
      // ranges set up in advance are not valid (PLAN.md D86) and are never on the page.
      const statements = result.locator("[data-hypothesis]");
      await expect(statements).toHaveCount(protocol.hypotheses.length);
      for (const [index, h] of protocol.hypotheses.entries()) {
        await expect(statements.nth(index)).toHaveAttribute("data-reading", h.result);
        await expect(statements.nth(index)).toContainText(copy.methods.results.protocol.result[h.result]);
        const note = statements.nth(index).locator(".note");
        if (h.interval) await expect(note).toContainText(copy.methods.results.protocol.postHoc);
        else if (h.value != null) await expect(note).toHaveText(copy.methods.lane.point);
      }
      await expect(result.locator("[data-protocol]")).toHaveAttribute("data-adopted", protocol.adopted ? "true" : "false");
      if (protocol.caveat) await expect(result.getByTestId("protocol-caveat")).toHaveText(protocol.caveat);
      await expect(result.getByTestId("not-scored").locator("li")).toHaveCount(protocol.not_scored.length);
    }
    for (const stage of [experiment.development, experiment.test]) {
      if (!stage) continue;
      const facts = result.locator(`[data-stage="${stage.stage}"]`);
      await expect(facts).toHaveAttribute("data-met", stage.met ? "true" : "false");
      await expect(facts).toContainText(stage.met ? copy.methods.results.met : copy.methods.results.missed);
      const url = repoFileUrl(stage.source.path);
      if (url) await expect(facts.locator(`a[data-report="${stage.source.path}"]`)).toHaveAttribute("href", url);
    }
  }
});

test("the methods page is in the nav, and its contents links land on each lane", async ({ page }) => {
  const methods = await methodsData(page.request);
  await page.goto("methods/");
  const nav = page.getByRole("navigation", { name: copy.site.nav.label });
  await expect(nav.locator('[aria-current="page"]')).toHaveText(copy.site.nav.methods);
  const last = methods.lanes[methods.lanes.length - 1]!;
  await page.locator(`main nav a[href="#lane-${last.lane}"]`).click();
  await expect(page).toHaveURL(new RegExp(`#lane-${last.lane}$`));
  await expect(page.locator(`#lane-${last.lane} h2`)).toBeInViewport();
});
