import { expect, test } from "@playwright/test";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { copy, fill } from "@/lib/copy";
import type { GeneratedTexts } from "@/lib/types/contract";
import { OUT_DIR, players, possessionIndex } from "./site";

/**
 * Machine-written texts (PLAN.md P3) on the built site: a page with a text shows it, labelled as machine-written
 * with its model and prompt version; a page without one shows no empty block. The data is whatever the site was
 * built from, so a build with no genai file checks the second half only.
 */
function texts(kind: "scouting" | "possession"): GeneratedTexts | null {
  const file = path.join(OUT_DIR, "data", "genai", `${kind}.json`);
  return existsSync(file) ? (JSON.parse(readFileSync(file, "utf8")) as GeneratedTexts) : null;
}

const machine = copy.site.machineText;

test("a player page shows its scouting report labelled as machine-written, and a page with none shows nothing", async ({
  page,
}) => {
  const scouting = texts("scouting");
  const keys = new Set(scouting?.items.map((item) => item.key) ?? []);
  const item = scouting?.items[0];
  if (item) {
    await page.goto(`prep/${item.key}/`);
    const block = page.getByTestId("machine-text");
    await expect(block).toHaveCount(1);
    await expect(block.getByRole("heading", { name: machine.title })).toBeVisible();
    await expect(block).toContainText(machine.label);
    await expect(block).toContainText(item.text);
    await expect(block).toContainText(fill(machine.note, { model: item.model, version: item.prompt_version }));
  }
  const without = players.players.find((player) => !keys.has(player.slug));
  if (without) {
    await page.goto(`prep/${without.slug}/`);
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(without.name);
    await expect(page.getByTestId("machine-text")).toHaveCount(0);
  }
});

test("a possession page shows its account in words, and a page with none shows nothing", async ({ page }) => {
  const possession = texts("possession");
  const keys = new Set(possession?.items.map((item) => item.key) ?? []);
  const item = possession?.items[0];
  if (item) {
    await page.goto(`review/${item.key}/`);
    const block = page.getByTestId("machine-text");
    await expect(block).toHaveCount(1);
    await expect(block).toContainText(machine.label);
    await expect(block).toContainText(item.text);
  }
  const slug = (item?.key ?? "").split("/")[0] || null;
  if (slug) {
    const row = possessionIndex(slug).possessions.find((candidate) => !keys.has(`${slug}/${candidate.possession}`));
    if (row) {
      await page.goto(`review/${slug}/${row.possession}/`);
      await expect(page.getByTestId("machine-text")).toHaveCount(0);
    }
  }
});
