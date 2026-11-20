import { expect, test } from "@playwright/test";
import { existsSync } from "node:fs";
import path from "node:path";
import { storyData } from "@/components/coldopen/story";
import { stillUrl, stillWebpUrl } from "@/lib/media";
import type { DemoFrames } from "@/lib/types/contract";
import { WEB_DIR } from "../helpers/scripts";

// The stills of the cold open (D116): the data names the JPEG; the WebP beside it, when the build holds one, is
// what a browser that takes WebP is given, and the JPEG stays the fallback.

const HAS_WEBP = existsSync(path.join(WEB_DIR, "public", "stills", "scene-1.webp"));

test("the WebP beside a still is found by its name, and nothing else is", () => {
  test.skip(!HAS_WEBP, "this checkout holds no WebP still");
  expect(stillWebpUrl("stills/scene-1.jpg")).toBe(stillUrl("stills/scene-1.webp"));
  expect(stillWebpUrl("stills/scene-1.jpg")).toMatch(/stills\/scene-1\.webp$/);
});

for (const [name, still] of [
  ["a still the build does not hold", "stills/no-such-scene.jpg"],
  ["a still that is not a JPEG", "stills/scene-1.png"],
  ["a path out of public/", "../stills/scene-1.jpg"],
  ["no still", null],
] as const) {
  test(`no WebP for ${name}`, () => {
    expect(stillWebpUrl(still)).toBeNull();
  });
}

test("the story carries the WebP only beside a JPEG this build holds", () => {
  const frames = {
    scenes: [{ clip: "scene-1.mp4", still: "stills/scene-1.jpg", width: 1280, height: 720, frames: [], still_t: 0 }],
  } as unknown as DemoFrames;
  const story = storyData({ frames }, "public", () => "/stills/scene-1.jpg", () => "/stills/scene-1.webp");
  expect(story.media?.stillWebp).toBe("/stills/scene-1.webp");
  expect(storyData({ frames }, "public", () => null, () => "/stills/scene-1.webp").media?.stillWebp).toBeNull();
  expect(storyData({ frames }, "public", () => "/stills/scene-1.jpg").media?.stillWebp).toBeNull();
});
