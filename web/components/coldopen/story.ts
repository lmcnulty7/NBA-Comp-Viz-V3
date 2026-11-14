/**
 * The cold open of docs/SCRIPT.md as props: the copy of each scene with its tokens filled, and the data its
 * pictures need. Runs on the server while the site is built; pure, so a unit test can run it on any data root.
 */
import { copy } from "@/lib/copy";
import { assetUrl, type Variant } from "@/lib/site";
import { renderLine, type TokenData } from "@/lib/tokens";
import type { StoryScene, StorySceneKey } from "./ColdOpen";
import { gapReadouts, pairRows, positionRow, scorecardBars, stillFrameIndex, storyScene } from "./model";
import type { StoryData } from "./Visuals";

const open = copy.home.coldOpen;

export type SceneKey = "headline" | StorySceneKey | "next";

/** The fallback line of a step, by docs/SCRIPT.md: step 4.2 and steps 5.3 to 5.5 have one. */
export function fallbackOf(key: SceneKey, index: number): string | null {
  if (key === "truth" && index === 1) return open.truth.fallback;
  if (key === "swap" && index >= 2 && index <= 4) return open.swap.fallback;
  return null;
}

/**
 * The steps of one scene with their tokens filled. A token that names no field of the token table throws, so
 * the build fails. A step whose value this data build lacks shows its fallback line; a step with no fallback is
 * left out, and a comparison scene that lost a step says that this build has no comparison.
 */
export function storyLines(key: SceneKey, data: Partial<TokenData>): string[] {
  const lines: string[] = [];
  let lost = false;
  open[key].steps.forEach((step: string, index: number) => {
    const line = renderLine(step, fallbackOf(key, index), data);
    if (line === null) lost = true;
    else lines.push(line);
  });
  if (lost && key === "swap") lines.splice(1, 0, copy.home.ui.noSwap);
  return lines;
}

const STORY_KEYS: StorySceneKey[] = ["clip", "boxes", "map", "truth", "swap"];

export function storyScenes(data: Partial<TokenData>, variant: Variant): StoryScene[] {
  return STORY_KEYS.map((key) => ({
    key,
    steps: storyLines(key, data),
    caption: key === "clip" && variant === "public" ? open.clip.caption : null,
  }));
}

/** The data of the pictures. `stillUrl` says whether this build holds the still (lib/media.ts on the server). */
export function storyData(
  data: Partial<TokenData>,
  variant: Variant,
  stillUrl: (still: string) => string | null,
): StoryData {
  const scene = storyScene(data);
  const fillShare = renderLine("{swap.fill_share}", null, data);
  return {
    variant,
    media: scene
      ? {
          clip: variant === "private" ? assetUrl(`clips/${scene.clip}`) : null,
          still: stillUrl(scene.still),
          width: scene.width,
          height: scene.height,
        }
      : null,
    frames: scene?.frames ?? [],
    stillIndex: scene ? stillFrameIndex(scene) : 0,
    bars: data.scorecard ? scorecardBars(data.scorecard) : [],
    position: positionRow(data),
    readouts: gapReadouts(data),
    swap:
      data.swap && fillShare !== null
        ? { fillShare, share: data.swap.fill_share, pairs: pairRows(data.swap, scene) }
        : null,
  };
}
