"use client";

import { useRef, type ReactNode } from "react";
import { copy, fill } from "@/lib/copy";
import { useActiveStep, useReducedMotion } from "./hooks";
import { BoxesVisual, ClipVisual, MapVisual, SwapVisual, TruthVisual, type StoryData } from "./Visuals";
import styles from "./coldopen.module.css";

export type StorySceneKey = "clip" | "boxes" | "map" | "truth" | "swap";

export interface StoryScene {
  key: StorySceneKey;
  /** The scroll steps, tokens already filled (or replaced by their fallback lines). */
  steps: string[];
  /** A fixed caption under the picture: the public build's request-access note. */
  caption?: string | null;
}

function visualOf(key: StorySceneKey, data: StoryData, stage: number): ReactNode {
  switch (key) {
    case "clip":
      return <ClipVisual data={data} />;
    case "boxes":
      return <BoxesVisual data={data} stage={stage} />;
    case "map":
      return <MapVisual data={data} />;
    case "truth":
      return <TruthVisual data={data} />;
    case "swap":
      return <SwapVisual data={data} />;
  }
}

function Scene({ scene, data, number, total }: { scene: StoryScene; data: StoryData; number: number; total: number }) {
  const ref = useRef<HTMLElement>(null);
  const [active, setActive] = useActiveStep(ref, scene.steps.length);
  const reduced = useReducedMotion();
  // With reduced motion every picture shows its last state at once.
  const stage = reduced ? scene.steps.length - 1 : active;
  return (
    <section
      ref={ref}
      className={styles.scene}
      aria-label={fill(copy.home.ui.scene, { n: number, total })}
      data-scene={scene.key}
      data-active-step={active}
    >
      <div className={styles.visual}>
        <div className={styles.sticky}>
          {visualOf(scene.key, data, stage)}
          {scene.caption ? (
            <p className={styles.caption} data-testid="access-note">
              {scene.caption}
            </p>
          ) : null}
        </div>
      </div>
      <ol className={styles.steps}>
        {scene.steps.map((step, index) => (
          <li
            key={step}
            className={styles.step}
            data-step={index}
            data-current={index === active ? "true" : undefined}
            tabIndex={0}
            onFocus={() => setActive(index)}
          >
            <p>{step}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}

/**
 * The scroll-driven cold open (PLAN.md M4, D25; docs/SCRIPT.md scenes 1 to 5). Each scene is a picture that
 * stays in view beside its scroll steps. A step that is read, or that takes keyboard focus, sets what the picture
 * shows. With reduced motion nothing plays on its own and each picture shows its final state.
 */
export function ColdOpen({ scenes, data }: { scenes: StoryScene[]; data: StoryData }) {
  return (
    <div className={styles.story} aria-label={copy.home.ui.story} role="group" data-testid="cold-open">
      {scenes.map((scene, index) => (
        <Scene key={scene.key} scene={scene} data={data} number={index + 1} total={scenes.length} />
      ))}
    </div>
  );
}
