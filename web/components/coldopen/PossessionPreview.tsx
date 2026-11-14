"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Court } from "@/components/court/Court";
import { EMPTY_SCENE, defaultFrame, lastShotSpot, sceneFromPossession } from "@/components/court/scene";
import { copy } from "@/lib/copy";
import { getBrowserSource } from "@/lib/data/browser";
import type { Possession } from "@/lib/types/contract";
import { useInView, usePlayback, useReducedMotion } from "./hooks";
import styles from "./coldopen.module.css";

const NO_TIMES: number[] = [];

/** A featured possession playing on the court renderer: the small live picture on the review card. */
export function PossessionPreview({ gameSlug, possession }: { gameSlug: string; possession: number }) {
  const [loaded, setLoaded] = useState<Possession | null>(null);
  const [failed, setFailed] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref);
  const reduced = useReducedMotion();

  useEffect(() => {
    let cancelled = false;
    getBrowserSource()
      .possession(gameSlug, possession)
      .then(
        (document) => {
          if (!cancelled) setLoaded(document);
        },
        () => {
          if (!cancelled) setFailed(true);
        },
      );
    return () => {
      cancelled = true;
    };
  }, [gameSlug, possession]);

  const times = loaded?.t ?? NO_TIMES;
  const start = loaded ? defaultFrame(loaded) : 0;
  const playback = usePlayback(times, { active: inView && loaded !== null, autoplay: !reduced, start });
  const frame = playback.playing || playback.index !== 0 ? playback.index : start;
  const scene = useMemo(() => {
    if (!loaded) return EMPTY_SCENE;
    const spot = lastShotSpot(loaded);
    return sceneFromPossession(loaded, frame, { markers: spot ? [{ kind: "spot", ...spot }] : [] });
  }, [loaded, frame]);

  if (failed) return <p className="small muted">{copy.home.ui.previewError}</p>;
  return (
    <div ref={ref} className={styles.preview} aria-busy={loaded === null}>
      <Court scene={scene} view="half" attackX={loaded?.attack_x ?? null} label={copy.home.ui.preview} layers={{ labels: false }} />
    </div>
  );
}
