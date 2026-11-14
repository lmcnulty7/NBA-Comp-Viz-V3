"use client";

import { useCallback, useEffect, useRef, useState, useSyncExternalStore, type RefObject } from "react";
import { frameAtTime } from "@/components/court/scene";

const REDUCE = "(prefers-reduced-motion: reduce)";

function subscribeMotion(onChange: () => void): () => void {
  const query = window.matchMedia(REDUCE);
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}

/**
 * True when the reader asked for reduced motion. The server and the first paint assume they did, so nothing
 * starts to move before the browser has said it may.
 */
export function useReducedMotion(): boolean {
  return useSyncExternalStore(
    subscribeMotion,
    () => window.matchMedia(REDUCE).matches,
    () => true,
  );
}

/** True while the element is at least partly in the window. */
export function useInView<T extends Element>(ref: RefObject<T | null>, margin = "0px"): boolean {
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const element = ref.current;
    if (!element || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver((entries) => setInView(entries.some((entry) => entry.isIntersecting)), {
      rootMargin: margin,
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref, margin]);
  return inView;
}

export interface Playback {
  /** The frame on screen. */
  index: number;
  playing: boolean;
  setIndex: (index: number) => void;
  toggle: () => void;
  /** Follow an outside clock (a playing video): the frame closest to `t` seconds. */
  follow: (t: number) => void;
}

/**
 * Plays a run of sampled frames in real time, looping, while `active` is true. It starts on its own only when
 * `autoplay` is true (never for a reader who asked for reduced motion); Play and Pause always work.
 */
export function usePlayback(times: readonly number[], options: { active: boolean; autoplay: boolean; start?: number }): Playback {
  const { active, autoplay, start = 0 } = options;
  const [index, setIndexState] = useState(start);
  const [chosen, setChosen] = useState<boolean | null>(null);
  const clock = useRef(times[start] ?? 0);
  // On its own it plays only while in view; a reader who pressed Play gets play at once.
  const playing = times.length > 1 && (chosen ?? autoplay) && (active || chosen === true);
  const first = times[0] ?? 0;
  const last = times[times.length - 1] ?? 0;

  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    let before: number | null = null;
    const span = Math.max(last - first, 1e-6);
    const step = (now: number) => {
      if (before !== null) {
        let t = clock.current + (now - before) / 1000;
        if (t > last) t = first + ((t - first) % span);
        clock.current = t;
        const next = frameAtTime(times, t);
        setIndexState((current) => (current === next ? current : next));
      }
      before = now;
      frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [playing, times, first, last]);

  const setIndex = useCallback(
    (next: number) => {
      const bounded = Math.max(0, Math.min(times.length - 1, Math.round(next)));
      clock.current = times[bounded] ?? 0;
      setIndexState(bounded);
    },
    [times],
  );
  const toggle = useCallback(() => setChosen(!playing), [playing]);
  const follow = useCallback(
    (t: number) => {
      clock.current = t;
      const next = frameAtTime(times, t);
      setIndexState((current) => (current === next ? current : next));
    },
    [times],
  );
  return { index, playing, setIndex, toggle, follow };
}

/**
 * The step of a scene that is being read: the last step whose top has passed the middle of the window. A step
 * that takes keyboard focus becomes the active one, so a keyboard reader drives the picture too.
 */
export function useActiveStep(container: RefObject<HTMLElement | null>, count: number): [number, (step: number) => void] {
  const [active, setActive] = useState(0);
  useEffect(() => {
    const root = container.current;
    if (!root || typeof IntersectionObserver === "undefined") return;
    const steps = [...root.querySelectorAll<HTMLElement>("[data-step]")];
    const visible = new Set<number>();
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const step = Number((entry.target as HTMLElement).dataset.step);
          if (entry.isIntersecting) visible.add(step);
          else visible.delete(step);
        }
        if (visible.size) setActive(Math.max(...visible));
      },
      { rootMargin: "-45% 0px -45% 0px" },
    );
    steps.forEach((step) => observer.observe(step));
    return () => observer.disconnect();
  }, [container, count]);
  return [active, setActive];
}
