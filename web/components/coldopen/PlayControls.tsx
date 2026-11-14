"use client";

import { useId } from "react";
import { copy, fill } from "@/lib/copy";
import { formatSeconds } from "@/lib/format";
import type { Playback } from "./hooks";
import styles from "./coldopen.module.css";

const ui = copy.home.ui;

/** Play or pause a stretch and drag its time bar. Both work with the keyboard. */
export function PlayControls({ playback, times }: { playback: Playback; times: readonly number[] }) {
  const id = useId();
  const total = times.length;
  const now = times[playback.index] ?? 0;
  const end = times[total - 1] ?? 0;
  const value = fill(ui.timeValue, { t: formatSeconds(now), total: formatSeconds(end) });
  if (total < 2) return null;
  return (
    <div className={styles.controls}>
      <button type="button" className={styles.play} onClick={playback.toggle} aria-pressed={playback.playing}>
        {playback.playing ? ui.pause : ui.play}
      </button>
      <label htmlFor={id} className="visually-hidden">
        {ui.time}
      </label>
      <input
        id={id}
        type="range"
        min={0}
        max={total - 1}
        step={1}
        value={playback.index}
        aria-valuetext={value}
        onChange={(event) => playback.setIndex(Number(event.target.value))}
      />
      <span className={`num ${styles.time}`}>{value}</span>
    </div>
  );
}
