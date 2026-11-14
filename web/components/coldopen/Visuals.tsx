"use client";

import { useMemo, useRef, useState } from "react";
import { Court } from "@/components/court/Court";
import { copy, fill } from "@/lib/copy";
import { formatSeconds } from "@/lib/format";
import type { DemoFrame } from "@/lib/types/contract";
import { useInView, usePlayback, useReducedMotion } from "./hooks";
import { MediaFrame, type StoryMedia } from "./MediaFrame";
import { broadcastCourt, filledCourt, sensorCourt, type Bar, type PairRow, type PositionRow } from "./model";
import { FillShare, GapPanel, PairTable, ScoreBars, TruthLegend } from "./Panels";
import { PlayControls } from "./PlayControls";
import styles from "./coldopen.module.css";

const ui = copy.home.ui;
const open = copy.home.coldOpen;

/** Everything the pictures of the story need, worked out on the server (components/coldopen/model.ts). */
export interface StoryData {
  variant: "public" | "private";
  media: StoryMedia | null;
  /** The sampled frames of the scene the story plays. */
  frames: DemoFrame[];
  /** The frame of the still. */
  stillIndex: number;
  bars: Bar[];
  position: PositionRow | null;
  /** The readout of scene 4 for each frame. */
  readouts: string[];
  swap: { fillShare: string; share: number; pairs: PairRow[] } | null;
}

function NoDemo() {
  return <p className={`small ${styles.noPicture}`}>{ui.noDemo}</p>;
}

/** Scene 1: the clip on the private build, the still on the public one. */
export function ClipVisual({ data }: { data: StoryData }) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref);
  const reduced = useReducedMotion();
  if (!data.media) return <NoDemo />;
  return (
    <div ref={ref} className={styles.visualStack} data-visual="clip">
      <MediaFrame media={data.media} boxes={null} moving={data.variant === "private"} play={inView && !reduced} />
    </div>
  );
}

/** Scene 2: the frame frozen, the boxes on it, then the three tracking rows against their targets. */
export function BoxesVisual({ data, stage }: { data: StoryData; stage: number }) {
  if (!data.media) return <NoDemo />;
  const frame = data.frames[data.stillIndex];
  return (
    <div className={styles.visualStack} data-visual="boxes" data-stage={stage}>
      <MediaFrame media={data.media} boxes={frame?.boxes ?? null} outlineUnmatched={stage >= 1} />
      <ScoreBars bars={data.bars} shown={stage >= 1} />
    </div>
  );
}

function useStretch(data: StoryData, start: number, driven = false) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref);
  const reduced = useReducedMotion();
  const times = useMemo(() => data.frames.map((frame) => frame.t), [data.frames]);
  // A playing clip drives the frame on screen itself; the timer runs only when nothing else does.
  const playback = usePlayback(times, { active: inView && !(driven && !reduced), autoplay: !reduced, start });
  return { ref, times, playback, inView, reduced };
}

/**
 * Scene 3: the frame with its boxes beside the court map. On the private build the clip plays and the dots
 * follow it; on the public build the still stays and the map plays through the sampled frames.
 */
export function MapVisual({ data }: { data: StoryData }) {
  const [clipFailed, setClipFailed] = useState(false);
  const moving = data.variant === "private" && data.media?.clip != null && !clipFailed;
  const { ref, times, playback, inView, reduced } = useStretch(data, data.stillIndex, moving);
  const frame = data.frames[playback.index];
  const scene = useMemo(() => broadcastCourt(frame, false), [frame]);
  if (!data.media || data.frames.length === 0) return <NoDemo />;
  const clipPlays = moving && !reduced;
  return (
    <div ref={ref} className={styles.visualStack} data-visual="map">
      <div className={styles.split}>
        <MediaFrame
          media={data.media}
          boxes={(clipPlays ? frame : data.frames[data.stillIndex])?.boxes ?? null}
          outlineUnmatched
          moving={clipPlays}
          play={clipPlays && inView}
          onTime={clipPlays ? playback.follow : undefined}
          onFailed={() => setClipFailed(true)}
        />
        <Court scene={scene} label={fill(ui.mapLabel, { t: formatSeconds(frame?.t ?? 0) })} />
      </div>
      {clipPlays ? null : <PlayControls playback={playback} times={times} />}
    </div>
  );
}

/**
 * The public build holds no sensor position (PLAN.md D60): where a court would draw them, it shows the still
 * and says that the court sits on the private copy (D21).
 */
function PrivateStill({ data, note }: { data: StoryData; note: string }) {
  if (!data.media) return <NoDemo />;
  return (
    <figure className={styles.courtFigure} data-testid="private-still">
      <MediaFrame media={data.media} boxes={null} />
      <figcaption className={styles.caption} data-testid="access-note">
        {note}
      </figcaption>
    </figure>
  );
}

/**
 * Scene 4: the sensor positions on the same map, the gap on screen and the scorecard row beside it. The public
 * build shows the still in place of the map, with the gap of the still's frame beside the scorecard row (D50).
 */
export function TruthVisual({ data }: { data: StoryData }) {
  return data.variant === "public" ? <TruthStill data={data} /> : <TruthPlayback data={data} />;
}

function TruthStill({ data }: { data: StoryData }) {
  if (data.frames.length === 0) return <NoDemo />;
  return (
    <div className={styles.visualStack} data-visual="truth">
      <PrivateStill data={data} note={ui.truthPrivate} />
      <GapPanel readout={data.readouts[data.stillIndex] ?? open.truth.labels.noGap} row={data.position} />
    </div>
  );
}

function TruthPlayback({ data }: { data: StoryData }) {
  const { ref, times, playback } = useStretch(data, data.stillIndex);
  const frame = data.frames[playback.index];
  const scene = useMemo(() => broadcastCourt(frame, true), [frame]);
  if (data.frames.length === 0) return <NoDemo />;
  return (
    <div ref={ref} className={styles.visualStack} data-visual="truth">
      <Court scene={scene} label={fill(ui.truthLabel, { t: formatSeconds(frame?.t ?? 0) })} />
      <TruthLegend />
      <PlayControls playback={playback} times={times} />
      <GapPanel readout={data.readouts[playback.index] ?? open.truth.labels.noGap} row={data.position} />
    </div>
  );
}

/** The numbers of scene 5: the fill share and the pairs of values, or the words that this build has none. */
function SwapNumbers({ data }: { data: StoryData }) {
  return data.swap ? (
    <>
      <FillShare value={data.swap.fillShare} share={data.swap.share} />
      <PairTable rows={data.swap.pairs} />
    </>
  ) : (
    <p className="small">{ui.noSwap}</p>
  );
}

/**
 * Scene 5: the same moment from sensor and from broadcast positions, marked as an illustration. The public build
 * shows the still in place of the two courts (PLAN.md D60); the numbers of the comparison stay.
 */
export function SwapVisual({ data }: { data: StoryData }) {
  if (data.variant === "public") {
    return (
      <div className={styles.visualStack} data-visual="swap">
        <p className={styles.badge} data-testid="illustration-badge">
          {open.swap.labels.badge}
        </p>
        {data.frames.length > 0 ? <PrivateStill data={data} note={ui.swapPrivate} /> : null}
        <SwapNumbers data={data} />
      </div>
    );
  }
  return <SwapPlayback data={data} />;
}

function SwapPlayback({ data }: { data: StoryData }) {
  const { ref, times, playback } = useStretch(data, data.stillIndex);
  const frame = data.frames[playback.index];
  const sensor = useMemo(() => sensorCourt(frame), [frame]);
  const broadcast = useMemo(() => filledCourt(frame), [frame]);
  const t = formatSeconds(frame?.t ?? 0);
  return (
    <div ref={ref} className={styles.visualStack} data-visual="swap">
      <p className={styles.badge} data-testid="illustration-badge">
        {open.swap.labels.badge}
      </p>
      {data.frames.length > 0 ? (
        <>
          <div className={styles.split}>
            <figure className={styles.courtFigure}>
              <Court scene={sensor} label={fill(ui.sensorLabel, { t })} layers={{ truth: false }} />
              <figcaption>{open.swap.labels.sensor}</figcaption>
            </figure>
            <figure className={styles.courtFigure}>
              <Court scene={broadcast} label={fill(ui.broadcastLabel, { t })} />
              <figcaption>{open.swap.labels.broadcast}</figcaption>
            </figure>
          </div>
          <PlayControls playback={playback} times={times} />
        </>
      ) : null}
      <SwapNumbers data={data} />
    </div>
  );
}
