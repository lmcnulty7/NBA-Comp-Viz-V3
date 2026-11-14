"use client";

import { memo, useCallback, useEffect, useId, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { EpvChart } from "@/components/charts/EpvChart";
import { LaneMetric } from "@/components/charts/LaneMetric";
import { XfgBar } from "@/components/charts/XfgBar";
import { Court } from "@/components/court/Court";
import type { CourtView } from "@/components/court/geometry";
import { EMPTY_SCENE, defaultFrame, frameCount, lastShotSpot, untrackedPlayers } from "@/components/court/scene";
import { Evidence } from "@/components/evidence/Evidence";
import type { EvidenceGroups } from "@/components/evidence/groups";
import { copy, fill } from "@/lib/copy";
import { getBrowserSource } from "@/lib/data/browser";
import { formatClock, formatNumber, formatProbability, formatSeconds } from "@/lib/format";
import type { LaneResult, Possession, Shot } from "@/lib/types/contract";
import { SPEEDS, advance, endTime, frameAtOrBefore, sceneAtTime, speedRate, timeOfFrame, type SpeedKey } from "./timeline";
import styles from "./possession.module.css";

const text = copy.review.possession;

type Load =
  | { status: "loading" }
  | { status: "error" }
  | { status: "ready"; possession: Possession; firstTime: number };

/** The frame named by ?frame= in the address, when it is a whole number inside the possession. */
function frameFromAddress(possession: Possession): number | null {
  const raw = new URLSearchParams(window.location.search).get("frame");
  if (raw === null || !/^\d+$/.test(raw)) return null;
  const frame = Number(raw);
  return frame < frameCount(possession) ? frame : null;
}

const REDUCED_MOTION = "(prefers-reduced-motion: reduce)";

function subscribeReducedMotion(onChange: () => void): () => void {
  const query = window.matchMedia(REDUCED_MOTION);
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}

/** True while the reader's system asks for reduced motion. */
function useReducedMotion(): boolean {
  return useSyncExternalStore(
    subscribeReducedMotion,
    () => window.matchMedia(REDUCED_MOTION).matches,
    () => false,
  );
}

/** The last shot of the possession that has a location: the one the court marks. */
function markedShot(possession: Possession): Shot | null {
  for (let index = possession.shots.length - 1; index >= 0; index -= 1) {
    const shot = possession.shots[index];
    if (shot && shot.x != null && shot.y != null) return shot;
  }
  return null;
}

export interface StageLanes {
  xfg: LaneResult | null;
  epv: LaneResult | null;
  ghosting: LaneResult | null;
}

export interface PossessionStageProps {
  gameSlug: string;
  possession: number;
  /** The model lanes behind the values shown, from results.json, for their held-out metric. */
  lanes: StageLanes;
  /** The possession's sources, sorted by the value they feed. */
  evidence: EvidenceGroups;
}

/**
 * The possession review stage (PLAN.md M5): the court with playback (play, pause, scrub, speed), the EPV curve
 * synced to the playback time, the shot marker with its xFG, and ghost defenders behind a toggle labelled
 * "average defense". The possession is loaded through the data source.
 */
export function PossessionStage({ gameSlug, possession: possessionNumber, lanes, evidence }: PossessionStageProps) {
  const [load, setLoad] = useState<Load>({ status: "loading" });
  const [chosenTime, setChosenTime] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<SpeedKey>("normal");
  const [view, setView] = useState<CourtView>("full");
  const [showGhosts, setShowGhosts] = useState(true);
  const reduced = useReducedMotion();
  const sliderId = useId();
  const speedId = useId();
  const viewName = useId();
  const ghostId = useId();
  const panelId = useId();

  useEffect(() => {
    let cancelled = false;
    getBrowserSource()
      .possession(gameSlug, possessionNumber)
      .then(
        (possession) => {
          if (cancelled) return;
          const frame = frameFromAddress(possession) ?? defaultFrame(possession);
          setLoad({ status: "ready", possession, firstTime: timeOfFrame(possession, frame) });
        },
        () => {
          if (!cancelled) setLoad({ status: "error" });
        },
      );
    return () => {
      cancelled = true;
    };
  }, [gameSlug, possessionNumber]);

  const possession = load.status === "ready" ? load.possession : null;
  const end = possession ? endTime(possession) : 0;
  const time = possession ? Math.min(end, Math.max(0, chosenTime ?? (load.status === "ready" ? load.firstTime : 0))) : 0;
  const frame = possession ? frameAtOrBefore(possession.t, time) : 0;
  const total = possession ? frameCount(possession) : 0;
  const playable = total > 1 && end > 0;

  // The playback clock. The loop reads the time through a ref, so one animation loop runs for a whole play.
  const timeRef = useRef(time);
  useEffect(() => {
    timeRef.current = time;
  }, [time]);
  useEffect(() => {
    if (!playing || !possession) return;
    const rate = speedRate(speed);
    let last: number | null = null;
    let handle = 0;
    const step = (now: number) => {
      const elapsed = last === null ? 0 : (now - last) / 1000;
      last = now;
      const next = advance(timeRef.current, elapsed, rate, end);
      timeRef.current = next.time;
      setChosenTime(next.time);
      if (next.done) {
        setPlaying(false);
        return;
      }
      handle = requestAnimationFrame(step);
    };
    handle = requestAnimationFrame(step);
    return () => cancelAnimationFrame(handle);
  }, [playing, speed, end, possession]);

  const togglePlay = useCallback(() => {
    if (!playable) return;
    if (playing) {
      setPlaying(false);
      return;
    }
    if (timeRef.current >= end) {
      timeRef.current = 0;
      setChosenTime(0);
    }
    setPlaying(true);
  }, [playable, playing, end]);

  const seek = useCallback(
    (next: number) => {
      const clamped = Math.min(end, Math.max(0, next));
      timeRef.current = clamped;
      setChosenTime(clamped);
    },
    [end],
  );

  const shot = possession ? markedShot(possession) : null;
  const markers = useMemo(() => {
    if (!possession) return [];
    const spot = lastShotSpot(possession);
    return spot ? [{ kind: "spot" as const, ...spot }] : [];
  }, [possession]);

  // While it plays, the court moves between samples, unless the reader asked for reduced motion. A paused
  // court always shows a sampled frame.
  const tween = playing && !reduced;
  const scene = useMemo(() => {
    if (!possession) return EMPTY_SCENE;
    return sceneAtTime(possession, tween ? time : timeOfFrame(possession, frame), { tween, markers });
  }, [possession, time, frame, tween, markers]);

  if (load.status === "error") {
    return (
      <p role="alert" className="label">
        {text.court.error}
      </p>
    );
  }

  const frameTime = possession ? timeOfFrame(possession, frame) : 0;
  const courtLabel = possession
    ? fill(text.court.label, {
        n: possession.possession,
        t: formatSeconds(frameTime),
        offense: possession.offense,
        defense: possession.defense,
      })
    : text.court.loadingLabel;
  const frameValue = possession
    ? fill(text.court.frameValue, { frame: frame + 1, total, t: formatSeconds(frameTime) })
    : text.court.loading;
  const untracked = possession ? untrackedPlayers(possession, frame) : [];
  const ghosts = possession?.ghosts ?? null;
  const shotTimes = possession ? possession.shots.map((item) => item.t).filter((t): t is number => t != null) : [];
  let playLabel = text.playback.play;
  if (playing) playLabel = text.playback.pause;
  else if (playable && time >= end) playLabel = text.playback.replay;

  return (
    <div
      className={styles.stage}
      aria-busy={load.status === "loading"}
      data-testid="possession-stage"
      data-playing={playing ? "true" : "false"}
      data-tween={tween ? "true" : "false"}
    >
      <figure className={styles.figure}>
        <Court
          scene={scene}
          view={view}
          attackX={possession?.attack_x ?? null}
          label={courtLabel}
          layers={{ ghosts: showGhosts }}
        />
        <figcaption className={styles.caption}>
          <span role={possession ? undefined : "status"} data-testid="frame-value">
            {frameValue}
          </span>
          {possession ? (
            <span className={styles.clocks}>
              <span>
                {fill(text.court.clock, { clock: formatClock(possession.game_clock_s[frame]) || copy.site.none })}
              </span>
              <span>
                {fill(text.court.shotClock, {
                  clock: formatSeconds(possession.shot_clock_s[frame]) || copy.site.none,
                })}
              </span>
            </span>
          ) : null}
        </figcaption>
      </figure>

      {possession ? (
        <>
          <div className={styles.controls}>
            <div className={styles.transport}>
              <button
                type="button"
                className={styles.play}
                onClick={togglePlay}
                disabled={!playable}
                aria-pressed={playing}
                data-testid="play-toggle"
              >
                <span className={playing ? styles.pauseGlyph : styles.playGlyph} aria-hidden="true" />
                {playLabel}
              </button>
              <label className={styles.speed} htmlFor={speedId}>
                <span>{text.playback.speed}</span>
                <select
                  id={speedId}
                  value={speed}
                  onChange={(event) => setSpeed(event.target.value as SpeedKey)}
                  data-testid="speed"
                >
                  {SPEEDS.map((option) => (
                    <option key={option.key} value={option.key}>
                      {text.playback.speeds[option.key]}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className={styles.slider}>
              <label htmlFor={sliderId}>{text.court.frame}</label>
              <input
                id={sliderId}
                type="range"
                min={0}
                max={Math.max(0, total - 1)}
                step={1}
                value={frame}
                disabled={!playable}
                aria-valuetext={frameValue}
                onChange={(event) => seek(timeOfFrame(possession, Number(event.target.value)))}
              />
            </div>
            <fieldset className={styles.views}>
              <legend>{text.court.view}</legend>
              <label>
                <input
                  type="radio"
                  name={viewName}
                  value="full"
                  checked={view === "full"}
                  onChange={() => setView("full")}
                />
                {text.court.full}
              </label>
              <label>
                <input
                  type="radio"
                  name={viewName}
                  value="half"
                  checked={view === "half"}
                  onChange={() => setView("half")}
                />
                {text.court.half}
              </label>
            </fieldset>
            <div className={styles.toggle}>
              <input
                id={ghostId}
                type="checkbox"
                checked={showGhosts && ghosts !== null}
                disabled={ghosts === null}
                onChange={(event) => setShowGhosts(event.target.checked)}
                data-testid="ghost-toggle"
              />
              <label htmlFor={ghostId}>{fill(text.ghosts.toggle, { label: ghosts?.label ?? text.ghosts.label })}</label>
            </div>
          </div>
          {!playable ? <p className="small">{text.playback.single}</p> : null}
          {reduced ? <p className="small muted">{text.playback.reduced}</p> : null}

          <ul className={styles.legend} aria-label={text.court.legend}>
            <li>
              <span className={`${styles.swatch} ${styles.offense}`} aria-hidden="true" />
              {fill(text.court.offense, { team: possession.offense })}
            </li>
            <li>
              <span className={`${styles.swatch} ${styles.defense}`} aria-hidden="true" />
              {fill(text.court.defense, { team: possession.defense })}
            </li>
            <li>
              <span className={`${styles.swatch} ${styles.ball}`} aria-hidden="true" />
              {text.court.ball}
            </li>
            {ghosts && showGhosts ? (
              <li>
                <span className={`${styles.swatch} ${styles.ghost}`} aria-hidden="true" />
                {fill(text.court.ghost, { label: ghosts.label })}
              </li>
            ) : null}
            {shot ? (
              <li>
                <span className={`${styles.swatch} ${styles.marker}`} aria-hidden="true" />
                {text.court.shot}
              </li>
            ) : null}
          </ul>

          <div className={styles.panels}>
            <section className={styles.panel} aria-labelledby={`${panelId}-epv`}>
              <h3 id={`${panelId}-epv`}>{text.epv.heading}</h3>
              {possession.epv && possession.epv.t.length > 0 && possession.epv.value.length > 0 ? (
                <EpvChart
                  curve={possession.epv}
                  end={end}
                  time={time}
                  shotTimes={shotTimes}
                  onSeek={playable ? seek : undefined}
                />
              ) : (
                <p className="small" data-testid="epv-none">
                  {text.epv.none}
                </p>
              )}
              <LaneMetric lane={lanes.epv} model={text.epv.model} />
              <Evidence what={text.epv.heading} sources={evidence.epv} />
            </section>

            <section className={styles.panel} aria-labelledby={`${panelId}-shot`} data-testid="shot-card">
              <h3 id={`${panelId}-shot`}>{text.shot.heading}</h3>
              {shot ? (
                <>
                  <p className="small">
                    {fill(text.shot.what, {
                      shooter: shot.shooter ?? copy.site.none,
                      outcome: shot.made ? text.shots.made : text.shots.missed,
                    })}{" "}
                    {shot.t != null ? fill(text.shot.when, { t: formatSeconds(shot.t) }) : null}
                  </p>
                  {shot.xfg != null ? (
                    <div className={styles.xfgRow}>
                      <span className={styles.xfgValue} data-testid="shot-xfg">
                        <span>{text.shot.xfg}</span> <span className="num">{formatProbability(shot.xfg)}</span>{" "}
                        <span className="muted">{copy.site.estimate.point}</span>
                      </span>
                      <XfgBar xfg={shot.xfg} />
                    </div>
                  ) : (
                    <p className="small">{text.shot.noXfg}</p>
                  )}
                  {shot.location_source === "shot_chart" ? (
                    <p className="small muted">{text.shots.locationFromShotChart}</p>
                  ) : null}
                </>
              ) : (
                <p className="small">{text.shot.none}</p>
              )}
              <LaneMetric lane={lanes.xfg} model={text.shot.model} />
              <Evidence what={text.shot.heading} sources={evidence.shots} />
            </section>
          </div>

          <div className="stack stack-tight small" data-testid="ghost-notes">
            {ghosts ? (
              <>
                <p>
                  {fill(text.ghosts.note, {
                    horizon: formatNumber(ghosts.horizon_s, 1),
                    label: ghosts.label,
                  })}
                </p>
                {ghosts.mean_deviation_ft != null ? (
                  <p>
                    {fill(text.ghosts.deviation, { value: formatNumber(ghosts.mean_deviation_ft, 1) })}{" "}
                    <span className="muted">{copy.site.estimate.point}</span>
                  </p>
                ) : null}
                <LaneMetric lane={lanes.ghosting} model={text.ghosts.model} />
              </>
            ) : (
              <p>{text.ghosts.none}</p>
            )}
            <Evidence what={text.ghosts.heading} sources={evidence.ghosts} />
            {untracked.length > 0 ? (
              <p>{fill(text.court.untracked, { names: untracked.map((track) => track.name).join(", ") })}</p>
            ) : null}
          </div>

          <Positions possession={possession} frame={frame} />
          <Evidence what={text.positions.title} sources={evidence.positions} />
          {evidence.other.length > 0 ? <Evidence what={text.otherSources} sources={evidence.other} /> : null}
        </>
      ) : null}
    </div>
  );
}

/** The positions at a frame as a table: the court's text alternative. It changes only when the frame does. */
const Positions = memo(function Positions({ possession, frame }: { possession: Possession; frame: number }) {
  const ballX = possession.ball.x[frame] ?? null;
  const ballY = possession.ball.y[frame] ?? null;
  return (
    <details className="sources">
      <summary>{text.positions.title}</summary>
      <div className="table-wrap">
        <table>
          <caption>{fill(text.positions.caption, { frame: frame + 1 })}</caption>
          <thead>
            <tr>
              <th scope="col">{text.positions.player}</th>
              <th scope="col">{text.positions.team}</th>
              <th scope="col">{text.positions.role}</th>
              <th scope="col">{text.positions.x}</th>
              <th scope="col">{text.positions.y}</th>
            </tr>
          </thead>
          <tbody>
            {possession.players.map((track) => {
              const x = track.x[frame] ?? null;
              const y = track.y[frame] ?? null;
              return (
                <tr key={track.player_id}>
                  <th scope="row">{track.name}</th>
                  <td>{track.team}</td>
                  <td>{track.offense ? text.positions.offense : text.positions.defense}</td>
                  {x === null || y === null ? (
                    <td colSpan={2}>{text.positions.missing}</td>
                  ) : (
                    <>
                      <td className="num">{formatNumber(x, 1)}</td>
                      <td className="num">{formatNumber(y, 1)}</td>
                    </>
                  )}
                </tr>
              );
            })}
            <tr>
              <th scope="row">{text.positions.ball}</th>
              <td />
              <td />
              {ballX === null || ballY === null ? (
                <td colSpan={2}>{text.positions.missing}</td>
              ) : (
                <>
                  <td className="num">{formatNumber(ballX, 1)}</td>
                  <td className="num">{formatNumber(ballY, 1)}</td>
                </>
              )}
            </tr>
          </tbody>
        </table>
      </div>
    </details>
  );
});
