"use client";

import { useEffect, useId, useMemo, useState } from "react";
import { Court } from "@/components/court/Court";
import type { CourtView } from "@/components/court/geometry";
import {
  EMPTY_SCENE,
  clampFrame,
  defaultFrame,
  frameCount,
  lastShotSpot,
  sceneFromPossession,
  untrackedPlayers,
} from "@/components/court/scene";
import { copy, fill } from "@/lib/copy";
import { getBrowserSource } from "@/lib/data/browser";
import { formatClock, formatNumber, formatSeconds } from "@/lib/format";
import type { Possession } from "@/lib/types/contract";
import styles from "./possession.module.css";

const text = copy.review.possession;

type Load =
  | { status: "loading" }
  | { status: "error" }
  | { status: "ready"; possession: Possession; firstFrame: number };

/** The frame named by ?frame= in the address, when it is a whole number inside the possession. */
function frameFromAddress(possession: Possession): number | null {
  const raw = new URLSearchParams(window.location.search).get("frame");
  if (raw === null || !/^\d+$/.test(raw)) return null;
  const frame = Number(raw);
  return frame < frameCount(possession) ? frame : null;
}

export interface PossessionStageProps {
  gameSlug: string;
  possession: number;
}

/**
 * The court of one possession at one frame, with a frame slider and a full or half court view. The possession
 * is loaded through the data source; playback, the value curve and the tour come with PLAN.md M5.
 */
export function PossessionStage({ gameSlug, possession: possessionNumber }: PossessionStageProps) {
  const [load, setLoad] = useState<Load>({ status: "loading" });
  const [chosenFrame, setChosenFrame] = useState<number | null>(null);
  const [view, setView] = useState<CourtView>("full");
  const sliderId = useId();
  const viewName = useId();

  useEffect(() => {
    let cancelled = false;
    getBrowserSource()
      .possession(gameSlug, possessionNumber)
      .then(
        (possession) => {
          if (cancelled) return;
          setLoad({
            status: "ready",
            possession,
            firstFrame: frameFromAddress(possession) ?? defaultFrame(possession),
          });
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
  const frame = possession ? clampFrame(possession, chosenFrame ?? (load.status === "ready" ? load.firstFrame : 0)) : 0;

  const scene = useMemo(() => {
    if (!possession) return EMPTY_SCENE;
    const spot = lastShotSpot(possession);
    return sceneFromPossession(possession, frame, { markers: spot ? [{ kind: "spot", ...spot }] : [] });
  }, [possession, frame]);

  if (load.status === "error") {
    return (
      <p role="alert" className="label">
        {text.court.error}
      </p>
    );
  }

  const total = possession ? frameCount(possession) : 0;
  const time = possession ? (possession.t[frame] ?? 0) : 0;
  const label = possession
    ? fill(text.court.label, {
        n: possession.possession,
        t: formatSeconds(time),
        offense: possession.offense,
        defense: possession.defense,
      })
    : text.court.loadingLabel;
  const frameValue = possession
    ? fill(text.court.frameValue, { frame: frame + 1, total, t: formatSeconds(time) })
    : text.court.loading;
  const untracked = possession ? untrackedPlayers(possession, frame) : [];
  const hasShotSpot = possession ? lastShotSpot(possession) !== null : false;

  return (
    <div className={styles.stage} aria-busy={load.status === "loading"} data-testid="possession-stage">
      <figure className={styles.figure}>
        <Court scene={scene} view={view} attackX={possession?.attack_x ?? null} label={label} />
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
            <div className={styles.slider}>
              <label htmlFor={sliderId}>{text.court.frame}</label>
              <input
                id={sliderId}
                type="range"
                min={0}
                max={Math.max(0, total - 1)}
                step={1}
                value={frame}
                aria-valuetext={frameValue}
                onChange={(event) => setChosenFrame(Number(event.target.value))}
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
          </div>

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
            {possession.ghosts ? (
              <li>
                <span className={`${styles.swatch} ${styles.ghost}`} aria-hidden="true" />
                {fill(text.court.ghost, { label: possession.ghosts.label })}
              </li>
            ) : null}
            {hasShotSpot ? (
              <li>
                <span className={`${styles.swatch} ${styles.marker}`} aria-hidden="true" />
                {text.court.shot}
              </li>
            ) : null}
          </ul>

          <div className="stack stack-tight small">
            {possession.ghosts ? (
              <>
                <p>
                  {fill(text.ghosts.note, {
                    horizon: formatNumber(possession.ghosts.horizon_s, 1),
                    label: possession.ghosts.label,
                  })}
                </p>
                {possession.ghosts.mean_deviation_ft != null ? (
                  <p>{fill(text.ghosts.deviation, { value: formatNumber(possession.ghosts.mean_deviation_ft, 1) })}</p>
                ) : null}
              </>
            ) : (
              <p>{text.ghosts.none}</p>
            )}
            {untracked.length > 0 ? (
              <p>{fill(text.court.untracked, { names: untracked.map((track) => track.name).join(", ") })}</p>
            ) : null}
          </div>

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
                    {scene.ball ? (
                      <>
                        <td className="num">{formatNumber(scene.ball.x, 1)}</td>
                        <td className="num">{formatNumber(scene.ball.y, 1)}</td>
                      </>
                    ) : (
                      <td colSpan={2}>{text.positions.missing}</td>
                    )}
                  </tr>
                </tbody>
              </table>
            </div>
          </details>
        </>
      ) : null}
    </div>
  );
}
