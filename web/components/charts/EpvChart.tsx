"use client";

import { useEffect, useRef, useState, type PointerEvent } from "react";
import { epvStateAt } from "@/components/review/timeline";
import { copy, fill } from "@/lib/copy";
import { formatNumber, formatSeconds } from "@/lib/format";
import { showValue } from "@/lib/tokens";
import type { EpvCurve } from "@/lib/types/contract";
import { linear, niceDomain, niceStep, stepDecimals, ticks } from "./scale";
import styles from "./charts.module.css";

const HEIGHT = 176;
const MARGIN = { top: 22, right: 14, bottom: 26, left: 40 };

export interface EpvChartProps {
  curve: EpvCurve;
  /** The length of the possession in seconds: the x axis runs from zero to it. */
  end: number;
  /** The playback time, in seconds from the start. */
  time: number;
  /** Release times of the shots, drawn as marks on the time axis. */
  shotTimes: readonly number[];
  /** Move playback to a time. */
  onSeek?: (time: number) => void;
  /** No playback on this copy of the site (PLAN.md D60): the cursor stays at the end. */
  still?: boolean;
}

/**
 * The EPV curve of a possession with a cursor at the playback time (PLAN.md M5). Expected points for the
 * offense from each state, the frozen model's raw output, one value about every second. Every value is a point
 * estimate (D53). Pressing or dragging on the chart moves playback to that moment.
 *
 * The line joins the states with straight segments, a picture only. The readout and its dot show a stored state
 * as the file holds it: the last state at or before the playback time, and the last state at the end of playback
 * (timeline.ts, epvStateAt).
 */
export function EpvChart({ curve, end, time, shotTimes, onSeek, still = false }: EpvChartProps) {
  const text = copy.review.possession.epv;
  const boxRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(600);
  const dragging = useRef(false);

  useEffect(() => {
    const box = boxRef.current;
    if (!box) return;
    const measure = () => {
      const next = Math.round(box.getBoundingClientRect().width);
      if (next > 0) setWidth(next);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(box);
    return () => observer.disconnect();
  }, []);

  const n = Math.min(curve.t.length, curve.value.length);
  const values = curve.value.slice(0, n);
  const times = curve.t.slice(0, n);
  const span = Math.max(end, times[n - 1] ?? 0, 0.001);
  const [y0, y1] = niceDomain(Math.min(0, ...values), Math.max(...values), 3);
  const x = linear([0, span], [MARGIN.left, Math.max(MARGIN.left + 1, width - MARGIN.right)]);
  const y = linear([y0, y1], [HEIGHT - MARGIN.bottom, MARGIN.top]);
  const yTicks = ticks(y0, y1, 3);
  const yDigits = stepDecimals(niceStep(y1 - y0, 3));
  const xTicks = ticks(0, span, width < 420 ? 3 : 6);
  const path = times.map((t, index) => `${index === 0 ? "M" : "L"}${x(t).toFixed(1)},${y(values[index] ?? 0).toFixed(1)}`).join("");
  const clamped = Math.min(Math.max(time, 0), span);
  const now = epvStateAt(curve, clamped, end);
  const stored = (value: number | undefined) => (value === undefined ? "" : showValue(value, "points, as stored"));

  const seekFrom = (event: PointerEvent<SVGSVGElement>) => {
    if (!onSeek) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * width;
    const share = (px - MARGIN.left) / Math.max(1, width - MARGIN.left - MARGIN.right);
    onSeek(Math.min(span, Math.max(0, share * span)));
  };

  const first = values[0];
  const last = values[n - 1];
  const name = fill(text.label, {
    start: stored(first),
    end: stored(last),
    n,
  });

  return (
    <figure className={styles.chart} data-testid="epv-chart">
      <figcaption className={styles.chartHead}>
        <span className={styles.chartTitle}>{text.title}</span>
        <span className={styles.readout} data-testid="epv-readout" data-state={now?.index}>
          {now === null ? null : fill(text.readout, { t: formatSeconds(now.t), value: stored(now.value) })}
          <span className={styles.point}>{copy.site.estimate.point}</span>
        </span>
      </figcaption>
      <div ref={boxRef} className={styles.plot}>
        <svg
          width={width}
          height={HEIGHT}
          viewBox={`0 0 ${width} ${HEIGHT}`}
          role="img"
          aria-label={name}
          className={onSeek ? styles.seekable : undefined}
          onPointerDown={(event) => {
            dragging.current = true;
            event.currentTarget.setPointerCapture?.(event.pointerId);
            seekFrom(event);
          }}
          onPointerMove={(event) => {
            if (dragging.current) seekFrom(event);
          }}
          onPointerUp={() => {
            dragging.current = false;
          }}
          onPointerCancel={() => {
            dragging.current = false;
          }}
          data-cursor-t={clamped.toFixed(3)}
        >
          {yTicks.map((tick) => (
            <g key={`y${tick}`}>
              <line
                x1={MARGIN.left}
                x2={width - MARGIN.right}
                y1={y(tick)}
                y2={y(tick)}
                className={tick === 0 ? styles.zero : styles.grid}
              />
              <text x={MARGIN.left - 6} y={y(tick)} className={styles.tickY}>
                {formatNumber(tick, yDigits)}
              </text>
            </g>
          ))}
          {xTicks.map((tick) => (
            <text key={`x${tick}`} x={x(tick)} y={HEIGHT - 8} className={styles.tickX}>
              {fill(text.tick, { t: formatNumber(tick, 0) })}
            </text>
          ))}
          {shotTimes.map((t, index) => (
            <g key={`shot${index}`} data-shot-t={t}>
              <line x1={x(t)} x2={x(t)} y1={MARGIN.top - 6} y2={HEIGHT - MARGIN.bottom} className={styles.shotLine} />
              <text x={x(t)} y={MARGIN.top - 10} className={styles.shotText}>
                {text.shot}
              </text>
            </g>
          ))}
          {n === 1 ? (
            <circle cx={x(times[0] ?? 0)} cy={y(values[0] ?? 0)} r={4} className={styles.lineDot} />
          ) : (
            <path d={path} className={styles.line} />
          )}
          <line
            x1={x(clamped)}
            x2={x(clamped)}
            y1={MARGIN.top - 4}
            y2={HEIGHT - MARGIN.bottom}
            className={styles.cursor}
            data-testid="epv-cursor"
          />
          {now === null ? null : <circle cx={x(now.t)} cy={y(now.value)} r={4.5} className={styles.cursorDot} />}
        </svg>
      </div>
      <p className={styles.hint}>{still ? text.hintStill : onSeek ? text.hint : text.hintStatic}</p>
      <details className="sources">
        <summary>{text.table}</summary>
        <div className="table-wrap">
          <table>
            <caption className="visually-hidden">{text.table}</caption>
            <thead>
              <tr>
                <th scope="col">{text.time}</th>
                <th scope="col">{text.value}</th>
              </tr>
            </thead>
            <tbody>
              {times.map((t, index) => (
                <tr key={`${t}-${index}`}>
                  <td className="num">{formatSeconds(t)}</td>
                  <td className="num">{stored(values[index])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}
