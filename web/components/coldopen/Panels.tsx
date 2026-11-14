import { Status } from "@/components/ui/Status";
import { copy, fill } from "@/lib/copy";
import type { Bar, PairRow, PositionRow } from "./model";
import styles from "./coldopen.module.css";

const ui = copy.home.ui;
const open = copy.home.coldOpen;

/**
 * The words that a panel's values are point estimates (D53): the cold open shows no interval, so every panel says
 * so, as the results page does for a value with no interval.
 */
function PointNote({ text = ui.pointEstimates, shown = true }: { text?: string; shown?: boolean }) {
  return (
    <p className={styles.pointNote} data-shown={shown ? "true" : "false"} data-testid="point-note">
      {text}
    </p>
  );
}

/** Scene 2: each tracking row as a bar against its target line, with its status in words. */
export function ScoreBars({ bars, shown = true }: { bars: Bar[]; shown?: boolean }) {
  if (bars.length === 0) return null;
  return (
    <div className={styles.panel}>
      <ul className={styles.bars} aria-label={ui.barsLabel} data-shown={shown ? "true" : "false"} data-testid="score-bars">
        {bars.map((bar) => (
          <li key={bar.key} className={styles.bar} data-row={bar.key}>
            <span className={styles.barHead}>
              <span className={styles.barName}>{bar.label}</span>
              <span className={`num ${styles.barValue}`}>{bar.value}</span>
            </span>
            <span className={styles.track} aria-hidden="true">
              <span className={styles.fillBar} style={{ width: `${bar.share * 100}%` }} />
              {bar.targetShare !== null ? (
                <span className={styles.targetLine} style={{ left: `${bar.targetShare * 100}%` }} />
              ) : null}
            </span>
            <span className={styles.barFoot}>
              <span>
                {open.boxes.labels.target}
                {": "}
                <span className="num">{bar.target}</span>
              </span>
              <Status tone={bar.tone} text={bar.status} />
            </span>
          </li>
        ))}
      </ul>
      <PointNote shown={shown} />
    </div>
  );
}

/**
 * Scene 4: the gap on the frame on screen, always beside the scorecard row of the game (D50), then the typical
 * gap of the stretch and the share of frames that could be checked.
 */
export function GapPanel({ readout, row }: { readout: string; row: PositionRow | null }) {
  return (
    <div className={styles.panel} data-testid="gap-panel">
      <p className={`${styles.readout} num`} aria-live="off" data-testid="gap-readout">
        {readout}
      </p>
      {row ? (
        <dl className={styles.facts} data-testid="scorecard-row">
          <div>
            <dt>{row.name}</dt>
            <dd>
              <span className="num">{row.values}</span>
              <span className={styles.factNote}>
                {open.truth.labels.target}
                {": "}
                <span className="num">{row.target}</span>
              </span>
              <Status tone={row.tone} text={row.status} />
            </dd>
          </div>
          {row.stretch ? (
            <div>
              <dt>{ui.stretch}</dt>
              <dd className="num">{row.stretch}</dd>
            </div>
          ) : null}
          {row.checked ? (
            <div>
              <dt>{open.truth.labels.checked}</dt>
              <dd>
                <span className="num">{row.checked.text}</span>
                <span className={styles.thin} aria-hidden="true">
                  <span style={{ width: `${row.checked.share * 100}%` }} />
                </span>
              </dd>
            </div>
          ) : null}
        </dl>
      ) : null}
      <PointNote />
    </div>
  );
}

/** Scene 5: the fill share as a bar. */
export function FillShare({ value, share }: { value: string; share: number }) {
  return (
    <div className={styles.fill} data-testid="fill-share">
      <span className={styles.barHead}>
        <span className={styles.legendItem}>
          <span className={`${styles.swatch} ${styles.swatchHollow}`} aria-hidden="true" />
          {open.swap.labels.filled}
        </span>
        <span className="num">{fill(ui.percent, { value })}</span>
      </span>
      <span className={styles.track} aria-hidden="true">
        <span className={styles.fillBar} style={{ width: `${Math.min(1, Math.max(0, share)) * 100}%` }} />
      </span>
      <PointNote text={ui.pointShare} />
    </div>
  );
}

/** Scene 5: the items of the comparison in this stretch, each as two values and the gap between them. */
export function PairTable({ rows }: { rows: PairRow[] }) {
  if (rows.length === 0) return <p className="small muted">{ui.pairNone}</p>;
  return (
    <div className={styles.panel} data-testid="pair-table">
      <div className={`table-wrap ${styles.pairs}`} role="region" aria-label={ui.pairs} tabIndex={0}>
        <table>
          <caption className="visually-hidden">{ui.pairs}</caption>
          <thead>
            <tr>
              <th scope="col">{ui.pairWhat}</th>
              <th scope="col">{ui.pairSensor}</th>
              <th scope="col">{ui.pairBroadcast}</th>
              <th scope="col">{ui.pairGap}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.key} data-kind={row.kind}>
                <th scope="row">{row.label}</th>
                <td className="num">{row.sensor}</td>
                <td className="num">{row.broadcast}</td>
                <td className="num">{row.gap}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <PointNote />
    </div>
  );
}

/** The two dots of the truth overlay, named. */
export function TruthLegend() {
  return (
    <ul className={styles.legend}>
      <li className={styles.legendItem}>
        <span className={`${styles.swatch} ${styles.swatchBroadcast}`} aria-hidden="true" />
        {open.truth.labels.broadcast}
      </li>
      <li className={styles.legendItem}>
        <span className={`${styles.swatch} ${styles.swatchTruth}`} aria-hidden="true" />
        {open.truth.labels.sensor}
      </li>
    </ul>
  );
}
