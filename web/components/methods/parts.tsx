import { DataText, DataValue } from "@/components/results/DataValue";
import { RepoPath } from "@/components/results/Sources";
import type { StatusTone } from "@/components/ui/Status";
import { copy, fillNodes, label } from "@/lib/copy";
import { numberText } from "@/lib/format";
import type {
  Benchmark,
  Experiment,
  OurValue,
  PostHocInterval,
  ProtocolResult,
  StageResult,
  StageValue,
} from "@/lib/types/contract";
import { Status } from "@/components/ui/Status";
import styles from "./methods.module.css";

export const FILE = "methods.json";
const text = copy.methods;

export const KIND_TONE: Record<Benchmark["comparability_class"], StatusTone> = {
  roughly: "pass",
  not_directly: "warn",
  none: "plain",
};

export const STATUS_TONE: Record<Experiment["status"], StatusTone> = {
  pre_registered: "plain",
  development_met_test_pending: "warn",
  not_adopted_development: "fail",
  adopted: "pass",
  not_adopted_test: "fail",
};

/** The plain name of a lane, or the lane key with its underscores as spaces when copy names no such lane. */
export function laneName(lane: string): string {
  return label(text.lanes, lane).replace(/_/g, " ");
}

/** A number the data file stores as text, printed as stored (a leading hyphen becomes the minus sign). */
export function NumText({ field, value }: { field: string; value: string }) {
  return (
    <span className="num" data-file={FILE} data-field={field} data-shown="number text">
      {numberText(value)}
    </span>
  );
}

/** One number of ours: its label, its value, its interval or the words point estimate, and its report. */
export function Ours({ value, field }: { value: OurValue; field: string }) {
  return (
    <li className={styles.value} data-rows={value.rows}>
      <DataText file={FILE} field={`${field}.label`} text={value.label} />
      <span className="estimate">
        <NumText field={`${field}.value`} value={value.value} />
        <span className="note" data-interval={value.lo != null ? "true" : "false"}>
          {value.lo != null && value.hi != null ? (
            fillNodes(copy.site.estimate.range, {
              lo: <NumText field={`${field}.lo`} value={value.lo} />,
              hi: <NumText field={`${field}.hi`} value={value.hi} />,
            })
          ) : value.kind === "setting" ? (
            text.lane.setting
          ) : (
            text.lane.point
          )}
        </span>
      </span>
      <span className="note">{fillNodes(text.lane.from, { path: <RepoPath path={value.source} /> })}</span>
    </li>
  );
}

function StageNumber({ value, field }: { value: StageValue; field: string }) {
  return (
    <li className={styles.value}>
      <DataText file={FILE} field={`${field}.label`} text={value.label} />
      <span className="estimate">
        <DataValue file={FILE} field={`${field}.value`} value={value.value} signed={!value.point_estimate} />
        <span className="note" data-interval={value.point_estimate ? "false" : "true"}>
          {value.lo != null && value.hi != null
            ? fillNodes(copy.site.estimate.range, {
                lo: <DataValue file={FILE} field={`${field}.lo`} value={value.lo} />,
                hi: <DataValue file={FILE} field={`${field}.hi`} value={value.hi} />,
              })
            : text.lane.point}
        </span>
      </span>
    </li>
  );
}

/** One stage of a test that has run: the paired difference against its bar, the sample, and the file. */
export function StageFacts({ stage, field }: { stage: StageResult; field: string }) {
  const r = text.results;
  return (
    <div className="stack stack-tight" data-stage={stage.stage} data-met={stage.met ? "true" : "false"}>
      <h3 className={styles.stage}>{label(r.stage, stage.stage)}</h3>
      <dl className="facts">
        <div>
          <dt>{r.difference}</dt>
          <dd>
            <span className="estimate">
              <DataValue file={FILE} field={`${field}.diff`} value={stage.diff} signed />
              <span className="note" data-interval="true">
                {fillNodes(copy.site.estimate.range, {
                  lo: <DataValue file={FILE} field={`${field}.lo`} value={stage.lo} />,
                  hi: <DataValue file={FILE} field={`${field}.hi`} value={stage.hi} />,
                })}
              </span>
            </span>
          </dd>
        </div>
        <div>
          <dt>{r.upper}</dt>
          <dd>
            <DataValue file={FILE} field={`${field}.upper`} value={stage.upper} signed />
          </dd>
        </div>
        <div>
          <dt>{r.threshold}</dt>
          <dd>
            <DataValue file={FILE} field={`${field}.threshold`} value={stage.threshold} signed />
          </dd>
        </div>
        <div>
          <dt>{r.verdict}</dt>
          <dd>
            <Status tone={stage.met ? "pass" : "fail"} text={stage.met ? r.met : r.missed} />
          </dd>
        </div>
        <div>
          <dt>{r.sample}</dt>
          <dd>
            <span className={styles.sample}>
              {fillNodes(copy.site.estimate.sample, {
                n: <DataValue file={FILE} field={`${field}.rows`} value={stage.rows} shown="count, as stored" />,
                unit: <DataText file={FILE} field={`${field}.rows_unit`} text={stage.rows_unit} />,
              })}
            </span>
            <span className="note">
              {fillNodes(r.games, {
                n: <DataValue file={FILE} field={`${field}.games`} value={stage.games} shown="count, as stored" />,
              })}
            </span>
          </dd>
        </div>
      </dl>
      <p className="small">
        {r.metric}
        {": "}
        <DataText file={FILE} field={`${field}.metric`} text={stage.metric} />
      </p>
      <p className="small">
        {r.method}
        {": "}
        <DataText file={FILE} field={`${field}.interval`} text={stage.interval} />
      </p>
      <p className="small">{fillNodes(r.file, { path: <RepoPath path={stage.source.path} /> })}</p>
      {stage.values.length > 0 ? (
        <details className="sources">
          <summary>{r.more}</summary>
          <ul className={styles.values}>
            {stage.values.map((value, index) => (
              <StageNumber key={value.label} value={value} field={`${field}.values[${index}]`} />
            ))}
          </ul>
        </details>
      ) : null}
    </div>
  );
}

/** The plain label of an experiment's state, as a status mark. */
export function ExperimentStatus({ experiment }: { experiment: Experiment }) {
  return (
    <span className={styles.state}>
      <Status tone={STATUS_TONE[experiment.status]} text={label(text.status, experiment.status)} />
    </span>
  );
}

function PostHocNote({ point, interval, field }: { point: boolean; interval: PostHocInterval | null | undefined; field: string }) {
  const p = text.results.protocol;
  return (
    <span className="note" data-interval={interval ? "true" : "false"} data-post-hoc={interval ? "true" : undefined}>
      {interval ? (
        <>
          {fillNodes(copy.site.estimate.range, {
            lo: <DataValue file={FILE} field={`${field}.interval.lo`} value={interval.lo} />,
            hi: <DataValue file={FILE} field={`${field}.interval.hi`} value={interval.hi} />,
          })}
          {", "}
          <em>{p.postHoc}</em>
        </>
      ) : point ? (
        text.lane.point
      ) : null}
    </span>
  );
}

/**
 * The result of a test set up as a protocol of statements (PLAN.md D86): each statement read on its value, with
 * its valid post hoc range or the words point estimate, then the rule to adopt and the plan's own reading. The
 * ranges set up in advance are not valid confidence intervals and are never shown.
 */
export function ProtocolFacts({ protocol, field }: { protocol: ProtocolResult; field: string }) {
  const r = text.results;
  const p = r.protocol;
  const intervalFile = [...protocol.hypotheses.map((h) => h.interval), ...protocol.adoption.map((a) => a.interval)].find(
    (interval) => interval,
  )?.source.path;
  return (
    <div className="stack stack-tight" data-protocol={protocol.stage} data-adopted={protocol.adopted ? "true" : "false"}>
      <h3 className={styles.stage}>{p.stage}</h3>
      <p className="small">{p.intro}</p>
      <p className="small">
        {fillNodes(p.sample, {
          n: <DataValue file={FILE} field={`${field}.rows`} value={protocol.rows} shown="count, as stored" />,
          unit: <DataText file={FILE} field={`${field}.rows_unit`} text={protocol.rows_unit} />,
          games: <DataValue file={FILE} field={`${field}.games`} value={protocol.games} shown="count, as stored" />,
        })}
      </p>
      <h4 className={styles.side}>{p.statements}</h4>
      <ul className={styles.values}>
        {protocol.hypotheses.map((h, index) => {
          const at = `${field}.hypotheses[${index}]`;
          return (
            <li key={h.label} className={styles.value} data-hypothesis={h.label} data-reading={h.result}>
              <span>
                <strong className={styles.lead}>
                  <DataText file={FILE} field={`${at}.label`} text={h.label} />
                </strong>
                {": "}
                <DataText file={FILE} field={`${at}.statement`} text={h.statement} />
              </span>
              <span className="estimate">
                {h.value != null ? <DataValue file={FILE} field={`${at}.value`} value={h.value} /> : null}
                <PostHocNote point={h.point_estimate && h.value != null} interval={h.interval} field={at} />
              </span>
              <span className={styles.state}>
                <Status tone="plain" text={label(p.result, h.result)} />
              </span>
            </li>
          );
        })}
      </ul>
      <h4 className={styles.side}>{p.rule}</h4>
      <p className={`${styles.prose} small`}>
        <DataText file={FILE} field={`${field}.rule`} text={protocol.rule} />
      </p>
      <ul className={styles.values}>
        {protocol.adoption.map((a, index) => {
          const at = `${field}.adoption[${index}]`;
          return (
            <li key={a.label} className={styles.value}>
              <DataText file={FILE} field={`${at}.label`} text={a.label} />
              <span className="estimate">
                <DataValue file={FILE} field={`${at}.value`} value={a.value} />
                <PostHocNote point={a.point_estimate} interval={a.interval} field={at} />
              </span>
            </li>
          );
        })}
      </ul>
      <p>
        <Status tone={protocol.adopted ? "pass" : "fail"} text={protocol.adopted ? p.adopted : p.notAdopted} />
      </p>
      <p className="small">
        <span className={styles.lead}>{p.reading}</span>
        {": "}
        <DataText file={FILE} field={`${field}.reading`} text={protocol.reading} />
      </p>
      {protocol.caveat ? (
        <div className="stack stack-tight">
          <h4 className={styles.side}>{p.caveat}</h4>
          <q className={styles.quote} data-testid="protocol-caveat">
            <DataText file={FILE} field={`${field}.caveat`} text={protocol.caveat} />
          </q>
        </div>
      ) : null}
      {protocol.not_scored.length > 0 ? (
        <div className="stack stack-tight" data-testid="not-scored">
          <h4 className={styles.side}>{p.notScored}</h4>
          <ul className="bullets small">
            {protocol.not_scored.map((sentence, index) => (
              <li key={sentence}>
                <DataText file={FILE} field={`${field}.not_scored[${index}]`} text={sentence} />
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      <p className="small">{p.postHocNote}</p>
      <p className="small">
        {fillNodes(r.file, { path: <RepoPath path={protocol.source.path} /> })}
        {intervalFile ? <span className="note">{fillNodes(p.intervals, { path: <RepoPath path={intervalFile} /> })}</span> : null}
      </p>
    </div>
  );
}
