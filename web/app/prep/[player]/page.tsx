import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import type { ReactNode } from "react";
import { IntervalBar, domainOf } from "@/components/charts/IntervalBar";
import { Evidence } from "@/components/evidence/Evidence";
import { EvidenceScope } from "@/components/evidence/EvidenceScope";
import { MachineText } from "@/components/genai/MachineText";
import { metricEstimate, type SortKey } from "@/components/prep/sort";
import { EstimateValue } from "@/components/ui/EstimateValue";
import { SourceList } from "@/components/ui/SourceList";
import { copy, fill, label } from "@/lib/copy";
import { getServerSource } from "@/lib/data/server";
import { formatCount, formatNumber, formatPercent, formatSigned } from "@/lib/format";
import { checkedText, playerFacts, playerNames } from "@/lib/genai";
import { routes } from "@/lib/site";
import type { Estimate, MetricNote, PlayersFile, RebuildCheck } from "@/lib/types/contract";
import styles from "./player.module.css";

const text = copy.prep;
const view = copy.prep.player;

type Metric = MetricNote["metric"];

export const dynamicParams = false;

export async function generateStaticParams() {
  const players = await getServerSource().players();
  return players.players.map((player) => ({ player: player.slug }));
}

async function load(slug: string) {
  const players = await getServerSource().players();
  const player = players.players.find((candidate) => candidate.slug === slug);
  if (!player) notFound();
  return { players, player };
}

export async function generateMetadata({ params }: PageProps<"/prep/[player]">): Promise<Metadata> {
  const { player: slug } = await params;
  const { player } = await load(slug);
  return { title: player.name };
}

/** The heading of one view, the model status the data gives for it, and its notes and evidence. */
function MetricSection({
  metric,
  players,
  children,
}: {
  metric: Metric;
  players: PlayersFile;
  children: ReactNode;
}) {
  const note = players.metrics.find((candidate) => candidate.metric === metric);
  const name = label(text.metrics, metric);
  return (
    <section className="section" aria-labelledby={metric} data-metric={metric}>
      <h2 id={metric}>{name}</h2>
      {children}
      {note ? (
        <div className="stack stack-tight small">
          <p>
            <span className={styles.term}>{view.status}</span> {note.verdict}
          </p>
          {note.notes.length > 0 ? (
            <ul className="bullets">
              {note.notes.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          ) : null}
          <Evidence what={name} sources={note.sources} />
        </div>
      ) : (
        <Evidence what={name} sources={[]} />
      )}
    </section>
  );
}

/** Why a view is not shown: below its minimum sample, or withheld for the data problem the export names. */
function Hidden({ n, min, unit, withheld }: { n: number; min: number; unit: string; withheld?: string | null }) {
  return (
    <p className="label muted" data-hidden="true">
      {withheld
        ? fill(view.withheld, { reason: withheld })
        : fill(view.hidden, { n: formatCount(n), unit, min: formatCount(min) })}
    </p>
  );
}

/** The minimum-sample rule of a view next to the player's own sample. */
function MinSample({ n, min, unit }: { n: number; min: number; unit: string }) {
  return (
    <p className="small muted" data-testid="min-sample">
      {fill(view.minSample, { n: formatCount(n), unit, min: formatCount(min) })}
    </p>
  );
}

function Row({ name, children }: { name: string; children: ReactNode }) {
  return (
    <div>
      <dt>{name}</dt>
      <dd>{children}</dd>
    </div>
  );
}

/** A point estimate printed as one: the number and the words. */
function Point({ children }: { children: ReactNode }) {
  return (
    <span className="estimate">
      <span className="num">{children}</span>
      <span className="note">{copy.site.estimate.point}</span>
    </span>
  );
}

/** The estimates every shown player has for one view, for the axis of the interval bars. */
function axis(players: PlayersFile, key: SortKey, include: number[] = []): [number, number] {
  const estimates = players.players
    .map((player) => metricEstimate(player, key))
    .filter((estimate): estimate is Estimate => estimate !== null);
  return domainOf(estimates, include);
}

function Bar({ estimate, domain, reference }: { estimate: Estimate; domain: [number, number]; reference?: number | null }) {
  return (
    <div className={styles.bar}>
      <IntervalBar domain={domain} value={estimate.value} lo={estimate.lo} hi={estimate.hi} reference={reference} />
      <span className="small muted">{view.barNote}</span>
    </div>
  );
}

/** The reproduction check of a rebuilt output (D45, D49), in words. */
function Check({ check }: { check: RebuildCheck | undefined }) {
  if (!check) return null;
  return (
    <p className="small" data-testid="rebuild-check" data-passed={check.passed}>
      {fill(check.passed ? view.check.passed : view.check.failed, {
        compared: check.compared,
        diff: formatNumber(check.max_abs_diff, 6),
        tolerance: formatNumber(check.tolerance, 6),
        unit: check.unit,
      })}
    </p>
  );
}

export default async function PlayerPage({ params }: PageProps<"/prep/[player]">) {
  const { player: slug } = await params;
  const { players, player } = await load(slug);
  // A machine-written scouting report (PLAN.md P3), shown only when it passes this page's own number check.
  const texts = await getServerSource().texts("scouting");
  const scouting = checkedText(texts, player.slug, playerFacts(player), playerNames(player));
  const slugOf = new Map(players.players.map((candidate) => [candidate.player_id, candidate.slug]));
  const { shot_making: shots, gravity, rebounding, absence_risk: absence, similar } = player;
  const baseline = players.shot_making_baseline;
  const descriptive = <p className="small">{view.descriptive}</p>;
  const checks = new Map(players.checks.map((check) => [check.name, check]));
  return (
    <div className="column stack stack-loose">
      <header className="page-head">
        <nav className="crumbs" aria-label={text.title}>
          <Link href={routes.prep}>{view.back}</Link>
        </nav>
        <h1>{player.name}</h1>
        <p className="lede">
          {fill(view.meta, { teams: player.teams.join(", "), games: formatCount(player.games) })}
          {player.position ? <> {fill(view.position, { position: player.position })}</> : null}
        </p>
        <p className="small" data-testid="hidden-count">
          {fill(view.hiddenCount, { hidden: formatCount(players.n_hidden), total: formatCount(players.n_players) })}
        </p>
      </header>

      <EvidenceScope>
        <MachineText item={scouting} texts={texts} note={copy.site.machineText.descriptive} />

        <MetricSection metric="shot_making" players={players}>
          {shots.shown && shots.over_expected_per_100 ? (
            <>
              <dl className={styles.values} data-values="player">
                <Row name={view.shotMaking.value}>
                  <EstimateValue estimate={shots.over_expected_per_100} digits={1} />
                  <Bar estimate={shots.over_expected_per_100} domain={axis(players, "shot_making", [0])} reference={0} />
                </Row>
                {shots.baseline_error_per_100 != null ? (
                  <Row name={view.shotMaking.baselineError}>
                    <Point>{formatSigned(shots.baseline_error_per_100, 1)}</Point>
                    <span className="note small">{view.shotMaking.baselineErrorNote}</span>
                  </Row>
                ) : null}
                {shots.fg_pct != null ? (
                  <Row name={view.shotMaking.fg}>
                    <Point>{formatPercent(shots.fg_pct, 1)}</Point>
                  </Row>
                ) : null}
                {shots.expected_fg_pct != null ? (
                  <Row name={view.shotMaking.expected}>
                    <Point>{formatPercent(shots.expected_fg_pct, 1)}</Point>
                  </Row>
                ) : null}
                {shots.frozen_model_residual_per_100 != null ? (
                  <Row name={view.shotMaking.residual}>
                    <Point>{formatSigned(shots.frozen_model_residual_per_100, 1)}</Point>
                    <span className="note small">{view.shotMaking.residualNote}</span>
                  </Row>
                ) : null}
              </dl>
              <MinSample n={shots.attempts} min={shots.min_attempts} unit={view.shotMaking.unit} />
            </>
          ) : (
            <Hidden n={shots.attempts} min={shots.min_attempts} unit={view.shotMaking.unit} withheld={shots.withheld} />
          )}
          <div className="stack stack-tight small" data-testid="baseline">
            <p>
              {fill(view.shotMaking.baseline, {
                name: baseline.name,
                selected: baseline.selected ? view.shotMaking.selected : view.shotMaking.notSelected,
              })}
            </p>
            <dl className={styles.values} data-values="baseline">
              <Row name={view.shotMaking.ece}>
                <EstimateValue estimate={baseline.ece} digits={3} />
              </Row>
            </dl>
            {baseline.calibration.length > 0 ? (
              <details className="sources" data-testid="calibration">
                <summary>{view.shotMaking.calibration.title}</summary>
                <p>{view.shotMaking.calibration.intro}</p>
                <div className="table-wrap">
                  <table>
                    <caption className="visually-hidden">{view.shotMaking.calibration.title}</caption>
                    <thead>
                      <tr>
                        <th scope="col">{view.shotMaking.calibration.bin}</th>
                        <th scope="col">{view.shotMaking.calibration.n}</th>
                        <th scope="col">{view.shotMaking.calibration.predicted}</th>
                        <th scope="col">{view.shotMaking.calibration.observed}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {baseline.calibration.map((row) => (
                        <tr key={`${row.lo}-${row.hi}`}>
                          <td className="num">
                            {fill(copy.site.estimate.range, { lo: formatNumber(row.lo, 2), hi: formatNumber(row.hi, 2) })}
                          </td>
                          <td className="num">{formatCount(row.n)}</td>
                          <td className="num">{formatNumber(row.predicted, 3)}</td>
                          <td className="num">{formatNumber(row.observed, 3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            ) : null}
            <Evidence what={view.shotMaking.baselineName} sources={[baseline.source]} />
          </div>
        </MetricSection>

        <MetricSection metric="gravity" players={players}>
          {gravity.shown && gravity.gravity_ft ? (
            <>
              <dl className={styles.values} data-values="player">
                <Row name={view.gravity.value}>
                  <EstimateValue estimate={gravity.gravity_ft} digits={2} />
                  <Bar estimate={gravity.gravity_ft} domain={axis(players, "gravity", [0])} reference={0} />
                </Row>
              </dl>
              <MinSample n={gravity.possessions} min={gravity.min_possessions} unit={view.gravity.unit} />
            </>
          ) : (
            <Hidden n={gravity.possessions} min={gravity.min_possessions} unit={view.gravity.unit} />
          )}
          {descriptive}
        </MetricSection>

        <MetricSection metric="rebounding" players={players}>
          {rebounding.shown && rebounding.rate_per_100 ? (
            <>
              <dl className={styles.values} data-values="player">
                <Row name={view.rebounding.value}>
                  <EstimateValue estimate={rebounding.rate_per_100} digits={1} />
                  <Bar
                    estimate={rebounding.rate_per_100}
                    domain={axis(players, "rebounding")}
                    reference={rebounding.model_expected_per_100 ?? null}
                  />
                </Row>
                {rebounding.model_expected_per_100 != null ? (
                  <Row name={view.rebounding.expected}>
                    <Point>{formatNumber(rebounding.model_expected_per_100, 1)}</Point>
                  </Row>
                ) : null}
              </dl>
              <p className="small">{view.rebounding.gap}</p>
              <MinSample n={rebounding.chances} min={rebounding.min_chances} unit={view.rebounding.unit} />
            </>
          ) : (
            <Hidden
              n={rebounding.chances}
              min={rebounding.min_chances}
              unit={view.rebounding.unit}
              withheld={rebounding.withheld}
            />
          )}
        </MetricSection>

        <MetricSection metric="absence_risk" players={players}>
          {absence.shown ? (
            <>
              <dl className={styles.values} data-values="player">
                {absence.model_probability ? (
                  <Row name={view.absence.model}>
                    <EstimateValue estimate={absence.model_probability} digits={3} />
                  </Row>
                ) : null}
                {absence.observed_rate ? (
                  <Row name={view.absence.observed}>
                    <EstimateValue estimate={absence.observed_rate} digits={3} />
                    <span className="note small">{view.descriptive}</span>
                  </Row>
                ) : null}
                <Row name={view.absence.absent}>
                  <span className="num">{formatCount(absence.absent)}</span>
                  <span className="note small">{fill(view.absence.absentOf, { games: formatCount(absence.games) })}</span>
                </Row>
              </dl>
              <p className="small" data-testid="absence-caution">
                {absence.kind === "observed_only" ? <>{view.absence.observedOnly} </> : null}
                {view.absence.caution}
              </p>
              <MinSample n={absence.games} min={absence.min_games} unit={view.absence.unit} />
            </>
          ) : (
            <>
              <Hidden n={absence.games} min={absence.min_games} unit={view.absence.unit} />
              <p className="small" data-testid="absence-caution">
                {view.absence.caution}
              </p>
            </>
          )}
          {absence.kind === "model_out_of_fold" ? <Check check={checks.get("absence")} /> : null}
        </MetricSection>

        <MetricSection metric="similar" players={players}>
          {!similar.available ? (
            <p className="label muted">{view.similar.unavailable}</p>
          ) : similar.shown ? (
            <>
              <div className="table-wrap" role="region" aria-label={view.similar.caption} tabIndex={0}>
                <table>
                  <caption className="visually-hidden">{view.similar.caption}</caption>
                  <thead>
                    <tr>
                      <th scope="col">{view.similar.rank}</th>
                      <th scope="col">{view.similar.name}</th>
                      <th scope="col">{view.similar.distance}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {similar.neighbours.map((neighbour) => {
                      const neighbourSlug = slugOf.get(neighbour.player_id);
                      return (
                        <tr key={neighbour.rank}>
                          <td className="num">{neighbour.rank}</td>
                          <th scope="row">
                            {neighbourSlug ? (
                              <Link href={routes.player(neighbourSlug)} prefetch={false}>
                                {neighbour.name}
                              </Link>
                            ) : (
                              neighbour.name
                            )}
                          </th>
                          <td className="num">
                            {formatNumber(neighbour.distance, 3)}
                            <span className="note small">{copy.site.estimate.point}</span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="small muted">{view.similar.distanceNote}</p>
              <MinSample n={similar.possessions} min={similar.min_possessions} unit={view.similar.unit} />
            </>
          ) : (
            <Hidden n={similar.possessions} min={similar.min_possessions} unit={view.similar.unit} />
          )}
          {similar.available ? <Check check={checks.get("embeddings")} /> : null}
          {descriptive}
        </MetricSection>
      </EvidenceScope>

      <SourceList sources={players.sources} />
    </div>
  );
}
