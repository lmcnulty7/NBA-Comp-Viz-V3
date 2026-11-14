import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import type { ReactNode } from "react";
import { EstimateValue } from "@/components/ui/EstimateValue";
import { SourceList } from "@/components/ui/SourceList";
import { copy, fill, label } from "@/lib/copy";
import { getServerSource } from "@/lib/data/server";
import { formatCount, formatNumber, formatPercent, formatSigned } from "@/lib/format";
import { routes } from "@/lib/site";
import type { MetricNote, PlayersFile } from "@/lib/types/contract";
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

/** The heading of one view, the model status the data gives for it, and its notes and sources. */
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
  return (
    <section className="section" aria-labelledby={metric} data-metric={metric}>
      <h2 id={metric}>{label(text.metrics, metric)}</h2>
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
          <SourceList sources={note.sources} />
        </div>
      ) : null}
    </section>
  );
}

function Hidden({ n, min, unit }: { n: number; min: number; unit: string }) {
  return (
    <p className="label muted" data-hidden="true">
      {fill(view.hidden, { n: formatCount(n), unit, min: formatCount(min) })}
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

export default async function PlayerPage({ params }: PageProps<"/prep/[player]">) {
  const { player: slug } = await params;
  const { players, player } = await load(slug);
  const slugOf = new Map(players.players.map((candidate) => [candidate.player_id, candidate.slug]));
  const { shot_making: shots, gravity, rebounding, absence_risk: absence, similar } = player;
  const baseline = players.shot_making_baseline;
  const descriptive = <p className="small">{view.descriptive}</p>;
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
      </header>

      <MetricSection metric="shot_making" players={players}>
        {shots.shown && shots.over_expected_per_100 ? (
          <dl className={styles.values} data-values="player">
            <Row name={view.shotMaking.value}>
              <EstimateValue estimate={shots.over_expected_per_100} digits={1} />
            </Row>
            {shots.baseline_error_per_100 != null ? (
              <Row name={view.shotMaking.baselineError}>
                <span className="num">{formatSigned(shots.baseline_error_per_100, 1)}</span>
                <span className="note">{copy.site.estimate.point}</span>
              </Row>
            ) : null}
            {shots.fg_pct != null ? (
              <Row name={view.shotMaking.fg}>
                <span className="num">{formatPercent(shots.fg_pct, 1)}</span>
              </Row>
            ) : null}
            {shots.expected_fg_pct != null ? (
              <Row name={view.shotMaking.expected}>
                <span className="num">{formatPercent(shots.expected_fg_pct, 1)}</span>
              </Row>
            ) : null}
            {shots.frozen_model_residual_per_100 != null ? (
              <Row name={view.shotMaking.residual}>
                <span className="num">{formatNumber(shots.frozen_model_residual_per_100, 1)}</span>
                <span className="note">{copy.site.estimate.point}</span>
              </Row>
            ) : null}
          </dl>
        ) : (
          <Hidden n={shots.attempts} min={shots.min_attempts} unit={view.shotMaking.unit} />
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
        </div>
      </MetricSection>

      <MetricSection metric="gravity" players={players}>
        {gravity.shown && gravity.gravity_ft ? (
          <dl className={styles.values} data-values="player">
            <Row name={view.gravity.value}>
              <EstimateValue estimate={gravity.gravity_ft} digits={2} />
            </Row>
          </dl>
        ) : (
          <Hidden n={gravity.possessions} min={gravity.min_possessions} unit={view.gravity.unit} />
        )}
        {descriptive}
      </MetricSection>

      <MetricSection metric="rebounding" players={players}>
        {rebounding.shown && rebounding.rate_per_100 ? (
          <dl className={styles.values} data-values="player">
            <Row name={view.rebounding.value}>
              <EstimateValue estimate={rebounding.rate_per_100} digits={1} />
            </Row>
            {rebounding.model_expected_per_100 != null ? (
              <Row name={view.rebounding.expected}>
                <span className="num">{formatNumber(rebounding.model_expected_per_100, 1)}</span>
                <span className="note">{copy.site.estimate.point}</span>
              </Row>
            ) : null}
          </dl>
        ) : (
          <Hidden n={rebounding.chances} min={rebounding.min_chances} unit={view.rebounding.unit} />
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
                  <span className="note">{view.descriptive}</span>
                </Row>
              ) : null}
              <Row name={view.absence.absent}>
                <span className="num">{formatCount(absence.absent)}</span>
              </Row>
            </dl>
            <p className="small">
              {absence.kind === "observed_only" ? <>{view.absence.observedOnly} </> : null}
              {view.absence.caution}
            </p>
          </>
        ) : (
          <Hidden n={absence.games} min={absence.min_games} unit={view.absence.unit} />
        )}
      </MetricSection>

      <MetricSection metric="similar" players={players}>
        {!similar.available ? (
          <p className="label muted">{view.similar.unavailable}</p>
        ) : similar.shown ? (
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
                          <Link href={routes.player(neighbourSlug)}>{neighbour.name}</Link>
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
        ) : (
          <Hidden n={similar.possessions} min={similar.min_possessions} unit={view.similar.unit} />
        )}
        {descriptive}
      </MetricSection>
    </div>
  );
}
